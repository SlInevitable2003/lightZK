#pragma once
#include "msm_common.cuh"

#include <cuda/pipeline>
#include <cooperative_groups.h>

#define BSR_BLK_SIZE 32
#define BSR_BLK_PER_SM 28
#define SWZ 2

template<typename affine_t, typename point_t, typename bucket_t>
__global__ __launch_bounds__(BSR_BLK_SIZE, BSR_BLK_PER_SM) 
void bucket_segemented_reduction(uint32_t *bucket_off, uint32_t *indices, affine_t *points, point_t *bucket_sum) 
{
    namespace cg = cooperative_groups;

    cg::thread_block g = cg::this_thread_block();
    uint32_t bucket_id = g.group_index().x;
    uint32_t lane_id = g.thread_rank();

    uint32_t start = bucket_off[bucket_id];
    uint32_t end = bucket_off[bucket_id + 1];

    bucket_t acc; acc.inf();
    affine_t incr;

    __shared__ uint32_t shmem[8 * BSR_BLK_SIZE];
    uint32_t *ptr = shmem + 8 * ((lane_id + SWZ * lane_id) & 31);

    for (uint32_t i = start + lane_id; i < end; i += 32) {
        uint32_t j = indices[i];
        incr = points[j];
        acc.pacc_with_shmem(incr, ptr);
    }

    acc.to_jacobian();
    bucket_sum[bucket_id * 32 + lane_id] = reinterpret_cast<point_t*>(&acc)[0];
}

template<typename affine_t, typename point_t, typename bucket_t>
__global__ __launch_bounds__(BSR_BLK_SIZE, BSR_BLK_PER_SM) 
void bucket_segemented_reduction_increment(uint32_t *bucket_off, uint32_t *indices, affine_t *points, point_t *bucket_sum) 
{
    uint32_t bucket_id = blockIdx.x;
    uint32_t lane_id = threadIdx.x;

    uint32_t start = bucket_off[bucket_id];
    uint32_t end = bucket_off[bucket_id + 1];

    bucket_t acc; acc.inf();
    affine_t incr;

    __shared__ uint32_t shmem[8 * BSR_BLK_SIZE];
    uint32_t *ptr = shmem + 8 * ((lane_id + SWZ * lane_id) & 31);

    for (uint32_t i = start + lane_id; i < end; i += 32) {
        uint32_t j = indices[i];
        incr = points[j];
        acc.pacc_with_shmem(incr, ptr);
    }

    acc.to_jacobian();
    bucket_sum[bucket_id * 32 + lane_id].add(reinterpret_cast<point_t*>(&acc)[0]);
}

template<typename affine_t, typename point_t, typename bucket_t>
__global__ __launch_bounds__(BSR_BLK_SIZE, BSR_BLK_PER_SM) 
void bucket_segemented_reduction_last_window(uint32_t *bucket_off, uint32_t *indices, affine_t *points, point_t *bucket_sum, uint32_t warps_per_bucket) 
{
    uint32_t bucket_id = blockIdx.x / warps_per_bucket, warp_off = blockIdx.x % warps_per_bucket;
    uint32_t lane_id = threadIdx.x;

    uint32_t start = bucket_off[bucket_id];
    uint32_t end = bucket_off[bucket_id + 1];
    uint32_t len = end - start;
    uint32_t chunk = (len + warps_per_bucket - 1) / warps_per_bucket;

    start = start + warp_off * chunk;
    uint32_t end_ = start + chunk;
    end = (end_ < end) ? end_ : end;

    bucket_t acc; acc.inf();
    affine_t incr;

    __shared__ uint32_t shmem[16 * BSR_BLK_SIZE];

    for (uint32_t i = start + lane_id; i < end; i += 32) {
        uint32_t j = indices[i];
        incr = points[j];
        acc.pacc_with_shmem(incr, shmem + 16 * lane_id);
    }

    acc.to_jacobian();
    bucket_sum[blockIdx.x * 32 + lane_id] = reinterpret_cast<point_t*>(&acc)[0];
}