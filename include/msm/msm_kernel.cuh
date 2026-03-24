#pragma once
#include "msm_common.cuh"

#include <cuda/pipeline>
#include <cooperative_groups.h>

#define BSR_BLK_SIZE 32
#define BSR_BLK_PER_SM 28

#define WRONG_FOR_SPPED 1

template<typename affine_t, typename point_t, typename bucket_t, size_t threads_unit = 1>
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

    for (uint32_t i = start + lane_id / threads_unit; i < end; i += 32 / threads_unit) {
        uint32_t j = indices[i] * threads_unit + (lane_id & (threads_unit - 1));
        incr = points[j];
#ifdef WRONG_FOR_SPEED
        acc.pacc(incr);
#else
        acc.add(incr);
#endif
    }

    acc.to_jacobian();
    bucket_sum[bucket_id * 32 + lane_id] = reinterpret_cast<point_t*>(&acc)[0];
}

template<typename affine_t, typename point_t, typename bucket_t, size_t threads_unit = 1>
__global__ __launch_bounds__(BSR_BLK_SIZE, BSR_BLK_PER_SM) 
void bucket_segemented_reduction_increment(uint32_t *bucket_off, uint32_t *indices, affine_t *points, point_t *bucket_sum) 
{
    uint32_t bucket_id = blockIdx.x;
    uint32_t lane_id = threadIdx.x;

    uint32_t start = bucket_off[bucket_id];
    uint32_t end = bucket_off[bucket_id + 1];

    bucket_t acc; acc.inf();
    affine_t incr;

    for (uint32_t i = start + lane_id / threads_unit; i < end; i += 32 / threads_unit) {
        uint32_t j = indices[i] * threads_unit + (lane_id & (threads_unit - 1));
        incr = points[j];
#ifdef WRONG_FOR_SPEED
        acc.pacc(incr);
#else
        acc.add(incr);
#endif
    }

    acc.to_jacobian();
    bucket_sum[bucket_id * 32 + lane_id].add(reinterpret_cast<point_t*>(&acc)[0]);
}

template<typename affine_t, typename point_t, typename bucket_t, size_t threads_unit = 1>
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

    for (uint32_t i = start + lane_id / threads_unit; i < end; i += 32 / threads_unit) {
        uint32_t j = indices[i] * threads_unit + (lane_id & (threads_unit - 1));
        incr = points[j];
#ifdef WRONG_FOR_SPEED
        acc.pacc(incr);
#else
        acc.add(incr);
#endif
    }

    acc.to_jacobian();
    bucket_sum[blockIdx.x * 32 + lane_id] = reinterpret_cast<point_t*>(&acc)[0];
}