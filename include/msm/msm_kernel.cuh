#pragma once
#include "msm_common.cuh"

#define BSR_BLK_SIZE 32
#define BSR_BLK_PER_SM 28

template<typename affine_t, typename point_t, typename bucket_t>
__global__ __launch_bounds__(BSR_BLK_SIZE, BSR_BLK_PER_SM) 
void bucket_segemented_reduction(uint32_t *bucket_off, uint32_t *indices, affine_t *points, point_t *bucket_sum) 
{
    uint32_t bucket_id = blockIdx.x;
    uint32_t lane_id = threadIdx.x;

    uint32_t start = bucket_off[bucket_id];
    uint32_t end = bucket_off[bucket_id + 1];

    bucket_t acc; acc.inf();
    affine_t incr;

    for (uint32_t i = start + lane_id; i < end; i += 32) {
        uint32_t j = indices[i];
        incr = points[j];
        acc.pacc(incr);
    }

    acc.to_jacobian();
    bucket_sum[bucket_id * 32 + lane_id] = reinterpret_cast<point_t*>(&acc)[0];
}