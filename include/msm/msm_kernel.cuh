#pragma once
#include "msm_common.cuh"

#define PARALLEL_DEGREE 128
#define PARALLEL_GROUP 5

template<typename affine_t, typename point_t>
__global__ __launch_bounds__(PARALLEL_DEGREE, PARALLEL_GROUP) void inner_bucket_sum(
    size_t n, uint32_t num_buckets, uint32_t parallel_degree,
    affine_t *points, point_t *bucket_sum, uint32_t *bucket_start, uint32_t *bucket_end, uint32_t *indices
) 
{
    uint32_t parallel_id = threadIdx.x;
    uint32_t bucket_id = blockIdx.x % num_buckets;
    uint32_t window_id = blockIdx.x / num_buckets;

    uint32_t bucket_start_idx = bucket_start[window_id * num_buckets + bucket_id];
    uint32_t bucket_end_idx = bucket_end[window_id * num_buckets + bucket_id];
    uint32_t stride = blockDim.x;

    point_t acc; acc.inf();
    for (uint32_t i = bucket_start_idx + parallel_id; i < bucket_end_idx; i += stride) {
        acc.add(points[indices[window_id * n + i]]);
    }

    bucket_sum[window_id * (num_buckets * parallel_degree) + bucket_id * parallel_degree + parallel_id] = acc;
}

template<typename affine_t, typename point_t, typename bucket_t>
__global__ __launch_bounds__(PARALLEL_DEGREE, PARALLEL_GROUP) void inner_bucket_sum_with_xyzz_as_medium(
    size_t n, uint32_t num_buckets, uint32_t parallel_degree,
    affine_t *points, point_t *bucket_sum, uint32_t *bucket_start, uint32_t *bucket_end, uint32_t *indices
) 
{
    uint32_t parallel_id = threadIdx.x;
    // ASSUME num_buckets = 256
    // uint32_t bucket_id = blockIdx.x & (num_buckets - 1);
    uint32_t window_id = blockIdx.x >> 8;

    uint32_t bucket_start_idx = bucket_start[blockIdx.x];
    uint32_t bucket_end_idx = bucket_end[blockIdx.x];
    uint32_t stride = blockDim.x;

    bucket_t acc; acc.inf();
    for (uint32_t i = bucket_start_idx + parallel_id; i < bucket_end_idx; i += stride) {
        uint32_t j = indices[window_id * n + i];
        acc.add(points[j]);
    }
    
    acc.to_jacobian();
    bucket_sum[blockIdx.x * parallel_degree + parallel_id] = *reinterpret_cast<point_t*>(&acc);
}

template<typename point_t>
__global__ void parallel_bucket_reduce(
    size_t num_buckets, uint32_t parallel_degree,
    point_t *bucket_sum
) 
{
    uint32_t window_id = blockIdx.x / num_buckets;
    uint32_t bucket_id = blockIdx.x % num_buckets;
    uint32_t lane_id = threadIdx.x;

    point_t acc; acc.inf();
    for (uint32_t i = lane_id; i < parallel_degree; i += 32) {
        acc.add(bucket_sum[window_id * (num_buckets * parallel_degree) + bucket_id * parallel_degree + i]);
    }

    point_t incr;
    for (uint32_t i = 1; i < 32; i <<= 1) {
        incr = custom_shfl_xor(acc, i);
        acc.add(incr);
    }

    if (lane_id == 0)
        bucket_sum[window_id * (num_buckets * parallel_degree) + bucket_id * parallel_degree] = acc;
}