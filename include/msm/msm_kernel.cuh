#pragma once
#include "msm_common.cuh"

#define PARALLEL_DEGREE 32
#define PARALLEL_GROUP 28

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
    size_t n, uint32_t num_windows, uint32_t num_buckets, uint32_t parallel_degree, uint32_t lst_valid_buckets,
    affine_t *points, point_t *bucket_sum, uint32_t *bucket_start, uint32_t *bucket_end, uint32_t *indices
) 
{
    uint32_t parallel_id = threadIdx.x;
    uint32_t window_id = blockIdx.x / num_buckets;
    uint32_t bucket_start_idx, bucket_end_idx;

    // ASSUME sizeof(field_t) = 8 * sizeof(uint32_t)
    __shared__ uint32_t shmem[8 * PARALLEL_DEGREE];

    if (window_id < num_windows - 1) {
        bucket_start_idx = bucket_start[blockIdx.x];
        bucket_end_idx = bucket_end[blockIdx.x];
    } else {
        // ASSUME lst_valid_buckets is a power of 2
        uint32_t bucket_id = blockIdx.x % num_buckets;
        bucket_start_idx = bucket_start[window_id * num_buckets + bucket_id % lst_valid_buckets];
        bucket_end_idx = bucket_end[window_id * num_buckets + bucket_id % lst_valid_buckets];
         
        uint32_t slice_id = bucket_id / lst_valid_buckets;
        uint32_t slice = num_buckets / lst_valid_buckets; 
        
        uint32_t bucket_length = (bucket_end_idx - bucket_start_idx) / slice;
        bucket_start_idx += slice_id * bucket_length;
        if (slice_id < slice - 1) bucket_end_idx = bucket_start_idx + bucket_length;
    }

    uint32_t stride = blockDim.x;
    bucket_t acc; acc.inf();
    for (uint32_t i = bucket_start_idx + parallel_id; i < bucket_end_idx; i += stride) {
        uint32_t j = indices[window_id * n + i];
        affine_t incr = points[j];
        acc.pacc_with_shmem(incr, shmem + 8 * parallel_id);
        // acc.pacc(incr);
    }
    
    acc.to_jacobian();
    bucket_sum[blockIdx.x * parallel_degree + parallel_id] = *reinterpret_cast<point_t*>(&acc);
}

#define REDUCE_SIZ 128
template<typename point_t>
__global__ void parallel_bucket_reduce(
    size_t num_buckets, uint32_t parallel_degree,
    point_t *bucket_sum
) 
{
    uint32_t para_id = blockIdx.x * blockDim.x + threadIdx.x;

    point_t acc; acc.inf();
    for (uint32_t i = 0; i < parallel_degree; i++) {
        acc.add(bucket_sum[para_id * parallel_degree + i]);
    }
    
    bucket_sum[para_id * parallel_degree] = acc;
}