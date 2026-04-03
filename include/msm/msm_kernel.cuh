#pragma once
#include "msm_common.cuh"

#include <cuda/pipeline>
#include <cooperative_groups.h>

#define WRONG_FOR_SPEED 1

#define GA_BLK_SIZ 256

#define IBA_BLK_PER_SM 3
template<typename AffT, typename XYZZT>
__global__ __launch_bounds__(GA_BLK_SIZ, IBA_BLK_PER_SM) void intra_bucket_accumulation(
    uint32_t *bucket_off, uint32_t *indices, AffT *points, AffT *lifted_points, XYZZT *bucket_sum_WWR,
    uint32_t buckets_count, uint32_t windows_count, uint32_t half_windows_count, uint32_t last_window_buckets_count, uint32_t scale,
    uint32_t *g_task_id
) {
    uint32_t valid_buckets_count = buckets_count - 1;

    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    // static __device__ uint32_t g_task_id;
    __shared__ AffT shr[2][GA_BLK_SIZ];

    const uint32_t warp_id = g.thread_rank() / 32;
    const uint32_t lane_id = g.thread_rank() % 32;
    const uint32_t thread_id = g.thread_rank();
    const uint32_t buckets_per_block = GA_BLK_SIZ / 32;

    uint32_t window_id = g.group_index().x % windows_count;
    AffT *ptr = (window_id < half_windows_count) ? points : lifted_points;

    __shared__ uint32_t task_base_id;
    cg::invoke_one(g, [&] () { task_base_id = atomicAdd(&g_task_id[window_id], buckets_per_block); });
    g.sync();

    uint32_t bucket_id = task_base_id + warp_id;
    uint32_t cur_buckets_count = (window_id == windows_count - 1) ? last_window_buckets_count - 1 : valid_buckets_count;
    while (bucket_id < cur_buckets_count) {
        uint32_t start = bucket_off[window_id * buckets_count + bucket_id];
        uint32_t end = bucket_off[window_id * buckets_count + bucket_id + 1];

        XYZZT acc; acc.inf();
        AffT incr;

        uint32_t stage = 0, i = start + lane_id;
        if (i < end) {
            pipeline.producer_acquire();
            uint32_t j = indices[window_id * scale + i];
            cuda::memcpy_async(&shr[stage][thread_id], &ptr[j], sizeof(AffT), pipeline);
            pipeline.producer_commit();
        }

        for (; i < end; i += 32) {
            uint32_t nxt_i = i + 32;
            if (nxt_i < end) {
                pipeline.producer_acquire();
                uint32_t j = indices[window_id * scale + nxt_i];
                cuda::memcpy_async(&shr[1 - stage][thread_id], &ptr[j], sizeof(AffT), pipeline);
                pipeline.producer_commit();
            }
            
            cuda::pipeline_consumer_wait_prior<1>(pipeline);
            incr = shr[stage][thread_id];
#ifdef WRONG_FOR_SPEED
            acc.pacc(incr);
#else
            acc.add(incr);
#endif
            pipeline.consumer_release();

            stage = 1 - stage;
        }

        bucket_sum_WWR[window_id * buckets_count * 32 + bucket_id * 32 + lane_id] = acc;

        if (lane_id == 0) bucket_id = atomicAdd(&g_task_id[window_id], 1);
        bucket_id = __shfl_sync(0xffffffff, bucket_id, 0);
    }
}

#define WR_BLK_PER_SM 2
template<typename XYZZT>
__global__ __launch_bounds__(GA_BLK_SIZ, WR_BLK_PER_SM) void warp_reduce(
    XYZZT *buckets_sum_WWR, XYZZT *buckets_sum,
    uint32_t buckets_count, uint32_t windows_count, uint32_t half_windows_count, uint32_t last_window_buckets_count
) {
    uint32_t valid_buckets_count = buckets_count - 1;

    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();
    
    const uint32_t thread_id = g.thread_rank();

    __shared__ XYZZT buffer[GA_BLK_SIZ];

    uint32_t valid_thread_count = valid_buckets_count * half_windows_count;
    if (thread_id >= valid_thread_count) return;

    uint32_t bucket_id = thread_id % valid_buckets_count;
    uint32_t window_id = thread_id / valid_buckets_count;

    XYZZT acc, incr; acc.inf();
    for (uint32_t i = 0; i < 32; i++) acc.add(buckets_sum_WWR[window_id * buckets_count * 32 + bucket_id * 32 + i]);

    // if (lane_id == 0) buckets_sum[window_id * buckets_count + bucket_id] = acc;

    window_id += half_windows_count;
    if (window_id >= windows_count || (window_id == windows_count - 1 && bucket_id >= last_window_buckets_count - 1)) {
        buckets_sum[(window_id - half_windows_count) * buckets_count + bucket_id] = acc;
    } else {
        uint32_t blk_thread_id = thread_id % GA_BLK_SIZ;
        buffer[blk_thread_id] = acc;

        acc.inf();
        for (uint32_t i = 0; i < 32; i++) acc.add(buckets_sum_WWR[window_id * buckets_count * 32 + bucket_id * 32 + i]);

        incr = buffer[blk_thread_id];
        acc.add(incr);
        buckets_sum[(window_id - half_windows_count) * buckets_count + bucket_id] = acc;
    }
}

template<typename ProjT, typename XYZZT>
__global__ __launch_bounds__(GA_BLK_SIZ) void bucket_reduce(
    XYZZT *buckets_sum, ProjT *windows_sum,
    uint32_t buckets_count, uint32_t half_windows_count, uint32_t last_window_buckets_count
) {
    uint32_t valid_buckets_count = buckets_count - 1;

    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();

    const uint32_t window_id = g.group_index().x;
    const uint32_t lane_id = g.thread_rank();

    uint32_t buckets_per_thread = buckets_count / GA_BLK_SIZ;
    buckets_per_thread += (buckets_per_thread == 0);

    int start = lane_id * buckets_per_thread;
    int end = (lane_id + 1) * buckets_per_thread;
    end = (end < valid_buckets_count) ? end : valid_buckets_count;

    XYZZT sum, scan; sum.inf(), scan.inf();
    for (int i = end - 1; i >= start; i--)  {
        sum.add(buckets_sum[window_id * buckets_count + i]);
        scan.add(sum);
    }

    __shared__ XYZZT shmem[GA_BLK_SIZ];
    shmem[lane_id] = scan;
    g.sync();

    for (int i = 1; i < GA_BLK_SIZ; i <<= 1) {
        scan = shmem[lane_id ^ i];
        g.sync();
        
        shmem[lane_id].add(scan);
        g.sync();
    }

    scan = shmem[lane_id];
    scan.to_jacobian();
    g.sync();

    cg::invoke_one(g, [&] __device__ () { windows_sum[window_id] = *reinterpret_cast<ProjT*>(&scan); });
    g.sync();

    shmem[GA_BLK_SIZ - 1 - lane_id] = sum;
    g.sync();

    for (int i = 1; i < GA_BLK_SIZ; i <<= 1) {
        if (lane_id >= i) scan = shmem[lane_id - i];
        else scan.inf();
        g.sync();

        shmem[lane_id].add(scan);
        g.sync();
    }

    cg::invoke_one(g, [&] __device__ () { shmem[GA_BLK_SIZ - 1].inf(); });
    g.sync();

    for (int i = 1; i < GA_BLK_SIZ; i <<= 1) {
        scan = shmem[lane_id ^ i];
        g.sync();
        
        shmem[lane_id].add(scan);
        g.sync();
    }

    sum = shmem[lane_id];
    sum.to_jacobian();
    g.sync();

    cg::invoke_one(g, [&] __device__ () {
        ProjT res = *reinterpret_cast<ProjT*>(&sum);
        for (int i = 1; i < buckets_per_thread; i <<= 1) res.dbl();
        windows_sum[window_id].add(res);
    });
}