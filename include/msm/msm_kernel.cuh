#pragma once
#include "msm_common.cuh"

#include <cooperative_groups.h>

#define WRONG_FOR_SPEED 1

#define GA_BLK_SIZ 256

#define IBA_BLK_PER_SM 3
template<typename AffT, typename XYZZT>
__global__ __launch_bounds__(GA_BLK_SIZ, IBA_BLK_PER_SM) void intra_bucket_accumulation(
    uint32_t *bucket_off, uint32_t *indices, AffT *points, XYZZT *bucket_sum_WWR,
    uint32_t buckets_count, uint32_t windows_count, uint32_t scale, uint32_t last_window_buckets_count,
    uint32_t lo_windows_count, uint32_t lo_window_id,
    uint32_t *g_task_id
) {
    uint32_t valid_buckets_count = buckets_count - 1;

    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();

    const uint32_t thread_id = g.thread_rank();
    const uint32_t warp_id = thread_id / 32;
    const uint32_t lane_id = thread_id % 32;
    
    const uint32_t buckets_per_block = GA_BLK_SIZ / 32;

    __shared__ uint32_t task_base_id;
    cg::invoke_one(g, [&] () { task_base_id = atomicAdd(&g_task_id[lo_window_id], buckets_per_block); });
    g.sync();

    uint32_t bucket_id = task_base_id + warp_id;
    while (bucket_id < valid_buckets_count) {
        XYZZT acc; acc.inf();
        AffT *ptr = points;

        uint32_t cur_windows_count = windows_count - (bucket_id >= last_window_buckets_count - 1);
        for (uint32_t window_id = lo_window_id; window_id < cur_windows_count; window_id += lo_windows_count, ptr += scale) {
            uint32_t start = bucket_off[window_id * buckets_count + bucket_id];
            uint32_t end = bucket_off[window_id * buckets_count + bucket_id + 1];
            
            AffT incr;
            for (uint32_t i = start + lane_id; i < end; i += 32) {
                uint32_t j = indices[window_id * scale + i];
                incr = ptr[j];
#ifdef WRONG_FOR_SPEED
                acc.pacc(incr);
#else
                acc.add(incr);
#endif
            }
        }
        
        bucket_sum_WWR[lo_window_id * buckets_count * 32 + bucket_id * 32 + lane_id] = acc;

        if (lane_id == 0) bucket_id = atomicAdd(&g_task_id[lo_window_id], 1);
        bucket_id = __shfl_sync(0xffffffff, bucket_id, 0);
    }
}

#define WR_BLK_PER_SM 2
template<typename XYZZT>
__global__ __launch_bounds__(GA_BLK_SIZ, WR_BLK_PER_SM) void warp_reduce(
    XYZZT *buckets_sum_WWR, XYZZT *buckets_sum,
    uint32_t buckets_count, uint32_t windows_count, uint32_t lo_windows_count, uint32_t last_window_buckets_count
) {
    uint32_t valid_buckets_count = buckets_count - 1;

    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();
    const uint32_t thread_id = g.thread_rank();

    uint32_t valid_thread_count = lo_windows_count * valid_buckets_count;
    if (thread_id >= valid_thread_count) return;

    uint32_t bucket_id = thread_id % valid_buckets_count;
    uint32_t window_id = thread_id / valid_buckets_count;

    XYZZT acc, incr; acc.inf();
    for (uint32_t i = 0; i < 32; i++) acc.add(buckets_sum_WWR[window_id * buckets_count * 32 + bucket_id * 32 + i]);
    buckets_sum[window_id * buckets_count + bucket_id] = acc;
}

template<typename ProjT, typename XYZZT, size_t BLK_SIZ>
__global__ __launch_bounds__(BLK_SIZ) void bucket_reduce(
    XYZZT *buckets_sum, ProjT *windows_sum,
    uint32_t buckets_count, uint32_t last_window_buckets_count
) {
    uint32_t valid_buckets_count = buckets_count - 1;

    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();

    const uint32_t window_id = g.group_index().x;
    const uint32_t lane_id = g.thread_rank();

    uint32_t buckets_per_thread = buckets_count / BLK_SIZ;
    buckets_per_thread += (buckets_per_thread == 0);

    int start = lane_id * buckets_per_thread;
    int end = (lane_id + 1) * buckets_per_thread;
    end = (end < valid_buckets_count) ? end : valid_buckets_count;

    XYZZT sum, scan; sum.inf(), scan.inf();
    for (int i = end - 1; i >= start; i--)  {
        sum.add(buckets_sum[window_id * buckets_count + i]);
        scan.add(sum);
    }

    __shared__ XYZZT shmem[BLK_SIZ];
    shmem[lane_id] = scan;
    g.sync();

    for (int i = 1; i < BLK_SIZ; i <<= 1) {
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

    shmem[BLK_SIZ - 1 - lane_id] = sum;
    g.sync();

    for (int i = 1; i < BLK_SIZ; i <<= 1) {
        if (lane_id >= i) scan = shmem[lane_id - i];
        else scan.inf();
        g.sync();

        shmem[lane_id].add(scan);
        g.sync();
    }

    cg::invoke_one(g, [&] __device__ () { shmem[BLK_SIZ - 1].inf(); });
    g.sync();

    for (int i = 1; i < BLK_SIZ; i <<= 1) {
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