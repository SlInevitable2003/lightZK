#pragma once
#include "fields/bucket.cuh"
#include "msm/msm_kernel.cuh"
#include "utils.cuh"

#include <vector>

template <typename FieldT, typename AffT, typename ProjT, typename XYZZT, typename HostFT, typename HostPT>
class LargeMSMContext {
    ProjT infty;

    AffT *bases_pinned = nullptr;
    AffT *bases_ping, *bases_pong;

    XYZZT *buckets_sum_WWR;
    XYZZT *buckets_sum;
    ProjT *windows_sum;

    uint32_t *g_task_id;

    TypedGpuArena arena;
    GPUConfig gpu;

public:
    size_t scale;
    size_t window_bits;
    size_t windows_count, buckets_count;
    size_t elastic, tile_count;
    
    size_t lo_windows_count;

    LargeMSMContext(size_t scale_, size_t window_bits_, size_t elastic_, size_t tile_count_)
        : scale(scale_), window_bits(window_bits_),
        windows_count(ceil_div(FieldT::nbits, window_bits_)), buckets_count(1 << window_bits_),
        lo_windows_count(ceil_div(windows_count, elastic_)),
        elastic(elastic_), tile_count(tile_count_)
    {
        arena.register_alloc(bases_ping, tile_count * scale);
        arena.register_alloc(bases_pong, tile_count * scale);
        arena.register_alloc(buckets_sum_WWR, lo_windows_count * buckets_count * 32);
        arena.register_alloc(buckets_sum, lo_windows_count * buckets_count);
        arena.register_alloc(windows_sum, lo_windows_count);
        arena.register_alloc(g_task_id, lo_windows_count); // 可以共用1个g_task_id，但就这么写吧
        arena.commit("MSMContext");

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaHostAlloc(&bases_pinned, elastic * scale * sizeof(AffT), cudaHostAllocDefault));
    }

    ~LargeMSMContext() { if (bases_pinned) cudaFreeHost(bases_pinned); }

    void load_bases(const HostPT *host_bases)
    {
        memset(bases_pinned, 0, scale * sizeof(AffT));
        #pragma omp parallel for
        for (size_t i = 0; i < scale; i++) if (!host_bases[i].is_zero()) memcpy(&bases_pinned[i], &host_bases[i], sizeof(AffT));
        cudaMemcpy(bases_ping, bases_pinned, scale * sizeof(AffT), cudaMemcpyHostToDevice);
        
        for (size_t i = 1; i < elastic; i++) {
            AffT *bases_in = (i & 1) ? bases_ping : bases_pong;
            AffT *bases_out = (i & 1) ? bases_pong : bases_ping;
            cudaMemcpy(bases_out, bases_in, scale * sizeof(AffT), cudaMemcpyDeviceToDevice);

            size_t scale = this->scale;
            size_t dbl_off = lo_windows_count * window_bits;
            kernel<<<ceil_div(scale, GA_BLK_SIZ), GA_BLK_SIZ>>>([=] __device__ (AffT *points) {
                namespace cg = cooperative_groups;
                cg::grid_group g = cg::this_grid();
                int idx = g.thread_rank();
                if (idx >= scale) return;

                ProjT p = points[idx];
                for (int i = 0; i < dbl_off; i++) p.dbl();
                points[idx] = p;
            }, bases_out);

            cudaMemcpy(bases_pinned + i * scale, bases_out, scale * sizeof(AffT), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
        }
    }

public:
    void msm(BucketContext<FieldT, HostFT>& bkt_ctx, HostPT *result = 0)
    {
        size_t last_window_bits = (FieldT::nbits % window_bits != 0) ? (FieldT::nbits % window_bits) : window_bits;
        size_t last_window_buckets_count = 1 << last_window_bits;
        size_t valid_buckets_count = buckets_count - 1;

        cudaStream_t stream_copy, stream_comp;
        cudaStreamCreate(&stream_copy);
        cudaStreamCreate(&stream_comp);

        cudaEvent_t evt_ping_free, evt_pong_free;
        cudaEventCreate(&evt_ping_free);
        cudaEventCreate(&evt_pong_free);

        cudaEventRecord(evt_ping_free, stream_comp);
        cudaEventRecord(evt_pong_free, stream_comp);

        cudaEvent_t evt_copy_ready;
        cudaEventCreate(&evt_copy_ready);

        size_t num_tile_groups = ceil_div(elastic, tile_count);

        size_t copy_start_0 = 0;
        size_t num_copies_0 = std::min(tile_count, elastic);
        cudaMemcpyAsync(bases_ping, bases_pinned + copy_start_0 * scale, num_copies_0 * scale * sizeof(AffT), cudaMemcpyHostToDevice, stream_copy);
        cudaEventRecord(evt_copy_ready, stream_copy);

        for (size_t tg = 0; tg < num_tile_groups; tg++) {
            bool is_ping = (tg % 2 == 0);
            AffT *cur_bases = is_ping ? bases_ping : bases_pong;
            cudaEvent_t evt_my_free = is_ping ? evt_ping_free : evt_pong_free;

            size_t copy_start = tg * tile_count;
            size_t copy_end = std::min(copy_start + tile_count, elastic);
            size_t num_copies = copy_end - copy_start;

            cudaStreamWaitEvent(stream_comp, evt_copy_ready, 0);
            cudaMemsetAsync(g_task_id, 0, lo_windows_count * sizeof(uint32_t), stream_comp);

            IBAWriteMode mode = (tg == 0) ? IBAWriteMode::OVERWRITE : IBAWriteMode::ACCUMULATE;
            for (size_t i = 0; i < lo_windows_count; i++) {
                intra_bucket_accumulation_tiled<AffT, XYZZT><<<gpu.sm_count * IBA_BLK_PER_SM, GA_BLK_SIZ, 0, stream_comp>>>(
                    bkt_ctx.buckets_off, bkt_ctx.indices_as_vals, cur_bases, buckets_sum_WWR,
                    buckets_count, windows_count, scale, last_window_buckets_count,
                    lo_windows_count, i,
                    copy_start, num_copies, mode,
                    g_task_id
                );
            }

            cudaEventRecord(evt_my_free, stream_comp);

            if (tg + 1 < num_tile_groups) {
                bool next_is_ping = ((tg + 1) % 2 == 0);
                AffT *next_bases = next_is_ping ? bases_ping : bases_pong;
                cudaEvent_t evt_next_free = next_is_ping ? evt_ping_free : evt_pong_free;

                size_t next_start = (tg + 1) * tile_count;
                size_t next_end = std::min(next_start + tile_count, elastic);
                size_t next_num = next_end - next_start;

                cudaStreamWaitEvent(stream_copy, evt_next_free, 0);
                cudaMemcpyAsync(next_bases, bases_pinned + next_start * scale, next_num * scale * sizeof(AffT), cudaMemcpyHostToDevice, stream_copy);
                cudaEventRecord(evt_copy_ready, stream_copy);
            }
        }

        cudaStreamSynchronize(stream_comp);
        CUDA_CHECK(cudaGetLastError());

        warp_reduce<<<ceil_div(lo_windows_count * valid_buckets_count, GA_BLK_SIZ),
                    GA_BLK_SIZ>>>(
            buckets_sum_WWR, buckets_sum, buckets_count, windows_count,
            lo_windows_count, last_window_buckets_count);

        const size_t blk_siz = GA_BLK_SIZ / XYZZT::degree;
        (bucket_reduce<ProjT, XYZZT, blk_siz>)
            <<<lo_windows_count, blk_siz>>>(
                buckets_sum, windows_sum, buckets_count, last_window_buckets_count);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        cudaStreamDestroy(stream_copy);
        cudaStreamDestroy(stream_comp);
        cudaEventDestroy(evt_ping_free);
        cudaEventDestroy(evt_pong_free);
        cudaEventDestroy(evt_copy_ready);

        if (result) {
            std::vector<HostPT> buffer(lo_windows_count);
            cudaMemcpy(buffer.data(), windows_sum,
                    lo_windows_count * sizeof(ProjT), cudaMemcpyDeviceToHost);
            HostPT res = HostPT::zero();
            for (int k = lo_windows_count - 1; k >= 0; k--) {
                for (size_t l = 0; l < window_bits; l++) res = res.dbl();
                res = res + buffer[k];
            }
            *result = res;
        }
    }

};
