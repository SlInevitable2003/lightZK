#pragma once
#include "fields/bucket.cuh"
#include "msm/msm_kernel.cuh"
#include "utils.cuh"

#include <vector>

template <typename FieldT, typename AffT, typename ProjT, typename XYZZT,
          typename HostFT, typename HostPT, size_t instances = 1>
class MSMContext {
    ProjT infty;

    AffT *bases[instances], *lifted_bases[instances];

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
    size_t half_windows_count;

    static const size_t degree = XYZZT::degree;

    MSMContext(size_t scale, size_t window_bits)
        : scale(scale), window_bits(window_bits),
          windows_count(ceil_div(FieldT::nbits, window_bits)), buckets_count(1 << window_bits), half_windows_count(ceil_div(windows_count, 2))
    {
        for (size_t i = 0; i < instances; i++) {
            arena.register_alloc(bases[i], degree * scale);
            arena.register_alloc(lifted_bases[i], degree * scale);
        }
        arena.register_alloc(buckets_sum_WWR, windows_count * buckets_count * 32);
        arena.register_alloc(buckets_sum, degree * half_windows_count * buckets_count);
        arena.register_alloc(windows_sum, degree * half_windows_count);
        arena.register_alloc(g_task_id, windows_count);
        arena.commit("MSMContext");

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
    }

    void load_bases(const HostPT *host_bases, size_t instance_id = 0)
    {
        std::vector<AffT> buffer(degree * scale);
        memset(buffer.data(), 0, buffer.size() * sizeof(AffT));
        #pragma omp parallel for
        for (size_t i = 0; i < scale; i++) if (!host_bases[i].is_zero()) memcpy(&buffer[degree * i], &host_bases[i], degree * sizeof(AffT));
        cudaMemcpy(bases[instance_id], buffer.data(), degree * scale * sizeof(AffT), cudaMemcpyHostToDevice);

        // if (is_g2) {
        //     static_assert(sizeof(AffT) >= 2 * 4 * sizeof(uint32_t));
        //     kernel<<<ceil_div(scale, 256), 256>>>([=] __device__ (AffT *points) {
        //         namespace cg = cooperative_groups;
        //         cg::grid_group g = cg::this_grid();
        //         int idx = g.thread_rank();
        //         if (idx >= scale) return;

        //         uint4 *base = reinterpret_cast<uint4*>(&points[idx << 1]);
        //         const int stride = sizeof(AffT) / (2 * sizeof(uint4));
        //         uint4 buffer[stride];
        //         for (int i = 0; i < stride; i++) buffer[i] = base[stride + i];
        //         for (int i = 0; i < stride; i++) base[stride + i] = base[stride * 2+ i];
        //         for (int i = 0; i < stride; i++) base[stride * 2 + i] = buffer[i];
        //     }, bases[instance_id]);
        //     cudaDeviceSynchronize();
        //     CUDA_CHECK(cudaGetLastError());
        // }

        cudaMemcpy(lifted_bases[instance_id], bases[instance_id], degree * scale * sizeof(AffT), cudaMemcpyDeviceToDevice);
        size_t dbl_off = half_windows_count * window_bits;

        size_t scale = this->scale;
        kernel<<<ceil_div(scale, GA_BLK_SIZ), GA_BLK_SIZ>>>([=] __device__ (AffT *points) {
            namespace cg = cooperative_groups;
            cg::grid_group g = cg::this_grid();
            int idx = g.thread_rank();
            if (idx >= scale) return;

            ProjT p = points[idx];
            for (int i = 0; i < dbl_off; i++) p.dbl();
            points[idx] = p;
        }, lifted_bases[instance_id]);


        // if (!is_g2) {
        //     thrust::for_each(thrust::device, lifted_bases[instance_id], lifted_bases[instance_id] + scale, [=] __device__ (AffT &x) {
        //         ProjT p = x;
        //         for (int i = 0; i < dbl_off; i++) p.dbl();
        //         x = p;
        //     });
        // } else {
        //     kernel<<<ceil_div(scale, 32), 64>>>([=] __device__ (AffT *points) {
        //         namespace cg = cooperative_groups;
        //         cg::grid_group g = cg::this_grid();
        //         int idx = g.thread_rank();
        //         if (idx >= 2 * scale) return;

        //         ProjT p = points[idx];
        //         for (int i = 0; i < dbl_off; i++) p.dbl();
        //         points[idx] = p;
        //     }, lifted_bases[instance_id]);
        //     cudaDeviceSynchronize();
        //     CUDA_CHECK(cudaGetLastError());
        // }

    }

public:
    void msm(BucketContext<FieldT, HostFT>& bkt_ctx, HostPT *result = 0, size_t instance_id = 0)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        size_t last_window_bits = FieldT::nbits - (windows_count - 1) * window_bits;
        size_t last_window_buckets_count = 1 << last_window_bits;
        size_t valid_buckets_count = buckets_count - 1;

        cudaMemset(g_task_id, 0, windows_count * sizeof(uint32_t));
        (intra_bucket_accumulation<AffT, XYZZT>)<<<gpu.sm_count * IBA_BLK_PER_SM, GA_BLK_SIZ, 0, stream>>>
            (bkt_ctx.buckets_off, bkt_ctx.indices_as_vals, bases[instance_id], lifted_bases[instance_id], buckets_sum_WWR, buckets_count, windows_count, half_windows_count, last_window_buckets_count, scale, g_task_id);

        warp_reduce<<<ceil_div(half_windows_count * valid_buckets_count, GA_BLK_SIZ), GA_BLK_SIZ, 0, stream>>>
            (buckets_sum_WWR, buckets_sum, buckets_count, windows_count, half_windows_count, last_window_buckets_count);

        bucket_reduce<<<half_windows_count, GA_BLK_SIZ, 0, stream>>>
            (buckets_sum, windows_sum, buckets_count, half_windows_count, last_window_buckets_count);

        cudaStreamSynchronize(stream);
        CUDA_CHECK(cudaGetLastError());
        cudaStreamDestroy(stream);

        if (result) {
            std::vector<HostPT> buffer(half_windows_count);
            cudaMemcpy(buffer.data(), windows_sum, degree * half_windows_count * sizeof(ProjT), cudaMemcpyDeviceToHost);
            HostPT res = HostPT::zero();
            for (int k = half_windows_count - 1; k >= 0; k--) {
                for (int l = 0; l < window_bits; l++) res = res.dbl();
                res = res + buffer[k];
            }
            *result = res;
        }
    }
};
