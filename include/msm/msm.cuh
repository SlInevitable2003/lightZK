#pragma once
#include "fields/bucket.cuh"
#include "msm/msm_kernel.cuh"
#include "utils.cuh"

#include <vector>

template <typename FieldT, typename AffT, typename ProjT, typename XYZZT,
          typename HostFT, typename HostPT>
class MSMContext {
    ProjT infty;

    AffT *device_buffer;
    AffT *host_buffer;

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

    static const size_t degree = XYZZT::degree;

    MSMContext(size_t scale, size_t window_bits)
        : scale(scale), window_bits(window_bits),
          windows_count(ceil_div(FieldT::nbits, window_bits)), buckets_count(1 << window_bits)
    {
        cudaHostAlloc(&host_buffer, degree * windows_count * scale * sizeof(AffT), cudaHostAllocDefault);
        CUDA_CHECK(cudaGetLastError());

        arena.register_alloc(device_buffer, 2 * degree * scale);
        arena.register_alloc(buckets_sum_WWR, windows_count * buckets_count * 32);
        arena.register_alloc(buckets_sum, degree * buckets_count);
        arena.register_alloc(windows_sum, degree);
        arena.register_alloc(g_task_id, windows_count);
        arena.commit("MSMContext");

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
    }

    ~MSMContext() { cudaFreeHost(host_buffer); }

    void load_bases(const HostPT *host_bases, size_t instance_id = 0)
    {
        memset(host_buffer, 0, degree * scale * sizeof(AffT));
        #pragma omp parallel for
        for (size_t i = 0; i < scale; i++) if (!host_bases[i].is_zero()) memcpy(&host_buffer[degree * i], &host_bases[i], degree * sizeof(AffT));
        cudaMemcpy(device_buffer, host_buffer, degree * scale * sizeof(AffT), cudaMemcpyHostToDevice);

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
            // cudaDeviceSynchronize();
            // CUDA_CHECK(cudaGetLastError());
        // }

        for (size_t i = 1; i < windows_count; i++) {
            cudaMemcpy(device_buffer + (i % 2) * scale, device_buffer + ((i - 1) % 2) * scale, degree * scale * sizeof(AffT), cudaMemcpyDeviceToDevice);
            size_t scale = this->scale;
            size_t dbl_off = window_bits;
            kernel<<<ceil_div(scale, GA_BLK_SIZ), GA_BLK_SIZ>>>([=] __device__ (AffT *points) {
                namespace cg = cooperative_groups;
                cg::grid_group g = cg::this_grid();
                int idx = g.thread_rank();
                if (idx >= scale) return;

                ProjT p = points[idx];
                for (int i = 0; i < dbl_off; i++) p.dbl();
                points[idx] = p;
            }, device_buffer + (i % 2) * scale);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());

            cudaMemcpy(host_buffer + i * scale, device_buffer + (i % 2) * scale, degree * scale * sizeof(AffT), cudaMemcpyDeviceToHost);
        }

        cudaMemcpy(device_buffer, host_buffer, degree * scale * sizeof(AffT), cudaMemcpyHostToDevice);

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
    void msm(BucketContext<FieldT, HostFT>& bkt_ctx, HostPT *result = 0)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        size_t last_window_bits = FieldT::nbits - (windows_count - 1) * window_bits;
        size_t last_window_buckets_count = 1 << last_window_bits;
        size_t valid_buckets_count = buckets_count - 1;

        cudaMemset(g_task_id, 0, windows_count * sizeof(uint32_t));
        for (uint32_t window_id = 0; window_id < windows_count; window_id ++) {
            // cudaMemcpy(device_buffer + (window_id % 2) * scale, host_buffer + window_id * scale, scale * sizeof(AffT), cudaMemcpyHostToDevice);
            (intra_bucket_accumulation<AffT, XYZZT>)<<<gpu.sm_count * IBA_BLK_PER_SM, GA_BLK_SIZ, 0, stream>>>
                (bkt_ctx.buckets_off, bkt_ctx.indices_as_vals, device_buffer, buckets_sum_WWR, buckets_count, windows_count, window_id, last_window_buckets_count, scale, g_task_id);
            if (window_id + 1 < windows_count)
                cudaMemcpyAsync(device_buffer + ((window_id + 1) % 2) * scale, host_buffer + (window_id + 1) * scale, scale * sizeof(AffT), cudaMemcpyHostToDevice, stream);
        }

        warp_reduce<<<ceil_div(valid_buckets_count, GA_BLK_SIZ), GA_BLK_SIZ, 0, stream>>>
            (buckets_sum_WWR, buckets_sum, buckets_count, windows_count, last_window_buckets_count);

        bucket_reduce<<<1, GA_BLK_SIZ, 0, stream>>>
            (buckets_sum, windows_sum, buckets_count, last_window_buckets_count);

        cudaStreamSynchronize(stream);
        CUDA_CHECK(cudaGetLastError());
        cudaStreamDestroy(stream);

        if (result) cudaMemcpy(result, windows_sum, degree * sizeof(ProjT), cudaMemcpyDeviceToHost);
    }
};
