#pragma once
#include "fields/bucket.cuh"
#include "msm/msm_kernel.cuh"
#include "utils.cuh"

#include <vector>

#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

template <typename ProjT>
struct ProjT_ADD {
    __device__ ProjT operator()(const ProjT &lhs, const ProjT &rhs) const {
        ProjT res;
        ProjT::add(res, lhs, rhs);
        return res;
    }
};

template <typename FieldT, typename AffT, typename ProjT, typename XYZZT,
          typename HostFT, typename HostPT, size_t instances = 1>
class MSMContext {
    HostPT infty;

    AffT *bases[instances], *lifted_bases[instances];

    ProjT *buckets_sum_WWR;
    ProjT *buckets_sum, *buckets_sum_buffer;
    ProjT *windows_sum;

    TypedGpuArena arena;
    void *temp_storage;
    size_t temp_storage_bytes;

public:
    size_t scale;
    size_t window_bits;
    size_t windows_count, buckets_count;

    size_t last_window_buckets_count;
    size_t last_window_warps_per_bucket;

    MSMContext(size_t scale_, size_t window_bits_)
        : scale(scale_), window_bits(window_bits_),
          windows_count(ceil_div(FieldT::nbits, window_bits)), buckets_count(1 << window_bits),
          last_window_buckets_count(1 << (FieldT::nbits % window_bits)), last_window_warps_per_bucket(buckets_count / last_window_buckets_count),
          infty(HostPT::zero())
    {
        for (size_t i = 0; i < instances; i++) {
            arena.register_alloc(bases[i], scale);
            arena.register_alloc(lifted_bases[i], scale);
        }
        arena.register_alloc(buckets_sum_WWR, buckets_count * 32);
        arena.register_alloc(buckets_sum, buckets_count - 1);
        arena.register_alloc(buckets_sum_buffer, buckets_count - 1);
        arena.register_alloc(windows_sum, windows_count);
        arena.commit();

        temp_storage = 0, temp_storage_bytes = 0;

        size_t temp_storage_bytes_upd = 0;

        temp_storage_bytes_upd = 0;
        cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes_upd, buckets_sum, buckets_sum_buffer, ProjT_ADD<ProjT>(), buckets_count - 1);
        temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes);

        temp_storage_bytes_upd = 0;
        cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes_upd, buckets_sum_buffer, windows_sum, buckets_count - 1, ProjT_ADD<ProjT>(), reinterpret_cast<ProjT*>(&infty)[0]);
        temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes);

        cudaMalloc(&temp_storage, temp_storage_bytes);
    }

    ~MSMContext() { cudaFree(temp_storage); }

    void load_bases(HostPT *host_bases, bool lift = true, size_t instance_id = 0)
    {
        std::vector<AffT> buffer(scale);
        memset(buffer.data(), 0, buffer.size() * sizeof(AffT));
        #pragma omp parallel for
        for (size_t i = 0; i < scale; i++) if (!host_bases[i].is_zero()) memcpy(&buffer[i], &host_bases[i], sizeof(AffT));
        cudaMemcpy(bases[instance_id], buffer.data(), scale * sizeof(AffT), cudaMemcpyHostToDevice);

        if (lift) {
            cudaMemcpy(lifted_bases[instance_id], bases[instance_id], scale * sizeof(AffT), cudaMemcpyDeviceToDevice);
            size_t dbl_off = (windows_count / 2) * window_bits;
            thrust::for_each(thrust::device, lifted_bases[instance_id], lifted_bases[instance_id] + scale, [=] __device__ (AffT &x) {
                ProjT p = x;
                for (int i = 0; i < dbl_off; i++) p.dbl();
                x = p;
            });
        }
    }

    // ---------------- 内部实现 --------------------

    void warp_reduce(cudaStream_t stream, bool last_window = false, bool batch = false)
    {
        size_t last_window_warps_per_bucket = this->last_window_warps_per_bucket;

        size_t upper_bound = (last_window && !batch) ? last_window_buckets_count : buckets_count;
        size_t cur_window_buckets_count = last_window ? last_window_buckets_count : buckets_count;

        if (!last_window) {
            kernel<<<cur_window_buckets_count - 1, 32, 0, stream>>>([=] __device__ (ProjT *buckets_sum_WWR, ProjT *buckets_sum) {
                uint32_t bucket_id = blockIdx.x;
                uint32_t lane_id = threadIdx.x;
                ProjT acc = buckets_sum_WWR[bucket_id * 32 + lane_id], incr;
                
                for (uint32_t i = 1; i < 32; i <<= 1) {
                    incr = custom_shfl_xor(acc, i);
                    acc.add(incr);
                }

                if (lane_id == 0) buckets_sum[upper_bound - 2 - bucket_id] = acc;
            }, buckets_sum_WWR, buckets_sum);
        } else if (!batch) {
            kernel<<<cur_window_buckets_count - 1, 32, 0, stream>>>([=] __device__ (ProjT *buckets_sum_WWR, ProjT *buckets_sum) {
                uint32_t bucket_id = blockIdx.x;
                uint32_t lane_id = threadIdx.x;
                ProjT acc, incr; acc.inf();
                
                for (uint32_t i = 0; i < last_window_warps_per_bucket; i++) {
                    incr = buckets_sum_WWR[bucket_id * last_window_warps_per_bucket * 32 + i * 32 + lane_id];
                    acc.add(incr);
                }

                for (uint32_t i = 1; i < 32; i <<= 1) {
                    incr = custom_shfl_xor(acc, i);
                    acc.add(incr);
                }

                if (lane_id == 0) buckets_sum[upper_bound - 2 - bucket_id] = acc;
            }, buckets_sum_WWR, buckets_sum);
        } else {
            kernel<<<cur_window_buckets_count - 1, 32, 0, stream>>>([=] __device__ (ProjT *buckets_sum_WWR, ProjT *buckets_sum) {
                uint32_t bucket_id = blockIdx.x;
                uint32_t lane_id = threadIdx.x;
                ProjT acc, incr; acc.inf();
                
                for (uint32_t i = 0; i < last_window_warps_per_bucket; i++) {
                    incr = buckets_sum_WWR[bucket_id * last_window_warps_per_bucket * 32 + i * 32 + lane_id];
                    acc.add(incr);
                }

                for (uint32_t i = 1; i < 32; i <<= 1) {
                    incr = custom_shfl_xor(acc, i);
                    acc.add(incr);
                }

                if (lane_id == 0) buckets_sum[upper_bound - 2 - bucket_id].add(acc);
            }, buckets_sum_WWR, buckets_sum);
        }
    }

    void bucket_reduce(int window_id, cudaStream_t stream, bool last_window = false, bool batch = false)
    {
        size_t upper_bound = (last_window && !batch) ? last_window_buckets_count : buckets_count;
        cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, buckets_sum, buckets_sum_buffer, ProjT_ADD<ProjT>(), upper_bound - 1);
        cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, buckets_sum_buffer, windows_sum + window_id, upper_bound - 1, ProjT_ADD<ProjT>(), reinterpret_cast<ProjT*>(&infty)[0]);
    }

public:
    void msm(BucketContext<FieldT, HostFT>& bkt_ctx, HostPT *result = 0, size_t instance_id = 0)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        auto policy = thrust::cuda::par.on(stream);

        for (int i = 0, j = windows_count / 2; j < windows_count - 1; i++, j++) {
            (bucket_segemented_reduction<AffT, ProjT, XYZZT>)<<<buckets_count - 1, 32, 0, stream>>>
                (bkt_ctx.buckets_off + i * buckets_count, bkt_ctx.indices_as_vals + i * scale, bases[instance_id], buckets_sum_WWR);
            
            (bucket_segemented_reduction_increment<AffT, ProjT, XYZZT>)<<<buckets_count - 1, 32, 0, stream>>>
                (bkt_ctx.buckets_off + j * buckets_count, bkt_ctx.indices_as_vals + j * scale, lifted_bases[instance_id], buckets_sum_WWR);

            warp_reduce(stream);
            bucket_reduce(i, stream);
        }

        // cudaStreamSynchronize(stream);
        // CUDA_CHECK(cudaGetLastError());

        if (windows_count % 2 == 0) {
            int i = windows_count / 2 - 1, j = windows_count - 1;

            (bucket_segemented_reduction<AffT, ProjT, XYZZT>)<<<buckets_count - 1, 32, 0, stream>>>
                (bkt_ctx.buckets_off + i * buckets_count, bkt_ctx.indices_as_vals + i * scale, bases[instance_id], buckets_sum_WWR);
            warp_reduce(stream);

            (bucket_segemented_reduction_last_window<AffT, ProjT, XYZZT>)<<<last_window_warps_per_bucket * (last_window_buckets_count - 1), 32, 0, stream>>>
                (bkt_ctx.buckets_off + j * buckets_count, bkt_ctx.indices_as_vals + j * scale, lifted_bases[instance_id], buckets_sum_WWR, last_window_warps_per_bucket);
            warp_reduce(stream, true, true);

            bucket_reduce(i, stream, true, true);
        } else {
            int i = windows_count - 1;
            (bucket_segemented_reduction_last_window<AffT, ProjT, XYZZT>)<<<last_window_warps_per_bucket * (last_window_buckets_count - 1), 32, 0, stream>>>
                (bkt_ctx.buckets_off + i * buckets_count, bkt_ctx.indices_as_vals + i * scale, lifted_bases[instance_id], buckets_sum_WWR, last_window_warps_per_bucket);
            
            warp_reduce(stream, true);
            bucket_reduce(i, stream, true);
        }

        cudaStreamSynchronize(stream);
        CUDA_CHECK(cudaGetLastError());
        cudaStreamDestroy(stream);

        if (result) {
            std::vector<HostPT> buffer(windows_count);
            cudaMemcpy(buffer.data(), windows_sum, windows_count * sizeof(ProjT), cudaMemcpyDeviceToHost);
            HostPT res = HostPT::zero();
            for (int k = windows_count / 2 - (windows_count % 2 == 0); k >= 0; k--) {
                for (int l = 0; l < window_bits; l++) res = res.dbl();
                res = res + buffer[k];
            }
            *result = res;
        }
    }
};
