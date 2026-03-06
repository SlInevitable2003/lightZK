#pragma once
#include "msm/msm_kernel.cuh"
#include "utils.cuh"

#include <vector>

#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/device/device_histogram.cuh>
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
          typename HostFT, typename HostPT>
class MSMContext {
    HostPT infty;

    FieldT *scalars;
    AffT *bases, *lifted_bases;
    
    uint16_t *window_scalars_as_keys;
    uint32_t *indices_as_vals;

    uint32_t *buckets_size, *buckets_off;
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
        arena.register_alloc(scalars, scale);
        arena.register_alloc(bases, scale);
        arena.register_alloc(lifted_bases, scale);
        arena.register_alloc(window_scalars_as_keys, windows_count * scale);
        arena.register_alloc(indices_as_vals, windows_count * scale);
        arena.register_alloc(buckets_size, buckets_count);
        arena.register_alloc(buckets_off, buckets_count);
        arena.register_alloc(buckets_sum_WWR, buckets_count * 32);
        arena.register_alloc(buckets_sum, buckets_count - 1);
        arena.register_alloc(buckets_sum_buffer, buckets_count - 1);
        arena.register_alloc(windows_sum, windows_count);
        arena.commit();

        temp_storage = 0, temp_storage_bytes = 0;

        size_t temp_storage_bytes_upd = 0;
        cub::DeviceHistogram::HistogramEven(temp_storage, temp_storage_bytes_upd, window_scalars_as_keys, buckets_size, int(buckets_count + 1), 0, int(buckets_count), scale);
        temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes);

        temp_storage_bytes_upd = 0;
        cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes_upd, buckets_size, buckets_off, buckets_count);
        temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes);

        temp_storage_bytes_upd = 0;
        cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes_upd, buckets_sum, buckets_sum_buffer, ProjT_ADD<ProjT>(), buckets_count - 1);
        temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes);

        temp_storage_bytes_upd = 0;
        cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes_upd, buckets_sum_buffer, windows_sum, buckets_count - 1, ProjT_ADD<ProjT>(), reinterpret_cast<ProjT*>(&infty)[0]);
        temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes);

        cudaMalloc(&temp_storage, temp_storage_bytes);
    }

    ~MSMContext() { cudaFree(temp_storage); }

    void load_bases(HostPT *host_bases, bool lift = true)
    {
        std::vector<AffT> buffer(scale);
        memset(buffer.data(), 0, buffer.size() * sizeof(AffT));
        #pragma omp parallel for
        for (size_t i = 0; i < scale; i++) if (!host_bases[i].is_zero()) memcpy(&buffer[i], &host_bases[i], sizeof(AffT));
        cudaMemcpy(bases, buffer.data(), scale * sizeof(AffT), cudaMemcpyHostToDevice);

        if (lift) {
            cudaMemcpy(lifted_bases, bases, scale * sizeof(AffT), cudaMemcpyDeviceToDevice);
            size_t dbl_off = (windows_count / 2) * window_bits;
            thrust::for_each(thrust::device, lifted_bases, lifted_bases + scale, [=] __device__ (AffT &x) {
                ProjT p = x;
                for (int i = 0; i < dbl_off; i++) p.dbl();
                x = p;
            });
        }
    }

    void load_scalars(HostFT *host_scalars, bool from = true, bool extraction = true, bool sort = true)
    {
        if (host_scalars) cudaMemcpy(scalars, host_scalars, scale * sizeof(FieldT), cudaMemcpyHostToDevice);
        
        cudaStream_t stream[windows_count];
        for (int i = 0; i < windows_count; i++) cudaStreamCreate(&stream[i]);
        
        std::vector<decltype(thrust::cuda::par.on(stream[0]))> policy;
        policy.reserve(windows_count);
        for (int i = 0; i < windows_count; i++) policy.push_back(thrust::cuda::par.on(stream[i]));

        if (from) thrust::for_each(policy[0], scalars, scalars + scale, [] __device__ (FieldT &x) { x.from(); });

        if (extraction) {
            const size_t item_per_thread = sizeof(FieldT) / sizeof(uint32_t), block_size = 256;
            assert((scale & (scale - 1)) == 0 && "scale must be a power of 2");
            kernel<<<scale / block_size, block_size, 0, stream[0]>>>([=] __device__ (uint32_t *scalar_words, uint16_t *keys, uint32_t *vals) {
                using BlockLoad = cub::BlockLoad<uint32_t, block_size, item_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

                __shared__ typename BlockLoad::TempStorage temp_storage_load;

                uint32_t s[item_per_thread];
                uint32_t blk_off = blockIdx.x * block_size;
                uint32_t tid = threadIdx.x + blk_off;
                BlockLoad(temp_storage_load).Load(scalar_words + blk_off * item_per_thread, s);

                for (int j = 0; j < windows_count; j++) {
                    uint16_t key = get_window_by_ptr(reinterpret_cast<const FieldT*>(s), j * window_bits, window_bits);
                    keys[j * scale + tid] = key;
                    vals[j * scale + tid] = tid;
                }
            }, reinterpret_cast<uint32_t*>(scalars), window_scalars_as_keys, indices_as_vals);
        }

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        if (sort) {
            #pragma omp parallel for
            for (int i = 0; i < windows_count; i++) 
                thrust::sort_by_key(policy[i], window_scalars_as_keys + i * scale, window_scalars_as_keys + (i + 1) * scale, indices_as_vals + i * scale);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
        }

        for (int i = 0; i < windows_count; i++) cudaStreamDestroy(stream[i]);
    }

    // ---------------- 内部实现 --------------------
    void bucket_scatter(int window_id, cudaStream_t stream)
    {
        cub::DeviceHistogram::HistogramEven(temp_storage, temp_storage_bytes, window_scalars_as_keys + window_id * scale, buckets_size, int(buckets_count + 1), 0, int(buckets_count), scale, stream);
        cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, buckets_size, buckets_off, buckets_count, stream);
    }

    void warp_reduce(cudaStream_t stream, bool last_window = false, bool batch = false)
    {
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
    void msm(HostPT *result = 0)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        auto policy = thrust::cuda::par.on(stream);

        for (int i = 0, j = windows_count / 2; j < windows_count - 1; i++, j++) {
            bucket_scatter(i, stream);
            (bucket_segemented_reduction<AffT, ProjT, XYZZT>)<<<buckets_count - 1, 32, 0, stream>>>
                (buckets_off, indices_as_vals + i * scale, bases, buckets_sum_WWR);
            
            bucket_scatter(j, stream);
            (bucket_segemented_reduction_increment<AffT, ProjT, XYZZT>)<<<buckets_count - 1, 32, 0, stream>>>
                (buckets_off, indices_as_vals + j * scale, lifted_bases, buckets_sum_WWR);

            warp_reduce(stream);
            bucket_reduce(i, stream);
        }

        // cudaStreamSynchronize(stream);
        // CUDA_CHECK(cudaGetLastError());

        if (windows_count % 2 == 0) {
            int i = windows_count / 2 - 1, j = windows_count - 1;

            bucket_scatter(i, stream);
            (bucket_segemented_reduction<AffT, ProjT, XYZZT>)<<<buckets_count - 1, 32, 0, stream>>>
                (buckets_off, indices_as_vals + i * scale, bases, buckets_sum_WWR);
            warp_reduce(stream);

            bucket_scatter(j, stream);
            (bucket_segemented_reduction_last_window<AffT, ProjT, XYZZT>)<<<last_window_warps_per_bucket * (last_window_buckets_count - 1), 32, 0, stream>>>
                (buckets_off, indices_as_vals + j * scale, lifted_bases, buckets_sum_WWR, last_window_warps_per_bucket);
            warp_reduce(stream, true, true);

            bucket_reduce(i, stream, true, true);
        } else {
            int i = windows_count - 1;
            bucket_scatter(i, stream);
            (bucket_segemented_reduction_last_window<AffT, ProjT, XYZZT>)<<<last_window_warps_per_bucket * (last_window_buckets_count - 1), 32, 0, stream>>>
                (buckets_off, indices_as_vals + i * scale, lifted_bases, buckets_sum_WWR, last_window_warps_per_bucket);
            
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
