#pragma once
#include "utils.cuh"

#include <vector>

#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_histogram.cuh>

template<typename FieldT, typename HostFT>
class BucketContext {
    FieldT *scalars;

    TypedGpuArena arena;
    void *temp_storage;
    size_t temp_storage_bytes;

public:
    uint16_t *window_scalars_as_keys;
    uint32_t *indices_as_vals;

    uint32_t *buckets_size, *buckets_off;

    size_t scale;
    size_t window_bits;
    size_t windows_count, buckets_count;

    BucketContext(size_t scale_, size_t window_bits_) 
    : scale(scale_), window_bits(window_bits_), windows_count(ceil_div(FieldT::nbits, window_bits)), buckets_count(1 << window_bits)
    {
        arena.register_alloc(scalars, scale);
        arena.register_alloc(window_scalars_as_keys, windows_count * scale);
        arena.register_alloc(indices_as_vals, windows_count * scale);
        arena.register_alloc(buckets_size, windows_count * buckets_count);
        arena.register_alloc(buckets_off, windows_count * buckets_count);
        arena.commit();

        temp_storage = 0, temp_storage_bytes = 0;

        size_t temp_storage_bytes_upd = 0;
        cub::DeviceHistogram::HistogramEven(temp_storage, temp_storage_bytes_upd, window_scalars_as_keys, buckets_size, int(buckets_count + 1), 0, int(buckets_count), scale);
        temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes);

        temp_storage_bytes_upd = 0;
        cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes_upd, buckets_size, buckets_off, buckets_count);
        temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes);

        cudaMalloc(&temp_storage, temp_storage_bytes);
    }

    ~BucketContext() { cudaFree(temp_storage); }

    void load_scalars(HostFT *host_scalars) { cudaMemcpy(scalars, host_scalars, scale * sizeof(FieldT), cudaMemcpyHostToDevice); }

    void process(bool from = true, bool extraction = true, bool sort = true, bool scatter = true)
    {   
        cudaStream_t stream[windows_count];
        for (int i = 0; i < windows_count; i++) cudaStreamCreate(&stream[i]);
        
        std::vector<decltype(thrust::cuda::par.on(stream[0]))> policy;
        policy.reserve(windows_count);
        for (int i = 0; i < windows_count; i++) policy.push_back(thrust::cuda::par.on(stream[i]));

        if (from) thrust::for_each(policy[0], scalars, scalars + scale, [] __device__ (FieldT &x) { x.from(); });

        if (extraction) {
            size_t scale = this->scale;
            size_t window_bits = this->window_bits;
            size_t windows_count = this->windows_count;
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
            #pragma omp parallel for num_threads(8)
            for (int i = 0; i < windows_count; i++) 
                thrust::sort_by_key(policy[i], window_scalars_as_keys + i * scale, window_scalars_as_keys + (i + 1) * scale, indices_as_vals + i * scale);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
        }

        if (scatter) {
            for (int i = 0; i < windows_count; i++) {
                cub::DeviceHistogram::HistogramEven(temp_storage, temp_storage_bytes, window_scalars_as_keys + i * scale, buckets_size + i * buckets_count, int(buckets_count + 1), 0, int(buckets_count), scale, stream[0]);
                cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, buckets_size + i * buckets_count, buckets_off + i * buckets_count, buckets_count, stream[0]);
            }
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
        }

        for (int i = 0; i < windows_count; i++) cudaStreamDestroy(stream[i]);
    }
};