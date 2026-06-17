#pragma once
#include "utils.cuh"

#include <vector>

#include <cooperative_groups.h>

#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

template<typename FieldT, typename HostFT, typename key_t = uint16_t>
class BucketContext {
    TypedGpuArena arena;
    void *temp_storage;
    size_t temp_storage_bytes;

public:
    FieldT *scalars;

    key_t *window_scalars_as_keys;
    uint32_t *indices_as_vals;

    uint32_t *buckets_size, *buckets_off;
    uint32_t *scan_buffer;

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
        arena.commit("BucketContext");
    }

    ~BucketContext() { cudaFree(temp_storage); }

    void load_scalars(const HostFT *host_scalars) { cudaMemcpy(scalars, host_scalars, scale * sizeof(FieldT), cudaMemcpyHostToDevice); }
    void load_scalars(FieldT *device_scalars) { cudaMemcpy(scalars, device_scalars, scale * sizeof(FieldT), cudaMemcpyDeviceToDevice); }

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
            // assert((scale & (scale - 1)) == 0 && "scale must be a power of 2");
            kernel<<<ceil_div(scale, block_size), block_size, 0, stream[0]>>>([=] __device__ (FieldT *scalars, key_t *keys, uint32_t *vals) {
                // using BlockLoad = cub::BlockLoad<uint32_t, block_size, item_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

                // __shared__ typename BlockLoad::TempStorage temp_storage_load;

                // uint32_t s[item_per_thread];
                // uint32_t blk_off = blockIdx.x * block_size;
                // uint32_t tid = threadIdx.x + blk_off;

                // int valid_items = scale - blk_off;
                // if (valid_items > block_size) valid_items = block_size;

                // BlockLoad(temp_storage_load).Load(scalar_words + blk_off * item_per_thread, s, valid_items * item_per_thread, 0);

                namespace cg = cooperative_groups;
                cg::grid_group g = cg::this_grid();
                int idx = g.thread_rank();

                if (idx < scale) {
                    for (int j = 0; j < windows_count; j++) {
                        key_t key = get_window_by_ptr(scalars + idx, j * window_bits, window_bits);
                        keys[j * scale + idx] = key;
                        vals[j * scale + idx] = idx;
                    }
                }
            }, scalars, window_scalars_as_keys, indices_as_vals);
        }

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        if (sort) {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < windows_count; i++) 
                thrust::sort_by_key(policy[i], window_scalars_as_keys + i * scale, window_scalars_as_keys + (i + 1) * scale, indices_as_vals + i * scale);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
        }

        if (scatter) {
            size_t scale = this->scale;
            assert((scale & (scale - 1)) == 0 && "scale must be a power of 2!");
            size_t log_scale = __builtin_ctzll(scale);

            size_t windows_count = this->windows_count;
            size_t buckets_count = this->buckets_count;
            const size_t block_size = 256;
            kernel<<<ceil_div(windows_count * buckets_count, block_size), block_size>>>([=] __device__ (const key_t *keys, uint32_t *buckets_off) {
                namespace cg = cooperative_groups;
                cg::grid_group g = cg::this_grid();
                size_t tid = g.thread_rank();
                
                size_t window_id = tid / buckets_count;
                const key_t *cur_keys = keys + window_id * scale;
                key_t bucket_id = tid % buckets_count;

                uint32_t l = 0, r = scale;
                if (cur_keys[0] > bucket_id) r = 0;
                else {
                    for (int i = log_scale; i > 0; i--) {
                        uint32_t mid = l + ((r - l) >> 1);
                        bool flag = cur_keys[mid] > bucket_id;
                        l = flag ? l : mid;
                        r = flag ? mid : r;
                    }
                }
                buckets_off[tid] = r;
                
            }, window_scalars_as_keys, buckets_off);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
        }

        for (int i = 0; i < windows_count; i++) cudaStreamDestroy(stream[i]);
    }
};