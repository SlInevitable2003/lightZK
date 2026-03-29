#pragma once
#include "utils.cuh"
#include "bucket_kernel.cuh"

#include <vector>

#include <cooperative_groups.h>

template<typename FieldT, typename HostFT>
class BucketContext {
    TypedGpuArena arena;
    GPUConfig gpu;

public:
    FieldT *scalars;

    size_t scale;
    size_t window_bits;
    size_t windows_count, buckets_count;

    uint8_t *sppark_buffer;
    size_t temp_size, digits_size;

    vec2d_t<uint2> temp_buffer;
    vec2d_t<uint32_t> digits;

    uint32_t *sort_buffer;
    vec2d_t<uint32_t> histogram;

    BucketContext(size_t scale, size_t window_bits) 
    : scale(scale), window_bits(window_bits), windows_count(ceil_div(FieldT::nbits, window_bits)), buckets_count(1 << (window_bits - 1)),
      temp_size(scale * sizeof(uint2)), digits_size(windows_count * scale * sizeof(uint32_t))
    {
        arena.register_alloc(scalars, scale);
        arena.register_alloc(sppark_buffer, temp_size + digits_size);
        arena.register_alloc(sort_buffer, windows_count * buckets_count);
        arena.commit("BucketContext");

        temp_buffer = {&sppark_buffer[0], scale};
        digits = {&sppark_buffer[temp_size], scale};
        histogram = {sort_buffer, buckets_count};
    }

    void load_scalars(const HostFT *host_scalars) { cudaMemcpy(scalars, host_scalars, scale * sizeof(FieldT), cudaMemcpyHostToDevice); }
    void load_scalars(FieldT *device_scalars) { cudaMemcpy(scalars, device_scalars, scale * sizeof(FieldT), cudaMemcpyDeviceToDevice); }

    void process()
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        uint32_t grid_size = gpu.sm_count() / 3;
        while (grid_size & (grid_size - 1)) grid_size -= (grid_size & (0 - grid_size)); // grid_size <- largest 2^k s.t. 2^k <= grid_size, e.g., 1011 <- 1000

        breakdown<<<2 * grid_size, BD_BLK_SIZ, sizeof(FieldT) * BD_BLK_SIZ, stream>>>(digits, scalars, scale, windows_count, window_bits);
        cudaStreamSynchronize(stream);
        
        const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;
        uint32_t top = FieldT::nbits - window_bits * (windows_count - 1);
        for (size_t window_id = 0; window_id < windows_count; window_id++) {
            window_val_sort<<<grid_size, SORT_BLK_SIZ, shared_sz, stream>>>
                (digits[window_id], scale, window_id, temp_buffer[0], histogram[window_id], window_bits - 1, (window_id == windows_count - 1) ? window_bits - 1 : top - 1);
        }
        cudaStreamSynchronize(stream);
        CUDA_CHECK(cudaGetLastError());
    }
};