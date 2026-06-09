#pragma once
#include <cstdint>
#include <vector>

#include "mgpu/CuReduce.cuh"

namespace mgpu {
    #define SCAN_BLOCK_SIZE 256
    #define SCAN_GRAIN_SIZE 4

    enum class CuScanType { Inclusive, Exclusive };

    template <typename T, typename Op, CuScanType scan_type, size_t block_size = SCAN_BLOCK_SIZE>
    __device__ void CuScan_block(T &x, T *dbuffer, Op op, T neutral = 0) {
        size_t head = 0;
        const uint32_t tid = threadIdx.x;
        dbuffer[tid] = x;
        __syncthreads();
        
        if (block_size >= 2) { if (tid >= 1) { x = op(x, dbuffer[head + tid - 1]); } head = block_size - head; dbuffer[head + tid] = x; __syncthreads(); }
        if (block_size >= 4) { if (tid >= 2) { x = op(x, dbuffer[head + tid - 2]); } head = block_size - head; dbuffer[head + tid] = x; __syncthreads(); }
        if (block_size >= 8) { if (tid >= 4) { x = op(x, dbuffer[head + tid - 4]); } head = block_size - head; dbuffer[head + tid] = x; __syncthreads(); }
        if (block_size >= 16) { if (tid >= 8) { x = op(x, dbuffer[head + tid - 8]); } head = block_size - head; dbuffer[head + tid] = x; __syncthreads(); }
        if (block_size >= 32) { if (tid >= 16) { x = op(x, dbuffer[head + tid - 16]); } head = block_size - head; dbuffer[head + tid] = x; __syncthreads(); }
        if (block_size >= 64) { if (tid >= 32) { x = op(x, dbuffer[head + tid - 32]); } head = block_size - head; dbuffer[head + tid] = x; __syncthreads(); }
        if (block_size >= 128) { if (tid >= 64) { x = op(x, dbuffer[head + tid - 64]); } head = block_size - head; dbuffer[head + tid] = x; __syncthreads(); }
        if (block_size >= 256) { if (tid >= 128) { x = op(x, dbuffer[head + tid - 128]); } head = block_size - head; dbuffer[head + tid] = x; __syncthreads(); }
        if (block_size >= 512) { if (tid >= 256) { x = op(x, dbuffer[head + tid - 256]); } head = block_size - head; dbuffer[head + tid] = x; __syncthreads(); }

        if (scan_type == CuScanType::Exclusive) {
            dbuffer[tid] = x;
            __syncthreads();
            x = (tid == 0) ? neutral : dbuffer[tid - 1];
        }
    }

    template <typename T, typename Op, size_t block_size = SCAN_BLOCK_SIZE, size_t grain_size = SCAN_GRAIN_SIZE>
    __global__ void CuScan(T *in, size_t len, Op op, T neutral = 0) { // Inclusive
        __shared__ T sdata[2 * block_size];

        T reg[grain_size];
        GlobalToReg<T, grain_size>(in, len, reg, neutral);
        #pragma unroll
        for (size_t j = 1; j < grain_size; j++) reg[j] = op(reg[j], reg[j - 1]);
        
        T total = reg[grain_size - 1];
        CuScan_block<T, Op, CuScanType::Exclusive>(total, sdata, op, neutral);
        #pragma unroll
        for (size_t j = 0; j < grain_size; j++) reg[j] = op(reg[j], total);

        RegToGlobal<T, grain_size>(reg, in, len);
    }

    template <typename T, typename Op, size_t block_size = SCAN_BLOCK_SIZE, size_t grain_size = SCAN_GRAIN_SIZE>
    __global__ void CuScan_add(T *in, T* out, T *aux, size_t len, Op op, T neutral = 0) { // Inclusive
        __shared__ T sdata[2 * block_size];

        T reg[grain_size];
        GlobalToReg<T, grain_size>(in, len, reg, neutral);
        #pragma unroll
        for (size_t j = 1; j < grain_size; j++) reg[j] = op(reg[j], reg[j - 1]);
        
        T total = reg[grain_size - 1];
        CuScan_block<T, Op, CuScanType::Exclusive>(total, sdata, op, neutral);
        #pragma unroll
        for (size_t j = 0; j < grain_size; j++) reg[j] = op(reg[j], total);

        T extra = (blockIdx.x == 0) ? neutral : aux[blockIdx.x - 1];
        #pragma unroll
        for (size_t j = 0; j < grain_size; j++) reg[j] = op(reg[j], extra);

        RegToGlobal<T, grain_size>(reg, out, len);
    }

    // template <typename T, size_t block_size = SCAN_BLOCK_SIZE, size_t grain_size = SCAN_GRAIN_SIZE>
    // class CuScaner {
    //     CuTensor<T> d_data_, d_buffer_;
    // public:
    //     CuScaner() = default;
    //     CuScaner(CuContext &context, const std::vector<T> &host_data) {
    //         d_data_ = context.alloc<T>(host_data.size());
    //         d_buffer_ = context.alloc<T>(ceil_div(host_data.size(), block_size * grain_size));
    //         context.HtoD(host_data, d_data_);
    //     }

    //     template <typename Op, CuScanType scan_type>
    //     void scan(CuContext &context, std::vector<T> &host_result, Op op, T neutral = 0) {
    //         size_t grid_size = ceil_div(d_data_.count(), block_size * grain_size);
    //         CuReduce<T, Op, block_size, grain_size><<<grid_size, block_size>>>(
    //             d_data_.data(), d_buffer_.data(), d_data_.count(), op, neutral
    //         );
    //         CuScan<T, Op, block_size, grain_size><<<1, block_size>>>(
    //             d_buffer_.data(), grid_size, op, neutral
    //         );
    //         CuScan_add<T, Op, block_size, grain_size><<<grid_size, block_size>>>(
    //             d_data_.data(), d_buffer_.data(), d_data_.count(), op, neutral
    //         );
    //         if (scan_type == CuScanType::Exclusive) {
    //             // TODO
    //         }
    //         context.DtoH(d_data_, host_result);
    //     }

    //     template <typename Op>
    //     void inclusive_scan(CuContext &context, std::vector<T> &host_result, Op op, T neutral = 0) {
    //         scan<Op, CuScanType::Inclusive>(context, host_result, op, neutral);
    //     }

    //     template <typename Op>
    //     void exclusive_scan(CuContext &context, std::vector<T> &host_result, Op op, T neutral = 0) {
    //         scan<Op, CuScanType::Exclusive>(context, host_result, op, neutral);
    //     }
    // };
} // namespace mgpu