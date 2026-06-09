#pragma once
#include <cstdint>
#include <vector>

#include "mgpu/utils/CuLoadStore.cuh"

namespace mgpu {
    #define REDUCE_BLOCK_SIZE 256
    #define REDUCE_GRAIN_SIZE 4

    template <typename T, typename Op, size_t block_size = REDUCE_BLOCK_SIZE, size_t grain_size = REDUCE_GRAIN_SIZE>
    __global__ void CuReduce(T *in, T *out, size_t len, Op op, T neutral = 0) {
        __shared__ T sdata[block_size];

        const uint32_t tid = threadIdx.x;
        
        T reg[grain_size];
        GlobalToReg<T, grain_size>(in, len, reg, neutral);
        #pragma unroll
        for (size_t j = 1; j < grain_size; j++) reg[0] = op(reg[0], reg[j]);
        sdata[tid] = reg[0];
        __syncthreads();

        if (block_size >= 512) { if (tid < 256) { sdata[tid] = op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
        if (block_size >= 256) { if (tid < 128) { sdata[tid] = op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
        if (block_size >= 128) { if (tid < 64) { sdata[tid] = op(sdata[tid], sdata[tid + 64]); } __syncthreads(); } 

        if (tid < 32) {
            volatile T *vsmem = sdata;
            if (block_size >= 64) vsmem[tid] = op(vsmem[tid], vsmem[tid + 32]);
            if (block_size >= 32) vsmem[tid] = op(vsmem[tid], vsmem[tid + 16]);
            if (block_size >= 16) vsmem[tid] = op(vsmem[tid], vsmem[tid + 8]);
            if (block_size >= 8) vsmem[tid] = op(vsmem[tid], vsmem[tid + 4]);
            if (block_size >= 4) vsmem[tid] = op(vsmem[tid], vsmem[tid + 2]);
            if (block_size >= 2) vsmem[tid] = op(vsmem[tid], vsmem[tid + 1]);
        }

        if (tid == 0) out[blockIdx.x] = sdata[0];
    }
} // namespace mgpu