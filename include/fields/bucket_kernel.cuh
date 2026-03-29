// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "utils.cuh"

#define BD_BLK_SIZ 1024

template <typename FieldT>
class FieldTranspose {
    uint32_t val[sizeof(FieldT) / sizeof(uint32_t)][32];

public:
    __device__ const uint32_t& operator[](size_t i) const  { return val[i][0]; }
    __device__ FieldTranspose& view(uint32_t laneid) { return *reinterpret_cast<FieldTranspose*>(&val[0][laneid]); }
    __device__ void write_column(const FieldT& rhs)
    {
        for (size_t i = 0; i < sizeof(FieldT) / sizeof(uint32_t); i++) val[i][0] = rhs[i];
        return *this;
    }
};

template<typename FieldT>
__device__ __forceinline__
static uint32_t get_64bits(const FieldTranspose<FieldT>& scalar, uint32_t off, uint32_t top_i = (FieldT::nbits + 31) / 32 - 1)
{
    uint32_t i = off / 32;
    uint64_t ret = scalar[i];

    if (i < top_i) ret |= uint64_t(scalar[i + 1]) << 32;

    return ret >> (off % 32);
}

__device__ __forceinline__
static uint32_t booth_encode(uint32_t wval, uint32_t wmask, uint32_t wbits)
{
    uint32_t sign = (wval >> wbits) & 1;
    wval = ((wval + 1) & wmask) >> 1;
    return sign ? 0 - wval : wval;
    // wval = (low) 窗口(绝对)值[0:wbits-2] 符号位[wbits-1] (high)
    // 暗示以后计算 (-1)^{符号位} * 窗口(绝对)值 * 基点
}

template <typename FieldT>
__global__ void breakdown(vec2d_t<uint32_t> digits, FieldT* scalars, size_t scale, size_t windows_count, size_t window_bits)
{
    // assert(len <= (1U<<31) && wbits < 32);
    extern __shared__ FieldTranspose<FieldT> xchange[];

    const uint32_t tid = threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    const uint32_t top_word_idx = ceil_div(FieldT::nbits, 32) - 1;
    const uint32_t window_mask = uint32_t(-1) >> (31 - window_bits);
    FieldTranspose<FieldT> &scalar = xchange[tid / 32].view(tid % 32);

    #pragma unroll 1
    for (uint32_t i = gid; i < scale; i += stride) {
        FieldT s = scalars[i];
        s.from();
        uint32_t sign = s.abs() << 31;

        scalar.write_column(s);

        #pragma unroll 1
        for (uint32_t bit0 = (windows_count - 1) * window_bits - 1, window_id = windows_count; --window_id; bit0 -= window_bits) {
            uint32_t window_val = get_64bits(scalar, bit0, top_word_idx);
            window_val = booth_encode(window_val, window_mask, window_bits);
            if (window_val) window_val ^= sign;
            digits[window_id][i] = window_val;
        }

        uint32_t window_val = s[0] << 1;
        window_val = booth_encode(window_val, window_mask, window_bits);
        if (window_val) window_val ^= sign;
        digits[0][i] = window_val;
    }
}

#include "sort.cuh"

__launch_bounds__(SORT_BLK_SIZ)
__global__ void window_val_sort(uint32_t *inout, size_t scale, uint32_t window_id,
                                uint2 *buffer, uint32_t *histogram,
                                uint32_t window_bits, uint32_t lsbits)
{
    assert(scale <= (1U<<31) && window_bits <= 2 * DIGIT_BITS && gridDim.x <= WARP_SZ);
    uint32_t lg_gridDim = 31 - __clz(gridDim.x);

    if (window_bits > DIGIT_BITS || (lg_gridDim && window_bits > lg_gridDim+1)) {
        uint32_t top_bits = window_bits / 2;
        uint32_t low_bits = window_bits - top_bits;

        if (low_bits < lg_gridDim+1) {
            low_bits = lg_gridDim+1;
            top_bits = window_bits - low_bits;
        }

        upper_sort(buffer, inout, scale, lsbits, top_bits, low_bits, histogram);

        histogram += blockIdx.x << low_bits;

        #pragma unroll 1
        for (uint32_t i = blockIdx.x; i < 1 << top_bits; i += gridDim.x) {
            uint2 slice = *(uint2*)histogram;
            lower_sort(inout, buffer, slice.x, slice.y, histogram, low_bits);
            histogram += gridDim.x << low_bits;
        }
    } else if (blockIdx.x == 0) {
        counters[0] = 0;
        __syncthreads();

        uint32_t lshift = window_bits - lsbits;
        uint32_t pack_mask = 0xffffffffU << lshift;

        #pragma unroll 1
        for (uint32_t i = threadIdx.x; i < (uint32_t)scale; i += SORT_BLK_SIZ) {
            auto val = inout[(size_t)i];
            auto pck = pack(i, pack_mask, (val-1) << lshift);
            if (val) {
                auto idx = atomicAdd(&counters[0], 1);
                //uint32_t pid = pidx ? pidx[i] : i;
                buffer[idx] = uint2{pck, pack(i, 0x80000000, val)};
            }
        }

        __syncthreads();
        lower_sort(inout, buffer, 0, counters[0], histogram, window_bits);
    }
}