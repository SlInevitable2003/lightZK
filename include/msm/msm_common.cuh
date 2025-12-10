#pragma once
#include "fields/alt_bn128-fp2.cuh"
#include "curves/jacobian_t.cuh"

namespace alt_bn128 {
    typedef jacobian_t<fp_t> g1_t;
    typedef jacobian_t<fp2_t> g2_t;
}

template <class field_t>
__device__ inline bool get_bit(const field_t& scalar, size_t i)
{
    return reinterpret_cast<const uint32_t*>(&scalar)[i / 32] & (1u << (i % 32));
}

template <class field_t>
__host__ __device__ inline uint32_t get_window(const field_t& scalar, uint32_t offset, uint32_t window_bits)
{
    uint32_t top_word_offset = sizeof(scalar) / sizeof(uint32_t) - 1;

    uint64_t ret = 0;
    uint32_t word_idx = offset / 32, word_offset = offset % 32;
    ret = reinterpret_cast<const uint32_t*>(&scalar)[word_idx];
    if (word_idx + 1 <= top_word_offset) ret |= uint64_t(reinterpret_cast<const uint32_t*>(&scalar)[word_idx + 1]) << 32;
    ret = (ret >> word_offset) & ((1 << window_bits) - 1);
    return uint32_t(ret);
}

template <class field_t>
__host__ __device__ inline uint32_t get_window_by_ptr(field_t *scalar, uint32_t offset, uint32_t window_bits)
{
    uint32_t top_word_offset = sizeof(*scalar) / sizeof(uint32_t) - 1;

    uint64_t ret = 0;
    uint32_t word_idx = offset / 32, word_offset = offset % 32;
    ret = reinterpret_cast<const uint32_t*>(scalar)[word_idx];
    if (word_idx + 1 <= top_word_offset) ret |= uint64_t(reinterpret_cast<const uint32_t*>(scalar)[word_idx + 1]) << 32;
    ret = (ret >> word_offset) & ((1 << window_bits) - 1);
    return uint32_t(ret);
}

template <typename T> __device__ T custom_shfl_xor(const T& value, int lane_mask, unsigned int mask = 0xffffffff) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    static_assert(sizeof(T) % sizeof(int) == 0, "T size must be a multiple of 4 bytes");
    constexpr int num_ints = sizeof(T) / sizeof(int);
    int data[num_ints];
    memcpy(data, &value, sizeof(T));
#pragma unroll
    for (int i = 0; i < num_ints; ++i) {
        data[i] = __shfl_xor_sync(mask, data[i], lane_mask);
    }
    T result;
    memcpy(&result, data, sizeof(T));
    return result;
}

template<typename point_t, typename affine_t, typename field_t>
__device__ void dist_pacc(field_t *accxy, field_t *accz, field_t *incrxy)
{
    uint32_t pid = threadIdx.x & 1, offset = (threadIdx.x & 31) & (~1);
    field_t z1(*accz), // Z1
            m0(*incrxy), // X2 | Y2
            m1(*accxy), // X1 | Y1
            m2, z3; m2.zero();
    bool accinf = z1.is_zero(), incrinf = m0.is_zero();
    incrinf &= __shfl_xor_sync(0x11 << offset, incrinf, 1);
    field_t z1z1 = z1^2; // Z1^2
    field_t us; vec_select(&us, &field_t::one(), &z1, sizeof(field_t), pid == 0); us *= m0 * z1z1; // U2 = X2 * Z1Z1 | S2 = Z1 * Y2 * Z1Z1
    field_t hr = us - m1; // H = U2 - X1 | S2 - Y1
    // bool equal = hr.is_zero(); equal &= __shfl_xor_sync(0x11 << offset, equal, 1);
    // if (equal) { if (pid == 0) acc->dbl(); return; }
    field_t p2hr = hr + hr; // 2H | r = 2 * (S2 - Y1)
    field_t hhrr2 = p2hr^2; // I = 4H^2 | r^2
    field_t j = hr * hhrr2; // J = H * I | ?
    field_t j_tmp = custom_shfl_xor(j, 1); vec_select(&j, &j, &j_tmp, sizeof(field_t), pid == 0); // J | J
    field_t v; vec_select(&v, &hhrr2, &j, sizeof(field_t), pid == 0); v *= m1;
    m2 -= v; m2 = m2 + m2;
    field_t r2mj = custom_shfl_xor(hhrr2 - j, 1);
    if (pid == 0) m2 += r2mj;
    field_t vmx3 = custom_shfl_xor(v - m2, 1);
    if (pid == 0) {
        z3 = z1 + hr;
        z3 ^= 2;
        z3 -= (hr^2) + z1z1;
    }
    else if (pid == 1) m2 += p2hr * vmx3;
    field_t z3_tmp = custom_shfl_xor(z3, 1); vec_select(&z3, &z3_tmp, &j, sizeof(field_t), pid == 0);
    uint32_t flag = ((!accinf && !incrinf) << 1) | (accinf && !incrinf);
    vec_select_by_index(accxy, accxy, incrxy, &m2, sizeof(field_t), flag);
    vec_select_by_index(accz, accz, &field_t::one(), &z3, sizeof(field_t), flag);
}