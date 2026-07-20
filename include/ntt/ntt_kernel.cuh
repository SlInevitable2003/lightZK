#include "ntt_common.cuh"

#include <cooperative_groups.h>

#define NTT_BLK_SIZ 1024

template <typename FieldT>
__global__ void onchip_butterfly_transformation_1(FieldT *in, FieldT *out, FieldT *omegas, size_t scale)
{
    uint32_t round = 0;

    const size_t offset = threadIdx.x, batch_id = blockIdx.x;
    size_t idx = offset * gridDim.x + batch_id;
    
    __shared__ FieldT shmem[NTT_BLK_SIZ];

    FieldT coeff = in[idx], other;

    for (uint32_t adj_bit = NTT_BLK_SIZ >> 1; adj_bit >= 32; adj_bit >>= 1, round++) {
        shmem[offset] = coeff;
        __syncthreads();    

        idx = idx & ~(scale >>= 1);
        other = shmem[offset ^ adj_bit];
        coeff = ((offset & adj_bit) == 0) ? (coeff + other) : (other - coeff) * omegas[idx << round];
        __syncthreads();
    }

    uint32_t *coeff_ptr = reinterpret_cast<uint32_t*>(&coeff);
    uint32_t *other_ptr = reinterpret_cast<uint32_t*>(&other);

    for (uint32_t adj_bit = 16; adj_bit >= 1; adj_bit >>= 1, round++) {
        idx = idx & ~(scale >>= 1);
        for (int k = 0; k < sizeof(FieldT) / sizeof(uint32_t); k++) other_ptr[k] = __shfl_xor_sync(0xffffffff, coeff_ptr[k], adj_bit);
        coeff = ((offset & adj_bit) == 0) ? (coeff + other) : (other - coeff) * omegas[idx << round];
    }

    out[batch_id * NTT_BLK_SIZ + offset] = coeff;
}

template <typename FieldT>
__global__ void onchip_butterfly_transformation_2(FieldT *in, FieldT *out, FieldT *omegas, size_t scale)
{
    uint32_t round = 10;

    const size_t offset = threadIdx.x, batch_id = blockIdx.x;
    size_t idx = offset * gridDim.x + batch_id;
    
    scale >>= 10;

    __shared__ FieldT shmem[NTT_BLK_SIZ];

    FieldT coeff = in[idx], other; idx >>= 10;

    for (uint32_t adj_bit = NTT_BLK_SIZ >> 1; adj_bit >= 32; adj_bit >>= 1, round++) {
        shmem[offset] = coeff;
        __syncthreads();    

        idx = idx & ~(scale >>= 1);
        other = shmem[offset ^ adj_bit];
        coeff = ((offset & adj_bit) == 0) ? (coeff + other) : (other - coeff) * omegas[idx << round];
        __syncthreads();
    }

    uint32_t *coeff_ptr = reinterpret_cast<uint32_t*>(&coeff);
    uint32_t *other_ptr = reinterpret_cast<uint32_t*>(&other);

    for (uint32_t adj_bit = 16; adj_bit >= 1; adj_bit >>= 1, round++) {
        idx = idx & ~(scale >>= 1);
        for (int k = 0; k < sizeof(FieldT) / sizeof(uint32_t); k++) other_ptr[k] = __shfl_xor_sync(0xffffffff, coeff_ptr[k], adj_bit);
        coeff = ((offset & adj_bit) == 0) ? (coeff + other) : (other - coeff) * omegas[idx << round];
    }

    out[batch_id * NTT_BLK_SIZ + offset] = coeff;
}

template <typename FieldT>
__global__ void onchip_butterfly_transformation_3(FieldT *in, FieldT *out, FieldT *omegas, size_t scale, size_t batch_size)
{
    uint32_t round = 20;

    const size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t offset = thread_id % batch_size, batch_id = thread_id / batch_size;
    size_t idx = offset * (scale / batch_size) + batch_id;

    scale >>= 20;
    
    extern __shared__ FieldT shmem[];

    FieldT coeff = in[idx], other; idx >>= 20;

    for (uint32_t adj_bit = batch_size >> 1; adj_bit >= 32; adj_bit >>= 1, round++) {
        shmem[threadIdx.x] = coeff;
        __syncthreads();    

        idx = idx & ~(scale >>= 1);
        other = shmem[threadIdx.x ^ adj_bit];
        coeff = ((offset & adj_bit) == 0) ? (coeff + other) : (other - coeff) * omegas[idx << round];
        __syncthreads();
    }

    uint32_t *coeff_ptr = reinterpret_cast<uint32_t*>(&coeff);
    uint32_t *other_ptr = reinterpret_cast<uint32_t*>(&other);

    for (uint32_t adj_bit = ((batch_size < 32) ? (batch_size >> 1) : 16); adj_bit >= 1; adj_bit >>= 1, round++) {
        idx = idx & ~(scale >>= 1);
        for (int k = 0; k < sizeof(FieldT) / sizeof(uint32_t); k++) other_ptr[k] = __shfl_xor_sync(0xffffffff, coeff_ptr[k], adj_bit);
        coeff = ((offset & adj_bit) == 0) ? (coeff + other) : (other - coeff) * omegas[idx << round];
    }

    out[batch_id * batch_size + offset] = coeff;
}