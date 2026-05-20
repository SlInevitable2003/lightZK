#pragma once

#include "fields/alt_bn128-fp2.cuh"
#include "ntt/ntt_common.cuh"
#include "ntt/ntt_kernel.cuh"

#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

template <typename FieldT>
__global__ void power_computing(FieldT *arr, size_t len) 
{
    arr[0] = FieldT::one();
    FieldT fact = arr[1];
    for (int i = 2; i < len; i++) arr[i] = arr[i - 1] * fact;
}

template<typename FieldT, typename HostFT>
class NTTContext {
    size_t scale;
    FieldT *buffer;
    FieldT *scale_inverse;
    FieldT *omega_power_table, *coset_power_table;
    FieldT *imega_power_table, *ioset_power_table;
    
    TypedGpuArena arena;

public:
    NTTContext(size_t scale_, HostFT omega, HostFT coset) : scale(scale_) {
        assert((scale & (scale - 1)) == 0 && "Scale must be a power of 2");
        assert(scale >= (1 << 20) && "Scale must be at least 2^20");

        HostFT scale_in_field{scale};
        
        arena.register_alloc(scale_inverse, 1);
        arena.register_alloc(omega_power_table, scale);
        arena.register_alloc(coset_power_table, scale + 1);
        arena.register_alloc(imega_power_table, scale);
        arena.register_alloc(ioset_power_table, scale);

        arena.register_alloc(buffer, scale);
        arena.commit("NTTContext");

        static_assert(sizeof(HostFT) == sizeof(FieldT), "HostFT and FieldT must have the same size");
        scale_in_field.invert();
        cudaMemcpy(scale_inverse, &scale_in_field, sizeof(FieldT), cudaMemcpyHostToDevice);
        cudaMemcpy(omega_power_table + 1, &omega, sizeof(FieldT), cudaMemcpyHostToDevice);
        cudaMemcpy(coset_power_table + 1, &coset, sizeof(FieldT), cudaMemcpyHostToDevice);
        
        omega.invert();
        coset.invert();
        cudaMemcpy(imega_power_table + 1, &omega, sizeof(FieldT), cudaMemcpyHostToDevice);
        cudaMemcpy(ioset_power_table + 1, &coset, sizeof(FieldT), cudaMemcpyHostToDevice);

        power_computing<<<1, 1>>>(omega_power_table, scale);
        power_computing<<<1, 1>>>(coset_power_table, scale + 1);
        power_computing<<<1, 1>>>(imega_power_table, scale);
        power_computing<<<1, 1>>>(ioset_power_table, scale);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
    }

    void ntt(FieldT *poly, bool inverse = false) 
    {
        uint32_t rows = 1, cols = scale;
        while ((rows << 1) <= (cols >> 1)) rows <<= 1, cols >>= 1;
        // printf("%u, %u\n", rows, cols);

        FieldT *omegas = inverse ? imega_power_table : omega_power_table;

        // multi_row_ntt<<<1, FA_BLK_SIZ>>>(poly, omegas, 1, scale);
        // multi_row_bitrev_permutation<<<scale / FA_BLK_SIZ, FA_BLK_SIZ>>>(poly, 1, scale);
        
        transpose<<<{cols / TILE_DIM, rows / TILE_DIM}, {TILE_DIM, TILE_DIM}>>>(poly, buffer, rows, cols);
        multi_row_ntt<<<cols, FA_BLK_SIZ>>>(buffer, omegas, cols, rows);
        multi_row_bitrev_permutation<<<cols, FA_BLK_SIZ>>>(buffer, cols, rows);
        
        kernel<<<scale / FA_BLK_SIZ, FA_BLK_SIZ>>>([=] __device__ (FieldT *in, FieldT *out, FieldT *omegas) {
            namespace cg = cooperative_groups;
            cg::grid_group g = cg::this_grid();
            const uint32_t i = g.thread_rank();
            uint32_t j = i % rows, k = i / rows;
            out[i] = in[i] * omegas[j * k];
        }, buffer, poly, omegas);

        transpose<<<{rows / TILE_DIM, cols / TILE_DIM}, {TILE_DIM, TILE_DIM}>>>(poly, buffer, cols, rows);
        multi_row_ntt<<<rows, FA_BLK_SIZ>>>(buffer, omegas, rows, cols);
        multi_row_bitrev_permutation<<<rows, FA_BLK_SIZ>>>(buffer, rows, cols);

        transpose<<<{cols / TILE_DIM, rows / TILE_DIM}, {TILE_DIM, TILE_DIM}>>>(buffer, poly, rows, cols);

        if (inverse) {
            kernel<<<scale / FA_BLK_SIZ, FA_BLK_SIZ>>>([=] __device__ (FieldT *poly, FieldT *factor) {
                uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
                poly[i] *= factor[0];
            }, poly, scale_inverse);
        }

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());        
    }

    void intt(FieldT *poly) { ntt(poly, true); }

    void coset_ntt(FieldT *poly) {
        thrust::transform(thrust::device, poly, poly + scale, coset_power_table, poly, [] __device__ (FieldT a, FieldT b) { return a * b; });
        ntt(poly);
    }
    void coset_intt(FieldT *poly) {
        intt(poly);
        thrust::transform(thrust::device, poly, poly + scale, ioset_power_table, poly, [] __device__ (FieldT a, FieldT b) { return a * b; });
    }

    void A_times_B_minus_C_divided_by_Z(FieldT *polyA, FieldT *polyB, FieldT *polyC) {
        thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(polyA, polyB, polyC)),
            thrust::make_zip_iterator(thrust::make_tuple(polyA, polyB, polyC)) + scale,
            polyA,
            [] __device__ (const thrust::tuple<FieldT, FieldT, FieldT>& t) {
                FieldT a = thrust::get<0>(t);
                FieldT b = thrust::get<1>(t);
                FieldT c = thrust::get<2>(t);
                return a * b - c;
            }
        );

        FieldT *coset_power_table = this->coset_power_table;
        thrust::for_each(thrust::device, polyA, polyA + scale, [=] __device__ (FieldT &val) {
            FieldT factor = coset_power_table[scale] - FieldT::one();
            val /= factor;
        });
    }
};