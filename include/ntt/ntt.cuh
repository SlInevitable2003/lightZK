#include "fields/alt_bn128-fp2.cuh"
#include "ntt/ntt_common.cuh"

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
    FieldT *scale_inverse;
    FieldT *omega_power_table, *coset_power_table;
    FieldT *imega_power_table, *ioset_power_table;
    
    TypedGpuArena arena;

public:
    NTTContext(size_t scale_, HostFT omega, HostFT coset) : scale(scale_) {
        assert((scale & (scale - 1)) == 0 && "Scale must be a power of 2");
        assert(scale >= 2048 && "Scale must be at least 2048");

        HostFT scale_in_field{scale};
        
        arena.register_alloc(scale_inverse, 1);
        arena.register_alloc(omega_power_table, scale);
        arena.register_alloc(coset_power_table, scale);
        arena.register_alloc(imega_power_table, scale);
        arena.register_alloc(ioset_power_table, scale);
        arena.commit();

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
        power_computing<<<1, 1>>>(coset_power_table, scale);
        power_computing<<<1, 1>>>(imega_power_table, scale);
        power_computing<<<1, 1>>>(ioset_power_table, scale);
    }

    void ntt(FieldT *poly, bool inverse = false) 
    {
        auto butterfly_transformation = [] __device__ (FieldT *poly, FieldT *omegas, uint32_t round, uint32_t adjoint_bit) {
            uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            
            uint32_t current_index = i & (adjoint_bit - 1);
            uint32_t self_index = ((i - current_index) << 1) | current_index;
            uint32_t adjo_index = self_index | adjoint_bit;
            FieldT self = poly[self_index], adjo = poly[adjo_index];

            poly[self_index] = self + adjo;
            poly[adjo_index] = (self - adjo) * omegas[current_index << round];
        };
        auto bitrev_permutation = [] __device__ (FieldT *poly, uint32_t round) {
            uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            uint32_t j = bit_rev(i, round);
            if (i < j) {
                FieldT tmp = poly[i];
                poly[i] = poly[j];
                poly[j] = tmp;
            }
        };

        uint32_t round, adjoint_bit;
        FieldT *omegas = inverse ? imega_power_table : omega_power_table;
        for (round = 0, adjoint_bit = scale / 2; adjoint_bit > 0; round ++, adjoint_bit >>= 1)
            kernel<<<scale / 2048, 1024>>>(butterfly_transformation, poly, omegas, round, adjoint_bit);
        kernel<<<scale / 1024, 1024>>>(bitrev_permutation, poly, round);
        if (inverse) {
            kernel<<<scale / 1024, 1024>>>([=] __device__ (FieldT *poly, FieldT *factor) {
                uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
                poly[i] *= factor[0];
            }, poly, scale_inverse);
        }
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
};