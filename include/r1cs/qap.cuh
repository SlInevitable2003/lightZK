#pragma once
#include "fields/alt_bn128-fp2.cuh"
#include "utils/arena.cuh"
#include "utils/arith.cuh"
#include "utils/cuda_check.cuh"

template <typename FieldT>
__global__ void power_table_kernel(FieldT *table, size_t len)
{
    table[0] = FieldT::one();
    FieldT fact = table[1];
    for (size_t i = 2; i < len; i++) table[i] = table[i - 1] * fact;
}

template <typename FieldT>
__global__ void lagrange_polynomial_kernel(const FieldT *omega_powers, const FieldT *params, FieldT *u, size_t m)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    const FieldT t = params[0];
    const FieldT Z_over_m = params[1];
    const FieldT w = omega_powers[i];

    u[i] = Z_over_m * w / (t - w);
}

template <typename FieldT, typename HostFT>
class QAPContext {
    size_t m;

    FieldT *omega_powers;
    FieldT *u;
    FieldT *params;   // [0] = t, [1] = Z(t) / m

    TypedGpuArena arena;

public:
    QAPContext(size_t m_, HostFT omega) : m(m_)
    {
        assert((m & (m - 1)) == 0 && "domain size must be a power of two");

        arena.register_alloc(omega_powers, m);
        arena.register_alloc(u, m);
        arena.register_alloc(params, 2);
        arena.commit("QAPContext");

        cudaMemcpy(omega_powers + 1, &omega, sizeof(FieldT), cudaMemcpyHostToDevice);
        power_table_kernel<<<1, 1>>>(omega_powers, m);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
    }

    void prepare(HostFT t, HostFT Zt)
    {
        HostFT inv_m = HostFT(m).inverse();
        HostFT params_host[2] = { t, Zt * inv_m };
        cudaMemcpy(params, params_host, 2 * sizeof(FieldT), cudaMemcpyHostToDevice);
    }

    void compute_lagrange()
    {
        const size_t blk = 256;
        lagrange_polynomial_kernel<<<ceil_div(m, blk), blk>>>(omega_powers, params, u, m);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
    }

    FieldT *lagrange() { return u; }
};
