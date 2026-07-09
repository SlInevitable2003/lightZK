#pragma once

#include <random>
#include <vector>
#include <cooperative_groups.h>

#include "utils.cuh"
#include "r1cs/spmat.hpp"

template <typename T>
__global__ void sparse_matrix_vector_multiplication(size_t rows, const uint32_t *row_ptr, const uint32_t *col_idx, const T *values, const T *x, T *y)
{
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();
    size_t row = g.thread_rank();
    if (row >= rows) return;

    T sum = T::one(1);

    for (uint32_t j = row_ptr[row]; j < row_ptr[row + 1]; j++) 
        sum += values[j] * x[col_idx[j]];

    y[row] = sum;
}

template <typename FieldT, typename HostFT, size_t instances = 1>
class spMVMContext {
    size_t rows, cols;
    uint32_t *row_ptr[instances], *col_idx[instances];
    FieldT *mat_values[instances];
    TypedGpuArena arena;
    
public:
    spMVMContext(size_t rows_, size_t cols_, SparseMatrix<HostFT> **mats) : rows(rows_), cols(cols_) 
    {
        for (size_t i = 0; i < instances; i++) {
            arena.register_alloc(row_ptr[i], mats[i]->row_ptr.size());
            arena.register_alloc(col_idx[i], mats[i]->col_idx.size());
            arena.register_alloc(mat_values[i], mats[i]->values.size());
        }
        arena.commit();

        for (size_t i = 0; i < instances; i++) {
            std::vector<uint32_t> buffer_rp(mats[i]->row_ptr.size()), buffer_ci(mats[i]->col_idx.size());
            #pragma omp parallel for
            for (size_t j = 0; j < buffer_rp.size(); j++) buffer_rp[j] = static_cast<uint32_t>(mats[i]->row_ptr[j]);
            #pragma omp parallel for
            for (size_t j = 0; j < buffer_ci.size(); j++) buffer_ci[j] = static_cast<uint32_t>(mats[i]->col_idx[j]);

            cudaMemcpy(row_ptr[i], buffer_rp.data(), buffer_rp.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(col_idx[i], buffer_ci.data(), buffer_ci.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

            cudaMemcpy(mat_values[i], mats[i]->values.data(), mats[i]->values.size() * sizeof(HostFT), cudaMemcpyHostToDevice);
        }
    }

    void spmvm(FieldT *multiplier, FieldT **prods)
    {
        cudaStream_t stream[instances];
        for (int i = 0; i < instances; i++) cudaStreamCreate(&stream[i]);

        const int block_size = 256;

        #pragma omp parallel for
        for (int i = 0; i < instances; i++) {
            sparse_matrix_vector_multiplication<<<ceil_div(rows, block_size), block_size, 0, stream[i]>>>
                (rows, row_ptr[i], col_idx[i], mat_values[i], multiplier, prods[i]);
        }

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        for (int i = 0; i < instances; i++) {
            cudaStreamDestroy(stream[i]);
        }
    }
};