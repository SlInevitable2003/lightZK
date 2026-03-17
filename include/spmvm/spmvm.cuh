#pragma once

#include <random>
#include <vector>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "utils.cuh"

template <typename T>
struct SparseMatrix {
    std::vector<size_t> row_ptr, col_idx;
    std::vector<T> values;

    void randomize(size_t rows, size_t cols) {
        row_ptr.resize(rows + 1);
        col_idx.resize(rows * 2);
        values.resize(rows * 2);

        std::random_device rd;
        std::mt19937_64 rng(rd());
        std::uniform_int_distribution<size_t> dist(0, cols - 1);

        row_ptr[0] = 0;
        #pragma omp parallel for
        for (size_t r = 0; r < rows; r++) {
            size_t c1 = dist(rng);
            size_t c2 = dist(rng);
            while (c2 == c1) c2 = dist(rng);

            col_idx[r * 2] = c1;
            col_idx[r * 2 + 1] = c2;

            values[r * 2] = T::random_element();
            values[r * 2 + 1] = T::random_element();

            row_ptr[r + 1] = row_ptr[r] + 2;
        }
    }
};

template <typename FieldT, typename HostFT, size_t instances = 1>
class spMVMContext {
    size_t rows, cols;
    uint32_t *row_ptr[instances], *col_idx[instances];
    FieldT *mat_values[instances];
    TypedGpuArena arena;
    
public:
    spMVMContext(size_t rows_, size_t cols_, SparseMatrix<HostFT> *mats) : rows(rows_), cols(cols_) 
    {
        for (size_t i = 0; i < instances; i++) {
            arena.register_alloc(row_ptr[i], mats[i].row_ptr.size());
            arena.register_alloc(col_idx[i], mats[i].col_idx.size());
            arena.register_alloc(mat_values[i], mats[i].mat_values.size());
        }
        arena.commit();

        for (size_t i = 0; i < instances; i++) {
            std::vector<uint32_t> buffer_rp(mats[i].row_ptr.size()), buffer_ci(mats[i].col_idx.size());
            #pragma omp parallel for
            for (size_t j = 0; j < buffer_rp.size(); j++) buffer_rp[j] = static_cast<uint32_t>(mats[i].row_ptr[j]);
            #pragma omp parallel for
            for (size_t j = 0; j < buffer_ci.size(); j++) buffer_ci[j] = static_cast<uint32_t>(mats[i].col_idx[j]);
            
            thrust::copy(thrust::device, buffer_rp.begin(), buffer_rp.end(), row_ptr[i]);
            thrust::copy(thrust::device, buffer_ci.begin(), buffer_ci.end(), col_idx[i]);

            cudaMemcpy(mat_values[i], mats[i].values.data(), mats[i].values.size() * sizeof(HostFT), cudaMemcpyHostToDevice);
        }
    }

    void spmvm(FieldT *vector_values) 
    {
        cudaStream_t stream[instances];
        for (int i = 0; i < instances; i++) cudaStreamCreate(&stream[i]);

        // #pragma omp parallel
        // for (int i = 0; i < instances; i++);
    }
};