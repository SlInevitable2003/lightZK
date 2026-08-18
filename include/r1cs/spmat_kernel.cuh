#pragma once
#include <cstdint>

// y[col] = sum_r x[row] * vals[(col, row)]: a transposed SpMV over CSC data.
template <typename FieldT>
__global__ void transpose_spmv_kernel(const uint32_t *col_ptr, const uint32_t *row_idx, const FieldT *vals, const FieldT *x, FieldT *y, size_t cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    FieldT sum; sum.zero();
    for (uint32_t j = col_ptr[col]; j < col_ptr[col + 1]; j++)
        sum += vals[j] * x[row_idx[j]];
    y[col] = sum;
}
