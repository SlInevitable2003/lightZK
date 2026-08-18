#pragma once

#include <cstdint>
#include <random>
#include <vector>

template <typename S>
struct SparseMatrix {
    size_t num_cols;
    std::vector<size_t> row_ptr, col_idx;
    std::vector<S> values;

    void randomize(size_t rows, size_t cols) {
        row_ptr.resize(rows + 1);
        col_idx.resize(rows * 2);
        values.resize(rows * 2);

        std::random_device rd;
        std::mt19937_64 rng(rd());
        std::uniform_int_distribution<size_t> dist(0, cols - 1);

        #pragma omp parallel for
        for (size_t r = 0; r < rows; r++) {
            size_t c1 = dist(rng);
            size_t c2 = dist(rng);
            while (c2 == c1) c2 = dist(rng);

            col_idx[r * 2] = c1;
            col_idx[r * 2 + 1] = c2;

            values[r * 2] = S::random_element();
            values[r * 2 + 1] = S::random_element();
        }

        row_ptr[0] = 0;
        for (size_t r = 0; r < rows; r++) row_ptr[r + 1] = row_ptr[r] + 2;
    }
};

template <typename S>
struct CscMatrix {
    std::vector<uint32_t> col_ptr, row_idx;
    std::vector<S> values;

    static CscMatrix<S> from_csr(const SparseMatrix<S> &mat, size_t rows, size_t cols) {
        CscMatrix<S> csc;
        const size_t nnz = mat.col_idx.size();

        csc.col_ptr.assign(cols + 1, 0);
        for (size_t j = 0; j < nnz; j++) csc.col_ptr[mat.col_idx[j] + 1]++;
        for (size_t c = 0; c < cols; c++) csc.col_ptr[c + 1] += csc.col_ptr[c];

        csc.row_idx.resize(nnz);
        csc.values.resize(nnz);

        std::vector<uint32_t> cursor = csc.col_ptr;
        for (size_t r = 0; r < rows; r++)
            for (size_t j = mat.row_ptr[r]; j < mat.row_ptr[r + 1]; j++) {
                const size_t c = mat.col_idx[j];
                const size_t pos = cursor[c]++;
                csc.row_idx[pos] = static_cast<uint32_t>(r);
                csc.values[pos] = mat.values[j];
            }
        return csc;
    }
};