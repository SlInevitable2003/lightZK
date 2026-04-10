#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"

#include <omp.h>
using namespace std;

#include "api.h"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

template <typename ppT>
class SpMVTest {
    size_t rows, cols;
    vector<libff::Fr<ppT>> x;
    vector<libff::Fr<ppT>> cpu_result, gpu_result;

public:
    SparseMatrix<libff::Fr<ppT>> mat;

    SpMVTest(size_t rows_, size_t cols_, bool from_binary = false)
        : rows(rows_), cols(cols_)
    {
        size_t num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        cout << "Using " << num_threads << " threads for SpMV-Test preparation." << endl;

        if (from_binary) {
            libff::enter_block("Reading binary input");

            ifstream in("spmv_test_data.bin");
            if (!in) throw runtime_error("Failed to open input file.");

            // 读矩阵
            size_t nnz;
            in.read((char*)&nnz, sizeof(size_t));

            mat.row_ptr.resize(rows + 1);
            mat.col_idx.resize(nnz);
            mat.values.resize(nnz);

            in.read((char*)mat.row_ptr.data(), (rows + 1) * sizeof(size_t));
            in.read((char*)mat.col_idx.data(), nnz * sizeof(size_t));
            in.read((char*)mat.values.data(), nnz * sizeof(libff::Fr<ppT>));

            // 读向量
            x.resize(cols);
            in.read((char*)x.data(), cols * sizeof(libff::Fr<ppT>));

            in.close();
            libff::leave_block("Reading binary input");
        } else {
            libff::enter_block("Generating random sparse matrix");
            mat.randomize(rows, cols);
            libff::leave_block("Generating random sparse matrix");

            libff::enter_block("Generating random vector");
            x.resize(cols);
            #pragma omp parallel for
            for (size_t i = 0; i < cols; i++)
                x[i] = libff::Fr<ppT>::random_element();
            libff::leave_block("Generating random vector");

            // 写入 binary
            libff::enter_block("Writing binary input");
            ofstream out("spmv_test_data.bin");

            size_t nnz = mat.values.size();
            out.write((char*)&nnz, sizeof(size_t));
            out.write((char*)mat.row_ptr.data(), (rows + 1) * sizeof(size_t));
            out.write((char*)mat.col_idx.data(), nnz * sizeof(size_t));
            out.write((char*)mat.values.data(), nnz * sizeof(libff::Fr<ppT>));
            out.write((char*)x.data(), cols * sizeof(libff::Fr<ppT>));

            out.close();
            libff::leave_block("Writing binary input");
        }

        // ===== CPU reference =====
        if (from_binary) {
            libff::enter_block("Reading CPU result");

            ifstream in("spmv_test_result.bin");
            cpu_result.resize(rows);
            in.read((char*)cpu_result.data(), rows * sizeof(libff::Fr<ppT>));
            in.close();

            libff::leave_block("Reading CPU result");
        } else {
            libff::enter_block("Computing CPU SpMV");

            cpu_result.resize(rows);

            #pragma omp parallel for
            for (size_t r = 0; r < rows; r++) {
                libff::Fr<ppT> sum = libff::Fr<ppT>::zero();

                for (size_t j = mat.row_ptr[r]; j < mat.row_ptr[r + 1]; j++) {
                    sum += mat.values[j] * x[mat.col_idx[j]];
                }

                cpu_result[r] = sum;
            }

            libff::leave_block("Computing CPU SpMV");

            libff::enter_block("Writing CPU result");
            ofstream out("spmv_test_result.bin");
            out.write((char*)cpu_result.data(), rows * sizeof(libff::Fr<ppT>));
            out.close();
            libff::leave_block("Writing CPU result");
        }
    }

    template <typename GL, typename FS, typename FC, typename FL>
    void gpu_bench(GL &gpu_layout, FS bench_setup, FC bench_compute, FL bench_load)
    {
        cudaHostRegister(x.data(), x.size() * sizeof(libff::Fr<ppT>), cudaHostRegisterDefault);

        libff::enter_block("GPU SpMV Setup");
        bench_setup(mat, x, gpu_layout);
        libff::leave_block("GPU SpMV Setup");

        libff::enter_block("GPU SpMV Compute");
        bench_compute(gpu_layout);
        libff::leave_block("GPU SpMV Compute");

        libff::enter_block("GPU SpMV Load");
        bench_load(gpu_layout, gpu_result);
        libff::leave_block("GPU SpMV Load");

        int i;
        for (i = 0; i < rows; i++) {
            if (cpu_result[i] != gpu_result[i]) break;
        }

        if (i < rows) {
            cout << "Mismatch at " << i << endl;
            gpu_result[i].print();
            cpu_result[i].print();
            assert(false && "GPU SpMV result mismatch!");
        }

        cudaHostUnregister(x.data());
    }
};

#define INSTANCE 3

struct SpMVGPULayout {
    size_t rows, cols;

    fr_t *d_x, *d_y[INSTANCE];

    spMVMContext<fr_t, libff::Fr<ppT>, INSTANCE> spmvm_ctx;

    SpMVGPULayout(size_t r, size_t c, SparseMatrix<libff::Fr<ppT>> **mat)
        : rows(r), cols(c), spmvm_ctx(rows, cols, mat)
    {
        cudaMalloc(&d_x, cols * sizeof(fr_t));
        for (int i = 0; i < INSTANCE; i++) cudaMalloc(&d_y[i], rows * sizeof(fr_t));
    }

    ~SpMVGPULayout() { cudaFree(d_x); for (int i = 0; i < INSTANCE; i++) cudaFree(d_y[i]); }
};

void cuda_spmv_setup(SparseMatrix<libff::Fr<ppT>> &mat, vector<libff::Fr<ppT>> &x, SpMVGPULayout &gpu_layout)
{
    cudaMemcpy(gpu_layout.d_x, x.data(), gpu_layout.cols * sizeof(fr_t), cudaMemcpyHostToDevice);
}

void cuda_spmv_compute(SpMVGPULayout &gpu_layout)
{
    gpu_layout.spmvm_ctx.spmvm(gpu_layout.d_x, gpu_layout.d_y);
}

void cuda_spmv_load(SpMVGPULayout &gpu_layout, vector<libff::Fr<ppT>> &gpu_result)
{
    gpu_result.resize(gpu_layout.rows);
    cudaMemcpy(gpu_result.data(), gpu_layout.d_y[INSTANCE - 1], gpu_layout.rows * sizeof(fr_t), cudaMemcpyDeviceToHost);
}

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    string opt(argv[1]);
    assert(opt == "-regen" || opt == "-fast");

    size_t rows = 1 << 22;
    size_t cols = 1 << 22;

    SpMVTest<ppT> test(rows, cols, opt == "-fast");
    SparseMatrix<libff::Fr<ppT>> *mat_array[INSTANCE];
    for (int i = 0; i < INSTANCE; i++) mat_array[i] = &test.mat;
    SpMVGPULayout layout(rows, cols, mat_array);

    test.gpu_bench(layout, cuda_spmv_setup, cuda_spmv_compute, cuda_spmv_load);
}