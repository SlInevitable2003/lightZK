#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "libff/common/profiling.hpp"
#include <libff/common/utils.hpp>

#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include "libsnark/relations/constraint_satisfaction_problems/r1cs/examples/r1cs_examples.hpp"
#include "libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/examples/run_r1cs_gg_ppzksnark.hpp"

#include <omp.h>
using namespace std;

#include "api.h"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

template <typename T>
struct SparseMatrix {
    vector<size_t> row_ptr, col_idx;
    vector<T> values;

    void randomize(size_t rows, size_t cols) {
        row_ptr.resize(rows + 1);
        col_idx.resize(rows * 2);
        values.resize(rows * 2);

        std::random_device rd;
        std::mt19937_64 rng(rd());
        std::uniform_int_distribution<size_t> dist(0, cols - 1);

        row_ptr[0] = 0;
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

template <typename ppT>
struct Groth16Proof {
    libff::G1<ppT> Ar, Bs1, zK, qZ;
    libff::G2<ppT> Bs2;
};

template <typename ppT>
class Groth16ProveTest {
    size_t k, n, m;

    vector<libff::Fr<ppT>> z;
    SparseMatrix<libff::Fr<ppT>> mA, mB, mC;
    
    vector<libff::G1<ppT>> pkA1, pkB1, pkK, pkZ;
    vector<libff::G2<ppT>> pkB2;

    libff::G1<ppT> alpha1, beta1, delta1;
    libff::G2<ppT> beta2, delta2;

    Groth16Proof<ppT> cpu_proof, gpu_proof;
public:
    Groth16ProveTest(size_t k_, size_t n_, size_t m_) : k(k_), n(n_), m(m_) {
        size_t num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        cout << "Using " << num_threads << " threads for Groth16-Test preparation." << endl;

        if (from_binary) {
            libff::enter_block("Reading binary scalars and points");
            std::ifstream in("msm_test_data.bin");
            if (!in) throw std::runtime_error("Failed to open input file.");
            
            scalars.resize(n); points.resize(n);
            in.read(reinterpret_cast<char*>(scalars.data()), n * sizeof(libff::Fr<ppT>));
            in.read(reinterpret_cast<char*>(points.data()), n * sizeof(libff::G1<ppT>));

            in.close();
            libff::leave_block("Reading binary scalars and points");
        } else {
            libff::enter_block("Generating random proving keys");
            
            libff::leave_block("Generating random proving keys");

            libff::enter_block("Writing binary scalars and points");
            ofstream out("msm_test_data.bin");
            if (!out) throw runtime_error("Failed to open output file.");

            out.write(reinterpret_cast<char*>(scalars.data()), n * sizeof(libff::Fr<ppT>));
            out.write(reinterpret_cast<char*>(points.data()), n * sizeof(libff::G1<ppT>));

            out.close();
            libff::leave_block("Writing binary scalars and points");
        }

        if (from_binary) {
            libff::enter_block("Reading binary MSM result for reference");
            std::ifstream in("msm_test_result.bin");
            if (!in) throw std::runtime_error("Failed to open input file.");
            in.read(reinterpret_cast<char*>(&cpu_result), sizeof(libff::G1<ppT>));

            in.close();
            libff::leave_block("Reading binary MSM result for reference");
        } else {
            libff::enter_block("Computing CPU MSM result for reference");
            size_t batch_size = (n + num_threads - 1) / num_threads;
            vector<libff::G1<ppT>> results(num_threads, libff::G1<ppT>::zero());
            for (size_t i = 0; i < num_threads; i++) {
                size_t start = i * batch_size;
                size_t end = min(start + batch_size, n);
                if (start >= end) continue;
                results[i] = libff::multi_exp<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method::multi_exp_method_BDLO12>(
                    points.cbegin() + start,
                    points.cbegin() + end,
                    scalars.cbegin() + start,
                    scalars.cbegin() + end,
                    1
                );
            }
            cpu_result = libff::G1<ppT>::zero();
            for (const auto &res : results) cpu_result = cpu_result + res;
            libff::leave_block("Computing CPU MSM result for reference");

            libff::enter_block("Writing binary MSM result for reference");
            ofstream out("msm_test_result.bin");
            if (!out) throw runtime_error("Failed to open output file.");
            out.write(reinterpret_cast<char*>(&cpu_result), sizeof(libff::G1<ppT>));
            out.close();
            libff::leave_block("Writing binary MSM result for reference");
        }
    }

    template <typename GL, typename FS, typename FC>
    void gpu_bench(GL &gpu_layout, FS bench_setup, FC bench_compute) {
        libff::enter_block("GPU MSM Setup");
        bench_setup(scalars, points, gpu_layout);
        libff::leave_block("GPU MSM Setup");
        libff::enter_block("GPU MSM Compute");
        bench_compute(gpu_layout, gpu_result);
        libff::leave_block("GPU MSM Compute");
        if (gpu_result != cpu_result) {
            gpu_result.print(); cpu_result.print();
            assert(false && "GPU MSM result does not match CPU result!");
        }
    }
};

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    return 0;
}