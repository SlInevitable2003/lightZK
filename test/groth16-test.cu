#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "libff/common/profiling.hpp"
#include <libff/common/utils.hpp>

#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include "libfqfft/evaluation_domain/get_evaluation_domain.hpp"
#include "libff/algebra/scalar_multiplication/multiexp.hpp"

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

    Groth16Proof<ppT> cpu_proof, gpu_proof;
public:
    Groth16ProveTest(size_t k_, size_t n_, size_t m_, bool from_binary = false) : k(k_), n(n_), m(m_) {
        size_t num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        cout << "Using " << num_threads << " threads for Groth16-Test preparation." << endl;

        if (from_binary) {
            libff::enter_block("Reading random proving keys");
            std::ifstream in("groth16_test_data.bin");
            if (!in) throw std::runtime_error("Failed to open input file.");
            
            z.resize(1 + n);
            in.read(reinterpret_cast<char*>(z.data()), z.size() * sizeof(libff::Fr<ppT>));
            
            mA.row_ptr.resize(m + 1), mA.col_idx.resize(m * 2), mA.values.resize(m * 2);
            mB.row_ptr.resize(m + 1), mB.col_idx.resize(m * 2), mB.values.resize(m * 2);
            mC.row_ptr.resize(m + 1), mC.col_idx.resize(m * 2), mC.values.resize(m * 2);
            in.read(reinterpret_cast<char*>(mA.row_ptr.data()), mA.row_ptr.size() * sizeof(size_t));
            in.read(reinterpret_cast<char*>(mA.col_idx.data()), mA.col_idx.size() * sizeof(size_t));
            in.read(reinterpret_cast<char*>(mA.values.data()), mA.values.size() * sizeof(libff::Fr<ppT>));
            in.read(reinterpret_cast<char*>(mB.row_ptr.data()), mB.row_ptr.size() * sizeof(size_t));
            in.read(reinterpret_cast<char*>(mB.col_idx.data()), mB.col_idx.size() * sizeof(size_t));
            in.read(reinterpret_cast<char*>(mB.values.data()), mB.values.size() * sizeof(libff::Fr<ppT>));
            in.read(reinterpret_cast<char*>(mC.row_ptr.data()), mC.row_ptr.size() * sizeof(size_t));
            in.read(reinterpret_cast<char*>(mC.col_idx.data()), mC.col_idx.size() * sizeof(size_t));
            in.read(reinterpret_cast<char*>(mC.values.data()), mC.values.size() * sizeof(libff::Fr<ppT>));

            pkA1.resize(1 + n), pkB1.resize(1 + n), pkB2.resize(1 + n);
            in.read(reinterpret_cast<char*>(pkA1.data()), pkA1.size() * sizeof(libff::G1<ppT>));
            in.read(reinterpret_cast<char*>(pkB1.data()), pkB1.size() * sizeof(libff::G1<ppT>));
            in.read(reinterpret_cast<char*>(pkB2.data()), pkB2.size() * sizeof(libff::G2<ppT>));

            pkK.resize(n - k);
            in.read(reinterpret_cast<char*>(pkK.data()), pkK.size() * sizeof(libff::G1<ppT>));

            pkZ.resize(m - 1);
            in.read(reinterpret_cast<char*>(pkZ.data()), pkZ.size() * sizeof(libff::G1<ppT>));

            in.close();
            libff::leave_block("Reading random proving keys");
        } else {
            libff::enter_block("Generating random proving keys");
            
            z.resize(1 + n);
            #pragma omp parallel for
            for (size_t i = 0; i < z.size(); i++) z[i] = libff::Fr<ppT>::random_element();

            mA.randomize(m, 1 + n), mB.randomize(m, 1 + n), mC.randomize(m, 1 + n);

            pkA1.resize(1 + n), pkB1.resize(1 + n), pkB2.resize(1 + n);
            #pragma omp parallel for
            for (size_t i = 0; i < pkA1.size(); i++) {
                pkA1[i] = libff::G1<ppT>::random_element();
                pkB1[i] = libff::G1<ppT>::random_element();
                pkB2[i] = libff::G2<ppT>::random_element();
            }

            pkK.resize(n - k);
            #pragma omp parallel for
            for (size_t i = 0; i < pkK.size(); i++) pkK[i] = libff::G1<ppT>::random_element();

            pkZ.resize(m - 1);
            #pragma omp parallel for
            for (size_t i = 0; i < pkZ.size(); i++) pkZ[i] = libff::G1<ppT>::random_element();

            libff::leave_block("Generating random proving keys");

            libff::enter_block("Writing random proving keys");
            ofstream out("groth16_test_data.bin");
            if (!out) throw runtime_error("Failed to open output file.");

            out.write(reinterpret_cast<char*>(z.data()), z.size() * sizeof(libff::Fr<ppT>));
            out.write(reinterpret_cast<char*>(mA.row_ptr.data()), mA.row_ptr.size() * sizeof(size_t));
            out.write(reinterpret_cast<char*>(mA.col_idx.data()), mA.col_idx.size() * sizeof(size_t));
            out.write(reinterpret_cast<char*>(mA.values.data()), mA.values.size() * sizeof(libff::Fr<ppT>));
            out.write(reinterpret_cast<char*>(mB.row_ptr.data()), mB.row_ptr.size() * sizeof(size_t));
            out.write(reinterpret_cast<char*>(mB.col_idx.data()), mB.col_idx.size() * sizeof(size_t));
            out.write(reinterpret_cast<char*>(mB.values.data()), mB.values.size() * sizeof(libff::Fr<ppT>));
            out.write(reinterpret_cast<char*>(mC.row_ptr.data()), mC.row_ptr.size() * sizeof(size_t));
            out.write(reinterpret_cast<char*>(mC.col_idx.data()), mC.col_idx.size() * sizeof(size_t));
            out.write(reinterpret_cast<char*>(mC.values.data()), mC.values.size() * sizeof(libff::Fr<ppT>));
            out.write(reinterpret_cast<char*>(pkA1.data()), pkA1.size() * sizeof(libff::G1<ppT>));
            out.write(reinterpret_cast<char*>(pkB1.data()), pkB1.size() * sizeof(libff::G1<ppT>));
            out.write(reinterpret_cast<char*>(pkB2.data()), pkB2.size() * sizeof(libff::G2<ppT>));  
            out.write(reinterpret_cast<char*>(pkK.data()), pkK.size() * sizeof(libff::G1<ppT>));
            out.write(reinterpret_cast<char*>(pkZ.data()), pkZ.size() * sizeof(libff::G1<ppT>));

            out.close();
            libff::leave_block("Writing random proving keys");
        }

        if (from_binary) {
            libff::enter_block("Reading binary Groth16 proof result for reference");
            std::ifstream in("groth16_test_result.bin");
            if (!in) throw std::runtime_error("Failed to open input file.");
            in.read(reinterpret_cast<char*>(&cpu_proof), sizeof(cpu_proof));

            in.close();
            libff::leave_block("Reading binary Groth16 proof result for reference");
        } else {
            libff::enter_block("Generating Groth16 Proof on CPU for reference");

            vector<libff::Fr<ppT>> mAz(m, libff::Fr<ppT>::zero()), mBz(m, libff::Fr<ppT>::zero()), mCz(m, libff::Fr<ppT>::zero());
            #pragma omp parallel for
            for (size_t i = 0; i < m; i++) {
                for (size_t j = mA.row_ptr[i]; j < mA.row_ptr[i + 1]; j++) mAz[i] += mA.values[j] * z[mA.col_idx[j]];
                for (size_t j = mB.row_ptr[i]; j < mB.row_ptr[i + 1]; j++) mBz[i] += mB.values[j] * z[mB.col_idx[j]];
                for (size_t j = mC.row_ptr[i]; j < mC.row_ptr[i + 1]; j++) mCz[i] += mC.values[j] * z[mC.col_idx[j]];
            }

            const std::shared_ptr<libfqfft::evaluation_domain<libff::Fr<ppT>>> domain = libfqfft::get_evaluation_domain<libff::Fr<ppT>>(m);
            assert(domain->m == m);

            domain->iFFT(mAz), domain->iFFT(mBz), domain->iFFT(mCz);
            
            domain->cosetFFT(mAz, libff::Fr<ppT>::multiplicative_generator);
            domain->cosetFFT(mBz, libff::Fr<ppT>::multiplicative_generator);
            domain->cosetFFT(mCz, libff::Fr<ppT>::multiplicative_generator);

            #pragma omp parallel for
            for (size_t i = 0; i < m; i++) { mAz[i] = mAz[i] * mBz[i] - mCz[i]; }
            
            domain->divide_by_Z_on_coset(mAz);
            domain->icosetFFT(mAz, libff::Fr<ppT>::multiplicative_generator);

            cpu_proof = {
                libff::multi_exp<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method::multi_exp_method_BDLO12>(pkA1.cbegin(), pkA1.cend(), z.cbegin(), z.cend(), 1),
                libff::multi_exp<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method::multi_exp_method_BDLO12>(pkB1.cbegin(), pkB1.cend(), z.cbegin(), z.cend(), 1),
                libff::multi_exp<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method::multi_exp_method_BDLO12>(pkK.cbegin(), pkK.cend(), z.cbegin() + k, z.cend(), 1),
                libff::multi_exp<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method::multi_exp_method_BDLO12>(pkZ.cbegin(), pkZ.cend(), mAz.cbegin(), mAz.cend() + m - 1, 1),
                libff::multi_exp<libff::G2<ppT>, libff::Fr<ppT>, libff::multi_exp_method::multi_exp_method_BDLO12>(pkB2.cbegin(), pkB2.cend(), z.cbegin(), z.cend(), 1)
            };

            libff::leave_block("Generating Groth16 Proof on CPU for reference");

            libff::enter_block("Writing binary Proof result for reference");
            ofstream out("groth16_test_result.bin");
            if (!out) throw runtime_error("Failed to open output file.");
            out.write(reinterpret_cast<char*>(&cpu_proof), sizeof(cpu_proof));
            out.close();
            libff::leave_block("Writing binary Proof result for reference");
        }
    }

    template <typename GL, typename FS, typename FC>
    void gpu_bench(GL &gpu_layout, FS bench_setup, FC bench_compute) {
        libff::enter_block("GPU Groth16 Setup");
        bench_setup(*this, gpu_layout);
        libff::leave_block("GPU Groth16 Setup");
        libff::enter_block("GPU Groth16 Compute");
        bench_compute(gpu_layout, gpu_proof);
        libff::leave_block("GPU Groth16 Compute");
        if (gpu_proof.Ar != cpu_proof.Ar) { gpu_proof.Ar.print(); cpu_proof.Ar.print(); assert(false && "GPU Groth16 proof does not match CPU result!");}
        else if (gpu_proof.Bs1 != cpu_proof.Bs1) { gpu_proof.Bs1.print(); cpu_proof.Bs1.print(); assert(false && "GPU Groth16 proof does not match CPU result!");}
        else if (gpu_proof.Bs2 != cpu_proof.Bs2) { gpu_proof.Bs2.print(); cpu_proof.Bs2.print(); assert(false && "GPU Groth16 proof does not match CPU result!");}
        else if (gpu_proof.zK != cpu_proof.zK) { gpu_proof.zK.print(); cpu_proof.zK.print(); assert(false && "GPU Groth16 proof does not match CPU result!");}
        else if (gpu_proof.qZ != cpu_proof.qZ) { gpu_proof.qZ.print(); cpu_proof.qZ.print(); assert(false && "GPU Groth16 proof does not match CPU result!");}
        else cout << "GPU Groth16 proof matches CPU result." << endl;
    }
};

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    const size_t k = 1024, n = 1 << 20, m = 1 << 20;
    string pregen_option(argv[1]);
    assert(pregen_option == "-regen" || pregen_option == "-fast");
    Groth16ProveTest<ppT> groth16_prove_test(k, n, m, pregen_option == "-fast");

    return 0;
}