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

template <typename ppT>
struct Groth16Proof {
    libff::G1<ppT> Ar, Bs1, zK, qZ;
    libff::G2<ppT> Bs2;
};

template <typename ppT>
struct Groth16ProveTest {
    size_t k, n, m;

    vector<libff::Fr<ppT>> z;
    SparseMatrix<libff::Fr<ppT>> mA, mB, mC;
    
    vector<libff::G1<ppT>> pkA1, pkB1, pkK, pkZ;
    vector<libff::G2<ppT>> pkB2;

    Groth16Proof<ppT> cpu_proof, gpu_proof;

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
                pkA1[i] = libff::G1<ppT>::random_element(); pkA1[i].to_affine_coordinates();
                pkB1[i] = libff::G1<ppT>::random_element(); pkB1[i].to_affine_coordinates();
                pkB2[i] = libff::G2<ppT>::random_element(); pkB2[i].to_affine_coordinates();
            }

            pkK.resize(n - k);
            #pragma omp parallel for
            for (size_t i = 0; i < pkK.size(); i++) { pkK[i] = libff::G1<ppT>::random_element(); pkK[i].to_affine_coordinates(); }

            pkZ.resize(m - 1);
            #pragma omp parallel for
            for (size_t i = 0; i < pkZ.size(); i++) { pkZ[i] = libff::G1<ppT>::random_element(); pkZ[i].to_affine_coordinates(); }

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
                libff::multi_exp<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method::multi_exp_method_BDLO12>(pkK.cbegin(), pkK.cend(), z.cbegin() + k + 1, z.cend(), 1),
                libff::multi_exp<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method::multi_exp_method_BDLO12>(pkZ.cbegin(), pkZ.cend(), mAz.cbegin(), mAz.cbegin() + m - 1, 1),
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
        bench_compute(*this, gpu_layout, gpu_proof);
        libff::leave_block("GPU Groth16 Compute");
        if (gpu_proof.Ar != cpu_proof.Ar) { gpu_proof.Ar.print(); cpu_proof.Ar.print(); assert(false && "GPU Groth16 proof does not match CPU result! (Ar)");}
        else if (gpu_proof.Bs1 != cpu_proof.Bs1) { gpu_proof.Bs1.print(); cpu_proof.Bs1.print(); assert(false && "GPU Groth16 proof does not match CPU result! (Bs1)");}
        else if (gpu_proof.zK != cpu_proof.zK) { gpu_proof.zK.print(); cpu_proof.zK.print(); assert(false && "GPU Groth16 proof does not match CPU result! (zK)");}
        else if (gpu_proof.qZ != cpu_proof.qZ) { gpu_proof.qZ.print(); cpu_proof.qZ.print(); assert(false && "GPU Groth16 proof does not match CPU result! (qZ)");}
        else if (gpu_proof.Bs2 != cpu_proof.Bs2) { gpu_proof.Bs2.print(); cpu_proof.Bs2.print(); assert(false && "GPU Groth16 proof does not match CPU result! (Bs2)");}
        else cout << "GPU Groth16 proof matches CPU result." << endl;
    }
};

struct Groth16ProveGPULayout {
    BucketContext<fr_t, libff::Fr<ppT>> z_bucket_ctx, mz_bucket_ctx;
    spMVMContext<fr_t, libff::Fr<ppT>, 3> spmvm_ctx;
    NTTContext<fr_t, libff::Fr<ppT>> ntt_ctx;
    MSMContext<fr_t, g1_t::affine_t, g1_t, g1_bucket_t, libff::Fr<ppT>, libff::G1<ppT>, false, 3> g1_sparse_msm_ctx;
    MSMContext<fr_t, g1_t::affine_t, g1_t, g1_bucket_t, libff::Fr<ppT>, libff::G1<ppT>> g1_dense_msm_ctx;
    MSMContext<fr_t, g2_t::affine_t, g2_t, g2_bucket_t, libff::Fr<ppT>, libff::G2<ppT>, true> g2_msm_ctx;
    
    fr_t *polys[3];
    TypedGpuArena arena;

    Groth16ProveGPULayout(
        size_t n, size_t m, SparseMatrix<libff::Fr<ppT>> **mats, 
        libff::Fr<ppT> omega, libff::Fr<ppT> coset,
        size_t window_bits = 13) :
        z_bucket_ctx(1 + n, window_bits),
        mz_bucket_ctx(m, window_bits),
        spmvm_ctx(m, 1 + n, mats),
        ntt_ctx(m, omega, coset),
        g1_sparse_msm_ctx(1 + n, window_bits),
        g1_dense_msm_ctx(m, window_bits),
        g2_msm_ctx(1 + n, window_bits)
    {
        arena.register_alloc(polys[0], m);
        arena.register_alloc(polys[1], m);
        arena.register_alloc(polys[2], m);
        arena.commit("PolyBuffer");
    }
};

void cuda_prove_setup(Groth16ProveTest<ppT> &test, Groth16ProveGPULayout &gpu_layout)
{
    gpu_layout.g1_sparse_msm_ctx.load_bases(test.pkA1.data(), true, 0);
    gpu_layout.g1_sparse_msm_ctx.load_bases(test.pkB1.data(), true, 1);
    vector<libff::G1<ppT>> buffer1(1 + test.n, libff::G1<ppT>::zero());
    for (int i = test.k + 1; i <= test.n; i++) buffer1[i] = test.pkK[i - (test.k + 1)];
    gpu_layout.g1_sparse_msm_ctx.load_bases(buffer1.data(), true, 2);
    vector<libff::G1<ppT>> buffer2(test.m, libff::G1<ppT>::zero());
    for (int i = 0; i < test.m - 1; i++) buffer2[i] = test.pkZ[i];
    gpu_layout.g1_dense_msm_ctx.load_bases(buffer2.data());
}

void cuda_prove_compute(Groth16ProveTest<ppT> &test, Groth16ProveGPULayout &gpu_layout, Groth16Proof<ppT> &result)
{
    gpu_layout.z_bucket_ctx.load_scalars(test.z.data());
    gpu_layout.spmvm_ctx.spmvm(gpu_layout.z_bucket_ctx.scalars, gpu_layout.polys);
    
    gpu_layout.ntt_ctx.intt(gpu_layout.polys[0]);
    gpu_layout.ntt_ctx.intt(gpu_layout.polys[1]);
    gpu_layout.ntt_ctx.intt(gpu_layout.polys[2]);
    gpu_layout.ntt_ctx.coset_ntt(gpu_layout.polys[0]);
    gpu_layout.ntt_ctx.coset_ntt(gpu_layout.polys[1]);
    gpu_layout.ntt_ctx.coset_ntt(gpu_layout.polys[2]);
    gpu_layout.ntt_ctx.A_times_B_minus_C_divided_by_Z(gpu_layout.polys[0], gpu_layout.polys[1], gpu_layout.polys[2]);
    gpu_layout.ntt_ctx.coset_intt(gpu_layout.polys[0]);
    gpu_layout.mz_bucket_ctx.load_scalars(gpu_layout.polys[0]);
    
    gpu_layout.z_bucket_ctx.process();
    gpu_layout.g1_sparse_msm_ctx.msm(gpu_layout.z_bucket_ctx, &result.Ar);
    gpu_layout.g1_sparse_msm_ctx.msm(gpu_layout.z_bucket_ctx, &result.Bs1, 1);
    gpu_layout.g1_sparse_msm_ctx.msm(gpu_layout.z_bucket_ctx, &result.zK, 2);
    gpu_layout.g2_msm_ctx.msm(gpu_layout.z_bucket_ctx, &result.Bs2);
    
    gpu_layout.mz_bucket_ctx.process();
    gpu_layout.g1_dense_msm_ctx.msm(gpu_layout.mz_bucket_ctx, &result.qZ);
}


int main(int argc, char *argv[])
{
    ppT::init_public_params();

    const size_t exp = 20;
    const size_t k = 1024, n = (1 << exp) - 1, m = 1 << exp;
    string pregen_option(argv[1]);
    assert(pregen_option == "-regen" || pregen_option == "-fast");
    Groth16ProveTest<ppT> groth16_prove_test(k, n, m, pregen_option == "-fast");

    SparseMatrix<libff::Fr<ppT>> *mats[3] = {&groth16_prove_test.mA, &groth16_prove_test.mB, &groth16_prove_test.mC};
    Groth16ProveGPULayout gpu_layout(
        n, m, mats,
        reinterpret_cast<const libff::Fr<ppT>*>(forward_roots_of_unity)[exp], libff::Fr<ppT>::multiplicative_generator
    );
    groth16_prove_test.gpu_bench(gpu_layout, cuda_prove_setup, cuda_prove_compute);

    return 0;
}