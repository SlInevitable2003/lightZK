#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "libff/common/profiling.hpp"
#include <libff/common/utils.hpp>

#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include <libfqfft/evaluation_domain/get_evaluation_domain.hpp>

#include <omp.h>
using namespace std;

#include "api.h"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

/*
 * Groth16 proving key.
 *
 * The format follows groth16-test.cu exactly:
 *   - pkA1[1+n] : A_query  in G1,  one element per variable (index 0 is the constant '1')
 *   - pkB1[1+n] : B_query  in G1
 *   - pkB2[1+n] : B_query  in G2
 *   - pkK [1+n] : L_query  in G1,  zero-padded so private inputs k+1..n sit at
 *                 their natural column index; indices 0..k are the point at infinity
 *   - pkZ [m]   : H_query  in G1,  t^i * Z(t)/delta at indices 0..m-2, index m-1
 *                 is the point at infinity
 *
 * pkK and pkZ are already padded to the sizes expected by the bucket contexts in
 * groth16-test.cu, so no per-proof padding is needed before load_bases().
 *
 * Here k = |x| is the number of public inputs, n is the number of non-constant
 * variables (so the R1CS has 1+n columns) and m is the number of constraints.
 * Variable layout: column 0 is the constant '1', columns 1..k are the public
 * inputs and columns k+1..n are the private witnesses.
 */

template <typename ppT>
struct Groth16ProvingKeys {
    vector<libff::G1<ppT>> pkA1, pkB1, pkK, pkZ;
    vector<libff::G2<ppT>> pkB2;

    void print_size() const {
        libff::print_indent(); printf("* pkA1  (A_query, G1): %zu elements\n", pkA1.size());
        libff::print_indent(); printf("* pkB1  (B_query, G1): %zu elements\n", pkB1.size());
        libff::print_indent(); printf("* pkB2  (B_query, G2): %zu elements\n", pkB2.size());
        libff::print_indent(); printf("* pkK   (L_query, G1): %zu elements\n", pkK.size());
        libff::print_indent(); printf("* pkZ   (H_query, G1): %zu elements\n", pkZ.size());
    }
};

template <typename ppT>
Groth16ProvingKeys<ppT> groth16_setup_cpu(
    const SparseMatrix<libff::Fr<ppT>> &mA, const SparseMatrix<libff::Fr<ppT>> &mB, const SparseMatrix<libff::Fr<ppT>> &mC, size_t k,
    const libff::Fr<ppT> &t, const libff::Fr<ppT> &alpha, const libff::Fr<ppT> &beta, const libff::Fr<ppT> &delta_inv, const libff::Fr<ppT> &Zt)
{
    const size_t m = mA.row_ptr.size() - 1; // 行数 = 约束个数
    const size_t n = mA.num_cols - 1; // 列数 - 1 = |x| + |w|

    assert(mB.row_ptr.size() - 1 == m && mC.row_ptr.size() - 1 == m);
    assert(mB.num_cols - 1 == n && mC.num_cols - 1 == n);
    assert(k <= n);
    assert((m & (m - 1)) == 0 && "number of constraints must be a power of two");

    libff::enter_block("Call to groth16_setup_cpu");

    libff::enter_block("Compute evaluations of A, B, C, H at t");
    const std::shared_ptr<libfqfft::evaluation_domain<libff::Fr<ppT>>> domain = libfqfft::get_evaluation_domain<libff::Fr<ppT>>(m);
    assert(domain->m == m);
    const std::vector<libff::Fr<ppT>> u = domain->evaluate_all_lagrange_polynomials(t);
    std::vector<libff::Fr<ppT>> At(1 + n, libff::Fr<ppT>::zero());
    std::vector<libff::Fr<ppT>> Bt(1 + n, libff::Fr<ppT>::zero());
    std::vector<libff::Fr<ppT>> Ct(1 + n, libff::Fr<ppT>::zero());

    for (size_t r = 0; r < m; r++) {
        for (size_t j = mA.row_ptr[r]; j < mA.row_ptr[r + 1]; j++) At[mA.col_idx[j]] += u[r] * mA.values[j];
        for (size_t j = mB.row_ptr[r]; j < mB.row_ptr[r + 1]; j++) Bt[mB.col_idx[j]] += u[r] * mB.values[j];
        for (size_t j = mC.row_ptr[r]; j < mC.row_ptr[r + 1]; j++) Ct[mC.col_idx[j]] += u[r] * mC.values[j];
    }
    libff::leave_block("Compute evaluations of A, B, C, H at t");

    const libff::G1<ppT> g1 = libff::G1<ppT>::one();
    const libff::G2<ppT> g2 = libff::G2<ppT>::one();

    Groth16ProvingKeys<ppT> pk;

    libff::enter_block("Generate queries");
    libff::enter_block("Compute the A-query", false);
    pk.pkA1.resize(1 + n);
    #pragma omp parallel for
    for (size_t i = 0; i <= n; i++) pk.pkA1[i] = At[i] * g1;
    libff::leave_block("Compute the A-query", false);

    libff::enter_block("Compute the B-query", false);
    pk.pkB1.resize(1 + n);
    pk.pkB2.resize(1 + n);
    #pragma omp parallel for
    for (size_t i = 0; i <= n; i++) {
        pk.pkB1[i] = Bt[i] * g1;
        pk.pkB2[i] = Bt[i] * g2;
    }
    libff::leave_block("Compute the B-query", false);

    libff::enter_block("Compute the L-query", false);
    pk.pkK.resize(1 + n, libff::G1<ppT>::zero());
    #pragma omp parallel for
    for (size_t i = k + 1; i <= n; i++) pk.pkK[i] = ((beta * At[i] + alpha * Bt[i] + Ct[i]) * delta_inv) * g1;
    libff::leave_block("Compute the L-query", false);

    libff::enter_block("Compute the H-query", false);
    pk.pkZ.resize(m, libff::G1<ppT>::zero());
    const libff::Fr<ppT> Zt_delta_inv = Zt * delta_inv;

    libff::enter_block("Precompute powers of t", false);
    std::vector<libff::Fr<ppT>> tpows(m - 1);
    libff::Fr<ppT> ti = libff::Fr<ppT>::one();
    for (size_t i = 0; i < m - 1; i++) { tpows[i] = ti; ti *= t; }
    libff::leave_block("Precompute powers of t", false);

    #pragma omp parallel for
    for (size_t i = 0; i < m - 1; i++) pk.pkZ[i] = (tpows[i] * Zt_delta_inv) * g1;
    libff::leave_block("Compute the H-query", false);
    libff::leave_block("Generate queries");

    libff::leave_block("Call to groth16_setup_cpu");
    return pk;
}

template <typename ppT>
struct Groth16SetupTest {
    size_t k, n, m;

    SparseMatrix<libff::Fr<ppT>> mA, mB, mC;
    libff::Fr<ppT> t, alpha, beta, delta_inv, Zt;

    Groth16ProvingKeys<ppT> cpu_pk, gpu_pk;

    Groth16SetupTest(size_t k_, SparseMatrix<libff::Fr<ppT>> A, SparseMatrix<libff::Fr<ppT>> B, SparseMatrix<libff::Fr<ppT>> C) 
    : k(k_), mA(std::move(A)), mB(std::move(B)), mC(std::move(C))
    {
        size_t num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        cout << "Using " << num_threads << " threads for Groth16-Setup-Test preparation." << endl;

        m = mA.row_ptr.size() - 1;
        n = mA.num_cols - 1;
        assert(mB.row_ptr.size() - 1 == m && mC.row_ptr.size() - 1 == m);
        assert(mB.num_cols - 1 == n && mC.num_cols - 1 == n);
        assert(k <= n);

        libff::print_indent(); printf("* R1CS constraints: %zu\n", m);
        libff::print_indent(); printf("* R1CS variables:   %zu (public: %zu, private: %zu)\n", n + 1, k, n - k);

        libff::enter_block("Generate secret randomness");
        t = libff::Fr<ppT>::random_element();
        alpha = libff::Fr<ppT>::random_element();
        beta = libff::Fr<ppT>::random_element();
        const libff::Fr<ppT> delta = libff::Fr<ppT>::random_element();
        delta_inv = delta.inverse();
        libff::leave_block("Generate secret randomness");

        libff::enter_block("Compute vanishing polynomial Z(t)");
        Zt = (t ^ m) - libff::Fr<ppT>::one();
        libff::leave_block("Compute vanishing polynomial Z(t)");

        libff::enter_block("Generating Groth16 proving key on CPU");
        cpu_pk = groth16_setup_cpu<ppT>(mA, mB, mC, k, t, alpha, beta, delta_inv, Zt);
        cpu_pk.print_size();
        libff::leave_block("Generating Groth16 proving key on CPU");
    }

    template <typename GL, typename FS, typename FC, typename FL>
    void gpu_bench(GL &gpu_layout, FS bench_setup, FC bench_compute, FL bench_load) {
        libff::enter_block("GPU Groth16 Setup");
        bench_setup(*this, gpu_layout);
        libff::leave_block("GPU Groth16 Setup");
        libff::enter_block("GPU Groth16 Compute");
        bench_compute(*this, gpu_layout);
        libff::leave_block("GPU Groth16 Compute");
        libff::enter_block("GPU Groth16 Load");
        bench_load(*this, gpu_layout, gpu_pk);
        libff::leave_block("GPU Groth16 Load");

        auto check_g1 = [](const char *name, const vector<libff::G1<ppT>> &gpu, const vector<libff::G1<ppT>> &cpu) {
            assert(gpu.size() == cpu.size());
            for (size_t i = 0; i < cpu.size(); i++)
                if (gpu[i] != cpu[i]) {
                    printf("%s at index %zu\n", name, i);
                    gpu[i].print(); cpu[i].print();
                    assert(false);
                }
        };
        auto check_g2 = [](const char *name, const vector<libff::G2<ppT>> &gpu, const vector<libff::G2<ppT>> &cpu) {
            assert(gpu.size() == cpu.size());
            for (size_t i = 0; i < cpu.size(); i++)
                if (gpu[i] != cpu[i]) {
                    printf("%s at index %zu\n", name, i);
                    gpu[i].print(); cpu[i].print();
                    assert(false);
                }
        };

        check_g1("GPU pkA1 mismatch", gpu_pk.pkA1, cpu_pk.pkA1);
        check_g1("GPU pkB1 mismatch", gpu_pk.pkB1, cpu_pk.pkB1);
        check_g1("GPU pkK mismatch",  gpu_pk.pkK,  cpu_pk.pkK);
        check_g1("GPU pkZ mismatch",  gpu_pk.pkZ,  cpu_pk.pkZ);
        check_g2("GPU pkB2 mismatch", gpu_pk.pkB2, cpu_pk.pkB2);

        cout << "GPU Groth16 proving key matches CPU result." << endl;
    }
};

/*
 * GPU layout for the Groth16 setup.  Everything is uploaded in the setup phase
 * and all intermediates / results are produced in device memory during the
 * compute phase, so compute performs no host/device transfer.
 */
struct Groth16SetupGPULayout {
    size_t n, m;

    g1_t::affine_t *g1_base;
    g2_t::affine_t *g2_base;

    QAPContext<fr_t, libff::Fr<ppT>> qap;

    uint32_t *col_ptr[3], *row_idx[3];
    fr_t *mat_values[3];

    fr_t *tw;       /* [0]=t, [1]=alpha, [2]=beta, [3]=delta_inv, [4]=Z(t) */
    fr_t *t_pows;

    fr_t *At, *Bt, *Ct, *Lt, *Ht;
    g1_t *pkA1, *pkB1, *pkK, *pkZ;
    g2_t *pkB2;

    TypedGpuArena arena;

    Groth16SetupGPULayout(size_t n_, size_t m_, SparseMatrix<libff::Fr<ppT>> **mats, libff::Fr<ppT> omega)
        : n(n_), m(m_), qap(m, omega)
    {
        const size_t cols = 1 + n;

        arena.register_alloc(g1_base, 1);
        arena.register_alloc(g2_base, 1);
        arena.register_alloc(tw, 5);
        arena.register_alloc(t_pows, m);
        arena.register_alloc(At, cols);
        arena.register_alloc(Bt, cols);
        arena.register_alloc(Ct, cols);
        arena.register_alloc(Lt, cols);
        arena.register_alloc(Ht, m);
        arena.register_alloc(pkA1, cols);
        arena.register_alloc(pkB1, cols);
        arena.register_alloc(pkK, cols);
        arena.register_alloc(pkZ, m);
        arena.register_alloc(pkB2, cols);

        for (size_t i = 0; i < 3; i++) {
            arena.register_alloc(col_ptr[i], cols + 1);
            arena.register_alloc(row_idx[i], mats[i]->col_idx.size());
            arena.register_alloc(mat_values[i], mats[i]->col_idx.size());
        }
        arena.commit("Groth16SetupGPULayout");
    }
};

void cuda_setup_setup(Groth16SetupTest<ppT> &test, Groth16SetupGPULayout &gpu_layout)
{
    const size_t cols = 1 + test.n;

    libff::G1<ppT> g1 = libff::G1<ppT>::one(); g1.to_affine_coordinates();
    cudaMemcpy(gpu_layout.g1_base, &g1, sizeof(g1_t::affine_t), cudaMemcpyHostToDevice);
    libff::G2<ppT> g2 = libff::G2<ppT>::one(); g2.to_affine_coordinates();
    cudaMemcpy(gpu_layout.g2_base, &g2, sizeof(g2_t::affine_t), cudaMemcpyHostToDevice);

    libff::Fr<ppT> tw_host[5] = { test.t, test.alpha, test.beta, test.delta_inv, test.Zt };
    cudaMemcpy(gpu_layout.tw, tw_host, 5 * sizeof(fr_t), cudaMemcpyHostToDevice);

    gpu_layout.qap.prepare(test.t, test.Zt);

    SparseMatrix<libff::Fr<ppT>> *mats[3] = { &test.mA, &test.mB, &test.mC };
    for (size_t i = 0; i < 3; i++) {
        CscMatrix<libff::Fr<ppT>> csc = CscMatrix<libff::Fr<ppT>>::from_csr(*mats[i], test.m, cols);
        cudaMemcpy(gpu_layout.col_ptr[i], csc.col_ptr.data(), csc.col_ptr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_layout.row_idx[i], csc.row_idx.data(), csc.row_idx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_layout.mat_values[i], csc.values.data(), csc.values.size() * sizeof(fr_t), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(gpu_layout.t_pows + 1, &test.t, sizeof(fr_t), cudaMemcpyHostToDevice);
    power_table_kernel<<<1, 1>>>(gpu_layout.t_pows, test.m);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

void cuda_setup_compute(Groth16SetupTest<ppT> &test, Groth16SetupGPULayout &gpu_layout)
{
    const size_t cols = 1 + test.n, m = test.m, k = test.k;
    const size_t blk = 256;

    gpu_layout.qap.compute_lagrange();

    fr_t *out[3] = { gpu_layout.At, gpu_layout.Bt, gpu_layout.Ct };
    for (size_t i = 0; i < 3; i++) {
        transpose_spmv_kernel<<<ceil_div(cols, blk), blk>>>(
            gpu_layout.col_ptr[i], gpu_layout.row_idx[i], gpu_layout.mat_values[i],
            gpu_layout.qap.lagrange(), out[i], cols);
    }

    kernel<<<ceil_div(cols, blk), blk>>>([=] __device__ (const fr_t *At, const fr_t *Bt, const fr_t *Ct, const fr_t *tw, fr_t *Lt, size_t cols, size_t k) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= cols) return;
        if (i <= k) { Lt[i].zero(); return; }
        Lt[i] = (tw[2] * At[i] + tw[1] * Bt[i] + Ct[i]) * tw[3];
    }, gpu_layout.At, gpu_layout.Bt, gpu_layout.Ct, gpu_layout.tw, gpu_layout.Lt, cols, k);

    kernel<<<ceil_div(m, blk), blk>>>([=] __device__ (const fr_t *t_pows, const fr_t *tw, fr_t *Ht, size_t m) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= m) return;
        if (i >= m - 1) { Ht[i].zero(); return; }
        Ht[i] = t_pows[i] * tw[4] * tw[3];
    }, gpu_layout.t_pows, gpu_layout.tw, gpu_layout.Ht, m);

    fixmsm_kernel<fr_t, g1_t::affine_t, g1_t><<<ceil_div(cols, blk), blk>>>(gpu_layout.g1_base, gpu_layout.At, gpu_layout.pkA1, cols);
    fixmsm_kernel<fr_t, g1_t::affine_t, g1_t><<<ceil_div(cols, blk), blk>>>(gpu_layout.g1_base, gpu_layout.Bt, gpu_layout.pkB1, cols);
    fixmsm_kernel<fr_t, g2_t::affine_t, g2_t><<<ceil_div(cols, blk), blk>>>(gpu_layout.g2_base, gpu_layout.Bt, gpu_layout.pkB2, cols);
    fixmsm_kernel<fr_t, g1_t::affine_t, g1_t><<<ceil_div(cols, blk), blk>>>(gpu_layout.g1_base, gpu_layout.Lt, gpu_layout.pkK, cols);
    fixmsm_kernel<fr_t, g1_t::affine_t, g1_t><<<ceil_div(m, blk), blk>>>(gpu_layout.g1_base, gpu_layout.Ht, gpu_layout.pkZ, m);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

void cuda_setup_load(Groth16SetupTest<ppT> &test, Groth16SetupGPULayout &gpu_layout, Groth16ProvingKeys<ppT> &result)
{
    const size_t cols = 1 + test.n;

    result.pkA1.resize(cols);
    result.pkB1.resize(cols);
    result.pkK.resize(cols);
    result.pkZ.resize(test.m);
    result.pkB2.resize(cols);

    cudaMemcpy(result.pkA1.data(), gpu_layout.pkA1, cols * sizeof(g1_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.pkB1.data(), gpu_layout.pkB1, cols * sizeof(g1_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.pkK.data(),  gpu_layout.pkK,  cols * sizeof(g1_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.pkZ.data(),  gpu_layout.pkZ,  test.m * sizeof(g1_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.pkB2.data(), gpu_layout.pkB2, cols * sizeof(g2_t), cudaMemcpyDeviceToHost);
}

using namespace LightZK;
int main(int argc, char *argv[])
{
    ppT::init_public_params();

    size_t s = 100;
    if (argc > 1) s = atoi(argv[1]);

    R1CSManager<libff::Fr<ppT>> mgr;

    vector<Variable<libff::Fr<ppT>>> A, B, C, P;
    for (size_t i = 0; i < s * s; i++) A.push_back(Variable<libff::Fr<ppT>>(VariableType::Public, mgr));
    for (size_t i = 0; i < s * s; i++) B.push_back(Variable<libff::Fr<ppT>>(VariableType::Public, mgr));
    for (size_t i = 0; i < s * s; i++) C.push_back(Variable<libff::Fr<ppT>>(VariableType::Public, mgr));
    for (size_t i = 0; i < s * s * s; i++) P.push_back(Variable<libff::Fr<ppT>>(VariableType::Private, mgr));

    for (size_t i = 0; i < s; i++)
        for (size_t j = 0; j < s; j++)
            for (size_t l = 0; l < s; l++)
                mgr.add_constraint(P[i * s * s + j * s + l], A[i * s + l], B[l * s + j]);

    for (size_t i = 0; i < s; i++)
        for (size_t j = 0; j < s; j++) {
            LinearCombination<libff::Fr<ppT>> sum;
            for (size_t l = 0; l < s; l++) sum += P[i * s * s + j * s + l];
            mgr.add_constraint(C[i * s + j], sum);
        }

    SparseMatrix<libff::Fr<ppT>> A_mat, B_mat, C_mat;
    mgr.gen_spmat(A_mat, B_mat, C_mat, true);

    const size_t k = 3 * s * s;

    Groth16SetupTest<ppT> setup_test(k, std::move(A_mat), std::move(B_mat), std::move(C_mat));

    SparseMatrix<libff::Fr<ppT>> *mats[3] = { &setup_test.mA, &setup_test.mB, &setup_test.mC };
    Groth16SetupGPULayout gpu_layout(
        setup_test.n, setup_test.m, mats,
        reinterpret_cast<const libff::Fr<ppT>*>(forward_roots_of_unity)[log2_floor(setup_test.m)]
    );
    setup_test.gpu_bench(gpu_layout, cuda_setup_setup, cuda_setup_compute, cuda_setup_load);

    return 0;
}
