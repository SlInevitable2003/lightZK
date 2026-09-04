#include <cstdlib>
#include <iostream>
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

#ifndef APP_DATA_DIR
#define APP_DATA_DIR "app/data"
#endif

/*
    证明密钥
    member      size  curve  note
    pkA1        1+n   G1
    pkB1        1+n   G1
    pkB2        1+n   G2
    pkK         1+n   G1     以 0 填充 pkK[0..=k]
    pkZ         m     G1     pkZ[i] = [t^i * Z(t) / delta]_1 对于 i in 0..=m-2, 而 pkZ[m-1] = 0
    alpha_g1    1     G1     [alpha]_1
    beta_g1     1     G1     [beta]_1
    beta_g2     1     G2     [beta]_2
    delta_g1    1     G1     [delta]_1
    delta_g2    1     G2     [delta]_2
*/
template <typename ppT>
struct Groth16ProvingKeys {
    vector<libff::G1<ppT>> pkA1, pkB1, pkK, pkZ;
    vector<libff::G2<ppT>> pkB2;

    libff::G1<ppT> alpha_g1, beta_g1, delta_g1;
    libff::G2<ppT> beta_g2, delta_g2;

    void print_size() const {
        libff::print_indent(); printf("* pkA1  (A_query, G1): %zu elements\n", pkA1.size());
        libff::print_indent(); printf("* pkB1  (B_query, G1): %zu elements\n", pkB1.size());
        libff::print_indent(); printf("* pkB2  (B_query, G2): %zu elements\n", pkB2.size());
        libff::print_indent(); printf("* pkK   (L_query, G1): %zu elements\n", pkK.size());
        libff::print_indent(); printf("* pkZ   (H_query, G1): %zu elements\n", pkZ.size());
        libff::print_indent(); printf("* alpha_g1/beta_g1/delta_g1 (G1) + beta_g2/delta_g2 (G2): blinding bases\n");
    }
};

/*
    验证密钥
    member              size    curve   note
    vk_alpha_g1_beta_g2 1       GT      e([alpha]_1, [beta]_2)
    vk_gamma_g2         1       G2      [gamma]_2
    vk_delta_g2         1       G2      [delta]_2
    vk_IC               k+1     G1      vk_IC[i] = [(beta*A_i+alpha*B_i+C_i)/gamma]_1
*/
template <typename ppT>
struct Groth16VerificationKeys {
    libff::GT<ppT> alpha_g1_beta_g2;
    libff::G2<ppT> gamma_g2;
    libff::G2<ppT> delta_g2;
    vector<libff::G1<ppT>> IC;

    void print_size() const {
        libff::print_indent(); printf("* vk_alpha_g1_beta_g2 (GT): 1 element\n");
        libff::print_indent(); printf("* vk_gamma_g2        (G2): 1 element\n");
        libff::print_indent(); printf("* vk_delta_g2        (G2): 1 element\n");
        libff::print_indent(); printf("* vk_IC              (G1): %zu elements\n", IC.size());
    }
};

struct Groth16SetupGPULayout {
    size_t n, m;

    g1_t::affine_t *g1_base;
    g2_t::affine_t *g2_base;

    QAPContext<fr_t, libff::Fr<ppT>> qap;

    uint32_t *col_ptr[3], *row_idx[3];
    fr_t *mat_values[3];

    fr_t *tw; // [0] = t, [1] = alpha, [2] = beta, [3] = delta_inv, [4] = Z(t), [5] = gamma_inv
    fr_t *t_pows;

    fr_t *At, *Bt, *Ct, *Lt, *Ht;
    fr_t *ICt; // ICt[i] = (beta * Ai + alpha * Bi + Ci) * gamma_inv, i in 0..=k
    g1_t *pkA1, *pkB1, *pkK, *pkZ;
    g1_t *vkIC;
    g2_t *pkB2;

    TypedGpuArena arena;

    Groth16SetupGPULayout(size_t n_, size_t m_, const SparseMatrix<libff::Fr<ppT>> **mats, libff::Fr<ppT> omega)
        : n(n_), m(m_), qap(m, omega)
    {
        const size_t cols = 1 + n;

        arena.register_alloc(g1_base, 1); arena.register_alloc(g2_base, 1);
        arena.register_alloc(tw, 6); arena.register_alloc(t_pows, m);

        arena.register_alloc(At, cols); arena.register_alloc(Bt, cols); arena.register_alloc(Ct, cols);
        arena.register_alloc(Lt, cols); arena.register_alloc(ICt, cols); arena.register_alloc(Ht, m);
        
        arena.register_alloc(pkA1, cols); arena.register_alloc(pkB1, cols); arena.register_alloc(pkB2, cols);
        arena.register_alloc(pkK, cols); arena.register_alloc(pkZ, m); arena.register_alloc(vkIC, cols);

        for (size_t i = 0; i < 3; i++) {
            arena.register_alloc(col_ptr[i], cols + 1);
            arena.register_alloc(row_idx[i], mats[i]->col_idx.size());
            arena.register_alloc(mat_values[i], mats[i]->col_idx.size());
        }
        arena.commit("Groth16SetupGPULayout");
    }
};

template <typename ppT>
Groth16ProvingKeys<ppT> gpu_setup(
    const SparseMatrix<libff::Fr<ppT>> &mA, const SparseMatrix<libff::Fr<ppT>> &mB, const SparseMatrix<libff::Fr<ppT>> &mC, size_t k,
    const libff::Fr<ppT> &t, const libff::Fr<ppT> &alpha, const libff::Fr<ppT> &beta,
    const libff::Fr<ppT> &delta_inv, const libff::Fr<ppT> &gamma_inv, const libff::Fr<ppT> &Zt,
    vector<libff::G1<ppT>> &vk_IC)
{
    const size_t m = mA.row_ptr.size() - 1;
    const size_t n = mA.num_cols - 1;
    const size_t cols = 1 + n;

    const SparseMatrix<libff::Fr<ppT>> *mats[3] = { &mA, &mB, &mC };
    libff::Fr<ppT> omega = reinterpret_cast<const libff::Fr<ppT>*>(forward_roots_of_unity)[log2_floor(m)];

    Groth16SetupGPULayout layout(n, m, mats, omega);

    libff::enter_block("GPU Groth16 Setup");
    libff::G1<ppT> g1 = libff::G1<ppT>::one(); g1.to_affine_coordinates();
    cudaMemcpy(layout.g1_base, &g1, sizeof(g1_t::affine_t), cudaMemcpyHostToDevice);
    libff::G2<ppT> g2 = libff::G2<ppT>::one(); g2.to_affine_coordinates();
    cudaMemcpy(layout.g2_base, &g2, sizeof(g2_t::affine_t), cudaMemcpyHostToDevice);

    libff::Fr<ppT> tw_host[6] = { t, alpha, beta, delta_inv, Zt, gamma_inv };
    cudaMemcpy(layout.tw, tw_host, 6 * sizeof(fr_t), cudaMemcpyHostToDevice);

    layout.qap.prepare(t, Zt);

    for (size_t i = 0; i < 3; i++) {
        CscMatrix<libff::Fr<ppT>> csc = CscMatrix<libff::Fr<ppT>>::from_csr(*mats[i], m, cols);
        cudaMemcpy(layout.col_ptr[i], csc.col_ptr.data(), csc.col_ptr.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(layout.row_idx[i], csc.row_idx.data(), csc.row_idx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(layout.mat_values[i], csc.values.data(), csc.values.size() * sizeof(fr_t), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(layout.t_pows + 1, &t, sizeof(fr_t), cudaMemcpyHostToDevice);
    power_table_kernel<<<1, 1>>>(layout.t_pows, m);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    libff::leave_block("GPU Groth16 Setup");

    libff::enter_block("GPU Groth16 Compute");
    const size_t blk = 256;

    layout.qap.compute_lagrange();

    fr_t *out[3] = { layout.At, layout.Bt, layout.Ct };
    for (size_t i = 0; i < 3; i++) {
        transpose_spmv_kernel<<<ceil_div(cols, blk), blk>>>(
            layout.col_ptr[i], layout.row_idx[i], layout.mat_values[i],
            layout.qap.lagrange(), out[i], cols);
    }

    kernel<<<ceil_div(cols, blk), blk>>>([=] __device__ (const fr_t *At, const fr_t *Bt, const fr_t *Ct, const fr_t *tw, fr_t *Lt, size_t cols, size_t k) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= cols) return;
        if (i <= k) { Lt[i].zero(); return; }
        Lt[i] = (tw[2] * At[i] + tw[1] * Bt[i] + Ct[i]) * tw[3];
    }, layout.At, layout.Bt, layout.Ct, layout.tw, layout.Lt, cols, k);

    /* IC query for the verification key (constant + public inputs):
       (beta*At + alpha*Bt + Ct) * gamma_inv  for i in 0..k. */
    kernel<<<ceil_div(k + 1, blk), blk>>>([=] __device__ (const fr_t *At, const fr_t *Bt, const fr_t *Ct, const fr_t *tw, fr_t *ICt, size_t len) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= len) return;
        ICt[i] = (tw[2] * At[i] + tw[1] * Bt[i] + Ct[i]) * tw[5];
    }, layout.At, layout.Bt, layout.Ct, layout.tw, layout.ICt, k + 1);

    kernel<<<ceil_div(m, blk), blk>>>([=] __device__ (const fr_t *t_pows, const fr_t *tw, fr_t *Ht, size_t m) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= m) return;
        if (i >= m - 1) { Ht[i].zero(); return; }
        Ht[i] = t_pows[i] * tw[4] * tw[3];
    }, layout.t_pows, layout.tw, layout.Ht, m);

    fixmsm_kernel<fr_t, g1_t::affine_t, g1_t><<<ceil_div(cols, blk), blk>>>(layout.g1_base, layout.At, layout.pkA1, cols);
    fixmsm_kernel<fr_t, g1_t::affine_t, g1_t><<<ceil_div(cols, blk), blk>>>(layout.g1_base, layout.Bt, layout.pkB1, cols);
    fixmsm_kernel<fr_t, g2_t::affine_t, g2_t><<<ceil_div(cols, blk), blk>>>(layout.g2_base, layout.Bt, layout.pkB2, cols);
    fixmsm_kernel<fr_t, g1_t::affine_t, g1_t><<<ceil_div(cols, blk), blk>>>(layout.g1_base, layout.Lt, layout.pkK, cols);
    fixmsm_kernel<fr_t, g1_t::affine_t, g1_t><<<ceil_div(m, blk), blk>>>(layout.g1_base, layout.Ht, layout.pkZ, m);
    fixmsm_kernel<fr_t, g1_t::affine_t, g1_t><<<ceil_div(k + 1, blk), blk>>>(layout.g1_base, layout.ICt, layout.vkIC, k + 1);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    libff::leave_block("GPU Groth16 Compute");

    libff::enter_block("GPU Groth16 Load");
    Groth16ProvingKeys<ppT> pk;
    pk.pkA1.resize(cols);
    pk.pkB1.resize(cols);
    pk.pkK.resize(cols);
    pk.pkZ.resize(m);
    pk.pkB2.resize(cols);

    cudaMemcpy(pk.pkA1.data(), layout.pkA1, cols * sizeof(g1_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(pk.pkB1.data(), layout.pkB1, cols * sizeof(g1_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(pk.pkK.data(),  layout.pkK,  cols * sizeof(g1_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(pk.pkZ.data(),  layout.pkZ,  m * sizeof(g1_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(pk.pkB2.data(), layout.pkB2, cols * sizeof(g2_t), cudaMemcpyDeviceToHost);

    vk_IC.resize(k + 1);
    cudaMemcpy(vk_IC.data(), layout.vkIC, (k + 1) * sizeof(g1_t), cudaMemcpyDeviceToHost);

    /* fixmsm outputs Jacobian (projective) points, but the MSM load_bases()
       in prove.cu treats each point's leading bytes as AFFINE (X, Y) and only
       that is correct when Z = 1.  Normalize to affine before serializing,
       exactly like the pk points generated by the tests (to_affine_coordinates). */
    for (auto &p : pk.pkA1) p.to_affine_coordinates();
    for (auto &p : pk.pkB1) p.to_affine_coordinates();
    for (auto &p : pk.pkB2) p.to_affine_coordinates();
    for (auto &p : pk.pkK)  p.to_affine_coordinates();
    for (auto &p : pk.pkZ)  p.to_affine_coordinates();
    for (auto &p : vk_IC)   p.to_affine_coordinates();

    libff::leave_block("GPU Groth16 Load");

    return pk;
}

static void usage(const char *prog) {
    cerr << "Usage: " << prog << " [--path <dir>] <r1cs-file>\n"
         << "  Writes <stem>.pk.bin (proving key) and <stem>.vk.bin (verification key).\n";
    exit(1);
}

int main(int argc, char *argv[]) {
    ppT::init_public_params();

    string data_dir = APP_DATA_DIR;
    string filename;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--path") {
            if (i + 1 >= argc) usage(argv[0]);
            data_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
        } else if (!arg.empty() && arg[0] == '-') {
            cerr << "Unknown option: " << arg << "\n";
            usage(argv[0]);
        } else {
            filename = arg;
        }
    }
    if (filename.empty()) usage(argv[0]);

    const string path = data_dir + "/" + filename;

    libff::enter_block("Reading R1CS from file");
    BinaryArchive ar;
    ar.open_for_read(path);

    size_t k;
    ar.read(k);

    SparseMatrix<libff::Fr<ppT>> A, B, C;
    size_t num_cols;
    ar.read(num_cols);
    A.num_cols = num_cols;
    B.num_cols = num_cols;
    C.num_cols = num_cols;

    ar.read(A.row_ptr); ar.read(A.col_idx); ar.read(A.values);
    ar.read(B.row_ptr); ar.read(B.col_idx); ar.read(B.values);
    ar.read(C.row_ptr); ar.read(C.col_idx); ar.read(C.values);
    ar.close();
    libff::leave_block("Reading R1CS from file");

    cout << "Loaded R1CS from " << path << endl;

    const size_t m = A.row_ptr.size() - 1;

    libff::enter_block("Generate secret randomness");
    libff::Fr<ppT> t = libff::Fr<ppT>::random_element();
    libff::Fr<ppT> alpha = libff::Fr<ppT>::random_element();
    libff::Fr<ppT> beta = libff::Fr<ppT>::random_element();
    const libff::Fr<ppT> gamma = libff::Fr<ppT>::random_element();
    const libff::Fr<ppT> delta = libff::Fr<ppT>::random_element();
    const libff::Fr<ppT> delta_inv = delta.inverse();
    const libff::Fr<ppT> gamma_inv = gamma.inverse();
    const libff::Fr<ppT> Zt = (t ^ m) - libff::Fr<ppT>::one();
    libff::leave_block("Generate secret randomness");

    libff::enter_block("GPU setup");
    vector<libff::G1<ppT>> vk_IC;
    Groth16ProvingKeys<ppT> gpu_pk = gpu_setup<ppT>(A, B, C, k, t, alpha, beta, delta_inv, gamma_inv, Zt, vk_IC);
    libff::leave_block("GPU setup");

    /* prover blinding bases, on the same generator (G1::one()/G2::one())
       that gpu_setup uses for all the queries. */
    gpu_pk.alpha_g1 = alpha * libff::G1<ppT>::one();
    gpu_pk.beta_g1  = beta  * libff::G1<ppT>::one();
    gpu_pk.beta_g2  = beta  * libff::G2<ppT>::one();
    gpu_pk.delta_g1 = delta * libff::G1<ppT>::one();
    gpu_pk.delta_g2 = delta * libff::G2<ppT>::one();

    string stem = filename;
    const string ext = ".bin";
    if (stem.size() >= ext.size() && stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0)
        stem.erase(stem.size() - ext.size());
    const string pk_path  = data_dir + "/" + stem + ".pk.bin";
    const string vk_path  = data_dir + "/" + stem + ".vk.bin";

    BinaryArchive out;
    out.open_for_write(pk_path);
    out.write(gpu_pk.pkA1);
    out.write(gpu_pk.pkB1);
    out.write(gpu_pk.pkB2);
    out.write(gpu_pk.pkK);
    out.write(gpu_pk.pkZ);
    out.write(gpu_pk.alpha_g1);
    out.write(gpu_pk.beta_g1);
    out.write(gpu_pk.beta_g2);
    out.write(gpu_pk.delta_g1);
    out.write(gpu_pk.delta_g2);
    out.close();

    gpu_pk.print_size();
    cout << "Wrote proving key to " << pk_path << endl;

    /* ---- assemble the verification key (host side, cheap) ---- */
    Groth16VerificationKeys<ppT> vk;
    const libff::G1<ppT> alpha_g1 = alpha * libff::G1<ppT>::one();
    const libff::G2<ppT> beta_g2  = beta  * libff::G2<ppT>::one();
    vk.alpha_g1_beta_g2 = ppT::reduced_pairing(alpha_g1, beta_g2);
    vk.gamma_g2 = gamma * libff::G2<ppT>::one();
    vk.delta_g2 = delta * libff::G2<ppT>::one();
    vk.IC = vk_IC;   /* IC query computed on the GPU (index 0 = constant 1) */

    vk.print_size();

    BinaryArchive vout;
    vout.open_for_write(vk_path);
    vout.write(vk.alpha_g1_beta_g2);
    vout.write(vk.gamma_g2);
    vout.write(vk.delta_g2);
    vout.write(vk.IC);
    vout.close();

    cout << "Wrote verification key to " << vk_path << endl;
    return 0;
}
