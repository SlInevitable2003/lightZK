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
 * Groth16 proving key (same format as groth16-test.cu):
 *   - pkA1[1+n] : A_query  in G1
 *   - pkB1[1+n] : B_query  in G1
 *   - pkB2[1+n] : B_query  in G2
 *   - pkK [1+n] : L_query  in G1,  zero-padded at 0..k
 *   - pkZ [m]   : H_query  in G1,  t^i * Z(t)/delta at 0..m-2, zero at m-1
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

    Groth16SetupGPULayout(size_t n_, size_t m_, const SparseMatrix<libff::Fr<ppT>> **mats, libff::Fr<ppT> omega)
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

template <typename ppT>
Groth16ProvingKeys<ppT> gpu_setup(
    const SparseMatrix<libff::Fr<ppT>> &mA, const SparseMatrix<libff::Fr<ppT>> &mB, const SparseMatrix<libff::Fr<ppT>> &mC, size_t k,
    const libff::Fr<ppT> &t, const libff::Fr<ppT> &alpha, const libff::Fr<ppT> &beta, const libff::Fr<ppT> &delta_inv, const libff::Fr<ppT> &Zt)
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

    libff::Fr<ppT> tw_host[5] = { t, alpha, beta, delta_inv, Zt };
    cudaMemcpy(layout.tw, tw_host, 5 * sizeof(fr_t), cudaMemcpyHostToDevice);

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
    libff::leave_block("GPU Groth16 Load");

    return pk;
}

static void usage(const char *prog) {
    cerr << "Usage: " << prog << " [--path <dir>] <r1cs-file>\n";
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
    const libff::Fr<ppT> delta = libff::Fr<ppT>::random_element();
    const libff::Fr<ppT> delta_inv = delta.inverse();
    const libff::Fr<ppT> Zt = (t ^ m) - libff::Fr<ppT>::one();
    libff::leave_block("Generate secret randomness");

    libff::enter_block("GPU setup");
    Groth16ProvingKeys<ppT> gpu_pk = gpu_setup<ppT>(A, B, C, k, t, alpha, beta, delta_inv, Zt);
    libff::leave_block("GPU setup");

    string stem = filename;
    const string ext = ".bin";
    if (stem.size() >= ext.size() && stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0)
        stem.erase(stem.size() - ext.size());
    const string out_path = data_dir + "/" + stem + ".pk.bin";

    BinaryArchive out;
    out.open_for_write(out_path);
    out.write(gpu_pk.pkA1);
    out.write(gpu_pk.pkB1);
    out.write(gpu_pk.pkB2);
    out.write(gpu_pk.pkK);
    out.write(gpu_pk.pkZ);
    out.close();

    cout << "Wrote proving key to " << out_path << endl;
    return 0;
}
