#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "libff/common/profiling.hpp"
#include <libff/common/utils.hpp>

#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"

#include <omp.h>
using namespace std;

#include "api.h"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

#ifndef APP_DATA_DIR
#define APP_DATA_DIR "app/data"
#endif

/*
 * Groth16 proving (GPU) for the files produced by gen.cu / setup.cu.
 *
 *   input  : <r1cs>.bin     (R1CS written by gen.cu)
 *            <stem>.witness.bin (satisfying witness written by gen.cu)
 *            <stem>.pk.bin  (proving key written by setup.cu)
 *   output : <stem>.proof.bin (Groth16 proof)
 *
 * The GPU pipeline (spMV -> NTT -> MSM) is the same one used by
 * test/groth16-test.cu.
 */

/* Groth16 proof object (same layout as test/groth16-test.cu). */
template <typename ppT>
struct Groth16Proof {
    libff::G1<ppT> Ar, Bs1, zK, qZ;
    libff::G2<ppT> Bs2;
};

/* GPU proving layout (same as test/groth16-test.cu). */
struct Groth16ProveGPULayout {
    BucketContext<fr_t, libff::Fr<ppT>> z_bucket_ctx, mz_bucket_ctx;
    spMVMContext<fr_t, libff::Fr<ppT>, 3> spmvm_ctx;
    NTTContext<fr_t, libff::Fr<ppT>> ntt_ctx;
    MSMContext<fr_t, g1_t::affine_t, g1_t, g1_bucket_t, libff::Fr<ppT>, libff::G1<ppT>> g1_sparse_msm_ctx_0;
    MSMContext<fr_t, g1_t::affine_t, g1_t, g1_bucket_t, libff::Fr<ppT>, libff::G1<ppT>> g1_sparse_msm_ctx_1;
    MSMContext<fr_t, g1_t::affine_t, g1_t, g1_bucket_t, libff::Fr<ppT>, libff::G1<ppT>> g1_sparse_msm_ctx_2;
    MSMContext<fr_t, g1_t::affine_t, g1_t, g1_bucket_t, libff::Fr<ppT>, libff::G1<ppT>> g1_dense_msm_ctx;
    MSMContext<fr_t, g2_t::affine_t, g2_t, g2_bucket_t, libff::Fr<ppT>, libff::G2<ppT>> g2_sparse_msm_ctx;

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
        g1_sparse_msm_ctx_0(1 + n, window_bits),
        g1_sparse_msm_ctx_1(1 + n, window_bits),
        g1_sparse_msm_ctx_2(1 + n, window_bits),
        g1_dense_msm_ctx(m, window_bits),
        g2_sparse_msm_ctx(1 + n, window_bits)
    {
        arena.register_alloc(polys[0], m);
        arena.register_alloc(polys[1], m);
        arena.register_alloc(polys[2], m);
        arena.commit("PolyBuffer");
    }
};

/* Load the (already zero-padded) proving-key bases into the MSM contexts. */
static void cuda_prove_setup(Groth16ProveGPULayout &gpu_layout,
                             const vector<libff::G1<ppT>> &pkA1,
                             const vector<libff::G1<ppT>> &pkB1,
                             const vector<libff::G1<ppT>> &pkK,
                             const vector<libff::G1<ppT>> &pkZ,
                             const vector<libff::G2<ppT>> &pkB2)
{
    gpu_layout.g1_sparse_msm_ctx_0.load_bases(pkA1.data());
    gpu_layout.g1_sparse_msm_ctx_1.load_bases(pkB1.data());
    /* setup.cu writes pkK already zero-padded at columns 0..k, so the full
       (1+n)-sized array can be used directly as the zK base array. */
    gpu_layout.g1_sparse_msm_ctx_2.load_bases(pkK.data());
    /* setup.cu writes pkZ already zero-padded at index m-1 (full m size). */
    gpu_layout.g1_dense_msm_ctx.load_bases(pkZ.data());
    gpu_layout.g2_sparse_msm_ctx.load_bases(pkB2.data());
}

/* Run the proving computation on GPU: H = (A z * B z - C z) / Z via spMV+NTT,
   then Ar/Bs1/zK/Bs2 via sparse MSMs and qZ via the dense MSM. */
static void cuda_prove_compute(Groth16ProveGPULayout &gpu_layout,
                               const vector<libff::Fr<ppT>> &z,
                               Groth16Proof<ppT> &result)
{
    gpu_layout.z_bucket_ctx.load_scalars(z.data());
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
    gpu_layout.g1_sparse_msm_ctx_0.msm(gpu_layout.z_bucket_ctx, &result.Ar);
    gpu_layout.g1_sparse_msm_ctx_1.msm(gpu_layout.z_bucket_ctx, &result.Bs1);
    gpu_layout.g1_sparse_msm_ctx_2.msm(gpu_layout.z_bucket_ctx, &result.zK);
    gpu_layout.g2_sparse_msm_ctx.msm(gpu_layout.z_bucket_ctx, &result.Bs2);

    gpu_layout.mz_bucket_ctx.process();
    gpu_layout.g1_dense_msm_ctx.msm(gpu_layout.mz_bucket_ctx, &result.qZ);
}

/*
 * CPU sanity check: every row must satisfy A z * B z == C z.  This is
 * circuit-agnostic — it only checks the witness against the R1CS matrices
 * loaded from disk, which is a cheap way to catch a mismatched witness file.
 */
static bool check_witness(const SparseMatrix<libff::Fr<ppT>> &A,
                          const SparseMatrix<libff::Fr<ppT>> &B,
                          const SparseMatrix<libff::Fr<ppT>> &C,
                          const vector<libff::Fr<ppT>> &z)
{
    const size_t m = A.row_ptr.size() - 1;
    bool ok = true;
    for (size_t r = 0; r < m; r++) {
        libff::Fr<ppT> az = libff::Fr<ppT>::zero(), bz = libff::Fr<ppT>::zero(), cz = libff::Fr<ppT>::zero();
        for (size_t j = A.row_ptr[r]; j < A.row_ptr[r + 1]; j++) az += A.values[j] * z[A.col_idx[j]];
        for (size_t j = B.row_ptr[r]; j < B.row_ptr[r + 1]; j++) bz += B.values[j] * z[B.col_idx[j]];
        for (size_t j = C.row_ptr[r]; j < C.row_ptr[r + 1]; j++) cz += C.values[j] * z[C.col_idx[j]];
        if (az * bz != cz) {
            if (ok) cout << "Witness check FAILED at constraints:\n";
            cout << "  row " << r << "\n";
            ok = false;
        }
    }
    if (ok) cout << "Witness satisfies all R1CS constraints." << endl;
    return ok;
}

/* Resolve a user-supplied path against the data dir; absolute paths pass through. */
static string resolve_data_path(const string &data_dir, const string &p) {
    if (!p.empty() && p[0] == '/') return p;
    return data_dir + "/" + p;
}

static void usage(const char *prog) {
    cerr << "Usage: " << prog << " [--path <dir>] [--pk <file>] [--witness <file>] <r1cs-file>\n"
         << "  Reads the R1CS and witness written by gen.cu (<stem>.bin,\n"
         << "  <stem>.witness.bin) and the proving key written by setup.cu\n"
         << "  (<stem>.pk.bin), computes a Groth16 proof on the GPU and writes it\n"
         << "  to <stem>.proof.bin.\n"
         << "  --pk <file>      proving key file (default <stem>.pk.bin)\n"
         << "  --witness <file> witness file (default <stem>.witness.bin)\n"
         << "  Relative paths are resolved against --path / " APP_DATA_DIR ".\n";
    exit(1);
}

int main(int argc, char *argv[]) {
    ppT::init_public_params();

    string data_dir = APP_DATA_DIR;
    string filename, pk_override, witness_override;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--path") {
            if (i + 1 >= argc) usage(argv[0]);
            data_dir = argv[++i];
        } else if (arg == "--pk") {
            if (i + 1 >= argc) usage(argv[0]);
            pk_override = argv[++i];
        } else if (arg == "--witness") {
            if (i + 1 >= argc) usage(argv[0]);
            witness_override = argv[++i];
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

    /* derive sibling file names from <stem> */
    string stem = filename;
    const string ext = ".bin";
    if (stem.size() >= ext.size() && stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0)
        stem.erase(stem.size() - ext.size());

    /* pk and witness default to <stem>.pk.bin / <stem>.witness.bin next to the
       R1CS file; relative paths are resolved against the data dir. */
    const string r1cs_path    = resolve_data_path(data_dir, filename);
    const string pk_path      = resolve_data_path(data_dir, pk_override.empty() ? (stem + ".pk.bin") : pk_override);
    const string witness_path = resolve_data_path(data_dir, witness_override.empty() ? (stem + ".witness.bin") : witness_override);
    const string proof_path   = data_dir + "/" + stem + ".proof.bin";

    /* ---------------- read R1CS ---------------- */
    libff::enter_block("Reading R1CS from file");
    BinaryArchive ar;
    ar.open_for_read(r1cs_path);

    size_t k;
    ar.read(k);

    SparseMatrix<libff::Fr<ppT>> A, B, C;
    size_t num_cols;
    ar.read(num_cols);
    A.num_cols = B.num_cols = C.num_cols = num_cols;

    ar.read(A.row_ptr); ar.read(A.col_idx); ar.read(A.values);
    ar.read(B.row_ptr); ar.read(B.col_idx); ar.read(B.values);
    ar.read(C.row_ptr); ar.read(C.col_idx); ar.read(C.values);
    ar.close();
    libff::leave_block("Reading R1CS from file");

    const size_t m = A.row_ptr.size() - 1;  /* padded #constraints (power of 2) */
    const size_t n = num_cols - 1;          /* variables excluding constant 1   */

    cout << "Loaded R1CS from " << r1cs_path << endl;
    cout << "* public inputs : " << k << "\n"
         << "* variables     : " << n << " (private: " << (n - k) << ")\n"
         << "* constraints   : " << m << "\n";

    /* ---------------- read proving key ---------------- */
    libff::enter_block("Reading proving key from file");
    BinaryArchive par;
    par.open_for_read(pk_path);

    vector<libff::G1<ppT>> pkA1, pkB1, pkK, pkZ;
    vector<libff::G2<ppT>> pkB2;
    par.read(pkA1); par.read(pkB1); par.read(pkB2); par.read(pkK); par.read(pkZ);
    par.close();
    libff::leave_block("Reading proving key from file");

    cout << "Loaded proving key from " << pk_path << endl;
    if (pkA1.size() != 1 + n || pkB1.size() != 1 + n || pkB2.size() != 1 + n ||
        pkK.size() != 1 + n || pkZ.size() != m) {
        cerr << "Proving key size mismatch with R1CS "
                "(expected pkA1/pkB1/pkB2/pkK = 1+n and pkZ = m).\n";
        return 1;
    }

    /* ---------------- witness ---------------- */
    libff::enter_block("Reading witness from file");
    BinaryArchive war;
    war.open_for_read(witness_path);
    vector<libff::Fr<ppT>> z;
    war.read(z);
    war.close();
    libff::leave_block("Reading witness from file");
    if (z.size() != 1 + n) {
        cerr << "Witness size mismatch (expected 1+n = " << (1 + n)
             << ", got " << z.size() << ").\n";
        return 1;
    }

    libff::enter_block("Validating witness against R1CS");
    const bool witness_ok = check_witness(A, B, C, z);
    libff::leave_block("Validating witness against R1CS");
    if (!witness_ok)
        cerr << "Warning: witness does NOT satisfy the R1CS; proof will be invalid.\n";

    /* ---------------- GPU prove ---------------- */
    if (m < 2048) {
        cerr << "Number of constraints (" << m
             << ") is too small for the GPU NTT (needs >= 2048). "
                "Generate a larger circuit with gen.\n";
        return 1;
    }

    SparseMatrix<libff::Fr<ppT>> *mats[3] = { &A, &B, &C };
    const size_t exp = log2_floor(m);
    Groth16ProveGPULayout gpu_layout(
        n, m, mats,
        reinterpret_cast<const libff::Fr<ppT>*>(forward_roots_of_unity)[exp],
        libff::Fr<ppT>::multiplicative_generator);

    libff::enter_block("GPU Groth16 Prove Setup");
    cuda_prove_setup(gpu_layout, pkA1, pkB1, pkK, pkZ, pkB2);
    libff::leave_block("GPU Groth16 Prove Setup");

    Groth16Proof<ppT> proof;
    libff::enter_block("GPU Groth16 Prove Compute");
    cuda_prove_compute(gpu_layout, z, proof);
    libff::leave_block("GPU Groth16 Prove Compute");

    /* ---------------- write proof ---------------- */
    libff::enter_block("Writing proof to file");
    BinaryArchive out;
    out.open_for_write(proof_path);
    out.write(proof.Ar);
    out.write(proof.Bs1);
    out.write(proof.zK);
    out.write(proof.qZ);
    out.write(proof.Bs2);
    out.close();
    libff::leave_block("Writing proof to file");

    cout << "Wrote proof to " << proof_path << endl;
    return 0;
}
