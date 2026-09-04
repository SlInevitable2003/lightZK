#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "libff/common/profiling.hpp"
#include <libff/common/utils.hpp>
#include "libff/algebra/scalar_multiplication/multiexp.hpp"

#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"

#include "utils/binary_archive.cuh"   /* BinaryArchive (host-side I/O only) */

using namespace std;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

#ifndef APP_DATA_DIR
#define APP_DATA_DIR "app/data"
#endif

/*
 * Groth16 verification (host, libff pairings) for the files produced by
 * gen.cu / setup.cu / prove.cu.
 *
 *   input  : <r1cs>.bin     (R1CS written by gen.cu; only the header is read,
 *                            to learn k / num_cols for size checks)
 *            <stem>.witness.bin (satisfying witness written by gen.cu)
 *            <stem>.vk.bin  (verification key written by setup.cu)
 *            <stem>.proof.bin (randomized proof (A,B,C) written by prove.cu)
 *
 * Checks the canonical Groth16 equation (r1cs_gg_ppzksnark_online_verifier):
 *   acc = vk_IC[0] + sum_{i=1..k} x_i * vk_IC[i]
 *   accept iff  e(A, B) == e(alpha, beta) * e(acc, gamma) * e(C, delta)
 * The public inputs x_1..x_k are the first k entries of the witness
 * (variable layout: col 0 = constant 1, cols 1..k = public, rest = private).
 */

/* Same layout as prove.cu's output / setup.cu's vk. */
template <typename ppT>
struct Groth16Proof {
    libff::G1<ppT> g_A;
    libff::G2<ppT> g_B;
    libff::G1<ppT> g_C;
};

template <typename ppT>
struct Groth16VerificationKeys {
    libff::GT<ppT> alpha_g1_beta_g2;   /* e([alpha]_1, [beta]_2) */
    libff::G2<ppT> gamma_g2;           /* [gamma]_2 */
    libff::G2<ppT> delta_g2;           /* [delta]_2 */
    vector<libff::G1<ppT>> IC;         /* size k+1: [IC_0..IC_k], 0 为常量 1 */
};

/*
 * 从 witness z 中取出公开输入 x_i = z[i], i = 1..k
 * (z[0] = 1 是常量, 已在 acc 中通过 vk_IC[0] 体现).
 */
template <typename ppT>
static bool extract_public_input(const vector<libff::Fr<ppT>> &z, size_t k, vector<libff::Fr<ppT>> &pub)
{
    if (z.size() < 1 + k) return false;
    pub.assign(z.begin() + 1, z.begin() + 1 + k);
    return true;
}

template <typename ppT>
static bool verify_groth16(const Groth16VerificationKeys<ppT> &vk,
                           const vector<libff::Fr<ppT>> &public_input,   /* x_1..x_k */
                           const Groth16Proof<ppT> &proof)
{
    if (vk.IC.size() < 1 + public_input.size()) {
        cerr << "IC size (" << vk.IC.size() << ") < 1 + #public inputs ("
             << (1 + public_input.size()) << ").\n";
        return false;
    }

    /* acc = IC[0] + sum_{i=1..k} x_i * IC[i]  (IC[0] 对应常量 1) */
    libff::enter_block("Accumulate public input");
    libff::G1<ppT> acc = vk.IC[0];
    if (!public_input.empty()) {
        acc = acc + libff::multi_exp<libff::G1<ppT>, libff::Fr<ppT>,
                                     libff::multi_exp_method::multi_exp_method_BDLO12>(
            vk.IC.begin() + 1, vk.IC.end(), public_input.begin(), public_input.end(), 1);
    }
    libff::leave_block("Accumulate public input");

    /* e(A, B) == e(alpha, beta) * e(acc, gamma) * e(C, delta) */
    libff::enter_block("Pairing checks");
    const libff::GT<ppT> lhs = ppT::reduced_pairing(proof.g_A, proof.g_B);
    const libff::GT<ppT> rhs = vk.alpha_g1_beta_g2
                              * ppT::reduced_pairing(acc, vk.gamma_g2)
                              * ppT::reduced_pairing(proof.g_C, vk.delta_g2);
    libff::leave_block("Pairing checks");

    return lhs == rhs;
}

/* 相对路径统一解析到数据目录; 绝对路径原样使用. */
static string resolve_data_path(const string &data_dir, const string &p) {
    if (!p.empty() && p[0] == '/') return p;
    return data_dir + "/" + p;
}

static void usage(const char *prog) {
    cerr << "Usage: " << prog << " [--path <dir>] [--vk <file>] [--proof <file>] [--witness <file>] <r1cs-file>\n"
         << "  Verifies <stem>.proof.bin against the verification key <stem>.vk.bin\n"
         << "  (from setup.cu) and the public inputs in <stem>.witness.bin (from gen.cu).\n"
         << "  --vk <file>      verification key file (default <stem>.vk.bin)\n"
         << "  --proof <file>   proof file (default <stem>.proof.bin)\n"
         << "  --witness <file> witness file (default <stem>.witness.bin)\n"
         << "  Relative paths are resolved against --path / " APP_DATA_DIR ".\n";
    exit(1);
}

int main(int argc, char *argv[]) {
    ppT::init_public_params();

    string data_dir = APP_DATA_DIR;
    string filename, vk_override, proof_override, witness_override;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--path") {
            if (i + 1 >= argc) usage(argv[0]);
            data_dir = argv[++i];
        } else if (arg == "--vk") {
            if (i + 1 >= argc) usage(argv[0]);
            vk_override = argv[++i];
        } else if (arg == "--proof") {
            if (i + 1 >= argc) usage(argv[0]);
            proof_override = argv[++i];
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

    const string r1cs_path    = resolve_data_path(data_dir, filename);
    const string vk_path      = resolve_data_path(data_dir, vk_override.empty() ? (stem + ".vk.bin") : vk_override);
    const string proof_path   = resolve_data_path(data_dir, proof_override.empty() ? (stem + ".proof.bin") : proof_override);
    const string witness_path = resolve_data_path(data_dir, witness_override.empty() ? (stem + ".witness.bin") : witness_override);

    /* ---------------- read R1CS header (k, num_cols) ---------------- */
    libff::enter_block("Reading R1CS header");
    BinaryArchive ar;
    ar.open_for_read(r1cs_path);
    size_t k, num_cols;
    ar.read(k);
    ar.read(num_cols);
    ar.close();
    libff::leave_block("Reading R1CS header");
    cout << "Loaded R1CS header from " << r1cs_path << endl;
    cout << "* public inputs : " << k << "\n"
         << "* variables     : " << (num_cols - 1) << "\n";

    /* ---------------- read verification key ---------------- */
    libff::enter_block("Reading verification key from file");
    BinaryArchive var;
    var.open_for_read(vk_path);
    Groth16VerificationKeys<ppT> vk;
    var.read(vk.alpha_g1_beta_g2);
    var.read(vk.gamma_g2);
    var.read(vk.delta_g2);
    var.read(vk.IC);
    var.close();
    libff::leave_block("Reading verification key from file");

    cout << "Loaded verification key from " << vk_path << endl;
    if (vk.IC.size() != k + 1) {
        cerr << "Verification key / R1CS mismatch (vk_IC size " << vk.IC.size()
             << " != k+1 = " << (k + 1) << ").\n";
        return 1;
    }

    /* ---------------- read witness (public inputs) ---------------- */
    libff::enter_block("Reading witness from file");
    BinaryArchive war;
    war.open_for_read(witness_path);
    vector<libff::Fr<ppT>> z;
    war.read(z);
    war.close();
    libff::leave_block("Reading witness from file");

    if (z.size() != num_cols) {
        cerr << "Witness size mismatch (expected num_cols = " << num_cols
             << ", got " << z.size() << ").\n";
        return 1;
    }
    vector<libff::Fr<ppT>> public_input;
    if (!extract_public_input<ppT>(z, k, public_input)) {
        cerr << "Witness too short for " << k << " public inputs.\n";
        return 1;
    }

    /* ---------------- read proof ---------------- */
    libff::enter_block("Reading proof from file");
    BinaryArchive pr;
    pr.open_for_read(proof_path);
    Groth16Proof<ppT> proof;
    pr.read(proof.g_A);
    pr.read(proof.g_B);
    pr.read(proof.g_C);
    pr.close();
    libff::leave_block("Reading proof from file");
    cout << "Loaded proof from " << proof_path << endl;

    /* ---------------- verify ---------------- */
    cout << "* public inputs : " << k << "\n";
    libff::enter_block("Groth16 Verify");
    const bool ok = verify_groth16<ppT>(vk, public_input, proof);
    libff::leave_block("Groth16 Verify");

    cout << (ok ? "Proof verification succeeded." : "Proof verification FAILED.") << endl;
    return ok ? 0 : 1;
}
