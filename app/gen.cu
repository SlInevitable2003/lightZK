#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"

#include "api.h"

using namespace std;
using namespace LightZK;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;
using FieldT = libff::Fr<ppT>;

#ifndef APP_DATA_DIR
#define APP_DATA_DIR "app/data"
#endif

template <typename S>
struct CircuitOutput {
    size_t k; // 公共输入 x 的规模
    SparseMatrix<S> A, B, C;
    vector<S> z; // z = 1 || x || w
};

template <typename S>
static void finalize_circuit(R1CSManager<S> &mgr, const map<size_t, S> &assignment, CircuitOutput<S> &out)
{
    mgr.gen_spmat(out.A, out.B, out.C, true); // 填充到最近的 2 的幂次，使得 A/B/C 形如 (1+2^r) * 2^s
    out.k = mgr.num_public();
    out.z.assign(out.A.num_cols, S::zero());
    out.z[0] = S::one();
    for (const auto &[id, val] : assignment) out.z[mgr.dense_index(id)] = val;
}

static void build_matmul(R1CSManager<FieldT> &mgr, map<size_t, FieldT> &assignment, size_t s)
{
    vector<Variable<FieldT>> A, B, C, P;
    for (size_t i = 0; i < s * s; i++) A.push_back(Variable<FieldT>(VariableType::Public, mgr));
    for (size_t i = 0; i < s * s; i++) B.push_back(Variable<FieldT>(VariableType::Public, mgr));
    for (size_t i = 0; i < s * s; i++) C.push_back(Variable<FieldT>(VariableType::Public, mgr));
    for (size_t i = 0; i < s * s * s; i++) P.push_back(Variable<FieldT>(VariableType::Private, mgr));

    for (size_t i = 0; i < s; i++)
        for (size_t j = 0; j < s; j++)
            for (size_t l = 0; l < s; l++)
                mgr.add_constraint(P[i * s * s + j * s + l], A[i * s + l], B[l * s + j]);

    for (size_t i = 0; i < s; i++)
        for (size_t j = 0; j < s; j++) {
            LinearCombination<FieldT> sum;
            for (size_t l = 0; l < s; l++) sum += P[i * s * s + j * s + l];
            mgr.add_constraint(C[i * s + j], sum);
        }

    vector<FieldT> av(s * s), bv(s * s);
    #pragma omp parallel for
    for (size_t i = 0; i < s * s; i++) {
        av[i] = FieldT::random_element();
        bv[i] = FieldT::random_element();
    }

    for (size_t i = 0; i < s; i++)
        for (size_t j = 0; j < s; j++) {
            FieldT sum = FieldT::zero();
            for (size_t l = 0; l < s; l++) {
                const FieldT p = av[i * s + l] * bv[l * s + j];
                assignment[P[i * s * s + j * s + l].get_id()] = p;
                sum += p;
            }
            assignment[C[i * s + j].get_id()] = sum;
        }
    for (size_t i = 0; i < s * s; i++) {
        assignment[A[i].get_id()] = av[i];
        assignment[B[i].get_id()] = bv[i];
    }
}

static void usage(const char *prog) {
    cerr << "Usage: " << prog << " <case> [params...]\n"
         << "  cases:\n"
         << "    matmul <size>      matrix-multiplication R1CS of dimension <size>\n"
         << "  Outputs (in " APP_DATA_DIR "):\n"
         << "    <stem>.bin         R1CS (k, num_cols, A, B, C)\n"
         << "    <stem>.witness.bin satisfying witness z (size num_cols)\n";
    exit(1);
}

int main(int argc, char *argv[]) {
    ppT::init_public_params();

    if (argc < 2) usage(argv[0]);
    const string case_name = argv[1];

    R1CSManager<FieldT> mgr;
    map<size_t, FieldT> assignment;
    string stem;

    if (case_name == "matmul") {
        if (argc < 3) usage(argv[0]);
        const size_t s = (size_t)atoi(argv[2]);
        if (s == 0) usage(argv[0]);
        build_matmul(mgr, assignment, s);
        stem = "matmul_" + to_string(s);
    } else {
        cerr << "Unknown case: " << case_name << "\n";
        usage(argv[0]);
    }

    CircuitOutput<FieldT> out;
    finalize_circuit(mgr, assignment, out);

    printf("* R1CS constraints: %zu\n", out.A.row_ptr.size() - 1);
    printf("* R1CS variables:   %zu (public: %zu, private: %zu)\n", out.A.num_cols, out.k, out.A.num_cols - 1 - out.k);

    const string data_dir = APP_DATA_DIR;
    std::filesystem::create_directories(data_dir);
    const string r1cs_path = data_dir + "/" + stem + ".bin";
    const string witness_path = data_dir + "/" + stem + ".witness.bin";

    BinaryArchive ar;
    ar.open_for_write(r1cs_path);
    ar.write(out.k);
    ar.write(out.A.num_cols);
    ar.write(out.A.row_ptr); ar.write(out.A.col_idx); ar.write(out.A.values);
    ar.write(out.B.row_ptr); ar.write(out.B.col_idx); ar.write(out.B.values);
    ar.write(out.C.row_ptr); ar.write(out.C.col_idx); ar.write(out.C.values);
    ar.close();
    cout << "Wrote R1CS to " << r1cs_path << endl;

    BinaryArchive war;
    war.open_for_write(witness_path);
    war.write(out.z);
    war.close();
    cout << "Wrote witness to " << witness_path << endl;

    return 0;
}
