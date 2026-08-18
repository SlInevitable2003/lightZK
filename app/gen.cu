#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"

#include "api.h"

using namespace std;
using namespace LightZK;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

#ifndef APP_DATA_DIR
#define APP_DATA_DIR "app/data"
#endif

static void usage(const char *prog) {
    cerr << "Usage: " << prog << " <case> [params...]\n"
         << "  cases:\n"
         << "    matmul <size>   matrix-multiplication R1CS of dimension <size>\n";
    exit(1);
}

int main(int argc, char *argv[]) {
    ppT::init_public_params();

    if (argc < 2) usage(argv[0]);
    const string case_name = argv[1];

    size_t s = 0;
    if (case_name == "matmul") {
        if (argc < 3) usage(argv[0]);
        s = atoi(argv[2]);
        if (s == 0) usage(argv[0]);
    } else {
        cerr << "Unknown case: " << case_name << "\n";
        usage(argv[0]);
    }

    /* Build the matrix-multiplication R1CS: P[i,j,l] = A[i,l] * B[l,j]. */
    R1CSManager<libff::Fr<ppT>> mgr;

    vector<Variable<libff::Fr<ppT>>> A, B, C, P;
    for (size_t i = 0; i < s * s; i++) A.push_back(Variable<libff::Fr<ppT>>(VariableType::Public,  mgr));
    for (size_t i = 0; i < s * s; i++) B.push_back(Variable<libff::Fr<ppT>>(VariableType::Public,  mgr));
    for (size_t i = 0; i < s * s; i++) C.push_back(Variable<libff::Fr<ppT>>(VariableType::Public,  mgr));
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
    printf("* R1CS constraints: %zu\n", A_mat.row_ptr.size() - 1);
    printf("* R1CS variables:   %zu (public: %zu, private: %zu)\n", A_mat.num_cols, k, A_mat.num_cols - 1 - k);

    /* Output path: <APP_DATA_DIR>/matmul_<size>.bin */
    const string data_dir = APP_DATA_DIR;
    std::filesystem::create_directories(data_dir);
    const string path = data_dir + "/matmul_" + to_string(s) + ".bin";

    BinaryArchive ar;
    ar.open_for_write(path);

    ar.write(k);
    ar.write(A_mat.num_cols);
    ar.write(A_mat.row_ptr); ar.write(A_mat.col_idx); ar.write(A_mat.values);
    ar.write(B_mat.row_ptr); ar.write(B_mat.col_idx); ar.write(B_mat.values);
    ar.write(C_mat.row_ptr); ar.write(C_mat.col_idx); ar.write(C_mat.values);
    ar.close();

    cout << "Wrote R1CS to " << path << endl;
    return 0;
}
