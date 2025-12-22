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

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    assert(argc > 1);
    size_t log_m = atoi(argv[1]);
    size_t m = 1 << log_m;
    size_t k = min(m, size_t(1024));
    if (argc > 2) k = atoi(argv[2]);

    libff::enter_block("Generate R1CS example");
    libsnark::r1cs_example<libff::Fr<ppT>> example = libsnark::generate_r1cs_example_with_field_input<libff::Fr<ppT>>(m, k);
    libff::leave_block("Generate R1CS example");

    libff::print_header("(enter) Profile R1CS GG-ppzkSNARK");
    libsnark::run_r1cs_gg_ppzksnark<ppT>(example, true);
    libff::print_header("(leave) Profile R1CS GG-ppzkSNARK");

    return 0;
}