#include <cstdio>

#include "fields/alt_bn128-fp2.cuh"
#include "curves/jacobian_t.cuh"
#include "curves/xyzz_t.cuh"
#include "msm/msm.cuh"
#include "msm/large_msm.cuh"
#include "msm/fixmsm_kernel.cuh"
#include "ntt/ntt.cuh"
#include "spmvm/spmvm.cuh"
#include "r1cs/r1cs.hpp"
#include "r1cs/qap.cuh"
#include "r1cs/spmat_kernel.cuh"

#include "utils/arith.cuh"
#include "utils/arena.cuh"
#include "utils/binary_archive.cuh"
#include "utils/cuda_check.cuh"
#include "utils/gpu_config.cuh"
#include "utils/kernel.cuh"

namespace alt_bn128 {
    typedef jacobian_t<fp_t> g1_t;
    typedef jacobian_t<fp2_t> g2_t;
    typedef xyzz_t<fp_t> g1_bucket_t;
    typedef xyzz_t<fp2_t> g2_bucket_t;
}