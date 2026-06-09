#include <cstdio>

#include "fields/alt_bn128-fp2.cuh"
#include "fields/bls12_381.cuh"
#include "curves/jacobian_t.cuh"
#include "curves/xyzz_t.cuh"
#include "msm/msm.cuh"
#include "ntt/ntt.cuh"
#include "spmvm/spmvm.cuh"

#include "utils.cuh"

namespace alt_bn128 {
    typedef jacobian_t<fp_t> g1_t;
    typedef jacobian_t<fp2_t> g2_t;
    typedef xyzz_t<fp_t> g1_bucket_t;
    typedef xyzz_t<fp2_t> g2_bucket_t;
}

namespace bls12_381 {
    typedef jacobian_t<fp_t> g1_t;
    // typedef jacobian_t<fp2_t> g2_t;
    typedef xyzz_t<fp_t> g1_bucket_t;
    // typedef xyzz_t<fp2_t> g2_bucket_t;
}