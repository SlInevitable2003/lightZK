#include <cstdio>

#include "fields/alt_bn128-fp2.cuh"
#include "curves/jacobian_t.cuh"
#include "curves/xyzz_t.cuh"
#include "msm/msm_kernel.cuh"

namespace alt_bn128 {
    typedef jacobian_t<fp_t> g1_t;
    typedef jacobian_t<fp2_t> g2_t;
    typedef xyzz_t<fp_t> g1_bucket_t;
    typedef xyzz_t<fp2_t> g2_bucket_t;
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

template<typename T>
__device__ __host__ void print_mem(const T& val) {
   const uint8_t *ptr = reinterpret_cast<const uint8_t*>(&val);
   for (int i = 0; i < sizeof(T); i++) printf("%02x", ptr[i]);
   printf("\n");
}

template<typename F, typename... Args> __global__ void kernel(F func, Args... args) { func(args...); }