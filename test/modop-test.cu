#include <cstring>
#include <random>
#include <iomanip>
#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"

#include <gmp.h>
#include <omp.h>
using namespace std;

#include "api.h"
#include "arith-tester.cuh"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

#define ADD_TIMES 2

void modadd_compute(TestGPULayout<fr_t, fr_t> &gpu_layout)
{
    size_t n = gpu_layout.n;
    kernel<<<n / 512, 512>>>([=] __device__ (fr_t *array, fr_t *result) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        fr_t acc = array[2 * tid], inc = array[2 * tid + 1];
        #pragma unroll
        for (int i = 0; i < ADD_TIMES; i++) acc += inc;
        result[tid] = acc;
    }, gpu_layout.device_array, gpu_layout.device_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

__device__ __forceinline__ void dis_add8(uint32_t &a, const uint32_t &b, int &disoff)
{
    uint32_t g, p, g_, p_;
    asm("add.cc.u32 %0, %0, %1;" : "+r"(a) : "r"(b));
    asm("addc.u32 %0, 0, 0;" : "=r"(g));
    p = (a == -1);
    for (int i = 1; i < 8; i <<= 1) {
        g_ = __shfl_up_sync(0xffffffff, g, 1);
        p_ = __shfl_up_sync(0xffffffff, p, 1);
        if (disoff >= 2 * i) {
            g |= (g_ & p);
            p &= p_;
        }
    }
    g = __shfl_up_sync(0xffffffff, g, 1);
    if (disoff > 0) a += g;
}

void modadd_compute2(TestGPULayout<fr_t, fr_t> &gpu_layout)
{
    size_t n = gpu_layout.n;
    kernel<<<n * 8 / 512, 512>>>([=] __device__ (fr_t *array, fr_t *result) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int disid = tid / 8, disoff = tid % 8;
        uint32_t acc, inc;
        acc = reinterpret_cast<uint32_t*>(array + 2 * disid)[disoff];
        inc = reinterpret_cast<uint32_t*>(array + 2 * disid + 1)[disoff];
        #pragma unroll
        for (int i = 0; i < ADD_TIMES; i++) {
            dis_add8(acc, inc, disoff);
        }
        reinterpret_cast<uint32_t*>(result + disid)[disoff] = acc;
    }, gpu_layout.device_array, gpu_layout.device_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

void modmul_compute(TestGPULayout<fr_t, fr_t> &gpu_layout)
{
    size_t n = gpu_layout.n;
    kernel<<<n / 512, 512>>>([=] __device__ (fr_t *array, fr_t *result) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        fr_t read[2] = {array[2 * tid], array[2 * tid + 1]};
        result[tid] = read[0] * read[1];
    }, gpu_layout.device_array, gpu_layout.device_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char *argv[])
{
    size_t n = size_t(1) << 24;

    ppT::init_public_params();

    int t = 1;
    if (argc > 1) t = atoi(argv[1]);
    while (t--) {

        if (1) {
            Tester<libff::Fr<ppT>, libff::Fr<ppT>> tester(
                n,
                [] (libff::Fr<ppT> &x) { 
                    x = libff::Fr<ppT>::random_element();
                    x.mont_repr.data[x.num_limbs - 1] >>= 1;
                },
                [] (const libff::Fr<ppT> &x, const libff::Fr<ppT> &y) { 
                    libff::Fr<ppT> s = x;
                    #pragma unroll
                    for (int i = 0; i < ADD_TIMES; i++) s += y;
                    return s; 
                },
                "Addition Under Module"
            );
            TestGPULayout<fr_t, fr_t> gpu_layout, gpu_layout2;
            tester.gpu_bench(gpu_layout, test_gpu_setup<libff::Fr<ppT>, fr_t, fr_t>, modadd_compute);
            tester.gpu_bench(gpu_layout2, test_gpu_setup<libff::Fr<ppT>, fr_t, fr_t>, modadd_compute2);
        }

        if (0) {
            Tester<libff::Fr<ppT>, libff::Fr<ppT>> tester(
                n,
                [] (libff::Fr<ppT> &x) { x = libff::Fr<ppT>::random_element(); },
                [] (const libff::Fr<ppT> &x, const libff::Fr<ppT> &y) { return x * y; },
                "Multiplication Under Module"
            );
            TestGPULayout<fr_t, fr_t> gpu_layout;
            tester.gpu_bench(gpu_layout, test_gpu_setup<libff::Fr<ppT>, fr_t, fr_t>, modmul_compute);
        }

    }

    return 0;
}