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

#define ADD_TIMES 8
#define NO_SECOND_CARRY 1

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

__device__ __forceinline__ void carry_lookahead(uint32_t &gp, uint32_t &gp_, int &disoff, int disid = -1)
{
    for (int i = 1; i < 8; i <<= 1) {
        gp_ = __shfl_up_sync(0xffffffff, gp, i);
        if (disoff >= i) {
            uint32_t g = (gp | (gp_ & (gp >> 1))) & 1, p = (gp & gp_) & (~1);
            gp = g | p;
        }
    }
    gp &= 1;
    if (disid >= 0) gp_ = __shfl_sync(0xffffffff, gp, (disid % 4) * 8 + 7);
    gp = __shfl_up_sync(0xffffffff, gp, 1);
}

__device__ __forceinline__ void dis_add8(uint32_t &a, uint32_t &b, int &disoff)
{
    uint32_t gp, gp_;
    asm volatile("add.cc.u32 %0, %0, %2;"
                 "addc.u32 %1, 0, 0;"
                 : "+r"(a), "=r"(gp)
                 : "r"(b));
#ifndef NO_SECOND_CARRY
    gp |= (a == -1) << 1;
    carry_lookahead(gp, gp_, disoff);
#else
    gp = __shfl_up_sync(0xffffffff, gp, 1);
#endif
    if (disoff > 0) a += gp;
}

__device__ __forceinline__ void dis_fsub8(uint32_t &a, const uint32_t *MOD, int &disid, int &disoff)
{
    uint32_t gp, gp_, tmp;
    asm volatile("sub.cc.u32 %0, %2, %3;"
                 "subc.u32 %1, 0, 0;"
                 : "+r"(tmp), "=r"(gp)
                 : "r"(a), "r"(MOD[disoff]));
#ifndef NO_SECOND_CARRY
    gp = (-gp) | (tmp == 0) << 1;
    carry_lookahead(gp, gp_, disoff, disid);
    if (disoff > 0) tmp -= gp;
#else
    gp_ = __shfl_sync(0xffffffff, gp, (disid % 4) * 8 + 7);
    gp = __shfl_up_sync(0xffffffff, gp, 1);
    if (disoff > 0) tmp += gp;
#endif

    asm("{ .reg.pred %top;");
    asm("setp.eq.u32 %top, %0, 0;" :: "r"(gp_));
    asm("@%top mov.b32 %0, %1;" : "+r"(a) : "r"(tmp));
    asm("}");
}

__device__ __forceinline__ void dis_mul8(uint32_t &a, uint32_t &b, const uint32_t &M0, const uint32_t *MOD, int &disid, int *disoff, uint32_t *shmem)
{
    // for (int i = 0; i < 4; i += 2) {
    //     uint32_t bi = __shfl_sync(0xffffffff, b, (disoff % 4) * 8 + i);
    //     if (i == 0) {
    //         asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
    //             : "=r"(shmem[disoff * 2]), "=r"(shmem[disoff * 2 + 1])
    //             : "r"(a), "r"(bi));
    //     } else {
    //         uint32_t lo, hi;
    //         asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
    //             : "+r"(shmem[disoff * 2]), "+r"(shmem[disoff * 2 + 1])
    //             : "r"(a[0]), "r"(bi));
    //         __syncwarp();
    //     }
    //     __syncwarp();
    // }
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
            dis_fsub8(acc, device::ALT_BN128_r, disid, disoff);
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