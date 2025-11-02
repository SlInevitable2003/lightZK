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

void modadd_compute(TestGPULayout<fr_t, fr_t> &gpu_layout)
{
    size_t n = gpu_layout.n;
    kernel<<<n / 512, 512>>>([=] __device__ (fr_t *array, fr_t *result) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        result[tid] = array[2 * tid] + array[2 * tid + 1];
    }, gpu_layout.device_array, gpu_layout.device_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

void modmul_compute(TestGPULayout<fr_t, fr_t> &gpu_layout)
{
    size_t n = gpu_layout.n;
    kernel<<<n / 512, 512>>>([=] __device__ (fr_t *array, fr_t *result) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        result[tid] = array[2 * tid] * array[2 * tid + 1];
    }, gpu_layout.device_array, gpu_layout.device_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

void my_modmul(TestGPULayout<fr_t, fr_t> &gpu_layout)
{
    const size_t block_size = 512;
    size_t n = gpu_layout.n;
    kernel<<<n / block_size, block_size>>>([=] __device__ (fr_t *array, fr_t *result) {
        // ASSUME sizeof(fr_t) = 8 * sizeof(uint32_t)
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int warpid = tid / 32, laneid = tid % 32;
        int warpid_in_block = threadIdx.x / 32;
        // warp j is going to load target = array[64j : 64j + 63]
        __shared__ fr_t buffer[block_size * 2];
        // which can be seen as an array of uint4 with size 64 * 2 = 128
        fr_t *buffer_ptr = buffer + warpid_in_block * 64;
        fr_t *array_ptr = array + warpid * 64;
        for (int i = laneid; i < 128; i += 32) reinterpret_cast<uint4*>(buffer_ptr)[i] = reinterpret_cast<uint4*>(array_ptr)[i];
        __syncwarp();

        fr_t read[2];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int indice = (i + laneid / 2) % 4;
            reinterpret_cast<uint4*>(read)[indice] = reinterpret_cast<uint4*>(buffer_ptr + 2 * laneid)[indice];
        }
        __syncwarp();
        
        read[0] *= read[1];
        result[tid] = read[0];

    }, gpu_layout.device_array, gpu_layout.device_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

int main(int argc, char *argv[])
{
    size_t n = size_t(1) << 24;

    ppT::init_public_params();

    int t = 10;
    if (argc > 0) t = atoi(argv[1]);
    while (t--) {

        if (0) {
            Tester<libff::Fr<ppT>, libff::Fr<ppT>> tester(
                n,
                [] (libff::Fr<ppT> &x) { x = libff::Fr<ppT>::random_element(); },
                [] (const libff::Fr<ppT> &x, const libff::Fr<ppT> &y) { return x + y; },
                "Addition Under Module"
            );
            TestGPULayout<fr_t, fr_t> gpu_layout;
            tester.gpu_bench(gpu_layout, test_gpu_setup<libff::Fr<ppT>, fr_t, fr_t>, modadd_compute);
        }

        if (1) {
            Tester<libff::Fr<ppT>, libff::Fr<ppT>> tester(
                n,
                [] (libff::Fr<ppT> &x) { x = libff::Fr<ppT>::random_element(); },
                [] (const libff::Fr<ppT> &x, const libff::Fr<ppT> &y) { return x * y; },
                "Multiplication Under Module"
            );
            TestGPULayout<fr_t, fr_t> gpu_layout;
            tester.gpu_bench(gpu_layout, test_gpu_setup<libff::Fr<ppT>, fr_t, fr_t>, modmul_compute);
            
        }

        if (1) {
            Tester<libff::Fr<ppT>, libff::Fr<ppT>> tester(
                n,
                [] (libff::Fr<ppT> &x) { x = libff::Fr<ppT>::random_element(); },
                [] (const libff::Fr<ppT> &x, const libff::Fr<ppT> &y) { return x * y; },
                "Multiplication Under Module"
            );
            TestGPULayout<fr_t, fr_t> gpu_layout;
            tester.gpu_bench(gpu_layout, test_gpu_setup<libff::Fr<ppT>, fr_t, fr_t>, my_modmul);
        }

    }

    return 0;
}