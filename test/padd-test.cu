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

void padd_compute(TestGPULayout<g1_t::affine_t, g1_t> &gpu_layout)
{
    size_t n = gpu_layout.n;
    kernel<<<n/512, 512>>>([=] __device__ (g1_t::affine_t *array, g1_t *result) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        g1_t acc = array[2 * tid];
        acc.add(array[2 * tid + 1]);
        result[tid] = acc;
    }, gpu_layout.device_array, gpu_layout.device_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

void padd_compute2(TestGPULayout<g1_t::affine_t, g1_t> &gpu_layout)
{
    size_t n = gpu_layout.n;
    kernel<<<n/512, 512>>>([=] __device__ (g1_t::affine_t *array, g1_t *result) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        fp_t ax, ay, bx, by;
        ax = reinterpret_cast<fp_t*>(array + 2 * tid)[0];
        ay = reinterpret_cast<fp_t*>(array + 2 * tid)[1];
        bx = reinterpret_cast<fp_t*>(array + 2 * tid + 1)[0];
        by = reinterpret_cast<fp_t*>(array + 2 * tid + 1)[1];

        fp_t lambda = (ay - by) / (ax - bx);
        g1_t::affine_t sum;
        fp_t rx = (lambda^2) - ax - bx;
        reinterpret_cast<fp_t*>(&sum)[1] = lambda * (ax - rx) - ay;
        reinterpret_cast<fp_t*>(&sum)[0] = rx;            
        result[tid] = sum;
    }, gpu_layout.device_array, gpu_layout.device_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

void point_gpu_setup(vector<libff::G1<ppT>> array, TestGPULayout<g1_t::affine_t, g1_t> &gpu_layout)
{
    gpu_layout.n = array.size() / 2;
    cudaMalloc(&gpu_layout.device_array, 2 * gpu_layout.n * sizeof(g1_t::affine_t));
    cudaMalloc(&gpu_layout.device_result, gpu_layout.n * sizeof(g1_t));

    vector<g1_t::affine_t> temp(array.size());
    memset(temp.data(), 0, temp.size() * sizeof(g1_t::affine_t));
    #pragma omp parallel for
    for (size_t i = 0; i < array.size(); i++) if (!array[i].is_zero()) memcpy(&temp[i], &array[i], sizeof(g1_t::affine_t));

    cudaMemcpy(gpu_layout.device_array, temp.data(), array.size() * sizeof(g1_t::affine_t), cudaMemcpyHostToDevice);
}

int main(int argc, char *argv[])
{
    size_t n = size_t(1) << 22;

    ppT::init_public_params();

    int t = 1;
    string pregen_option = "-regen";

    if (argc > 1) t = atoi(argv[1]);
    if (argc > 2) pregen_option = argv[2];
    assert(pregen_option == "-regen" || pregen_option == "-fast");
    while (t--) {

        if (1) {
            Tester<libff::G1<ppT>, libff::G1<ppT>> tester(
                n,
                [] (libff::G1<ppT> &x) { 
                    x = libff::G1<ppT>::random_element();
                    x.to_affine_coordinates();
                },
                [] (const libff::G1<ppT> &x, const libff::G1<ppT> &y) { return x + y; },
                "padd_test",
                pregen_option == "-fast"
            );
            TestGPULayout<g1_t::affine_t, g1_t> gpu_layout, gpu_layout2;
            tester.gpu_bench(gpu_layout, point_gpu_setup, padd_compute);
            tester.gpu_bench(gpu_layout2, point_gpu_setup, padd_compute2);
        }
    }

    return 0;
}