#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"

#include <omp.h>
using namespace std;

#include "api.h"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    size_t exp = 25;
    size_t n = size_t(1) << exp;

    size_t num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    vector<libff::Fr<ppT>> vec_a(n), vec_b(n);
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) vec_a[i] = libff::Fr<ppT>::random_element(), vec_b[i] = libff::Fr<ppT>::random_element();

    TypedGpuArena arena;
    fr_t *d_vec_a, *d_vec_b;
    arena.register_alloc(d_vec_a, n);
    arena.register_alloc(d_vec_b, n);
    arena.commit();
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(d_vec_a, vec_a.data(), n * sizeof(libff::Fr<ppT>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_b, vec_b.data(), n * sizeof(libff::Fr<ppT>), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start_ip, stop_ip;
    cudaEventCreate(&start_ip);
    cudaEventCreate(&stop_ip);

    cudaEventRecord(start_ip, 0);
    
    libff::Fr<ppT> zero = libff::Fr<ppT>::zero();
    fr_t gpu_result = thrust::inner_product(
        thrust::device_pointer_cast(d_vec_a),
        thrust::device_pointer_cast(d_vec_a + n),
        thrust::device_pointer_cast(d_vec_b),
        *reinterpret_cast<fr_t*>(&zero),
        thrust::plus<fr_t>(),
        thrust::multiplies<fr_t>());
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(stop_ip, 0);

    float ms_ip = 0.0f;
    cudaEventElapsedTime(&ms_ip, start_ip, stop_ip);
    std::cout << "GPU inner product only: " << ms_ip << " ms" << std::endl;
}