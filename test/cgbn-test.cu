#include <iostream>
#include <vector>
#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include "libff/algebra/scalar_multiplication/multiexp.hpp"
#include "cgbn.h"

#include <omp.h>
using namespace std;

#include "api.h"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

template <typename ppT>
class PointAddTest {
    size_t n;

    vector<libff::G1<ppT>> points;
    libff::G1<ppT> cpu_result, gpu_result;
public:
    PointAddTest(size_t n_) : n(n_) {
        size_t num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        printf("Using %lu threads for data preparation...\n", num_threads);

        libff::enter_block("Generating random points");
        points.resize(n);
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            points[i] = libff::G1<ppT>::random_element();
            points[i].to_affine_coordinates();
        }
        libff::leave_block("Generating random points");

        libff::enter_block("Computing reference result");
        cpu_result = libff::G1<ppT>::zero();
        vector<libff::G1<ppT>> local_sum(num_threads, libff::G1<ppT>::zero());
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            size_t thread_id = omp_get_thread_num();
            local_sum[thread_id] = local_sum[thread_id] + points[i];
        }
        for (auto s : local_sum) cpu_result = cpu_result + s;
        libff::leave_block("Computing reference result");
    }

    template <typename GL, typename FS, typename FC>
    void gpu_bench(GL &gpu_layout, FS bench_setup, FC bench_compute) {
        libff::enter_block("GPU PointAdd Setup");
        bench_setup(points, gpu_layout);
        libff::leave_block("GPU PointAdd Setup");
        libff::enter_block("GPU PointAdd Compute");
        bench_compute(gpu_layout, gpu_result);
        libff::leave_block("GPU PointAdd Compute");
        if (gpu_result != cpu_result) {
            gpu_result.print(); cpu_result.print();
            assert(false && "GPU PointAdd result does not match CPU result!");
        }
    }
};

struct PointAddGPULayout {
    size_t n;
    g1_t::affine_t *points = 0;
    g1_t *warp_result;

    PointAddGPULayout() { 
        cudaMalloc(&warp_result, 32 * sizeof(g1_t));
    }

    ~PointAddGPULayout() {
        if (points) cudaFree(points);

        cudaFree(warp_result);
    }
};

void cuda_pointadd_setup(vector<libff::G1<ppT>> points, PointAddGPULayout &gpu_layout)
{
    gpu_layout.n = points.size();
    cudaMalloc(&gpu_layout.points, gpu_layout.n * sizeof(g1_t::affine_t));

    vector<g1_t::affine_t> temp(gpu_layout.n);
    memset(temp.data(), 0, temp.size() * sizeof(g1_t::affine_t));
    #pragma omp parallel for
    for (size_t i = 0; i < gpu_layout.n; i++) if (!points[i].is_zero()) memcpy(&temp[i], &points[i], sizeof(g1_t::affine_t));

    cudaMemcpy(gpu_layout.points, temp.data(), gpu_layout.n * sizeof(g1_t::affine_t), cudaMemcpyHostToDevice);
}

void cuda_pointadd_compute(PointAddGPULayout &gpu_layout, libff::G1<ppT> &result)
{
    size_t n = gpu_layout.n;
    kernel<<<1, 32>>>([=] __device__ (g1_t::affine_t *points, g1_t *warp_result) {
        g1_bucket_t acc; acc.inf();
        for (uint32_t i = threadIdx.x; i < n; i += 32) acc.add(points[i]);
        acc.to_jacobian();
        warp_result[threadIdx.x] = *reinterpret_cast<g1_t*>(&acc);
    }, gpu_layout.points, gpu_layout.warp_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    kernel<<<1, 32>>>([=] __device__ (g1_t *warp_result) {
        g1_t acc = warp_result[threadIdx.x], incr;
        for (uint32_t i = 1; i < 32; i <<= 1) {
            incr = custom_shfl_xor(acc, i);
            acc.add(incr);
        }
        warp_result[threadIdx.x] = acc;
    }, gpu_layout.warp_result);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(&result, gpu_layout.warp_result, sizeof(libff::G1<ppT>), cudaMemcpyDeviceToHost);
}

// #define TPI 8
// #define BITS 256
// typedef cgbn_mem_t<BITS> memory_t;
// typedef cgbn_context_t<TPI> context_t;
// typedef cgbn_env_t<context_t, BITS> env_t;
// void cuda_polyeval_cgbn(PolyEvalGPULayout &gpu_layout, libff::Fr<ppT> &result)
// {
//     size_t n = gpu_layout.n;
//     cgbn_error_report_t *report;
//     cgbn_error_report_alloc(&report);
//     kernel<<<1, TPI>>>([=] __device__ (cgbn_error_report_t *report, fr_t *x, fr_t *coeffs, fr_t *result) {
//         if (threadIdx.x == 0) result[0].one(1);
        
//         context_t bn_context(cgbn_report_monitor, report, 0);
//         env_t bn_env(bn_context.env<env_t>());
//         env_t::cgbn_t bn_x, bn_c, bn_r, bn_mod;
//         uint32_t np0 = device::ALT_BN128_m0;

//         cgbn_load(bn_env, bn_x, reinterpret_cast<memory_t*>(x));
//         cgbn_load(bn_env, bn_r, reinterpret_cast<memory_t*>(result));
//         cgbn_load(bn_env, bn_mod, (memory_t*)(device::ALT_BN128_r));
//         for (uint32_t i = n; i > 0; i--) {
//             cgbn_mont_mul(bn_env, bn_r, bn_r, bn_x, bn_mod, np0);
//             cgbn_load(bn_env, bn_c, reinterpret_cast<memory_t*>(&coeffs[i - 1]));
//             cgbn_add(bn_env, bn_r, bn_r, bn_c);
//             int32_t carry = cgbn_sub(bn_env, bn_c, bn_r, bn_mod);
//             cgbn_set(bn_env, bn_r, carry ? bn_r : bn_c);
//         }
//         cgbn_store(bn_env, reinterpret_cast<memory_t*>(result), bn_r);
//     }, report, gpu_layout.x, gpu_layout.coeffs, gpu_layout.result);
//     cudaDeviceSynchronize();
//     CUDA_CHECK(cudaGetLastError());
//     assert(!cgbn_error_report_check(report));
//     cudaMemcpy(&result, gpu_layout.result, sizeof(libff::Fr<ppT>), cudaMemcpyDeviceToHost);
// }

int main(int argc, char *argv[])
{
    ppT::init_public_params();
    PointAddTest<ppT> pointadd_test(1 << 14);
    { // sppark-based
        PointAddGPULayout gpu_layout;
        pointadd_test.gpu_bench(gpu_layout, cuda_pointadd_setup, cuda_pointadd_compute);
    }

    return 0;
}