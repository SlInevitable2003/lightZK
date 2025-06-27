#include <iostream>
#include <vector>
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include "libff/common/profiling.hpp"

#include <omp.h>
using namespace std;

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform_reduce.h>

#include "api.h"
using namespace alt_bn128;

typedef libsnark::default_r1cs_ppzksnark_pp ppT;

const size_t n = 1 << 20;

int main(int argc, char *argv[])
{
    ppT::init_public_params();
    size_t num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    assert(num_threads == 16); // Ensure we are using 16 threads

    vector<libff::Fr<ppT>> scalars(n);
    vector<libff::G1<ppT>> points(n);

    libff::enter_block("Data generation on CPU with 16 cores");
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        scalars[i] = libff::Fr<ppT>::random_element();
        points[i] = libff::G1<ppT>::random_element();
    }
    libff::leave_block("Data generation on CPU with 16 cores");

    thrust::device_vector<fr_t> d_scalars(n);
    thrust::device_vector<g1_t> d_points(n);

    // Copy data to device vectors
    fr_t* scalars_ptr = reinterpret_cast<fr_t*>(scalars.data());
    g1_t* points_ptr = reinterpret_cast<g1_t*>(points.data());
    thrust::copy(scalars_ptr, scalars_ptr + n, d_scalars.begin());
    thrust::copy(points_ptr, points_ptr + n, d_points.begin());

    // Perform the multi-scalar multiplication (MSM) on the GPU
    libff::enter_block("MSM on GPU");
    libff::G1<ppT> gpu_result, init_value(libff::G1<ppT>::zero());
    *reinterpret_cast<g1_t*>(&gpu_result) = thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(d_scalars.begin(), d_points.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_scalars.end(), d_points.end())),
        [] __device__  (const thrust::tuple<fr_t, g1_t>& tup) -> g1_t {
            return scalar_mult(thrust::get<0>(tup), thrust::get<1>(tup));
        },
        *reinterpret_cast<g1_t*>(&init_value),
        [] __device__ (const g1_t& a, const g1_t& b) -> g1_t {
            g1_t res;
            g1_t::add(res, a, b);
            return res;
        }
    );
    libff::leave_block("MSM on GPU");

    vector<libff::G1<ppT>> results(num_threads, libff::G1<ppT>::zero());
    libff::enter_block("MSM on CPU with 16 cores");
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        size_t thread_id = omp_get_thread_num();
        results[thread_id] = results[thread_id] + scalars[i] * points[i];
    }
    libff::leave_block("MSM on CPU with 16 cores");

    libff::G1<ppT> final_result = libff::G1<ppT>::zero();
    for (const auto& res : results) final_result = final_result + res;

    assert(final_result == gpu_result && "Results do not match!");

    return 0;
}