#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include "libff/algebra/scalar_multiplication/multiexp.hpp"

#include <omp.h>
using namespace std;

#include "api.h"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

template <typename ppT>
struct MSMTest { // TODO: replace struct with class
    size_t n;

    vector<libff::Fr<ppT>> scalars;
    vector<libff::G1<ppT>> points;

    libff::G1<ppT> cpu_result, gpu_result;
public:
    MSMTest(size_t n_, bool from_binary = false) : n(n_) {
        size_t num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        cout << "Using " << num_threads << " threads for MSM-Test preparation." << endl;

        if (from_binary) {
            libff::enter_block("Reading binary scalars and points");
            std::ifstream in("msm_test_data.bin");
            if (!in) throw std::runtime_error("Failed to open input file.");
            
            scalars.resize(n); points.resize(n);
            in.read(reinterpret_cast<char*>(scalars.data()), n * sizeof(libff::Fr<ppT>));
            in.read(reinterpret_cast<char*>(points.data()), n * sizeof(libff::G1<ppT>));

            in.close();
            libff::leave_block("Reading binary scalars and points");
        } else {
            libff::enter_block("Generating random scalars and points");
            scalars.resize(n); points.resize(n);
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                scalars[i] = libff::Fr<ppT>::random_element();
                points[i] = libff::G1<ppT>::random_element();
                points[i].to_affine_coordinates();
            }
            libff::leave_block("Generating random scalars and points");

            libff::enter_block("Writing binary scalars and points");
            ofstream out("msm_test_data.bin");
            if (!out) throw runtime_error("Failed to open output file.");

            out.write(reinterpret_cast<char*>(scalars.data()), n * sizeof(libff::Fr<ppT>));
            out.write(reinterpret_cast<char*>(points.data()), n * sizeof(libff::G1<ppT>));

            out.close();
            libff::leave_block("Writing binary scalars and points");
        }

        if (from_binary) {
            libff::enter_block("Reading binary MSM result for reference");
            std::ifstream in("msm_test_result.bin");
            if (!in) throw std::runtime_error("Failed to open input file.");
            in.read(reinterpret_cast<char*>(&cpu_result), sizeof(libff::G1<ppT>));

            in.close();
            libff::leave_block("Reading binary MSM result for reference");
        } else {
            libff::enter_block("Computing CPU MSM result for reference");
            size_t batch_size = (n + num_threads - 1) / num_threads;
            vector<libff::G1<ppT>> results(num_threads, libff::G1<ppT>::zero());
            for (size_t i = 0; i < num_threads; i++) {
                size_t start = i * batch_size;
                size_t end = min(start + batch_size, n);
                if (start >= end) continue;
                results[i] = libff::multi_exp<libff::G1<ppT>, libff::Fr<ppT>, libff::multi_exp_method::multi_exp_method_BDLO12>(
                    points.cbegin() + start,
                    points.cbegin() + end,
                    scalars.cbegin() + start,
                    scalars.cbegin() + end,
                    1
                );
            }
            cpu_result = libff::G1<ppT>::zero();
            for (const auto &res : results) cpu_result = cpu_result + res;
            libff::leave_block("Computing CPU MSM result for reference");

            libff::enter_block("Writing binary MSM result for reference");
            ofstream out("msm_test_result.bin");
            if (!out) throw runtime_error("Failed to open output file.");
            out.write(reinterpret_cast<char*>(&cpu_result), sizeof(libff::G1<ppT>));
            out.close();
            libff::leave_block("Writing binary MSM result for reference");
        }
    }

    template <typename GL, typename FS, typename FC>
    void gpu_bench(GL &gpu_layout, FS bench_setup, FC bench_compute) {
        libff::enter_block("GPU MSM Setup");
        bench_setup(scalars, points, gpu_layout);
        libff::leave_block("GPU MSM Setup");
        libff::enter_block("GPU MSM Compute");
        bench_compute(gpu_layout, gpu_result);
        libff::leave_block("GPU MSM Compute");
        if (gpu_result != cpu_result) {
            gpu_result.print(); cpu_result.print();
            assert(false && "GPU MSM result does not match CPU result!");
        }
    }
};

#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>

struct MSMGPULayout {
    size_t n;
    fr_t *scalars = 0;

    uint32_t *indices = 0;
    uint32_t *bucket_start, *bucket_size;

    const size_t window_bits;
    const size_t num_windows, num_buckets;

    MSMGPULayout(size_t window_bits_) :
        window_bits(window_bits_),
        num_windows((fr_t::nbits + window_bits - 1) / window_bits),
        num_buckets(1 << window_bits)
    {
        cudaMalloc(&bucket_start, num_windows * num_buckets * sizeof(uint32_t));
        cudaMalloc(&bucket_size, num_windows * num_buckets * sizeof(uint32_t));
    }

    ~MSMGPULayout()
    {
        if (scalars) cudaFree(scalars);
        if (indices) cudaFree(indices);

        cudaFree(bucket_start);
        cudaFree(bucket_size);
    }
};

void cuda_msm_setup(vector<libff::Fr<ppT>> scalars, MSMGPULayout gpu_layout)
{
    gpu_layout.n = scalars.size();
    cudaMalloc(&gpu_layout.scalars, gpu_layout.n * sizeof(fr_t));
    cudaMemcpy(gpu_layout.scalars, scalars.data(), gpu_layout.n * sizeof(fr_t), cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_layout.indices, gpu_layout.num_windows * gpu_layout.n * sizeof(uint32_t));
}

void cuda_msm_compute(MSMGPULayout &gpu_layout)
{
    // Get device properties
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    size_t sm_count = device_prop.multiProcessorCount;
    size_t threads_per_block = device_prop.maxThreadsPerBlock;
    size_t threads_per_sm = device_prop.maxThreadsPerMultiProcessor;
    size_t shared_mem_per_block = device_prop.sharedMemPerBlock;

    size_t n = gpu_layout.n;
    size_t window_bits = gpu_layout.window_bits;
    size_t num_windows = gpu_layout.num_windows;
    size_t num_buckets = gpu_layout.num_buckets;

    libff::enter_block("Bucket Scatter");
    fr_t *d_scalars = gpu_layout.scalars;
    thrust::for_each(thrust::device, d_scalars, d_scalars + n, [] __device__ (fr_t &s) { s.from(); });
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    auto indices_first = thrust::make_counting_iterator<uint32_t>(0);
    auto indices_last = thrust::make_counting_iterator<uint32_t>(num_windows * n);

    uint32_t *d_indices = gpu_layout.indices;
    thrust::transform(thrust::device, indices_first, indices_last, d_indices, [=] __device__ (uint32_t packed_id) { return packed_id % n; });
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    thrust::device_vector<uint32_t> d_window_vals(n);
    thrust::device_vector<uint32_t> du_bucket_id(num_buckets), du_bucket_size(num_buckets);
    
    uint32_t *d_bucket_size = gpu_layout.bucket_size;
    uint32_t *d_bucket_start = gpu_layout.bucket_start;
    thrust::fill(thrust::device, d_bucket_size, d_bucket_size + num_windows * num_buckets, 0);
    for (int i = 0; i < num_windows; i++) {
        auto get_window_val = [=] __device__ (uint32_t id) { return get_window(d_scalars[id], i * window_bits, window_bits); };
        thrust::transform(thrust::device, indices_first, indices_first + n, d_window_vals.begin(), get_window_val);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        thrust::sort_by_key(thrust::device, d_window_vals.begin(), d_window_vals.end(), d_indices + i * n);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        auto end = thrust::reduce_by_key(d_window_vals.begin(), d_window_vals.end(), thrust::make_constant_iterator<uint32_t>(1), du_bucket_id.begin(), du_bucket_size.begin());
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        int du_len = end.first - du_bucket_id.begin();

        thrust::scatter(thrust::device, du_bucket_size.begin(), du_bucket_size.begin() + du_len, du_bucket_id.begin(), d_bucket_size + i * num_buckets);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        thrust::exclusive_scan(thrust::device, d_bucket_size + i * num_buckets, d_bucket_size + (i + 1) * num_buckets, d_bucket_start + i * num_buckets);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
    }

    libff::leave_block("Bucket Scatter");

    
}

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    string pregen_option(argv[1]);
    assert(pregen_option == "-regen" || pregen_option == "-fast");
    MSMTest<ppT> msm_test(1 << 24, pregen_option == "-fast");
    MSMGPULayout gpu_layout(13);
    cuda_msm_setup(msm_test.scalars, gpu_layout);
    cuda_msm_compute(gpu_layout);
    // msm_test.gpu_bench(gpu_layout, cuda_msm_setup, cuda_msm_compute);

    return 0;
}
