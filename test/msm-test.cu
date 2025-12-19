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
class MSMTest {
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

struct MSMGPULayout {
    size_t n;
    fr_t *scalars = 0;
    g1_t::affine_t *points = 0;

    uint32_t *indices = 0;
    uint32_t *bucket_start, *bucket_end, *bucket_size;
    g1_t *bucket_sum;

    const size_t window_bits;
    const size_t num_windows, num_buckets;
    const size_t parallel_degree;

    MSMGPULayout(size_t window_bits_ = 8, size_t parallel_degree_ = PARALLEL_DEGREE) : 
        window_bits(window_bits_), 
        num_windows((fr_t::nbits + window_bits - 1) / window_bits), 
        num_buckets(1 << window_bits),
        parallel_degree(parallel_degree_)
    {
        cudaMalloc(&bucket_start, num_windows * num_buckets * sizeof(uint32_t));
        cudaMalloc(&bucket_end, num_windows * num_buckets * sizeof(uint32_t));
        cudaMalloc(&bucket_size, num_windows * num_buckets * sizeof(uint32_t));

        cudaMalloc(&bucket_sum, parallel_degree * num_windows * num_buckets * sizeof(g1_t));
    }

    ~MSMGPULayout() {
        if (scalars) cudaFree(scalars);
        if (points) cudaFree(points);
        if (indices) cudaFree(indices);

        cudaFree(bucket_start);
        cudaFree(bucket_end);
        cudaFree(bucket_size);
        cudaFree(bucket_sum);
    }
};

void cuda_msm_setup(vector<libff::Fr<ppT>> scalars, vector<libff::G1<ppT>> points, MSMGPULayout &gpu_layout)
{
    assert(scalars.size() == points.size() && "Scalars and points must have the same size");
    gpu_layout.n = scalars.size();
    cudaMalloc(&gpu_layout.scalars, gpu_layout.n * sizeof(fr_t));
    cudaMalloc(&gpu_layout.points, gpu_layout.n * sizeof(g1_t::affine_t));

    vector<g1_t::affine_t> temp(gpu_layout.n);
    memset(temp.data(), 0, temp.size() * sizeof(g1_t::affine_t));
    #pragma omp parallel for
    for (size_t i = 0; i < gpu_layout.n; i++) if (!points[i].is_zero()) memcpy(&temp[i], &points[i], sizeof(g1_t::affine_t));

    cudaMemcpy(gpu_layout.scalars, scalars.data(), gpu_layout.n * sizeof(fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_layout.points, temp.data(), gpu_layout.n * sizeof(g1_t::affine_t), cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_layout.indices, gpu_layout.num_windows * gpu_layout.n * sizeof(uint32_t));
}

void cuda_msm_compute(MSMGPULayout &gpu_layout, libff::G1<ppT> &result)
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
    size_t parallel_degree = gpu_layout.parallel_degree;

    // Bucket-Scatter
    libff::enter_block("Bucket Scatter");
    cudaMemset(gpu_layout.bucket_size, 0, num_windows * num_buckets * sizeof(uint32_t));
    kernel<<<sm_count * threads_per_sm / threads_per_block, threads_per_block>>>([=] __device__ (fr_t *scalars) {
        uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t stride = blockDim.x * gridDim.x;
        for (uint32_t i = thread_id; i < n; i += stride) scalars[i].from();
    }, gpu_layout.scalars);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    #define COUNT_SIZ 1024
    assert(n % COUNT_SIZ == 0 && "n must be divisible by COUNT_SIZ");
    assert(num_buckets * sizeof(uint32_t) <= shared_mem_per_block);
    kernel<<<num_windows, COUNT_SIZ, num_buckets * sizeof(uint32_t)>>>([=] __device__ (fr_t *scalars, uint32_t *bucket_size) {
        extern __shared__ uint32_t shm_bucket_size[];
        for (uint32_t i = threadIdx.x; i < num_buckets; i += COUNT_SIZ) shm_bucket_size[i] = 0;
        __syncthreads();

        uint32_t window_id = blockIdx.x;
        for (int i = threadIdx.x; i < n; i += COUNT_SIZ) {
            fr_t *ptr_s = scalars + i;
            uint32_t bucket_id = get_window_by_ptr(ptr_s, window_id * window_bits, window_bits);
            atomicAdd(&shm_bucket_size[bucket_id], 1);
        }
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < num_buckets; i += COUNT_SIZ) bucket_size[window_id * num_buckets + i] += shm_bucket_size[i];
    }, gpu_layout.scalars, gpu_layout.bucket_size);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    size_t block_size = min(threads_per_block, num_buckets);
    assert(num_windows * num_buckets % block_size == 0 && "num_windows * num_buckets must be divisible by block_size");
    size_t grid_size = num_windows * num_buckets / block_size;
    kernel<<<grid_size, block_size>>>([=] __device__ (uint32_t *bucket_size, uint32_t *bucket_start, uint32_t *bucket_end) {
        uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t window_id = thread_id / num_buckets;
        uint32_t bucket_id = thread_id % num_buckets;

        uint32_t start = 0;
        for (uint32_t i = 0; i < num_buckets; i++) start += ((i < bucket_id) ? bucket_size[window_id * num_buckets + i] : 0);

        bucket_start[window_id * num_buckets + bucket_id] = start;
        bucket_end[window_id * num_buckets + bucket_id] = start;
        // if (window_id == 0)
            // printf("Window %u, Bucket %u: Start = %u, End = %u\n", window_id, bucket_id, start, start + bucket_size[window_id * num_buckets + bucket_id]);
    }, gpu_layout.bucket_size, gpu_layout.bucket_start, gpu_layout.bucket_end);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // vector<uint32_t> host_bucket_size(num_windows * num_buckets);
    // cudaMemcpy(host_bucket_size.data(), gpu_layout.bucket_size, num_windows * num_buckets * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < num_windows; i++) {
    //     double mu = 0;
    //     uint32_t min_size = -1, max_size = 0;
    //     for (int j = 0; j < num_buckets; j++) {
    //         min_size = min(min_size, host_bucket_size[i * num_buckets + j]);
    //         max_size = max(max_size, host_bucket_size[i * num_buckets + j]);
    //         mu += 1. * host_bucket_size[i * num_buckets + j] / num_buckets;
    //     }
    //     double sigma = 0;
    //     for (int j = 0; j < num_buckets; j++) sigma += (host_bucket_size[i * num_buckets + j] - mu) * (host_bucket_size[i * num_buckets + j] - mu) / num_buckets;
    //     printf("Window %u: Min = %u, Max = %u, Mu = %.2f, Sigma = %.2f\n", i, min_size, max_size, mu, sigma);
    // }

    #define SCATTER_SIZ 512
    assert(n % SCATTER_SIZ == 0 && "n must be divisible by COUNT_SIZ");
    assert(num_windows <= 32);
    kernel<<<num_windows * n / SCATTER_SIZ, SCATTER_SIZ, num_buckets * sizeof(uint32_t)>>>([=] __device__ (
        fr_t *scalars, uint32_t *bucket_end, uint32_t *indices) 
    { 
        extern __shared__ uint32_t shm_bucket_size[];
        for (uint32_t i = threadIdx.x; i < num_buckets; i += blockDim.x) shm_bucket_size[i] = 0;
        __syncthreads();

        uint32_t window_id = blockIdx.x / (n / SCATTER_SIZ);
        uint32_t batch_id = blockIdx.x % (n / SCATTER_SIZ);
        uint32_t indice = batch_id * SCATTER_SIZ + threadIdx.x;
        fr_t *ptr_s = scalars + indice;
        uint32_t bucket_id = get_window_by_ptr(ptr_s, window_id * window_bits, window_bits);
        atomicAdd(&shm_bucket_size[bucket_id], 1);
        __syncthreads();

        uint32_t *shm_bucket_end = shm_bucket_size;
        for (uint32_t i = threadIdx.x; i < num_buckets; i += blockDim.x) shm_bucket_end[i] = atomicAdd(&bucket_end[window_id * num_buckets + i], shm_bucket_size[i]);
        __syncthreads();

        uint32_t index = atomicAdd(&shm_bucket_end[bucket_id], 1);
        indices[window_id * n + index] = indice;
    }, gpu_layout.scalars, gpu_layout.bucket_end, gpu_layout.indices);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    libff::leave_block("Bucket Scatter");
    #undef SCATTER_SIZ

    // Inner-Bucket-Sum
    uint32_t lst_valid_buckets = 1 << (fp_t::nbits % window_bits);
    libff::enter_block("Inner Bucket Sum");
    assert(parallel_degree % 32 == 0 && "parallel_degree must be divisible by 32");
    // inner_bucket_sum<<<num_windows * num_buckets, parallel_degree>>>(
    //     n, num_buckets, parallel_degree,
    //     gpu_layout.points, gpu_layout.bucket_sum, gpu_layout.bucket_start, gpu_layout.bucket_end, gpu_layout.indices
    // );
    (inner_bucket_sum_with_xyzz_as_medium<g1_t::affine_t, g1_t, g1_bucket_t>)<<<num_windows * num_buckets, parallel_degree>>>(
        n, num_windows, num_buckets, parallel_degree, lst_valid_buckets,
        gpu_layout.points, gpu_layout.bucket_sum, gpu_layout.bucket_start, gpu_layout.bucket_end, gpu_layout.indices
    );
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    libff::leave_block("Inner Bucket Sum");

    // exit(0);

    // Bucket & Window Reduce
    assert(num_windows * num_buckets % REDUCE_SIZ == 0 && "num_windows * num_buckets must be divisible by REDUCE_SIZ");
    libff::enter_block("Bucket Reduce");
    parallel_bucket_reduce<<<num_windows * num_buckets / REDUCE_SIZ, REDUCE_SIZ>>>(
        num_buckets, parallel_degree,
        gpu_layout.bucket_sum
    );
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    kernel<<<num_windows, REDUCE_SIZ>>>([=] __device__ (g1_t *bucket_sum) {
        uint32_t window_id = blockIdx.x;
        uint32_t thread_id = threadIdx.x;
        
        g1_t acc; acc.inf();
        for (uint32_t i = thread_id; i < num_buckets; i += REDUCE_SIZ) {
            uint32_t ii = (window_id < num_windows - 1) ? i : (i % lst_valid_buckets);
            g1_t addend = bucket_sum[window_id * (num_buckets * parallel_degree) + i * parallel_degree];
            g1_t incr; incr.inf();
            for (uint32_t j = window_bits; j > 0; j--) {
                incr.dbl();
                g1_t update = incr; update.add(addend);
                vec_select(&incr, &update, &incr, sizeof(g1_t), ii & (1 << (j - 1)));
            }
            acc.add(incr);
        }
        g1_t incr;
        for (uint32_t i = 1; i < 32; i <<= 1) {
            incr = custom_shfl_xor(acc, i);
            acc.add(incr);
        }

        if (thread_id % 32 == 0) bucket_sum[window_id * (num_buckets * parallel_degree) + thread_id / 32] = acc;
    }, gpu_layout.bucket_sum);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    kernel<<<1, 32>>>([=] __device__ (g1_t *bucket_sum) {
        uint32_t i = threadIdx.x;
        if (i >= num_windows) return;
        g1_t acc; acc.inf();
        for (int j = 0; j < REDUCE_SIZ / 32; j++) acc.add(bucket_sum[i * (num_buckets * parallel_degree) + j]);
        bucket_sum[i] = acc;
    }, gpu_layout.bucket_sum);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    libff::leave_block("Bucket Reduce");

    libff::enter_block("Window Reduce");
    vector<libff::G1<ppT>> window_sum(num_windows);
    cudaMemcpy(window_sum.data(), gpu_layout.bucket_sum, num_windows * sizeof(libff::G1<ppT>), cudaMemcpyDeviceToHost);
    result = libff::G1<ppT>::zero();
    for (size_t i = num_windows; i > 0; i--) {
        for (size_t j = 0; j < window_bits; j++) result = result.dbl();
        result = result +  window_sum[i - 1];
    }
    libff::leave_block("Window Reduce");
}

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    string pregen_option(argv[1]);
    assert(pregen_option == "-regen" || pregen_option == "-fast");
    MSMTest<ppT> msm_test(1 << 24, pregen_option == "-fast");
    MSMGPULayout gpu_layout(13);
    msm_test.gpu_bench(gpu_layout, cuda_msm_setup, cuda_msm_compute);

    return 0;
}
