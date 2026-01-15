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
    
    size_t window_bits;
    size_t n_windows, n_buckets;

    fr_t *d_scalars;
    libff::G1<ppT> *d_points;

    uint32_t *bucket_size, *bucket_off;
    libff::G1<ppT> *bucket_sum;

    uint32_t *indices = 0;
    uint32_t *keys = 0, *vals = 0;
    uint32_t *keys_ = 0, *vals_ = 0;

    uint32_t *tmp0;

    MSMGPULayout(size_t window_bits_) 
    : window_bits(window_bits_),
      n_windows((fr_t::nbits + window_bits - 1) / window_bits),
      n_buckets(1 << window_bits),
      bucket_sum(new libff::G1<ppT>[n_buckets - 1])
    {
        cudaMalloc(&bucket_size, n_buckets * sizeof(uint32_t));
        cudaMalloc(&bucket_off, n_buckets * sizeof(uint32_t));

        cudaMalloc(&tmp0, (n_windows + 1) * sizeof(uint32_t));
    }

    ~MSMGPULayout() 
    {
        cudaFree(bucket_size);
        cudaFree(bucket_off);
        delete[] bucket_sum;

        if (d_scalars) cudaFree(d_scalars);
        if (d_points) delete[] d_points;

        if (keys) cudaFree(keys); if (vals) cudaFree(vals);
        if (keys_) cudaFree(keys_); if (vals_) cudaFree(vals_);
        if (indices) delete[] indices;
    }
};

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

void cuda_msm_setup(vector<libff::Fr<ppT>> scalars, vector<libff::G1<ppT>> points, MSMGPULayout &gpu_layout)
{
    assert(scalars.size() == points.size() && "Scalars and points must have the same size.");
    gpu_layout.n = scalars.size();
    
    gpu_layout.d_points = new libff::G1<ppT>[gpu_layout.n];
    memcpy(gpu_layout.d_points, points.data(), gpu_layout.n * sizeof(libff::G1<ppT>));

    cudaMalloc(&gpu_layout.d_scalars, gpu_layout.n * sizeof(fr_t));
    cudaMemcpy(gpu_layout.d_scalars, scalars.data(), gpu_layout.n * sizeof(fr_t), cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_layout.keys, gpu_layout.n_windows * gpu_layout.n * sizeof(uint32_t));
    cudaMalloc(&gpu_layout.vals, gpu_layout.n_windows * gpu_layout.n * sizeof(uint32_t));
    cudaMalloc(&gpu_layout.keys_, gpu_layout.n_windows * gpu_layout.n * sizeof(uint32_t));
    cudaMalloc(&gpu_layout.vals_, gpu_layout.n_windows * gpu_layout.n * sizeof(uint32_t));

    gpu_layout.indices = new uint32_t[gpu_layout.n_windows * gpu_layout.n];

    thrust::sequence(thrust::device, gpu_layout.tmp0, gpu_layout.tmp0 + gpu_layout.n_windows + 1, uint32_t(0), uint32_t(gpu_layout.n));
}

#include <algorithm>

#include <thrust/copy.h>
#include <thrust/for_each.h>

#include <cub/block/block_load.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

void cuda_msm_compute(MSMGPULayout &gpu_layout, libff::G1<ppT> &result)
{
    size_t n = gpu_layout.n;

    // 1. Element-wise Mont.From
    libff::enter_block("Element-wise Mont.From");

    thrust::for_each(thrust::device, gpu_layout.d_scalars, gpu_layout.d_scalars + n, [] __device__ (fr_t &x) { x.from(); });
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    libff::leave_block("Element-wise Mont.From");

    // vector<libff::Fr<ppT>> h_scalars(n);
    // cudaMemcpy(h_scalars.data(), gpu_layout.d_scalars, n * sizeof(fr_t), cudaMemcpyDeviceToHost);

    size_t w = gpu_layout.window_bits;
    size_t n_windows = gpu_layout.n_windows;
    size_t n_buckets = gpu_layout.n_buckets;
    vector<libff::G1<ppT>> window_sum(n_windows);

    // 2. Element-wise window extraction
    libff::enter_block("Element-wise window extraction");
    
    {
        const size_t item_per_thread = sizeof(fr_t) / sizeof(uint32_t), block_size = 256;
        assert((n & (n-1)) == 0 && "n must be a power of 2");
        kernel<<<n / block_size, block_size>>>([=] __device__ (uint32_t *scalar_words, uint32_t *keys, uint32_t *vals) {
            using BlockLoad = cub::BlockLoad<uint32_t, block_size, item_per_thread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

            __shared__ typename BlockLoad::TempStorage temp_storage_load;

            uint32_t s[item_per_thread];
            uint32_t blk_off = blockIdx.x * block_size;
            uint32_t tid = threadIdx.x + blk_off;
            BlockLoad(temp_storage_load).Load(scalar_words + blk_off * item_per_thread, s);

            for (int j = 0; j < n_windows; j++) {
                uint32_t key = get_window_by_ptr(reinterpret_cast<const fr_t*>(s), j * w, w);
                keys[j * n + tid] = key;
                vals[j * n + tid] = tid;
            }
        }, reinterpret_cast<uint32_t*>(gpu_layout.d_scalars), gpu_layout.keys, gpu_layout.vals);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
    }
    
    libff::leave_block("Element-wise window extraction");

    libff::enter_block("Temporary Storage Allocation");

    // Temporary Storage Pre-allocation
    void *d_temp_storage = 0;
    size_t temp_storage_bytes = 0;

    size_t temp_storage_bytes_srs = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes_srs,
        gpu_layout.keys, gpu_layout.keys_, gpu_layout.vals, gpu_layout.vals_,
        n_windows * n, n_windows, gpu_layout.tmp0, gpu_layout.tmp0 + 1
    );
    temp_storage_bytes = max(temp_storage_bytes_srs, temp_storage_bytes);

    size_t temp_storage_bytes_h = 0;
    cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes_h,
        gpu_layout.keys_, gpu_layout.bucket_size, int(n_buckets + 1),
        0, int(n_buckets), n
    );
    temp_storage_bytes = max(temp_storage_bytes_h, temp_storage_bytes);

    size_t temp_storage_bytes_s = 0;
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes_s,
        gpu_layout.bucket_size, gpu_layout.bucket_off, n_buckets
    );
    temp_storage_bytes = max(temp_storage_bytes_s, temp_storage_bytes);    

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    libff::leave_block("Temporary Storage Allocation");

    // 3. Sorting
    libff::enter_block("Sorting");

    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes_srs,
        gpu_layout.keys, gpu_layout.keys_, gpu_layout.vals, gpu_layout.vals_,
        n_windows * n, n_windows, gpu_layout.tmp0, gpu_layout.tmp0 + 1
    );
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    
    libff::leave_block("Sorting");

    vector<uint32_t> h_keys(n_windows * n), h_vals(n_windows * n);
    cudaMemcpy(h_keys.data(), gpu_layout.keys_, n_windows * n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vals.data(), gpu_layout.vals_, n_windows * n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    vector<pair<uint32_t, uint32_t>> h_keys_vals(n_windows * n);
    for (int i = 0; i < n_windows * n; i++) h_keys_vals[i] = {h_keys[i], h_vals[i]};

    for (int j = 0; j < n_windows; j++) {
        libff::enter_block("Handling window " + to_string(j));

        pair<uint32_t, uint32_t> *cur_keys_vals = h_keys_vals.data() + j * n;

        // 4. Histogram
        libff::enter_block("Histogram");
        
        cub::DeviceHistogram::HistogramEven(
        d_temp_storage, temp_storage_bytes_h,
            gpu_layout.keys_ + j * n, gpu_layout.bucket_size, int(n_buckets + 1),
            0, int(n_buckets), n
        );
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
        
        libff::leave_block("Histogram");

        // 5. Inclusive Scan
        libff::enter_block("Inclusive Scan");
        
        cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes_s,
            gpu_layout.bucket_size, gpu_layout.bucket_off, n_buckets
        );
        
        libff::leave_block("Inclusive Scan");

        vector<uint32_t> h_bucket_size(n_buckets), h_bucket_off(n_buckets);
        thrust::copy(thrust::device, gpu_layout.bucket_size, gpu_layout.bucket_size + n_buckets, h_bucket_size.begin());
        thrust::copy(thrust::device, gpu_layout.bucket_off, gpu_layout.bucket_off + n_buckets, h_bucket_off.begin());

        // 6. Segemented Reduce
        libff::enter_block("Segmented Reduction");

        #pragma omp parallel for
        for (int k = 0; k < n_buckets - 1; k++) {
            libff::G1<ppT> sum = libff::G1<ppT>::zero();
            // bucket k is indexed from bucket_off[k] to bucket_off[k + 1]
            for (int i = h_bucket_off[k]; i < h_bucket_off[k + 1]; i++) {
                uint32_t idx = cur_keys_vals[i].second;
                sum = sum + gpu_layout.d_points[idx];
            }
            gpu_layout.bucket_sum[k] = sum;
        }

        libff::leave_block("Segmented Reduction");

        // 7. Double Round Scan (Bucket Reduce)
        libff::enter_block("Double Round Scan (Bucket Reduction)");

        window_sum[j] = libff::G1<ppT>::zero();
        for (int k = n_buckets - 3; k >= 0; k--) gpu_layout.bucket_sum[k] = gpu_layout.bucket_sum[k] + gpu_layout.bucket_sum[k + 1];
        for (int k = 0; k < n_buckets - 1; k++) window_sum[j] = window_sum[j] + gpu_layout.bucket_sum[k];

        libff::leave_block("Double Round Scan (Bucket Reduction)");

        libff::leave_block("Handling window " + to_string(j));
    }

    // 8. Window Reduce
    libff::enter_block("Window Reduce");

    result = window_sum[n_windows - 1];
    for (int k = n_windows - 2; k >= 0; k--) {
        for (int l = 0; l < w; l++) result = result.dbl();
        result = result + window_sum[k];
    }

    libff::leave_block("Window Reduce");

    gpu_layout.d_scalars = 0;
}

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    string pregen_option(argv[1]);
    assert(pregen_option == "-regen" || pregen_option == "-fast");
    MSMTest<ppT> msm_test(1 << 22, pregen_option == "-fast");
    MSMGPULayout gpu_layout(13);
    msm_test.gpu_bench(gpu_layout, cuda_msm_setup, cuda_msm_compute);

    return 0;
}
