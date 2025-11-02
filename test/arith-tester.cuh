#include <iostream>
#include <string>
#include <vector>

#include <omp.h>
using namespace std;

template <typename hostTs, typename hostTt>
class Tester {
    size_t n;

    vector<hostTs> host_array;
    vector<hostTt> host_result, device_result;
public:
    template <typename FI, typename FO> 
    Tester(size_t n_, FI init, FO op, string test_name = "test") : n(n_) {
        size_t num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        cout << "Using " << num_threads << " threads for " << test_name << " preparation." << endl;

        libff::enter_block("Generating random input data");
        host_array.resize(2 * n);
        #pragma omp parallel for
        for (size_t i = 0; i < 2 * n; i++) init(host_array[i]);
        libff::leave_block("Generating random input data");

        libff::enter_block("Computing CPU element-wise result for reference");
        host_result.resize(n);
        #pragma omp parallel for
        for (size_t i = 0; i < n; i++) host_result[i] = op(host_array[2 * i], host_array[2 * i + 1]);
        libff::leave_block("Computing CPU element-wise result for reference");
    }

    template <typename GL, typename FS, typename FC>
    void gpu_bench(GL &gpu_layout, FS bench_setup, FC bench_compute) {
        libff::enter_block("GPU Setup");
        bench_setup(host_array, gpu_layout);
        libff::leave_block("GPU Setup");
        vector<hostTt> gpu_result(host_result.size());
        libff::enter_block("GPU Compute");
        bench_compute(gpu_layout);
        libff::leave_block("GPU Compute");
        
        libff::enter_block("Move results from GPU to CPU");
        cudaMemcpy(gpu_result.data(), gpu_layout.device_result, gpu_result.size() * sizeof(hostTt), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
        libff::leave_block("Move results from GPU to CPU");

        bool match = true;
        assert(host_result.size() == gpu_result.size());
        size_t i;
        for (i = 0; i < host_result.size(); i++) if (host_result[i] != gpu_result[i]) { match = false; break; }
        if (!match) {
            cout << "GPU result does not match CPU result at index " << i << "!" << endl;
            cout << "CPU: " << host_result[i] << endl;
            cout << "GPU: " << gpu_result[i] << endl;
            exit(1);
        }
    }
};

template <typename deviceTs, typename deviceTt>
struct TestGPULayout {
    size_t n;
    deviceTs *device_array = 0;
    deviceTt *device_result = 0;

    TestGPULayout() {}
    ~TestGPULayout() {
        if (device_array) cudaFree(device_array);
        if (device_result) cudaFree(device_result);
    }
};

template <typename hostTs, typename deviceTs, typename deviceTt>
void test_gpu_setup(vector<hostTs> &host_array, TestGPULayout<deviceTs, deviceTt> &layout)
{
    assert(sizeof(hostTs) == sizeof(deviceTs));
    layout.n = host_array.size() / 2;
    cudaMalloc((void**)&layout.device_array, host_array.size() * sizeof(deviceTs));
    cudaMalloc((void**)&layout.device_result, layout.n * sizeof(deviceTt));

    cudaMemcpy(layout.device_array, host_array.data(), host_array.size() * sizeof(deviceTs), cudaMemcpyHostToDevice);
}