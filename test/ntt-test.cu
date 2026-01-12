#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include <libfqfft/evaluation_domain/get_evaluation_domain.hpp>

#include <omp.h>
using namespace std;

#include "api.h"
using namespace alt_bn128;
typedef libsnark::default_r1cs_ppzksnark_pp ppT;

template <typename ppT>
class NTTTest {
    size_t n;

    vector<libff::Fr<ppT>> coeff;
    libff::Fr<ppT> coset;

    vector<libff::Fr<ppT>> cpu_result, gpu_result;
public:
    NTTTest(size_t n_, bool from_binary = false) : n(n_), coset(libff::Fr<ppT>::multiplicative_generator) {
        size_t num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        cout << "Using " << num_threads << " threads for NTT-Test preparation." << endl;

        if (from_binary) {
            libff::enter_block("Reading binary coefficients");
            std::ifstream in("ntt_test_data.bin");
            if (!in) throw std::runtime_error("Failed to open input file.");
            
            coeff.resize(n);
            in.read(reinterpret_cast<char*>(coeff.data()), n * sizeof(libff::Fr<ppT>));

            in.close();
            libff::leave_block("Reading binary coefficients");
        } else {
            libff::enter_block("Generating random coefficients");
            coeff.resize(n);
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) coeff[i] = libff::Fr<ppT>::random_element();
            libff::leave_block("Generating random coefficients");

            libff::enter_block("Writing binary coefficients");
            ofstream out("ntt_test_data.bin");
            if (!out) throw runtime_error("Failed to open output file.");

            out.write(reinterpret_cast<char*>(coeff.data()), n * sizeof(libff::Fr<ppT>));

            out.close();
            libff::leave_block("Writing binary coefficients");
        }

        if (from_binary) {
            libff::enter_block("Reading binary NTT result for reference");
            std::ifstream in("ntt_test_result.bin");
            if (!in) throw std::runtime_error("Failed to open input file.");
            cpu_result.resize(n);
            in.read(reinterpret_cast<char*>(cpu_result.data()), n * sizeof(libff::Fr<ppT>));

            in.close();
            libff::leave_block("Reading binary NTT result for reference");
        } else {
            libff::enter_block("Computing CPU NTT result for reference");
            cpu_result = coeff;
            auto domain = libfqfft::get_evaluation_domain<libff::Fr<ppT>>(n);
            domain->cosetFFT(cpu_result, coset);
            libff::leave_block("Computing CPU NTT result for reference");

            libff::enter_block("Writing binary NTT result for reference");
            ofstream out("ntt_test_result.bin");
            if (!out) throw runtime_error("Failed to open output file.");
            out.write(reinterpret_cast<char*>(cpu_result.data()), n * sizeof(libff::Fr<ppT>));
            out.close();
            libff::leave_block("Writing binary NTT result for reference");
        }
    }

    template <typename GL, typename FS, typename FC, typename FL>
    void gpu_bench(GL &gpu_layout, FS bench_setup, FC bench_compute, FL bench_load) {
        libff::enter_block("GPU NTT Setup");
        bench_setup(coeff, coset, gpu_layout);
        libff::leave_block("GPU NTT Setup");
        libff::enter_block("GPU NTT Compute");
        bench_compute(gpu_layout);
        libff::leave_block("GPU NTT Compute");
        libff::enter_block("GPU NTT Load");
        bench_load(gpu_layout, gpu_result);
        libff::leave_block("GPU NTT Load");

        int i;
        for (i = 0; i < n; i++) if (cpu_result[i] != gpu_result[i]) break;
        if (i < n) {
            gpu_result[i].print(); cpu_result[i].print();
            assert(false && "GPU NTT result does not match CPU result!");
        }
    }
};

struct NTTGPULayout {
    size_t n;
    fr_t *coeff = 0, *power = 0;
    fr_t *coset = 0;

    NTTGPULayout() {}

    ~NTTGPULayout() {
        if (coeff) cudaFree(coeff);
        if (power) cudaFree(power);
        if (coset) cudaFree(coset);
    }
};

void cuda_ntt_setup(vector<libff::Fr<ppT>> coeff, libff::Fr<ppT> coset, NTTGPULayout &gpu_layout)
{
    gpu_layout.n = coeff.size();
    cudaMalloc(&gpu_layout.coeff, gpu_layout.n * sizeof(fr_t));
    cudaMalloc(&gpu_layout.power, gpu_layout.n * sizeof(fr_t));
    cudaMalloc(&gpu_layout.coset, gpu_layout.n * sizeof(fr_t));

    int log_n = __builtin_ctzl(gpu_layout.n);

    cudaMemcpy(gpu_layout.coeff, coeff.data(), gpu_layout.n * sizeof(fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_layout.power + 1, &forward_roots_of_unity[log_n], sizeof(fr_t), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_layout.coset + 1, &coset, sizeof(fr_t), cudaMemcpyHostToDevice);

    int n = gpu_layout.n;
    auto power_computing = [=] __device__ (fr_t *arr) {
        arr[0] = fr_t::one();
        fr_t fact = arr[1];
        for (int i = 2; i < n; i++) arr[i] = arr[i - 1] * fact;
    };
    kernel<<<1, 1>>>(power_computing, gpu_layout.power);
    kernel<<<1, 1>>>(power_computing, gpu_layout.coset);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

void cuda_ntt_compute(NTTGPULayout &gpu_layout)
{
    auto butterfly_transformation = [=] __device__ (fr_t *poly, fr_t *omegas, int round, int adjoint_bit) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int current_index = i & (adjoint_bit - 1);
        int self_index = ((i - current_index) << 1) | current_index;
        int adjo_index = self_index | adjoint_bit;
        fr_t self = poly[self_index], adjo = poly[adjo_index];
        poly[self_index] = self + adjo;
        poly[adjo_index] = (self - adjo) * omegas[current_index << round];
    };
    auto bitrev_permutation = [=] __device__ (fr_t *poly, int round) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = bit_rev(i, round);
        if (i < j) {
            fr_t tmp = poly[i];
            poly[i] = poly[j];
            poly[j] = tmp;
        }
    };

    int n = gpu_layout.n;

    thrust::transform(thrust::device, gpu_layout.coeff, gpu_layout.coeff + n, gpu_layout.coset, gpu_layout.coeff, [] __device__ (fr_t a, fr_t b) { return a * b; });
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    int round, adjoint_bit;
    for (round = 0, adjoint_bit = n / 2; adjoint_bit > 0; round ++, adjoint_bit >>= 1) {
        kernel<<<n / 2048, 1024>>>(butterfly_transformation, gpu_layout.coeff, gpu_layout.power, round, adjoint_bit);
        // cudaDeviceSynchronize();
        // CUDA_CHECK(cudaGetLastError());
    }

    kernel<<<n / 1024, 1024>>>(bitrev_permutation, gpu_layout.coeff, round);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

void cuda_ntt_load(NTTGPULayout &gpu_layout, vector<libff::Fr<ppT>> &gpu_result)
{
    gpu_result.resize(gpu_layout.n);
    cudaMemcpy(gpu_result.data(), gpu_layout.coeff, gpu_layout.n * sizeof(fr_t), cudaMemcpyDeviceToHost);
}

int main(int argc, char *argv[])
{
    ppT::init_public_params();

    string pregen_option(argv[1]);
    assert(pregen_option == "-regen" || pregen_option == "-fast");
    NTTTest<ppT> ntt_test(1 << 24, pregen_option == "-fast");
    NTTGPULayout gpu_layout;
    ntt_test.gpu_bench(gpu_layout, cuda_ntt_setup, cuda_ntt_compute, cuda_ntt_load);
}