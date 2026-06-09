#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <curand_kernel.h>
#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include "libff/algebra/scalar_multiplication/multiexp.hpp"

#include <omp.h>
using namespace std;

#include "api.h"
using namespace bls12_381;

namespace placeholder {
    struct Fr {
        uint32_t words[8];

        void randomize() {
            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
            for (int i = 0; i < 8; ++i) words[i] = dis(gen);
        }
    };

    struct G1 {
        uint32_t words[12 * 3];
        bool is_zero() const { for (int i = 12 * 2; i < 12 * 3; i++) if (words[i]) return false; return true; }
        
        static G1 one() {
            G1 gen;
            
            gen.words[0]  = 0xfd530c16; gen.words[1]  = 0x5cb38790;
            gen.words[2]  = 0x9976fff5; gen.words[3]  = 0x7817fc67;
            gen.words[4]  = 0x143ba1c1; gen.words[5]  = 0x154f95c7;
            gen.words[6]  = 0xf3d0e747; gen.words[7]  = 0xf0ae6acd;
            gen.words[8]  = 0x21dbf440; gen.words[9]  = 0xedce6ecc;
            gen.words[10] = 0x9e0bfb75; gen.words[11] = 0x12017741;

            gen.words[12] = 0x0ce72271; gen.words[13] = 0xbaac93d5;
            gen.words[14] = 0x7918fd8e; gen.words[15] = 0x8c22631a;
            gen.words[16] = 0x570725ce; gen.words[17] = 0xdd595f13;
            gen.words[18] = 0x50405194; gen.words[19] = 0x51ac5829;
            gen.words[20] = 0xad0059c0; gen.words[21] = 0x0e1c8c3f;
            gen.words[22] = 0x5008a26a; gen.words[23] = 0x0bbc3efc;

            gen.words[24] = 0x0002fffd; gen.words[25] = 0x76090000;
            gen.words[26] = 0xc40c0002; gen.words[27] = 0xebf4000b;
            gen.words[28] = 0x53c758ba; gen.words[29] = 0x5f489857;
            gen.words[30] = 0x70525745; gen.words[31] = 0x77ce5853;
            gen.words[32] = 0xa256ec6d; gen.words[33] = 0x5c071a97;
            gen.words[34] = 0xfa80e493; gen.words[35] = 0x15f65ec3; 

            return gen;
        }
    };
}

int main(int argc, char *argv[])
{
    assert(sizeof(g1_t) == sizeof(placeholder::G1));

    const size_t window_bits = 13;

    size_t scale = (1 << 20);
    if (argc > 2) scale = 1 << stoul(argv[2]);
    
    MSMContext<fr_t, g1_t::affine_t, g1_t, g1_bucket_t, placeholder::Fr, placeholder::G1> msm_ctx(scale, 13);
    BucketContext<fr_t, placeholder::Fr> bkt_ctx(scale, 13);

    libff::enter_block("Generating random scalars and points");

    vector<placeholder::Fr> host_scalars(scale);
    #pragma omp parallel for
    for (auto &x : host_scalars) x.randomize();

    vector<placeholder::G1> host_bases(scale, placeholder::G1::one());
    g1_t *device_bases;
    cudaMalloc((void**)&device_bases, scale * sizeof(g1_t));
    cudaMemcpy(device_bases, host_bases.data(), scale * sizeof(g1_t), cudaMemcpyHostToDevice);
    kernel<<<ceil_div(scale, 256), 256>>>([=] __device__ (g1_t *points) {
        namespace cg = cooperative_groups;
        cg::grid_group g = cg::this_grid();
        
        const uint32_t idx = g.thread_rank();
        if (idx >= scale) return;

        curandState local_state;
        curand_init(42, idx, 0, &local_state);

        g1_t p_base = points[idx];
        g1_t p_res = p_base;
        for (int i = 0; i < 256; i++) {
            p_res.dbl();
            if (curand(&local_state) & 1) p_res.add(p_base); 
        }

        p_res = g1_t::affine_t(p_res);
        points[idx] = p_res;
    }, device_bases);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(host_bases.data(), device_bases, scale * sizeof(g1_t), cudaMemcpyDeviceToHost);

    libff::leave_block("Generating random scalars and points");

    libff::enter_block("GPU MSM Setup");

    msm_ctx.load_bases(host_bases.data());
    bkt_ctx.load_scalars(host_scalars.data());

    libff::leave_block("GPU MSM Setup");

    libff::enter_block("GPU MSM Compute");

    bkt_ctx.process(true, false, false, false);
    bkt_ctx.process(false);
    msm_ctx.msm(bkt_ctx, &host_bases[0]);

    libff::leave_block("GPU MSM Compute");

    cudaFree(device_bases);

    return 0;
}