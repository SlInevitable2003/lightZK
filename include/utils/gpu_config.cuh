#pragma once
#include <cuda_runtime.h>

class GPUConfig {
public:
    int device_id;

    // device properties
    int sm_count;
    int warp_size;
    int max_threads_per_block;
    int max_threads_per_sm;
    int max_blocks_per_sm;
    size_t global_mem;
    size_t shared_mem_per_block;

    GPUConfig(int dev = 0) : device_id(dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        sm_count = prop.multiProcessorCount;
        warp_size = prop.warpSize;
        max_threads_per_block = prop.maxThreadsPerBlock;
        max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        global_mem = prop.totalGlobalMem;
        shared_mem_per_block = prop.sharedMemPerBlock;
        max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock;
    }
};
