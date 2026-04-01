#pragma once
#include <vector>
#include <cassert>
#include <cstdint>
#include <stdexcept>

inline size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }
inline size_t round_up(size_t a, size_t b) { return ceil_div(a, b) * b; }
inline size_t round_down(size_t a, size_t b) { return a / b * b; }

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

template<typename T>
__device__ __host__ void print_mem(const T& val) {
   const uint8_t *ptr = reinterpret_cast<const uint8_t*>(&val);
   for (int i = 0; i < sizeof(T); i++) printf("%02x", ptr[i]);
   printf("\n");
}

template<typename F, typename... Args> __global__ void kernel(F func, Args... args) { func(args...); }

struct TypedGpuArena {
    struct Request {
        void **out_ptr;
        size_t size;
        size_t alignment;
    };

    static size_t align_up(size_t offset, size_t alignment)
    {
        size_t mod = offset % alignment;
        return mod ? (offset + alignment - mod) : offset;
    }

    std::vector<Request> requests;

    void *base_ptr = nullptr;
    bool committed = false;

public:
    TypedGpuArena() = default;

    ~TypedGpuArena() { if (base_ptr) cudaFree(base_ptr); }

    TypedGpuArena(const TypedGpuArena&) = delete;
    TypedGpuArena& operator=(const TypedGpuArena&) = delete;

    template<typename T>
    void register_alloc(T*& out_ptr, size_t count)
    {
        assert(!committed && "Cannot register after commit");
        requests.push_back({reinterpret_cast<void**>(&out_ptr), sizeof(T) * count, alignof(T)});
    }

    void commit(const char *info = "TypedGpuArena")
    {
        assert(!committed && "Already committed");

        size_t total_size = 0;

        for (auto& r : requests) {
            total_size = align_up(total_size, r.alignment);
            total_size += r.size;
        }

        cudaError_t err = cudaMalloc(&base_ptr, total_size);
        if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");

        char* ptr = reinterpret_cast<char*>(base_ptr);
        size_t offset = 0;

        for (auto& r : requests) {
            offset = align_up(offset, r.alignment);
            *(r.out_ptr) = ptr + offset;
            offset += r.size;
        }

        committed = true;
        printf("[%s] Successfully alloc %f GB memory.\n", info, double(total_size) / double(1 << 30));
    }
};

void check_gpu_ptr(void* ptr);

#pragma once
#include <cuda_runtime.h>
#include <iostream>

class GPUConfig {
public:
    int device_id;

    // device properties
    int sm_count;
    int warp_size;
    int max_threads_per_block;
    int max_threads_per_sm;
    int max_blocks_per_sm;
    size_t shared_mem_per_block;

    GPUConfig(int dev = 0) : device_id(dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        sm_count = prop.multiProcessorCount;
        warp_size = prop.warpSize;
        max_threads_per_block = prop.maxThreadsPerBlock;
        max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        shared_mem_per_block = prop.sharedMemPerBlock;
        max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock;
    }
};