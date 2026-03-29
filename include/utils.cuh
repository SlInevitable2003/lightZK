#pragma once
#include <vector>
#include <cassert>
#include <cstdint>
#include <stdexcept>

__device__ __host__ inline size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

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

template<typename T, typename dim_t = unsigned int>
class vec2d_t {
    dim_t dim_x, dim_y_owned;
    T* ptr;

public:
    __host__ __device__ vec2d_t(T* data, dim_t x) : dim_x(x), dim_y_owned(0), ptr(data) {}
    __host__ __device__ vec2d_t(T* data, dim_t x, dim_t y) : dim_x(x), dim_y_owned(y<<1), ptr(data) {}

    vec2d_t(void* data, dim_t x) : dim_x(x), dim_y_owned(0), ptr((T*)data) {}
    vec2d_t(dim_t x, size_t y) : dim_x(x), dim_y_owned(((dim_t)y<<1) | 1), ptr(new T[x*y]) {}
    vec2d_t() : dim_x(0), dim_y_owned(0), ptr(nullptr) {}

#if !defined(__CUDA_ARCH__)
    vec2d_t(const vec2d_t& other) { *this = other; dim_y_owned &= ((dim_t)0-1) << 1; }
    ~vec2d_t() { if (dim_y_owned&1) delete[] ptr; }

    inline vec2d_t& operator=(const vec2d_t& other)
    {
        if (this == &other)
            return *this;

        dim_x = other.dim_x;
        dim_y_owned = other.dim_y_owned & ((dim_t)0 - 1) << 1;
        ptr = other.ptr;

        return *this;
    }
#endif

    inline operator void*() const { return ptr; }
    __host__ __device__
    inline T* operator[](size_t y) const { return ptr + dim_x*y; }
    __host__ __device__
    inline dim_t y() const { return dim_y_owned >> 1; }
    __host__ __device__
    inline dim_t x() const { return dim_x; }
};

class GPUConfig {
private:
    int device_id;
    cudaDeviceProp props;

public:
    GPUConfig(int dev_id = 0) : device_id(dev_id) { cudaGetDeviceProperties(&props, device_id); }
    inline int sm_count() const { return props.multiProcessorCount; }
    inline int max_threads_per_sm() const { return props.maxThreadsPerMultiProcessor; }
    inline int max_threads_per_block() const { return props.maxThreadsPerBlock; }
    inline size_t shared_mem_per_block() const { return props.sharedMemPerBlock; }
    inline size_t const_mem() const { return props.totalConstMem; }
    // inline int suggest_grid_size(int block_size) const {
    //     // 目标是让每个 SM 至少有几个活跃的 Blocks
    //     return sm_count() * (max_threads_per_sm() / block_size);
    // }
};