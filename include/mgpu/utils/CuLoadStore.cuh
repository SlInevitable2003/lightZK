#pragma once

namespace mgpu {

    template<size_t bytes> struct VectorType;
    template<> struct VectorType<4>  { using type = uint1; };
    template<> struct VectorType<8>  { using type = uint2; };
    template<> struct VectorType<16> { using type = uint4; };

    template <typename T, size_t grain_size>
    __device__ void GlobalToReg(const T* d_in, size_t len, T (&reg)[grain_size], T padding) {
        const size_t bytes_total = sizeof(T) * grain_size;
        const size_t bytes = bytes_total <= 16 ? bytes_total : 16;
        
        static_assert(bytes_total % bytes == 0, "grain_size in bytes must be a multiple of vector size.");

        using V_Type = typename VectorType<bytes>::type;
        const int vec_per_thread = bytes_total / bytes;

        const V_Type* vec_in = reinterpret_cast<const V_Type*>(d_in);
        V_Type* vec_reg = reinterpret_cast<V_Type*>(reg);

        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
        const size_t thread_start_idx = tid * grain_size;
        const size_t thread_end_idx = thread_start_idx + grain_size;

        if (thread_end_idx <= len) {
            #pragma unroll
            for (int i = 0; i < vec_per_thread; ++i) vec_reg[i] = vec_in[tid * vec_per_thread + i];
        } 
        else {
            #pragma unroll
            for (int i = 0; i < grain_size; ++i) {
                size_t global_idx = thread_start_idx + i;
                reg[i] = (global_idx < len) ? d_in[global_idx] : padding;
            }
        }
    }

    template <typename T, size_t grain_size>
    __device__ void RegToGlobal(const T (&reg)[grain_size], T* d_out, size_t len) {
        const size_t bytes_total = sizeof(T) * grain_size;
        const size_t bytes = bytes_total <= 16 ? bytes_total : 16;
        
        static_assert(bytes_total % bytes == 0, "grain_size in bytes must be a multiple of vector size.");

        using V_Type = typename VectorType<bytes>::type;
        const int vec_per_thread = bytes_total / bytes;

        V_Type* vec_out = reinterpret_cast<V_Type*>(d_out);
        const V_Type* vec_reg = reinterpret_cast<const V_Type*>(reg);

        const int tid = blockIdx.x * blockDim.x + threadIdx.x;

        const size_t thread_start_idx = tid * grain_size;
        const size_t thread_end_idx = thread_start_idx + grain_size;

        if (thread_end_idx <= len) {
            #pragma unroll
            for (int i = 0; i < vec_per_thread; ++i) vec_out[tid * vec_per_thread + i] = vec_reg[i];
        } 
        else {
            #pragma unroll
            for (int i = 0; i < grain_size; ++i) {
                size_t global_idx = thread_start_idx + i;
                if (global_idx < len) d_out[global_idx] = reg[i];
            }
        }
    }

} // namespace mgpu