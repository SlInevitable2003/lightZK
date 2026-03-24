#include "utils.cuh"

void check_gpu_ptr(void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err != cudaSuccess) {
        printf("非法指针或未分配的内存: %s\n", cudaGetErrorString(err));
    } else {
        if (attributes.type == cudaMemoryTypeDevice) {
            printf("合法 GPU 指针，设备 ID: %d\n", attributes.device);
        } else if (attributes.type == cudaMemoryTypeHost) {
            printf("这是主机 (CPU) 指针\n");
        }
    }
}