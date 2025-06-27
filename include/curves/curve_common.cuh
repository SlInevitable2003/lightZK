__device__ inline void vec_select(void* d, const void* a, const void* b, size_t size, bool cond) 
{
    const char* src = cond ? (const char*)a : (const char*)b;
    for (int i = 0; i < size; i++) ((char*)d)[i] = src[i];
}