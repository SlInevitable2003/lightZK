#pragma once

__device__ inline void vec_select(void* d, const void* a, const void* b, size_t size, bool cond) 
{
    const char* src = cond ? (const char*)a : (const char*)b;
    for (int i = 0; i < size; i++) ((char*)d)[i] = src[i];
}

__device__ inline void vec_select_by_index(void* d, const void* a, const void* b, const void *c, size_t size, int index) 
{
    const char* src = (index == 0) ? (const char*)a : ((index == 1) ? (const char*)b : (const char*)c);
    for (int i = 0; i < size; i++) ((char*)d)[i] = src[i];
}