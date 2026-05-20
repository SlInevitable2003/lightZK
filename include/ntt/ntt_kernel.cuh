#pragma once

#include <cuda/pipeline>
#include <cooperative_groups.h>

#define FA_BLK_SIZ 1024
#define DFA_BLK_SIZ 2048

#define TILE_DIM 32

template<typename FieldT>
__global__ void transpose(FieldT *in, FieldT *out, uint32_t rows, uint32_t cols)
{ // have been successfully tested for correctness
    __shared__ FieldT tile[TILE_DIM][TILE_DIM + 1];
    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();
    
    uint32_t x = g.group_index().x * TILE_DIM + g.thread_index().x;
    uint32_t y = g.group_index().y * TILE_DIM + g.thread_index().y;

    tile[g.thread_index().y][g.thread_index().x] = in[y * cols + x];
    
    g.sync();

    uint32_t new_x = g.group_index().y * TILE_DIM + g.thread_index().x; 
    uint32_t new_y = g.group_index().x * TILE_DIM + g.thread_index().y;

    out[new_y * rows + new_x] = tile[g.thread_index().x][g.thread_index().y];
}

template<typename FieldT>
__global__ __launch_bounds__(FA_BLK_SIZ) void multi_row_ntt(FieldT *poly, FieldT *omegas, uint32_t rows, uint32_t cols) 
{
    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();
    const uint32_t tid = g.thread_rank();
    const uint32_t bid = g.group_index().x;

    __shared__ FieldT shmem[FA_BLK_SIZ];
    uint32_t scale = rows * cols;
    uint32_t item_per_thread = cols / FA_BLK_SIZ;
    uint32_t round;

    for (int item_id = 0; item_id < item_per_thread; item_id++) {
        round = 0;
        
        uint32_t idx = tid * item_per_thread + item_id;
        shmem[tid] = poly[bid * cols + idx];
        g.sync();

        for (uint32_t offset = FA_BLK_SIZ / 2; offset; offset >>= 1) {
            uint32_t cur_idx = idx & (offset * item_per_thread - 1);

            FieldT c = shmem[tid];
            if (tid & offset) {
                c = (shmem[tid ^ offset] - c) * omegas[(rows * (cur_idx << round)) % scale];
            } else {
                c = c + shmem[tid ^ offset];
            }
            g.sync();

            shmem[tid] = c;
            round ++;
            g.sync();
        }

        poly[bid * cols + idx] = shmem[tid];
        g.sync();
    }

    uint32_t round_st = round;
    for (int idx = tid; idx < cols; idx += FA_BLK_SIZ) {
        round = round_st;

        shmem[tid] = poly[bid * cols + idx];
        g.sync();

        for (uint32_t offset = item_per_thread / 2; offset; offset >>= 1) {
            uint32_t cur_idx = idx & (offset - 1);

            FieldT c = shmem[tid];
            if (tid & offset) {
                c = (shmem[tid ^ offset] - c) * omegas[(rows * (cur_idx << round)) % scale];
            } else {
                c = c + shmem[tid ^ offset];
            }
            g.sync();

            shmem[tid] = c;
            round ++;
            g.sync();
        }

        poly[bid * cols + idx] = shmem[tid];
        g.sync();
    }
}

template<typename FieldT>
__global__ __launch_bounds__(FA_BLK_SIZ) void multi_row_bitrev_permutation(FieldT *poly, uint32_t rows, uint32_t cols)
{
    namespace cg = cooperative_groups;
    cg::thread_block g = cg::this_thread_block();
    const uint32_t tid = g.thread_rank();
    const uint32_t bid = g.group_index().x;

    uint32_t log_cols = 1;
    while ((1 << log_cols) < cols) log_cols ++;

    FieldT *row_ptr = &poly[bid * cols];
    for (uint32_t i = tid; i < cols; i += FA_BLK_SIZ) {
        uint32_t j = bit_rev(i, log_cols);
        
        if (i < j) {
            FieldT tmp = row_ptr[i];
            row_ptr[i] = row_ptr[j];
            row_ptr[j] = tmp;
        }
    }
}