#pragma once
#include "fields/bucket.cuh"
#include "msm/msm_kernel.cuh"
#include "utils.cuh"

#include <vector>

#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

template <typename ProjT>
struct ProjT_ADD {
    __device__ ProjT operator()(const ProjT &lhs, const ProjT &rhs) const {
        ProjT res;
        ProjT::add(res, lhs, rhs);
        return res;
    }
};

template <typename FieldT, typename AffT, typename ProjT, typename XYZZT,
          typename HostFT, typename HostPT, bool is_g2 = false, size_t instances = 1>
class MSMContext {
    ProjT infty;

    AffT *bases[instances], *lifted_bases[instances];

    ProjT *buckets_sum_WWR;
    ProjT *buckets_sum, *buckets_sum_buffer;
    ProjT *windows_sum;

    TypedGpuArena arena;
    void *temp_storage;
    size_t temp_storage_bytes;

public:
    size_t scale;
    size_t window_bits;
    size_t windows_count, buckets_count;

    size_t last_window_buckets_count;
    size_t last_window_warps_per_bucket;

    static const size_t threads_unit = is_g2 + 1;

    MSMContext(size_t scale_, size_t window_bits_)
        : scale(scale_), window_bits(window_bits_),
          windows_count(ceil_div(FieldT::nbits, window_bits)), buckets_count(1 << window_bits),
          last_window_buckets_count(1 << (FieldT::nbits % window_bits)), last_window_warps_per_bucket(buckets_count / last_window_buckets_count)
    {
        for (size_t i = 0; i < instances; i++) {
            arena.register_alloc(bases[i], threads_unit * scale);
            arena.register_alloc(lifted_bases[i], threads_unit * scale);
        }
        arena.register_alloc(buckets_sum_WWR, buckets_count * 32);
        arena.register_alloc(buckets_sum, threads_unit * (buckets_count - 1));
        arena.register_alloc(buckets_sum_buffer, threads_unit * (buckets_count - 1));
        arena.register_alloc(windows_sum, threads_unit * windows_count); 
        arena.commit("MSMContext"); 

        if (!is_g2) {
            HostPT host_infty = HostPT::zero();
            memcpy(&infty, &host_infty, sizeof(ProjT));

            temp_storage = 0, temp_storage_bytes = 0; 
            size_t temp_storage_bytes_upd = 0; 
            
            temp_storage_bytes_upd = 0; 
            cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes_upd, buckets_sum, buckets_sum_buffer, ProjT_ADD<ProjT>(), buckets_count - 1); 
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
            temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes); 

            temp_storage_bytes_upd = 0;
            ProjT local_init;
            cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes_upd, buckets_sum_buffer, windows_sum, buckets_count - 1, ProjT_ADD<ProjT>(), infty);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
            temp_storage_bytes = max(temp_storage_bytes_upd, temp_storage_bytes);
            
            cudaError_t err = cudaMalloc(&temp_storage, temp_storage_bytes); 
            if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed"); 
            printf("[MSMContext] Successfully alloc %f GB memory.\n", double(temp_storage_bytes) / double(1 << 30));
        }

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());
    }

    ~MSMContext() { if (temp_storage) cudaFree(temp_storage); }

    void load_bases(const HostPT *host_bases, bool lift = true, size_t instance_id = 0)
    {
        std::vector<AffT> buffer(threads_unit * scale);
        memset(buffer.data(), 0, buffer.size() * sizeof(AffT));
        #pragma omp parallel for
        for (size_t i = 0; i < scale; i++) if (!host_bases[i].is_zero()) memcpy(&buffer[threads_unit * i], &host_bases[i], threads_unit * sizeof(AffT));
        cudaMemcpy(bases[instance_id], buffer.data(), threads_unit * scale * sizeof(AffT), cudaMemcpyHostToDevice);

        if (is_g2) {
            static_assert(sizeof(AffT) >= 2 * 4 * sizeof(uint32_t));
            kernel<<<ceil_div(scale, 256), 256>>>([=] __device__ (AffT *points) {
                namespace cg = cooperative_groups;
                cg::grid_group g = cg::this_grid();
                int idx = g.thread_rank();
                if (idx >= scale) return;

                uint4 *base = reinterpret_cast<uint4*>(&points[idx << 1]);
                const int stride = sizeof(AffT) / (2 * sizeof(uint4));
                uint4 buffer[stride];
                for (int i = 0; i < stride; i++) buffer[i] = base[stride + i];
                for (int i = 0; i < stride; i++) base[stride + i] = base[stride * 2+ i];
                for (int i = 0; i < stride; i++) base[stride * 2 + i] = buffer[i];
            }, bases[instance_id]);
            cudaDeviceSynchronize();
            CUDA_CHECK(cudaGetLastError());
        }

        if (lift) {
            cudaMemcpy(lifted_bases[instance_id], bases[instance_id], threads_unit * scale * sizeof(AffT), cudaMemcpyDeviceToDevice);
            size_t dbl_off = (windows_count / 2) * window_bits;

            if (!is_g2) {
                thrust::for_each(thrust::device, lifted_bases[instance_id], lifted_bases[instance_id] + scale, [=] __device__ (AffT &x) {
                    ProjT p = x;
                    for (int i = 0; i < dbl_off; i++) p.dbl();
                    x = p;
                });
            } else {
                kernel<<<ceil_div(scale, 32), 64>>>([=] __device__ (AffT *points) {
                    namespace cg = cooperative_groups;
                    cg::grid_group g = cg::this_grid();
                    int idx = g.thread_rank();
                    if (idx >= 2 * scale) return;

                    ProjT p = points[idx];
                    for (int i = 0; i < dbl_off; i++) p.dbl();
                    points[idx] = p;
                }, lifted_bases[instance_id]);
                cudaDeviceSynchronize();
                CUDA_CHECK(cudaGetLastError());
            }
        }

    }

    // ---------------- 内部实现 --------------------

    void warp_reduce(cudaStream_t stream, bool last_window = false, bool batch = false)
    {
        size_t last_window_warps_per_bucket = this->last_window_warps_per_bucket;

        size_t upper_bound = (last_window && !batch) ? last_window_buckets_count : buckets_count;
        size_t cur_window_buckets_count = last_window ? last_window_buckets_count : buckets_count;

        size_t threads_unit = this->threads_unit;

        if (!last_window) {
            kernel<<<cur_window_buckets_count - 1, 32, 0, stream>>>([=] __device__ (ProjT *buckets_sum_WWR, ProjT *buckets_sum) {
                uint32_t bucket_id = blockIdx.x;
                uint32_t lane_id = threadIdx.x;
                ProjT acc = buckets_sum_WWR[bucket_id * 32 + lane_id], incr;
                
                for (uint32_t i = threads_unit; i < 32; i <<= 1) {
                    incr = custom_shfl_xor(acc, i);
                    acc.add(incr);
                }

                if (lane_id < threads_unit) buckets_sum[threads_unit * (upper_bound - 2 - bucket_id) + lane_id] = acc;
            }, buckets_sum_WWR, buckets_sum);
        } else if (!batch) {
            kernel<<<cur_window_buckets_count - 1, 32, 0, stream>>>([=] __device__ (ProjT *buckets_sum_WWR, ProjT *buckets_sum) {
                uint32_t bucket_id = blockIdx.x;
                uint32_t lane_id = threadIdx.x;
                ProjT acc, incr; acc.inf();
                
                for (uint32_t i = 0; i < last_window_warps_per_bucket; i++) {
                    incr = buckets_sum_WWR[bucket_id * last_window_warps_per_bucket * 32 + i * 32 + lane_id];
                    acc.add(incr);
                }

                for (uint32_t i = threads_unit; i < 32; i <<= 1) {
                    incr = custom_shfl_xor(acc, i);
                    acc.add(incr);
                }

                if (lane_id < threads_unit) buckets_sum[threads_unit * (upper_bound - 2 - bucket_id) + lane_id] = acc;
            }, buckets_sum_WWR, buckets_sum);
        } else {
            kernel<<<cur_window_buckets_count - 1, 32, 0, stream>>>([=] __device__ (ProjT *buckets_sum_WWR, ProjT *buckets_sum) {
                uint32_t bucket_id = blockIdx.x;
                uint32_t lane_id = threadIdx.x;
                ProjT acc, incr; acc.inf();
                
                for (uint32_t i = 0; i < last_window_warps_per_bucket; i++) {
                    incr = buckets_sum_WWR[bucket_id * last_window_warps_per_bucket * 32 + i * 32 + lane_id];
                    acc.add(incr);
                }

                for (uint32_t i = threads_unit; i < 32; i <<= 1) {
                    incr = custom_shfl_xor(acc, i);
                    acc.add(incr);
                }

                if (lane_id < threads_unit) buckets_sum[threads_unit * (upper_bound - 2 - bucket_id) + lane_id].add(acc);
            }, buckets_sum_WWR, buckets_sum);
        }
    }

    void bucket_reduce(int window_id, cudaStream_t stream, bool last_window = false, bool batch = false)
    {
        size_t upper_bound = (last_window && !batch) ? last_window_buckets_count : buckets_count;
        
        if (!is_g2) {
            cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, buckets_sum, buckets_sum_buffer, ProjT_ADD<ProjT>(), upper_bound - 1, stream);
            cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, buckets_sum_buffer, windows_sum + window_id, upper_bound - 1, ProjT_ADD<ProjT>(), infty, stream);
        } else {
            upper_bound -= 1;

            // TODO: The code segement below inccurs numerous UB
            const size_t blk_siz = 256;

            auto block_scan = [] __device__ (ProjT *shmem, uint32_t tid, uint32_t blk_siz) {
                for (int offset = 2; offset < blk_siz; offset <<= 1) {
                    ProjT val; val.inf();
                    if (tid >= offset) val = shmem[tid - offset];
                    __syncthreads();
                    
                    shmem[tid].add(val);
                    __syncthreads();
                }
            };

            auto block_reduce = [] __device__ (ProjT *shmem, uint32_t tid, uint32_t blk_siz) {
                for (int offset = blk_siz >> 1; offset >= 2; offset >>= 1) {
                    ProjT val = shmem[tid ^ offset];
                    __syncthreads();
                    
                    shmem[tid].add(val);
                    __syncthreads();
                }
            };

            size_t spines_count = ceil_div(upper_bound, blk_siz / 2);

            kernel<<<spines_count, blk_siz, 0, stream>>>([=] __device__ (ProjT *buckets_sum, ProjT *buckets_sum_buffer) {
                __shared__ ProjT shmem[blk_siz];

                namespace cg = cooperative_groups;
                cg::thread_block g = cg::this_thread_block();

                uint32_t gid = cg::this_grid().thread_rank();
                uint32_t tid = g.thread_rank();

                if (gid < upper_bound * 2) shmem[tid] = buckets_sum[gid];
                else shmem[tid].inf();
                cg::sync(g);

                block_scan(shmem, tid, blk_siz);

                if (tid >= blk_siz - 2) buckets_sum_buffer[g.group_index().x * 2 + (tid & 1)] = shmem[tid];

                if (gid < upper_bound * 2) buckets_sum[gid] = shmem[tid];
            }, buckets_sum, buckets_sum_buffer);

            size_t spines_count_2power = 1;
            while (spines_count_2power < spines_count) spines_count_2power <<= 1;
            kernel<<<1, spines_count_2power * 2, spines_count_2power * 2 * sizeof(ProjT), stream>>>([=] __device__ (ProjT *buckets_sum_buffer) {
                extern __shared__ char shmem_bytes[];
                ProjT* shmem = reinterpret_cast<ProjT*>(shmem_bytes);

                namespace cg = cooperative_groups;
                cg::thread_block g = cg::this_thread_block();

                uint32_t tid = g.thread_rank();

                if (tid < spines_count * 2) shmem[tid] = buckets_sum_buffer[tid];
                else shmem[tid].inf();
                cg::sync(g);

                block_scan(shmem, tid, spines_count_2power * 2);

                if (tid < spines_count * 2) buckets_sum_buffer[tid] = shmem[tid];
            }, buckets_sum_buffer);

            kernel<<<spines_count, blk_siz, 0, stream>>>([=] __device__ (ProjT *buckets_sum, ProjT *buckets_sum_buffer) {
                namespace cg = cooperative_groups;
                cg::thread_block g = cg::this_thread_block();

                uint32_t gid = cg::this_grid().thread_rank();
                uint32_t bid = g.group_index().x;
                uint32_t tid = g.thread_rank();

                __shared__ ProjT shmem[blk_siz];

                if (gid < upper_bound * 2) shmem[tid] = buckets_sum[gid];
                else shmem[tid].inf();
                cg::sync(g);

                if (bid > 0 && gid < upper_bound * 2) shmem[tid].add(buckets_sum_buffer[(bid - 1) * 2 + (tid & 1)]);
                
                block_reduce(shmem, tid, blk_siz);

                if (tid < 2) buckets_sum[gid] = shmem[tid];
            }, buckets_sum, buckets_sum_buffer);

            kernel<<<1, spines_count_2power * 2, spines_count_2power * 2 * sizeof(ProjT), stream>>>([=] __device__ (ProjT *buckets_sum, ProjT *windows_sum) {                
                extern __shared__ char shmem_bytes[];
                ProjT* shmem = reinterpret_cast<ProjT*>(shmem_bytes);

                namespace cg = cooperative_groups;
                cg::thread_block g = cg::this_thread_block();

                uint32_t tid = g.thread_rank();

                if (tid < spines_count * 2) shmem[tid] = buckets_sum_buffer[tid * blk_siz + (tid & 1)];
                else shmem[tid].inf();
                cg::sync(g);

                block_reduce(shmem, tid, spines_count_2power * 2);
                cg::sync(g);

                cg::invoke_one(g, [&] () {
                    uint4 *base = reinterpret_cast<uint4*>(&shmem[0]);
                    const int stride = sizeof(ProjT) / (3 * sizeof(uint4));
                    uint4 buffer[stride];
                    for (int i = 0; i < stride; i++) buffer[i] = base[stride + i];
                    for (int i = 0; i < stride; i++) base[stride + i] = base[3 * stride + i];
                    for (int i = 0; i < stride; i++) base[3 * stride + i] = base[4 * stride + i];
                    for (int i = 0; i < stride; i++) base[4 * stride + i] = base[2 * stride + i];
                    for (int i = 0; i < stride; i++) base[2 * stride + i] = buffer[i];
                    
                    windows_sum[window_id * 2] = shmem[0];
                    windows_sum[window_id * 2 + 1] = shmem[1];
                });
            }, buckets_sum, windows_sum);

            cudaStreamSynchronize(stream);
            CUDA_CHECK(cudaGetLastError());
        }
    }

public:
    void msm(BucketContext<FieldT, HostFT>& bkt_ctx, HostPT *result = 0, size_t instance_id = 0)
    {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        auto policy = thrust::cuda::par.on(stream);

        for (int i = 0, j = windows_count / 2; j < windows_count - 1; i++, j++) {
            (bucket_segemented_reduction<AffT, ProjT, XYZZT, threads_unit>)<<<buckets_count - 1, 32, 0, stream>>>
                (bkt_ctx.buckets_off + i * buckets_count, bkt_ctx.indices_as_vals + i * scale, bases[instance_id], buckets_sum_WWR);
            
            (bucket_segemented_reduction_increment<AffT, ProjT, XYZZT, threads_unit>)<<<buckets_count - 1, 32, 0, stream>>>
                (bkt_ctx.buckets_off + j * buckets_count, bkt_ctx.indices_as_vals + j * scale, lifted_bases[instance_id], buckets_sum_WWR);

            warp_reduce(stream);
            bucket_reduce(i, stream);
        }

        // cudaStreamSynchronize(stream);
        // CUDA_CHECK(cudaGetLastError());

        if (windows_count % 2 == 0) {
            int i = windows_count / 2 - 1, j = windows_count - 1;

            (bucket_segemented_reduction<AffT, ProjT, XYZZT, threads_unit>)<<<buckets_count - 1, 32, 0, stream>>>
                (bkt_ctx.buckets_off + i * buckets_count, bkt_ctx.indices_as_vals + i * scale, bases[instance_id], buckets_sum_WWR);
            warp_reduce(stream);

            (bucket_segemented_reduction_last_window<AffT, ProjT, XYZZT, threads_unit>)<<<last_window_warps_per_bucket * (last_window_buckets_count - 1), 32, 0, stream>>>
                (bkt_ctx.buckets_off + j * buckets_count, bkt_ctx.indices_as_vals + j * scale, lifted_bases[instance_id], buckets_sum_WWR, last_window_warps_per_bucket);
            warp_reduce(stream, true, true);

            bucket_reduce(i, stream, true, true);
        } else {
            int i = windows_count - 1;
            (bucket_segemented_reduction_last_window<AffT, ProjT, XYZZT, threads_unit>)<<<last_window_warps_per_bucket * (last_window_buckets_count - 1), 32, 0, stream>>>
                (bkt_ctx.buckets_off + i * buckets_count, bkt_ctx.indices_as_vals + i * scale, lifted_bases[instance_id], buckets_sum_WWR, last_window_warps_per_bucket);
            
            warp_reduce(stream, true);
            bucket_reduce(i, stream, true);
        }

        cudaStreamSynchronize(stream);
        CUDA_CHECK(cudaGetLastError());
        cudaStreamDestroy(stream);

        if (result) {
            std::vector<HostPT> buffer(windows_count);
            cudaMemcpy(buffer.data(), windows_sum, threads_unit * windows_count * sizeof(ProjT), cudaMemcpyDeviceToHost);
            HostPT res = HostPT::zero();
            for (int k = windows_count / 2 - (windows_count % 2 == 0); k >= 0; k--) {
                for (int l = 0; l < window_bits; l++) res = res.dbl();
                res = res + buffer[k];
            }
            *result = res;
        }
    }
};
