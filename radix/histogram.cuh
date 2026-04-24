// Histogram header
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "utils.cuh"


namespace rsort {

struct histogram_tuning {
    static constexpr uint32_t CARD_SMS                  = 92;
    static constexpr uint32_t GHIST_THREADS             = 256;
    static constexpr uint32_t GHIST_ITEM_PER_BLOCK      = GHIST_THREADS * 8;
    static constexpr uint32_t GHIST_ITEMS_PER_THREAD    = GHIST_ITEM_PER_BLOCK / GHIST_THREADS;
    static constexpr uint32_t HIST_BLOCKS               = CARD_SMS * 4;

    static constexpr uint32_t SCAN256_WARPS     = radix_consts::RADIX_BIN_SIZE / WARP_SIZE;
    static constexpr bool EXCLUSIVE_SCAN_256    = 
        ((GHIST_THREADS == 256) && (radix_consts::RADIX_BIN_SIZE == 256));

    static_assert(
        (GHIST_ITEM_PER_BLOCK % GHIST_THREADS) == 0,
        "GHIST_ITEM_PER_BLOCK must be divisible by GHIST_THREADS"
    );
};


// simple block-wide exclusive scan for nElement <= blockDim.x
template <typename T>
__device__ __forceinline__ T scan_exclusive_block(T prefix, T* s_mem, int n_element) {
    bool active = (int)threadIdx.x < n_element;
    T value = active ? s_mem[threadIdx.x] : 0;
    T x = value;
    
    for (uint32_t offset = 1; offset < (uint32_t)n_element; offset <<= 1) {
        if (active && (int)offset <= (int)threadIdx.x) {
            x += s_mem[threadIdx.x - offset];
        }
        __syncthreads();
        
        if (active) {
            s_mem[threadIdx.x] = x;
        }
        __syncthreads();
    }

    T sum = s_mem[n_element - 1];
    __syncthreads();
    
    if (active) {
        s_mem[threadIdx.x] = x + prefix - value;
    }
    __syncthreads();
    
    return sum;
}


// Block exclusive scan specialized for 256 threads (8 warps).
__device__ __forceinline__ uint32_t block_exclusive_scan_256(
    uint32_t x,
    uint32_t* warp_sums) {
    
    constexpr uint32_t SCAN256_WARPS = histogram_tuning::SCAN256_WARPS;

    int tid = (int)threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = (int)lane_id_u32();
    
    bool active = tid < (int)radix_consts::RADIX_BIN_SIZE;

    uint32_t base = active ? x : 0u;
    uint32_t v = base;
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        uint32_t y = __shfl_up_sync(0xFFFFFFFFu, v, offset);
        if (lane >= offset) v += y;
    }

    if (lane == (WARP_SIZE - 1) && warp < SCAN256_WARPS) {
        warp_sums[warp] = v;
    }
    __syncthreads();

    if (warp == 0) {
        uint32_t t = (lane < SCAN256_WARPS) ? warp_sums[lane] : 0u;
        #pragma unroll
        for (int offset = 1; offset < SCAN256_WARPS; offset <<= 1) {
            uint32_t y = __shfl_up_sync(0xFFFFFFFFu, t, offset);
            if (lane >= offset) t += y;
        }
        if (lane < SCAN256_WARPS) {
            warp_sums[lane] = t - warp_sums[lane];
        }
        
    }
    __syncthreads();

    return active ? (warp_sums[warp] + (v - base)) : 0u;
}


template <bool Descending, typename Lookback_T, typename Key_T, typename Len_T, bool DYNAMIC_WORK_STEAL = true>
__global__ void GHistogram_8bits(
    const Key_T* __restrict__ inputs,
    Len_T n,
    Lookback_T* __restrict__ gp_sum_buffer, // [RADIX_PASSES][RADIX_BIN_SIZE]
    uint32_t start_bits,
    uint32_t* __restrict__ counter) {

    using RT = radix_tuning<Key_T>;
    constexpr uint32_t RADIX_BIN_SIZE = RT::RADIX_BIN_SIZE;
    constexpr uint32_t RADIX_BITS = RT::RADIX_BITS;
    constexpr uint32_t RADIX_PASSES = sizeof(Key_T);

    using H = histogram_tuning;
    constexpr uint32_t GHIST_ITEMS_PER_THREAD = H::GHIST_ITEMS_PER_THREAD;
    constexpr uint32_t GHIST_ITEM_PER_BLOCK = H::GHIST_ITEM_PER_BLOCK;
    constexpr uint32_t SCAN256_WARPS = H::SCAN256_WARPS;
    constexpr uint32_t EXCLUSIVE_SCAN_256 = H::EXCLUSIVE_SCAN_256;


    __shared__ uint32_t local_counters[RADIX_PASSES][RADIX_BIN_SIZE];
    //if constexpr (EXCLUSIVE_SCAN_256) {
    __shared__ uint32_t scan_warp_sums[SCAN256_WARPS];
    //}


    // init
    #pragma unroll
    for (int p = 0; p < RADIX_PASSES; ++p) {
        #pragma unroll
        for (int j = (int)threadIdx.x; j < (int)RADIX_BIN_SIZE; j += (int)blockDim.x) {
            local_counters[p][j] = 0;
        }
    }
    __syncthreads();

    Len_T num_blocks = div_round_up<Len_T>(n, GHIST_ITEM_PER_BLOCK);

    using U = get_unsigned_of<Key_T>;
    if constexpr (DYNAMIC_WORK_STEAL) { // Dynamic work stealing
        for (;;) {
            __shared__ uint32_t i_block; // TODOs: should block and counter be key_t
            if (threadIdx.x == 0) {
                i_block = atomicInc(counter, 0xFFFFFFFFu);
            }
            __syncthreads();
            if ((Len_T)i_block >= num_blocks) {
                break;
            }

            #pragma unroll
            for (int j = 0; j < GHIST_ITEMS_PER_THREAD; ++j) {
                Len_T idx = i_block * GHIST_ITEM_PER_BLOCK + threadIdx.x * GHIST_ITEMS_PER_THREAD + (Len_T)j;
                if (idx < n) {
                    U item = twiddle_in<Descending, Key_T>(inputs[idx]);
                    #pragma unroll
                    for (int p = 0; p < RADIX_PASSES; ++p) {
                        U bit = start_bits + (U)p * RADIX_BITS;
                        U b = extract_byte<U>(item, bit);
                        atomicAdd(&local_counters[p][b], 1u);
                    }
                }
            }
            __syncthreads();
        }
    } else {
        // Fixed CTA striding. With 32 bit n, the maximum items seen by one CTA is
        // ceil(num_blocks / gridDim.x) * GHIST_ITEM_PER_BLOCK <= n, so uint32 locals stay safe.
        for (uint32_t i_block_fixed = (uint32_t)blockIdx.x; i_block_fixed < num_blocks; i_block_fixed += (uint32_t)gridDim.x) {
            #pragma unroll
            for (int j = 0; j < GHIST_ITEMS_PER_THREAD; ++j) {
                Len_T idx = i_block_fixed * GHIST_ITEM_PER_BLOCK + threadIdx.x * GHIST_ITEMS_PER_THREAD + (Len_T)j;
                if (idx < n) {
                    U item = twiddle_in<Descending, Key_T>(inputs[idx]);
                    #pragma unroll
                    for (int p = 0; p < RADIX_PASSES; ++p) {
                        U bit = start_bits + (U)p * RADIX_BITS;
                        U b = extract_byte<U>(item, bit);
                        atomicAdd(&local_counters[p][b], 1u);
                    }
                }
            }
            __syncthreads();
        }
    }

    // exclusive scan per pass (in shared)
    int block_dim = (int)blockDim.x;
    #pragma unroll
    for (int p = 0; p < RADIX_PASSES; ++p) {
        if constexpr (EXCLUSIVE_SCAN_256) {
            uint32_t x = local_counters[p][threadIdx.x];
            uint32_t excl = block_exclusive_scan_256(x, scan_warp_sums);
            local_counters[p][threadIdx.x] = excl;
            __syncthreads();
        } else {
            uint32_t prefix = 0u;
            for (int i = 0; i < (int)RADIX_BIN_SIZE; i += block_dim) {
                int n_el = (int)RADIX_BIN_SIZE - i;
                if (n_el > block_dim) {
                    n_el = block_dim;
                }
                prefix += scan_exclusive_block<uint32_t>(prefix, &local_counters[p][i], n_el);
            }
        }
    }

    // accumulate into global gp_sum_buffer
    #pragma unroll
    for (int p = 0; p < RADIX_PASSES; ++p) {
        for (int j = (int)threadIdx.x; j < (int)RADIX_BIN_SIZE; j += block_dim) {
            atomic_add_wrap(&gp_sum_buffer[p * RADIX_BIN_SIZE + j], (Lookback_T)local_counters[p][j]);
        }
    }
}

} // namespace rsort
