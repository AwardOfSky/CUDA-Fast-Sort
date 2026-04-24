// Ranker header
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "histogram.cuh"
#include "lookback.cuh"
#include "utils.cuh"


namespace rsort {

// Same as in CUB header but with no ternary operator (retval at 0xFFFFFFFFu)
__device__ __forceinline__ uint32_t cub_match_any_8_u32(uint32_t label) {
    uint32_t retval = 0xFFFFFFFFu;
    //uint32_t retval = 0u;
    #pragma unroll
    for (int bit = 0; bit < radix_consts::RADIX_BITS; ++bit) {
        uint32_t mask;
        uint32_t current_bit = 1u << bit;
        asm("{\n"
            "    .reg .pred p;\n"
            "    and.b32 %0, %1, %2;\n"
            "    setp.ne.u32 p, %0, 0;\n"
            "    vote.sync.ballot.b32 %0, p, 0xFFFFFFFF;\n"
            "    @!p not.b32 %0, %0;\n"
            "}\n"
            : "=r"(mask)
            : "r"(label), "r"(current_bit)
        );
        //retval = (bit == 0) ? mask : (retval & mask); // init retval = 0u
        retval &= mask;
    }
    return retval;
}


// This is pretty much the same kernel as CUB, just translated and simplified a bit
template<typename Key_T>
struct Radix_Ranker {

    using RT = radix_tuning<Key_T>;
    using U = get_unsigned_of<Key_T>;
    static constexpr uint32_t REORDER_WARPS = RT::REORDER_WARPS;
    static constexpr uint32_t ITEMS_PER_THREAD = RT::REORDER_ITEMS_PER_THREAD;
    static constexpr uint32_t RADIX_BIN_SIZE = RT::RADIX_BIN_SIZE;
    static constexpr uint32_t RADIX_BITS = RT::RADIX_BITS;
    static constexpr uint32_t SCAN256_WARPS = histogram_tuning::SCAN256_WARPS;


    struct Temp_Storage {
        int warp_offsets[REORDER_WARPS][RADIX_BIN_SIZE];
        uint32_t scan_warp_sums[SCAN256_WARPS];
    };

    template <bool Full_Tile, bool Descending, typename Lookback_Policy>
    static __device__ __forceinline__ void match_early_counts(
        Temp_Storage& temp_storage,
        U (&keys)[ITEMS_PER_THREAD],
        int (&ranks)[ITEMS_PER_THREAD],
        uint32_t bit_location,
        int& exclusive_digit_prefix,
        uint16_t* bin_count,
        volatile typename Lookback_Policy::T* lookBack_partial,
        uint32_t block_index,
        uint32_t invalid_items,
        typename Lookback_Policy::T lookback_epoch_bits) {

        const int warp = threadIdx.x / WARP_SIZE;
        const int lane = lane_id_u32();
        const int bin_owner = threadIdx.x < RADIX_BIN_SIZE;
        uint8_t digits[ITEMS_PER_THREAD];

        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
            digits[u] = (uint8_t)extract_byte<U>(keys[u], bit_location);
        }

        // Warp-private histogram
        int* warp_offsets = &temp_storage.warp_offsets[warp][0];
        #pragma unroll
        for (int bin = lane; bin < RADIX_BIN_SIZE; bin += WARP_SIZE) {
            warp_offsets[bin] = 0;
        }
        __syncwarp(0xFFFFFFFFu); // TODOs is this needed, why does it work with 0?

        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
            atomicAdd(&warp_offsets[(uint32_t)digits[u]], 1);
        }

        // Block-wide upsweep + callback
        __syncthreads();
        int bins = 0;
        if (bin_owner) {
            const int bin = (int)threadIdx.x;
            int* warp_bin_ptr = &temp_storage.warp_offsets[0][bin];
            int* warp_bin_it = warp_bin_ptr;
            #pragma unroll
            for (int j = 0; j < REORDER_WARPS; ++j) {
                int count = *warp_bin_it;
                *warp_bin_it = bins;
                bins += count;
                warp_bin_it += RADIX_BIN_SIZE;
            }

            uint32_t b = (uint32_t)bin;
            uint32_t published = (uint32_t)bins;
            if constexpr (!Full_Tile) {
                if (b == (uint32_t)(RADIX_BIN_SIZE - 1)) {
                    published -= invalid_items;
                }
            }
            bin_count[b] = (uint16_t)published;
            int p_index = (int)((block_index << RADIX_BITS) + b);
            
            lookBack_partial[p_index] = Lookback<Lookback_Policy>::pack_partial(
                published, lookback_epoch_bits
            );
        }

        // Block scan over per-bin counts
        //(threadIdx.x < Limit) ? (bin_owner ? (uint32_t)bins : 0u) : 0u
        exclusive_digit_prefix = (int)block_exclusive_scan_256(
            bin_owner ? (uint32_t)bins : 0u,
            temp_storage.scan_warp_sums
        );

        // Downsweep: convert per-warp counts into per-warp bases
        if (bin_owner) {
            const int bin = (int)threadIdx.x;
            int* warp_bin_ptr = &temp_storage.warp_offsets[0][bin];
            int* warp_bin_it = warp_bin_ptr;
            #pragma unroll
            for (int j = 0; j < REORDER_WARPS; ++j) {
                *warp_bin_it += exclusive_digit_prefix;
                warp_bin_it += RADIX_BIN_SIZE;
            }
        }
        __syncthreads();

        // Per-item rank within the warp/bin
        warp_offsets = &temp_storage.warp_offsets[warp][0];
        const int lane_mask_le = (int)lane_mask_le_u32();

        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
            uint32_t key_bin = (uint32_t)digits[u];

            int bin_mask = (int)cub_match_any_8_u32(key_bin);

            int leader = (WARP_SIZE - 1) - __clz((unsigned)bin_mask);
            int warp_offset = 0;
            int popc = __popc(bin_mask & lane_mask_le);
            if (lane == leader) {
                warp_offset = atomicAdd(&warp_offsets[key_bin], popc);
            }
            warp_offset = __shfl_sync(0xFFFFFFFFu, warp_offset, leader);
            ranks[u] = warp_offset + popc - 1;
        }
    }
};

} // namespace rsort

