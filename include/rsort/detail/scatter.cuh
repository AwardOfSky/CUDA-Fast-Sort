// Staging (and scattering) header
// nvcc -O3 -std=c++17 -arch=sm_86 radix_gpu.cu bench_parser.cpp -o radix_gpu.exe &&
// cuobjdump --dump-sass radix_gpu.exe > sass1.txt &&
// diff sass.txt sass1.txt

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <type_traits>

#include "ranker.cuh"
#include "tuning.cuh"


namespace rsort::detail {

template<
    typename Key_T,
    typename U,
    typename Lookback_T,
    typename Len_T,
    typename Value_T = no_value_t,
    bool Short_Mode = false,
    bool Is_Long_Double = false
>
struct Scatter {


    using RT = radix_tuning<Key_T, Value_T, Short_Mode>;
    using RTraits = radix_traits<Key_T, Is_Long_Double>;
    using Ranker = Radix_Ranker<Key_T, Value_T, Short_Mode>;
    using Stage_Ind_T = typename RT::Stage_Ind_T;

    static constexpr uint32_t STAGE_PAIRS = RT::sorting_pairs * sizeof(Lookback_T) / sizeof(U);
    static constexpr uint32_t LOGICAL_BLOCK_SIZE = RT::REORDER_LOGICAL_BLOCK_SIZE;
    static constexpr uint32_t RADIX_BIN_SIZE = RT::RADIX_BIN_SIZE;
    static constexpr uint32_t ITEMS_PER_THREAD = RT::REORDER_ITEMS_PER_THREAD;
    static constexpr uint32_t REORDER_THREADS = RT::REORDER_THREADS;
    static constexpr bool SORTING_PAIRS = RT::sorting_pairs;


    template<typename T, typename Ind_T, size_t N, enum staging_modes mode>
    struct conditional_arr; 

    template<typename T, typename Ind_T, size_t N>
    struct conditional_arr<T, Ind_T, N, staging_modes::disabled> {
        // empty
    };

    template<typename T, typename Ind_T, size_t N>
    struct conditional_arr<T, Ind_T, N, staging_modes::indices> {
        Ind_T v[N];
    };

    template<typename T, typename Ind_T, size_t N>
    struct conditional_arr<T, Ind_T, N, staging_modes::direct> {
        T v[N];
    };


    // Shared memory struct for main kernel
#define UNIONIZE_SMEM   1
    struct alignas(16) SMem {


        static constexpr enum staging_modes keys_staging = RT::staging.keys;
        static constexpr enum staging_modes vals_staging = RT::staging.vals;


#if UNIONIZE_SMEM
        union
#endif
        {
            typename Ranker::Temp_Storage rank_temp;
            struct {
                conditional_arr<U, Stage_Ind_T, LOGICAL_BLOCK_SIZE, keys_staging> staged_keys;
                conditional_arr<Value_T, Stage_Ind_T, LOGICAL_BLOCK_SIZE, vals_staging> staged_vals;
            };
        };
        uint16_t bin_count[RADIX_BIN_SIZE];     
        Lookback_T bin_offset[RADIX_BIN_SIZE];
          
        
        // Staging kernels
        template<bool Full_Block>
        __device__ RSORT_FORCEINLINE void stage_init(
            const U (&keys)[ITEMS_PER_THREAD],
            const int (&ranks)[ITEMS_PER_THREAD],
            int k,
            Len_T block_base,
            uint32_t actual_tile_items,
            size_t src_lane_base = 0,
            Value_T* __restrict__ in_vals = nullptr
        ) {
            [[maybe_unused]] Stage_Ind_T src_local = (Stage_Ind_T)(src_lane_base + (size_t)k * WARP_SIZE);

            // init keys
            if constexpr (keys_staging == staging_modes::direct) {
                staged_keys.v[ranks[k]] = keys[k];
            } else if constexpr (keys_staging == staging_modes::indices) {
                staged_keys.v[ranks[k]] = src_local;
            }

            // init vals
            if constexpr (vals_staging == staging_modes::direct) {
                if constexpr (Full_Block) {
                    staged_vals.v[ranks[k]] = in_vals[block_base + (Len_T)src_local];
                } else {
                    if (src_local < actual_tile_items) {
                        staged_vals.v[ranks[k]] = in_vals[block_base + (Len_T)src_local];
                    }
                }
            } else if constexpr (vals_staging == staging_modes::indices) {
                staged_vals.v[ranks[k]] = src_local;
            }
        }


        template<bool Descending>
        __device__ RSORT_FORCEINLINE void scatter(
            const Key_T* __restrict__ in_keys,
            Key_T* __restrict__ out_keys,
            Len_T block_base,
            Len_T idx,
            uint32_t bit_location,
            Value_T* __restrict__ out_vals = nullptr,
            Value_T* __restrict__ in_vals = nullptr
        ) {
            // keys
            U key;
            if constexpr (keys_staging == staging_modes::direct) {
                key = staged_keys.v[idx];
            } else if constexpr (keys_staging == staging_modes::indices) {
                Stage_Ind_T src_local = staged_keys.v[idx];
                key = RTraits::twiddle_in<Descending>(in_keys[block_base + (Len_T)src_local]);
            }

            U digit = RTraits::extract_key(key, bit_location);
            Len_T ind = idx + (Len_T)bin_offset[digit];
            out_keys[ind] = RTraits::twiddle_out<Descending>(key);

            // vals
            if constexpr (vals_staging == staging_modes::direct) {
                out_vals[ind] = staged_vals.v[idx];
            } else if constexpr (vals_staging == staging_modes::indices) {
                out_vals[ind] = in_vals[block_base + staged_vals.v[idx]];
            } else if constexpr (SORTING_PAIRS) {
                out_vals[ind] = in_vals[block_base + (Len_T)staged_keys.v[idx]];
            }
        }

    };


    // scatter kernel with full block support and using simple storaging
    template<
        bool Full_Block,
        bool Descending,
        typename SMem_T
    >
    static __device__ RSORT_FORCEINLINE void scatter_staged(
        SMem_T* __restrict__ smem,
        const Key_T* __restrict__ in_keys,
        const U (&keys)[ITEMS_PER_THREAD],
        const int (&ranks)[ITEMS_PER_THREAD],
        Key_T* __restrict__ out_keys,
        uint32_t actual_tile_items,
        uint32_t bit_location,
        Len_T block_base,
        size_t src_lane_base = 0,
        Value_T* __restrict__ in_vals = nullptr,
        Value_T* __restrict__ out_vals = nullptr
    ) {

        // stage in local rank order.
        // init staging
        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
            smem->stage_init<Full_Block>(keys, ranks, k, block_base, actual_tile_items, src_lane_base, in_vals);

        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
            Len_T idx = threadIdx.x + (Len_T)k * REORDER_THREADS;

            if constexpr (Full_Block) {
                smem->scatter<Descending>(in_keys, out_keys, block_base, idx, bit_location, out_vals, in_vals);

            } else {
                if (idx < (Len_T)actual_tile_items) {
                    smem->scatter<Descending>(in_keys, out_keys, block_base, idx, bit_location, out_vals, in_vals);

                }
            }
            __syncwarp(0xFFFFFFFFu);
        }
    }
    
};

} // namespace rsort::detail


/*
the conditional array struct still declares 1 element.
to void this, there is the idea of specializing the template
around the whole staging struct like this:

    struct keys_tag {};
    struct vals_tag {};

    template<typename Tag, typename T, size_t N, staging_modes Mode>
    struct staging_field;

    template<typename Tag, typename T, size_t N>
    struct staging_field<Tag, T, N, staging_modes::disabled> {
        // empty
    };

    template<typename Tag, typename T, size_t N>
    struct staging_field<Tag, T, N, staging_modes::indices> {
        uint32_t v[N];
    };

    template<typename Tag, typename T, size_t N>
    struct staging_field<Tag, T, N, staging_modes::direct> {
        T v[N];
    };

Then the staging struct becomes:

    struct stage_storage
        : staging_field<keys_tag, U, LOGICAL_BLOCK_SIZE, keys_staging>
        , staging_field<vals_tag, Value_T, LOGICAL_BLOCK_SIZE, vals_staging_>
    {
        using keys_base = staging_field<keys_tag, U, LOGICAL_BLOCK_SIZE, keys_staging>;

        using vals_base = staging_field<vals_tag, Value_T, LOGICAL_BLOCK_SIZE, vals_staging_>;

        __device__ RSORT_FORCEINLINE auto* keys() {
            return keys_base::v;
        }

        __device__ RSORT_FORCEINLINE auto* vals() {
            return vals_base::v;
        }
    } staged;


and we access the elements with smem->staged.keys()[i] and smem->staged.vals()[i]
However, this was shown to reduce performance slghtly even 
when caching the pointer before "U* __restrict__ staged_keys = smem->staged.keys();",
so it was reverted (Not a zero cost abstraction)
Other than that, Manually specializing templates for all staging modes
does not seem like good practise.
*/
