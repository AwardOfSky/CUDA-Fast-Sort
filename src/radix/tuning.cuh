// Radix Tuning header
#pragma once

#include <cuda_runtime.h>
#include <type_traits>
#include <cstdint>


// user knobs
#define FORCE_32BIT_STAGING     1
#define STAGE_KEYS_OVERRIDE     staging_modes::automatic
#define STAGE_VALS_OVERRIDE     staging_modes::automatic


#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif


namespace rsort {

// Staging modes, policy and logic could be in staging.cuh, but makes more sense to
// define it here at the base because it also influences kernel geometry and tuning.
// This header defines modes and tuning, including staging. staging.cuh defines how 
// those staging modes behave, along with scattering.
enum class staging_modes : uint32_t {
    automatic   = 0,
    disabled    = 1,
    indices     = 2,
    direct      = 3
};


struct no_value_t {};

struct radix_consts {
    static constexpr uint32_t RADIX_BITS = 8;
    static constexpr uint32_t RADIX_BIN_SIZE = 1u << RADIX_BITS;
    static constexpr uint32_t RADIX_MASK = RADIX_BIN_SIZE - 1u;
};

struct staging_pair {
    staging_modes keys;
    staging_modes vals;
};


// Tuning (extends constants)
template<
    typename Key_T,
    typename Value_T = no_value_t
>
struct radix_tuning : radix_consts {

    static constexpr bool sorting_pairs = !std::is_same_v<Value_T, no_value_t>;

    // The staging policy is a very simple heuristic for now. Basically:
    //
    // Keys will always be staged directly to avoid double twiddling,
    // except when the size is bogus, like more than 128-bit
    // (which is not even supported anyways).
    // So, effectively, keys are always staged directly.
    // 
    // Values will be staged depending on remaining size budget
    // we will set value staging to direct unless we're  already overshooting
    // our size budget per element (128 bits for now), in which case we revert
    // to index staging.
    //
    // Both staging modes can be overwritten
    //
    // Notes:
    // - Key Staging can't be disabled
    // - Value staging can be disabled,
    //   but if so, we have to revert key staging to indices
    static constexpr staging_pair staging_policy() {

        // default policy
        constexpr staging_modes default_keys =
            (sizeof(Key_T) <= 16)
                ? staging_modes::direct
                : staging_modes::indices;

        constexpr uint32_t key_size =
            (default_keys == staging_modes::direct)
                ? sizeof(Key_T)
                : sizeof(uint32_t);

        constexpr uint32_t size_budget = 16;
        constexpr staging_modes default_vals =
            ((key_size + sizeof(Value_T)) <= size_budget)
                ? staging_modes::direct
                : staging_modes::indices;

        // override
        constexpr staging_modes keys_staging =
            (STAGE_KEYS_OVERRIDE != staging_modes::automatic)
                ? STAGE_KEYS_OVERRIDE
                : default_keys;
        
        constexpr staging_modes vals_staging =
            (STAGE_VALS_OVERRIDE != staging_modes::automatic)
                ? STAGE_VALS_OVERRIDE
                : default_vals;

                
        // sanity checks
        static_assert(
                keys_staging != staging_modes::disabled,
                "Key staging can't be disabled!"
        );
        constexpr staging_modes final_vals = 
            sorting_pairs ? vals_staging : staging_modes::disabled;

        constexpr staging_modes final_keys =
            (sorting_pairs && (final_vals == staging_modes::disabled))
                ? staging_modes::indices
                : keys_staging;

        return {final_keys, final_vals};
    }

    static constexpr staging_pair staging = staging_policy();

    // These CTA_MULTIPLIER and REORDER_WARPS are for sm_86
    // In the end, these two constants should be a lookup table
    // according to the sm_xx of the card, but I only have this one
    // 32-bit - 21/12
    // 64-bit - 6/14 5/17 22/4 7/22(this one?) 26/6 27/6
    // 128-bit - optimize!

    // key-value pairs
    
    // 16/16: 13/22 13/24 
    // 32/16: 11/21 11/22 (this one) 10/24
    // 32/32: 12/11(this one)   15/9    15/12 10/14     8/16 7/19 (this one) 6/23
    // 64/32: 4/20
    // 32/64: 4/20
    // 64/64: 3/19
    static constexpr uint32_t CTA_MULTIPLIER_NO_PAIR =  (sizeof(Key_T) <= 4) ? 21 : // 21
                                                        (sizeof(Key_T) <= 8) ? 7  : // 7
                                                        2; // TODOs: optimize this
    static constexpr uint32_t REORDER_WARPS_NO_PAIR =   (sizeof(Key_T) <= 4) ? 12 : // 12 
                                                        (sizeof(Key_T) <= 8) ? 22 : // 22
                                                        20;
    

    static constexpr uint32_t size_el_key = 
        (staging.keys == staging_modes::direct) ? sizeof(Key_T) : sizeof(uint32_t);
    static constexpr uint32_t size_el_val = 
        sorting_pairs ? 
            ((staging.vals == staging_modes::direct) ?
                sizeof(Value_T) :
                sizeof(uint32_t)) :
            0;
    static constexpr uint32_t size_el = size_el_key + size_el_val;

    // 32/32 - 12/14
    static constexpr uint32_t get_cta_geometry() {
        if constexpr (!sorting_pairs) {
            return CTA_MULTIPLIER_NO_PAIR;
        }

        return  (size_el <= 4) ? 13 :
                (size_el <= 6) ? 11 :
                (size_el <= 8) ? 7 :
                (size_el <= 12) ? 4 :
                (size_el <= 16) ? 3 :
                2;
    }


    static constexpr uint32_t get_warp_geometry() {
        if constexpr (!sorting_pairs) {
            return REORDER_WARPS_NO_PAIR;
        }

        return  (size_el <= 4) ? 23 :
                (size_el <= 6) ? 22 :
                (size_el <= 8) ? 19 :
                (size_el <= 12) ? 20 :
                (size_el <= 16) ? 19 :
                20;
    }


    static constexpr uint32_t CTA_MULTIPLIER    = get_cta_geometry();
    static constexpr uint32_t REORDER_WARPS     = get_warp_geometry();
    
    static constexpr uint32_t RADIX_PASSES = sizeof(Key_T);
    static constexpr uint32_t REORDER_THREADS = REORDER_WARPS * WARP_SIZE;
    static constexpr uint32_t SORT_BLOCK_SIZE = CTA_MULTIPLIER * REORDER_THREADS; // items / CTA
    static constexpr uint32_t REORDER_ITEMS_PER_THREAD = SORT_BLOCK_SIZE / REORDER_THREADS;
    static constexpr uint32_t REORDER_ITEMS_PER_WARP = REORDER_ITEMS_PER_THREAD * WARP_SIZE;
    static constexpr uint32_t REORDER_LOGICAL_BLOCK_SIZE = SORT_BLOCK_SIZE;

    static_assert(
        (SORT_BLOCK_SIZE % REORDER_THREADS) == 0,
        "SORT_BLOCK_SIZE must be divisible by REORDER_THREADS"
    );

    using Stage_Ind_T = std::conditional_t<
        (REORDER_LOGICAL_BLOCK_SIZE <= 0xFFFFu) && !FORCE_32BIT_STAGING, uint16_t, uint32_t
    >;
    //using Stage_Ind_T = uint32_t;
};


template<typename T>
struct radix_traits {

    // 128 bit support (for integer types in GCC and Clang)
#if defined(__SIZEOF_INT128__)
    using native_u128 = unsigned __int128;
    using native_i128 = __int128;
    static constexpr bool has_native_u128 = true;
#else
    using native_u128 = uint64_t;
    using native_i128 = int64_t;
    static constexpr bool has_native_u128 = false;
#endif
    static constexpr bool type_is_valid_128 = 
        std::is_same_v<T, native_u128> || std::is_same_v<T, native_i128>;


    // type compliance asserts 
    static_assert(
        std::is_arithmetic_v<T> || type_is_valid_128,
        "Radix type must be a numeric."
    );
    static_assert(sizeof(T) <= 16, "Radix type too large.");
    static_assert(
        (sizeof(T) != 16) || has_native_u128,
        "128-bit keys require native same-width unsigned type support."
    );
    static_assert(
        (sizeof(T) != 16) || type_is_valid_128,
        "Only 128-bit integer keys are supported!"
    );
    // Are you on Linux and want to sort long doubles on your GPU? Ask Stalin Sort


    // set quivalent unsigned type
    using unsigned_of =
        std::conditional_t<sizeof(T) == 1, uint8_t,
        std::conditional_t<sizeof(T) == 2, uint16_t,
        std::conditional_t<sizeof(T) == 4, uint32_t,
        std::conditional_t<sizeof(T) == 8, uint64_t,
        native_u128>>>>;
        
    static constexpr bool is_signed_integral = std::is_signed_v<T> && std::is_integral_v<T>;
    static constexpr unsigned_of sign_mask_of = unsigned_of(1) << (sizeof(T) * 8 - 1);

    
    // unsigned and type T bit convertion (for twiddling)  
    static __device__ __forceinline__ T bits_to_type(unsigned_of x) {
        if constexpr(std::is_same_v<T, float>) {
            return __uint_as_float(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return __longlong_as_double(int64_t(x));
        } else {
            return T(x);
        }
    }


    static __device__ __forceinline__ unsigned_of type_to_bits(T x) {
        if constexpr(std::is_same_v<T, float>) {
            return __float_as_uint(x);
        } else if constexpr(std::is_same_v<T, double>) {
            return unsigned_of(__double_as_longlong(x));
        } else {
            return unsigned_of(x);
        }
    }


    // twiddling kernels
    template <bool Descending>
    static __host__ __device__ __forceinline__ unsigned_of twiddle_in(T x) {
        using U = unsigned_of;

        U bits = type_to_bits(x);
        if constexpr (std::is_floating_point_v<T>) {
            bits = (bits & sign_mask_of) ? ~bits : (bits ^ sign_mask_of);
        } else if constexpr (is_signed_integral) {
            bits ^= sign_mask_of;
        }

        if constexpr (Descending) {
            bits = ~bits;
        }

        return bits;
    }


    template<bool Descending>
    static __host__ __device__ __forceinline__ T twiddle_out(unsigned_of bits) {

        if constexpr (Descending) {
            bits = ~bits;
        }

        if constexpr (std::is_floating_point_v<T>) {
            bits = (bits & sign_mask_of) ? (bits ^ sign_mask_of) : ~bits;
            return bits_to_type(bits);
        } else if constexpr (is_signed_integral) {
            return T(bits ^ sign_mask_of);
        } else {
            return T(bits);
        }
    }

    
    // Filler for partial blocks
    static __host__ __device__ __forceinline__ constexpr unsigned_of tail_filler_bits() {
        return ~unsigned_of{0};
    }


    // Key extractor
    //template<bool Descending>
    static __device__ __forceinline__ uint32_t extract_key(unsigned_of x, unsigned_of bit) {
        //if constexpr (Descending) x = ~x;
        return (x >> bit) & radix_consts::RADIX_MASK;
    }
};


// alternative interface
template<typename T>
using get_unsigned_of = typename radix_traits<T>::unsigned_of;

} // namespace rsort
