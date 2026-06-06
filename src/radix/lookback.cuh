// Lookback header
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "utils.cuh"

// user knobs
#define LOOKBACK_OVERRIDE               0
#define LOOKBACK_EPOCH_TAG              1
#define REUSE_LOOKBACK_PER_PASS         1

#define USE_INLINE_LOOKBACK             0
#define LOOKBACK_DYMANIC_WAITING        1

// more SPINS for few CTAs, 1 for more CTAs
#define LOOKBACK_USE_NANOSLEEP_BACKOFF  1
#define LOOKBACK_NANOSLEEP_INITIAL      4
#define LOOKBACK_NANOSLEEP_MAX          256
#define LOOKBACK_SPINS                  1


namespace rsort {

// Lookback templating
template <typename T, bool Epoch> struct Lookback_Config;

// 32-bit lookback specialization
template<bool Epoch>
struct Lookback_Config<uint32_t, Epoch> {
    static constexpr uint32_t PARTIAL_MASK  = 1u << 30;
    static constexpr uint32_t GLOBAL_MASK   = 1u << 31;
    static constexpr uint32_t EPOCH_BITS    = Epoch ? 2u : 0u;
    static constexpr uint32_t EPOCH_SHIFT   = Epoch ? 28u : 30u;
    
    static constexpr uint32_t EPOCH_VALUE_MASK  = (1u << EPOCH_BITS) - 1u;
    static constexpr uint32_t EPOCH_MASK        = EPOCH_VALUE_MASK << EPOCH_SHIFT;
    static constexpr uint32_t VALUE_MASK        = (1u << EPOCH_SHIFT) - 1u;
    static constexpr uint32_t EPOCH_TAG         = ((~(PARTIAL_MASK | GLOBAL_MASK)) >> 24) & 0xFFu;
    static constexpr uint32_t EPOCH_TAG_WORD    = EPOCH_TAG * 0x01010101u;
};

// 64-bit lookback specialization
template<bool Epoch>
struct Lookback_Config<uint64_t, Epoch> {
    static constexpr uint64_t PARTIAL_MASK  = 1ull << 62;
    static constexpr uint64_t GLOBAL_MASK   = 1ull << 63;
    static constexpr uint32_t EPOCH_BITS    = Epoch ? 6ull : 0ull;
    static constexpr uint32_t EPOCH_SHIFT   = Epoch ? 56ull : 62ull;
    
    static constexpr uint64_t EPOCH_VALUE_MASK  = (1ull << EPOCH_BITS) - 1ull;
    static constexpr uint64_t EPOCH_MASK        = EPOCH_VALUE_MASK << EPOCH_SHIFT;
    static constexpr uint64_t VALUE_MASK        = (1ull << EPOCH_SHIFT) - 1ull;
    static constexpr uint64_t EPOCH_TAG         = ((~(PARTIAL_MASK | GLOBAL_MASK)) >> 56) & 0xFFull;
    static constexpr uint64_t EPOCH_TAG_WORD    = EPOCH_TAG * 0x0101010101010101ull;
};


// Policy Base
template <typename T_, bool Epoch, bool Reuse>
struct Lookback_Policy_Base {
    using T = T_;
    static constexpr bool epoch = 
        LOOKBACK_OVERRIDE ? LOOKBACK_EPOCH_TAG : Epoch;
    static constexpr bool reuse =
        (LOOKBACK_OVERRIDE ? REUSE_LOOKBACK_PER_PASS : Reuse) || epoch;
    using conf = Lookback_Config<T, epoch>;

    static constexpr inline void print_stats() {
        printf("Epoch\t=\t%u\n"
            "Reuse\t=\t%u\n"
            "Custom\t=\t%u\n",

            epoch,
            reuse,
            LOOKBACK_OVERRIDE
        );
    }
};


enum class Lookback_Modes : uint32_t {
    u32_epoch,
    u32_plain,
    u64_epoch
};

template <Lookback_Modes Mode> struct Lookback_Policy;


// ================== Lookback Policy definitions ================== 

// Policy Specializations <Type, Epoch, Reuse>

template <>
struct Lookback_Policy<Lookback_Modes::u32_epoch>
    : Lookback_Policy_Base<uint32_t, true, true> {};

// <uint32_t, false, false> for slightly faster(?), more memory expensive
template <>
struct Lookback_Policy<Lookback_Modes::u32_plain>
    : Lookback_Policy_Base<uint32_t, false, true> {};

template <>
struct Lookback_Policy<Lookback_Modes::u64_epoch>
    : Lookback_Policy_Base<uint64_t, true, true> {};
// =================================================================


using Faster_LB_Policy  = Lookback_Policy<Lookback_Modes::u32_epoch>;
using Fast_LB_Policy    = Lookback_Policy<Lookback_Modes::u32_plain>;
using General_LB_Policy = Lookback_Policy<Lookback_Modes::u64_epoch>;


// Policy selector
static constexpr inline void print_lookback_policy(Lookback_Modes mode) { 
    switch(mode) {
        case Lookback_Modes::u32_epoch:
            Faster_LB_Policy::print_stats();
            break;
        case Lookback_Modes::u32_plain:
            Fast_LB_Policy::print_stats();
            break;
        case Lookback_Modes::u64_epoch:
            General_LB_Policy::print_stats();
            break;
        default:
            break;
    }
}


// Note: LOOKBACK_OVERRIDE forces behaviour within a mode, does not change the policy. Be careful!!!
static inline Lookback_Modes get_lookback_mode(size_t n) {
    using LB = Lookback_Config<uint32_t, true>;
    const uint32_t turn_epoch_off = LB::VALUE_MASK;
    const uint32_t do_64t_looback = (1 << (LB::EPOCH_BITS + LB::EPOCH_SHIFT)) - 1;

    if (n <= turn_epoch_off) {
        return Lookback_Modes::u32_epoch;
    }
    if (n <= do_64t_looback) {
        return Lookback_Modes::u32_plain;
    }
    return Lookback_Modes::u64_epoch;
}


template<typename Policy>
struct Lookback {

    using T = typename Policy::T;
    using LB = typename Policy::conf;
    static constexpr auto VALUE_MASK = LB::VALUE_MASK;
    static constexpr auto GLOBAL_MASK = LB::GLOBAL_MASK;
    static constexpr bool Epoch = Policy::epoch;
    static constexpr uint32_t RADIX_BIN_SIZE = radix_consts::RADIX_BIN_SIZE;
    static constexpr uint32_t RADIX_BITS = radix_consts::RADIX_BITS;

    static_assert(
        !(LOOKBACK_EPOCH_TAG && !REUSE_LOOKBACK_PER_PASS && LOOKBACK_OVERRIDE),
        "Epoch-tagged lookback requires reused per-pass lookback slices."
    );


    // packing functions, always pack epoch bits along with publish state
    static __device__ RSORT_FORCEINLINE T pack_global(T val, T epoch_bits) {
        if constexpr(Epoch) {
            return (val & LB::VALUE_MASK) | LB::GLOBAL_MASK | epoch_bits;
        } else {
            return (val & LB::VALUE_MASK) | LB::GLOBAL_MASK;
        }
    }

    static __device__ RSORT_FORCEINLINE T pack_partial(T val, T epoch_bits) {
        if constexpr(Epoch) {
            return (val & LB::VALUE_MASK) | LB::PARTIAL_MASK | epoch_bits;
        } else {
            return (val & LB::VALUE_MASK) | LB::PARTIAL_MASK;
        }
    }


    // Rule #1 of high performance programming:
    // the more underscores your code has the faster it goes
    static __device__ __host__ RSORT_FORCEINLINE T pack_epoch(uint32_t epoch) {
        if constexpr(Epoch) {
            using LB = Lookback_Config<T, Epoch>;
            return ((T)epoch & LB::EPOCH_VALUE_MASK) << LB::EPOCH_SHIFT;
        } else {
            return 0;        
        }
    }


    // Note: No need to check for 0 if using Epoch bits because
    // lookback will be EPOCH_TAG filled for the first pass (and on epoch wrap)
    static __device__ RSORT_FORCEINLINE bool invalid_lookback_state(T raw, T lb) {
        if constexpr(Epoch) {
            //return ((raw & LB::EPOCH_MASK) != lb) || (raw == 0u);
            return (raw & LB::EPOCH_MASK) != lb;
        } else {
            return raw == 0u;
        }
    }


    // standard exponential backoff strategy
    static __device__ RSORT_FORCEINLINE void nano_wait(T* raw, const volatile T* ptr, T epoch_bits) {
#if LOOKBACK_USE_NANOSLEEP_BACKOFF && (__CUDA_ARCH__ >= 700)
        if (invalid_lookback_state(*raw, epoch_bits)) {

            uint32_t backoff = LOOKBACK_NANOSLEEP_INITIAL;

            while (invalid_lookback_state(*raw, epoch_bits)) {
                __nanosleep(backoff);
                *raw = *ptr;
                if (backoff < LOOKBACK_NANOSLEEP_MAX) {
                    
                    //backoff += LOOKBACK_NANOSLEEP_INITIAL; // linear
                    backoff <<= 2;

                }
            }
        }
#endif
    }


    // helpers
    static __device__ RSORT_FORCEINLINE void wait_valid_lookback_state(T* raw, const volatile T* ptr, T epoch_bits) {
        do {
            *raw = *ptr;
            nano_wait(raw, ptr, epoch_bits);
        } while (invalid_lookback_state(*raw, epoch_bits));
    }


    static __device__ RSORT_FORCEINLINE bool valid_global_lookback_state(T raw, T epoch_bits) {
        if constexpr(Epoch) {
            return ((raw & LB::GLOBAL_MASK) != 0) && ((raw & LB::EPOCH_MASK) == epoch_bits) && (raw != 0);
        } else {
            return ((raw & LB::GLOBAL_MASK) != 0) && (raw != 0);
        }
    }


    // Main lookback function, decoupled, partial publishing
    // less branching (more register pressure?)
    static __device__ RSORT_FORCEINLINE typename Policy::T lookback_prefix(
        const volatile typename Policy::T* __restrict__ lookback_bin,
        uint32_t block_index,
        typename Policy::T lb_epoch_bits) {

        T p = 0;
        if (block_index == 0) {
            return 0;
        }
        
        int i_block = (int)block_index - 1;
        const volatile T* ptr = lookback_bin + ((block_index - 1) << RADIX_BITS);
        T raw = *ptr;


        // change spin/verifications according to how hot the path is
        // 0 SPINS we lose the oppurtiny to exit
        // more than 1 we just spin our wheels, might be good for less items per thread
#if LOOKBACK_DYMANIC_WAITING && defined(_WIN32)
        // for first few skip global check, else only do 1
        if (block_index >= 4) {
#else
        // always LOOKBACK_SPINS
        #pragma unroll
        for (int i = 0; i < LOOKBACK_SPINS; ++i) {
#endif
            if (valid_global_lookback_state(raw, lb_epoch_bits)) {
                return (raw & VALUE_MASK);
            }
            raw = *ptr;
        }

        wait_valid_lookback_state(&raw, ptr, lb_epoch_bits);

        p += (raw & VALUE_MASK);
        if ((raw & GLOBAL_MASK) == 0) {
            for (--i_block; i_block >= 0; --i_block) {
                ptr -= RADIX_BIN_SIZE;

                wait_valid_lookback_state(&raw, ptr, lb_epoch_bits);

                p += (raw & VALUE_MASK);
                if (raw & GLOBAL_MASK) {
                    break;
                }
            }
        }

        return p;
    }


    // old version of the lookback:
    // seems to be slightly worse in most cases because of the extra branch,
    // but it's mostly codegen. Still, worth testing in new configurations 
    static __device__ RSORT_FORCEINLINE typename Policy::T lookback_prefix_old(
        const volatile typename Policy::T* __restrict__ lookback_bin,
        uint32_t block_index,
        uint32_t b,
        typename Policy::T lb_epoch_bits) {
        
        T p = 0;
        if (block_index == 0) {
            return 0;
        }

        int i_block = (int)block_index - 1;
        int lb = (int)(RADIX_BIN_SIZE * (uint32_t)i_block + b);
        T raw = 0;
        bool got_global = false;


#if LOOKBACK_DYMANIC_WAITING && defined(_WIN32)
        if (block_index >= 4) {
#else
        #pragma unroll
        for (int spin = 0; spin < LOOKBACK_SPINS; ++spin) {
#endif
            raw = *((const volatile T*)&lookback_bin[lb]);
            if (valid_global_lookback_state(raw, lb_epoch_bits)) {
                p = (raw & VALUE_MASK);
                got_global = true;
#if !LOOKBACK_DYMANIC_WAITING || !defined(_WIN32)
                break;
#endif
            }
        }


        if (!got_global) {
            wait_valid_lookback_state(&raw, (const volatile T*)&lookback_bin[lb], lb_epoch_bits);
            p += (raw & VALUE_MASK);
            if ((raw & GLOBAL_MASK) == 0) {
                for (--i_block; i_block >= 0; --i_block) {
                    lb = (int)(RADIX_BIN_SIZE * (uint32_t)i_block + b);
                    wait_valid_lookback_state(&raw, (const volatile T*)&lookback_bin[lb], lb_epoch_bits);
                    p += (raw & VALUE_MASK);
                    if (raw & GLOBAL_MASK) {
                        break;
                    }
                }
            }
        }

        return p;
    }
};

} // namespace rsort
