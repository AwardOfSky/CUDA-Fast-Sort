// Utils header
#pragma once

#include <cuda_runtime.h>
#include <string_view>
#include <limits>
#include <cstdint>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <time.h>
#endif

#if defined(_MSC_VER)
    #define RSORT_FORCEINLINE __forceinline
#else
    #define RSORT_FORCEINLINE __inline__ __attribute__((__always_inline__))
#endif

#define WARP_SIZE       32
static_assert(
    WARP_SIZE == 32,
    "This implementation assumes 32-lane NVIDIA hardware warps"
);

#include "tuning.cuh"



namespace rsort::detail {

// ---- memory helpers ----
template<size_t ALIGN, typename T>
static inline T align_up_power(T x) {
    static_assert(
        (ALIGN & (ALIGN - 1)) == 0,
        "[align_up_power]: ALIGN must be a power of two."
    );
    static_assert(
        std::is_unsigned_v<T>,
        "[align_up_power]: requires unsigned integer types."
    );
    return (x + (ALIGN - 1)) & ~(ALIGN - 1);
}


template<size_t ALIGN>
static inline size_t reserve_aligned(size_t* off, size_t bytes) {
    static_assert(
        (ALIGN & (ALIGN - 1)) == 0,
        "[reserve_aligned]: ALIGN must be a power of two"
    );
    *off = align_up_power<ALIGN>(*off);
    size_t at = *off;
    *off += bytes;
    return at;
}


// ---- cuda helpers ----
#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do {                                  \
    cudaError_t err__ = (x);                                \
    if (err__ != cudaSuccess) {                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n",           \
            __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1);                                            \
    }                                                       \
} while(0)
#endif


// Linux needs these wrappers for 64-bit counters
__device__ RSORT_FORCEINLINE uint32_t atomic_add_wrap(uint32_t* ptr, uint32_t val) {
    return atomicAdd(ptr, val);
}


__device__ RSORT_FORCEINLINE uint64_t atomic_add_wrap(uint64_t* ptr, uint64_t val) {
    return (uint64_t)atomicAdd(
        reinterpret_cast<unsigned long long int*>(ptr),
        (unsigned long long int)val
    );
}


template<uint32_t Limit>
__device__ RSORT_FORCEINLINE int active_thread_limit(uint32_t x) {
    uint32_t base = (threadIdx.x < Limit) ? x : 0u;
    return base;
}


__device__ RSORT_FORCEINLINE uint32_t lane_id_i32() {
    int x;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(x));
    return x;
}


__device__ RSORT_FORCEINLINE uint32_t lane_mask_le_i32() {
    int x;
    asm volatile("mov.u32 %0, %%lanemask_le;" : "=r"(x));
    return x;
}


__device__ __host__ RSORT_FORCEINLINE uint32_t div_round_up_u32(uint32_t v, uint32_t d) {
    return v / d + ((v % d) != 0); // overflow safe
}


template<typename T>
__device__ __host__ RSORT_FORCEINLINE T div_round_up(T v, T d) {
    return v / d + ((v % d) != 0); // overflow safe
}


// ---- type helpers ----

// get name of type - namespace compliant
// ChatGPT and stackoverflow black magic :D
template<typename T>
constexpr std::string_view type_name() {
    std::string_view name = __PRETTY_FUNCTION__;

#if defined(__clang__)
    auto start = name.find("T = ");
    auto end = name.rfind(']');
    start += 4;
    return name.substr(start, end - start);

#elif defined(__GNUC__)
    auto start = name.find("with T = ");
    auto end = name.find(';', start);
    if (end == std::string_view::npos) {
        end = name.rfind(']');
    }
    start += 9;
    return name.substr(start, end - start);

#elif defined(_MSC_VER)
    auto start = name.find("type_name<");
    auto end = name.rfind(">(void)");
    start += 10;
    return name.substr(start, end - start);
#endif
}


// ---- radix helpers ----

template<typename T>
inline uint32_t ceil_log2_size(T n) {
    static_assert(
        std::is_integral_v<T> && !std::is_same_v<T, bool>,
        "[ceil_log2_size]: N must be an integer."
    );

    if (n <= 1) {
        return 0;
    }

    int r = 0;
    T p = 1;

    while (p < n) {
        p <<= 1;
        ++r;
    }

    return r;
}


// quick and dirty array size estimation
uint32_t __host__ max_array_bits(const uint32_t vram_gb, const uint32_t sizeof_kv) {
    if (vram_gb == 0) {
        return 0;
    }

    uint32_t gib_bits = 0;
    uint32_t x = vram_gb;
    while (x > 1) {
        x >>= 1;
        ++gib_bits;
    }

    int32_t elem_adjust = 3 - ceil_log2_size(sizeof_kv);
    uint32_t base_bits_u64 = 26; // 2 arrays of 64-bit elements fit in 1 GiB

    int32_t bits = base_bits_u64 + elem_adjust + gib_bits;
    
    return uint32_t(bits >= 0 ? bits : 0);
}


// ---- generic sort helpers ----

enum class Order : bool {
    ascending = false,
    descending = true
};


enum class Array_Modes : uint32_t {
    start,
    random,
    byte_skip,
    asc,
    end,
};


constexpr const char* arr_modes_to_string(Array_Modes mode) {
    switch (mode) {
        case Array_Modes::byte_skip:      return "byte_skip";
        case Array_Modes::asc:              return "ascending";
        case Array_Modes::random: default:  return "random";
    }
    return "unknown";
}


#define RNG_AVALANCHE_MIX    1
__host__ __device__ RSORT_FORCEINLINE uint32_t mix32(uint32_t x) {
    
#if RNG_AVALANCHE_MIX
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
#else
    x = 1664525u * x + 1013904223u;
    x ^= x >> 16;
#endif
    
    return x;
}


template<
    typename Key_T,
    typename Len_T,
    bool Is_Long_Double = false,
    bool Pass = false
>
__global__ void init_keys(
    Key_T* a,
    Len_T n,
    uint32_t seed,
    Array_Modes arr_mode = Array_Modes::random
) {


    using RTraits = radix_traits<Key_T, Is_Long_Double>;
    using U = typename RTraits::unsigned_of;

    static_assert(
        (sizeof(Key_T) <= 8) || (sizeof(Key_T) == 16 && RTraits::has_native_u128), 
        "[init_keys]: Key type must be at most 64-bit, or 128-bit if native support exists."
    );
    

    Len_T i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        U x;

        if (arr_mode != Array_Modes::asc) {
            // random init based on type
            if constexpr ((sizeof(U) <= 4) && !Is_Long_Double) {
                x = mix32(i ^ seed);
            } else if constexpr ((sizeof(U) <= 8) && !Is_Long_Double) {
                x = (uint64_t(mix32(i + 0x00000000u) ^ seed) << 32) |
                    (uint64_t(mix32(i + 0x9e3779b9u) ^ seed) << 0);
            } else if constexpr (RTraits::has_native_u128) {
                using U128 = typename RTraits::native_u128;
                x = (U128(mix32(i + 0x00000000u) ^ seed) << 96) |
                    (U128(mix32(i + 0x9e3779b9u) ^ seed) << 64) |
                    (U128(mix32(i + 0x3c6ef372u) ^ seed) << 32) |
                    (U128(mix32(i + 0xdaa66d2bu) ^ seed) << 0);
            } else {
                return;
            }

            // skip every other bytes
            if (arr_mode == Array_Modes::byte_skip) {
                if constexpr ((sizeof(U) <= 8) && !Is_Long_Double) {
                    x = (x & U(0xFF00FF00FF00FF00ull)) | U(0x0055005500550055ull);
                } else if constexpr (RTraits::has_native_u128) {
                    using U128 = typename RTraits::native_u128;

                    U128 keep =
                        (U128(0xFF00FF00FF00FF00ull) << 64) |
                        U128(0xFF00FF00FF00FF00ull);
                    U128 fill =
                        (U128(0x0055005500550055ull) << 64) |
                        U128(0x0055005500550055ull);
                    
                    x = U((U128(x) & keep) | fill);
                } else {
                    return;
                }
            }
        } else {
            x = U(i);
        }

        // copy bits to dest (Pass == don't intialize)
        if constexpr (!Pass) {
            a[i] = RTraits::bits_to_type(x);
        }
    }
}


/*
    Post-sort order verification

    Order is checked on manifested values rather than radix identities,
    except for long doubles (see below).

    Note that IEEE NaNs are unordered: comparisons involving NaNs
    always evaluate to false. As a result, NaNs are not considered
    when validating sortedness. Exact bit-level preservation of NaNs
    (including payloads) is instead verified by the key-value integrity
    checks, which compare key identities rather than manifested values.
    So here, NaN's are ignore because we're checking for false statements
    and they evaluate to false with > and < operators.

    With long doubles, we compare on identify, as the inexistent support
    for long doubles on CUDA makes it impossible to correctly compare
    128-bit floating values.
    Twiddle in is applied on that case. The cast is redundant as, in these
    verification kernels, a long double is already reinterpreted as
    unsigned, to not be demoted to 64-bit.
*/
template<bool Descending, typename T, bool Is_Long_Double = false>
__global__ void check_sorted(const T* a, size_t n, int* ok) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) {

        T l_key, r_key;
        if constexpr (Is_Long_Double) {
            l_key = (T)radix_traits<T, Is_Long_Double>::twiddle_in<false>(a[i]);
            r_key = (T)radix_traits<T, Is_Long_Double>::twiddle_in<false>(a[i + 1]);
        } else {
            l_key = a[i];
            r_key = a[i + 1];
        }        

        if constexpr (Descending) {
            if (l_key < r_key) {
                atomicExch(ok, 0);
            }
        } else {
            if (l_key > r_key)  {
                atomicExch(ok, 0);
            }
        }
    }
}


/*  
    Post-sort KV-pair verification
    
    Performs verifications on unsigned representation (value identity).
    This is achieve by twiddling in without the Descending transform.
    The reason is that for floating types there can be differences,
    even though they are sorted and same value (NaN != NaN).
    We want to know: is the key the same as the original?
*/
template<
    bool Descending,
    typename Key_T,
    typename Len_T,
    typename Value_T,
    bool Is_Long_Double = false
>
__global__ void check_pairings(
    const Key_T* keys_or, 
    const Value_T* vals_or, 
    const Key_T* keys_sor, 
    const Value_T* vals_sor, 
    Len_T n,
    int* ok
) {


    using RTraits = radix_traits<Key_T, Is_Long_Double>;
    using U = typename RTraits::unsigned_of;


    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    // This should not happen on ascending value data, something is wrong
    if (vals_sor[i] >= n) {
        atomicExch(ok, 0);
        return;
    }

    // no need to twiddle for vals because they're indices here in the verification anyway
    U keys_sor_i = RTraits::twiddle_in<false>(keys_sor[i]);
    U keys_or_vals_sor_i = RTraits::twiddle_in<false>(keys_or[vals_sor[i]]);

    // check pairings both ways
    if (keys_sor_i != keys_or_vals_sor_i) {
        atomicExch(ok, 0);
        return;
    }
    if (vals_sor[i] != vals_or[vals_sor[i]]) {
        atomicExch(ok, 0);
        return;
    }

    // check stability
    if (i < n - 1) {
        U keys_sor_i1 = RTraits::twiddle_in<false>(keys_sor[i + 1]);

        if ((keys_sor_i == keys_sor_i1) && (vals_sor[i] > vals_sor[i + 1])) {
            atomicExch(ok, 0);
        }
    }
}


int set_seed_radix(int index) {
    return (index + 81) * 17;
}


// ---- other helpers ----
template<typename T>
double stdev(const T* arr, double avg, size_t n, uint32_t ddof = 0) {
    static_assert(
        std::is_arithmetic_v<T>,
        "[stdev]: Array type must be arithmetic."
    );

    if ((arr == nullptr) || (n <= ddof)) {
        return 0.0;
    }

    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double temp = (double)arr[i] - avg;
        acc += temp * temp;
    }

    return std::sqrt(acc / double(n - ddof));
}


void sleep_ms(long milliseconds) {
#if defined(_WIN32)
    Sleep(milliseconds);
#else
    struct timespec ts;
    ts.tv_sec = milliseconds / 1000;
    ts.tv_nsec = (milliseconds % 1000) * 1000000L;

    nanosleep(&ts, NULL);
#endif
}

} // namespace rsort::detail
