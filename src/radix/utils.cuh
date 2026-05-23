// Utils header
#pragma once

#include <cuda_runtime.h>
#include <string_view>
#include <limits>
#include <cstdint>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <time.h>
#endif

#define WARP_SIZE       32
static_assert(WARP_SIZE == 32, "This code assumes 32-lane NVIDIA hardware warps");

#include "tuning.cuh"


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
__device__ __forceinline__ uint32_t atomic_add_wrap(uint32_t* ptr, uint32_t val) {
    return atomicAdd(ptr, val);
}


__device__ __forceinline__ uint64_t atomic_add_wrap(uint64_t* ptr, uint64_t val) {
    return (uint64_t)atomicAdd(
        reinterpret_cast<unsigned long long int*>(ptr),
        (unsigned long long int)val
    );
}


template<uint32_t Limit>
__device__ __forceinline__ int active_thread_limit(uint32_t x) {
    uint32_t base = (threadIdx.x < Limit) ? x : 0u;
    return base;
}


__device__ __forceinline__ uint32_t lane_id_i32() {
    int x;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(x));
    return x;
}


__device__ __forceinline__ uint32_t lane_mask_le_i32() {
    int x;
    asm volatile("mov.u32 %0, %%lanemask_le;" : "=r"(x));
    return x;
}


__device__ __host__ __forceinline__ uint32_t div_round_up_u32(uint32_t v, uint32_t d) {
    return v / d + ((v % d) != 0); // overflow safe
}


template<typename T>
__device__ __host__ __forceinline__ T div_round_up(T v, T d) {
    return v / d + ((v % d) != 0); // overflow safe
}


// ---- type helpers ----

// get name of type - namespace compliant
// ChatGPT and stackoverflow black magic :D
template <typename T>
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


namespace rsort {

template <typename T>
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
enum class Array_Modes : uint32_t {
    start,
    random,
    blank_bytes,
    end,
};


constexpr const char* arr_modes_to_string(Array_Modes mode) {
    switch (mode) {
        case Array_Modes::random:       return "random";
        case Array_Modes::blank_bytes:  return "byte_skip";
    }
    return "unknown";
}


#define RNG_AVALANCHE_MIX    1
__host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) {
    
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


template<typename Key_T, typename Len_T, bool Pass = false>
__global__ void init_keys(
    Key_T* a,
    Len_T n,
    uint32_t seed,
    Array_Modes arr_mode = Array_Modes::random
) {


    using U = get_unsigned_of<Key_T>;
    using RTraits = radix_traits<Key_T>;
    

    Len_T i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        U x;

        // random init based on type
        if constexpr (sizeof(U) <= 4) {
            x = mix32(i ^ seed);
        } else if constexpr (sizeof(U) <= 8) {
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
        if (arr_mode == Array_Modes::blank_bytes) {
            if constexpr (sizeof(U) <= 8) {
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

        // copy bits to dest
        if constexpr (!Pass) {
            a[i] = RTraits::bits_to_type(x);
        }
    }
}


template<bool Descending, typename T>
__global__ void check_sorted(const T* a, size_t n, int* ok) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) {
        if constexpr (Descending) {
            if (a[i] < a[i+1]) {
                atomicExch(ok, 0);
            }
        } else {
            if (a[i] > a[i+1]) {
                atomicExch(ok, 0);
            }
        }
    }
}


int set_seed_radix(int index) {
    return (index + 81) * 17;
}


// ---- other helpers ----
template<typename T, bool ddof = false>
double stdev(const T* arr, double avg, size_t n) {
    if (n == 0) {
        return 0.0;
    }

    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double temp = (double)arr[i] - avg;
        acc += temp * temp;
    }

    if constexpr (ddof) {
        return (n > 1) ? std::sqrt(acc / (double)(n - 1)) : 0.0;
    } else {
        return std::sqrt(acc / (double)n);
    }
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

} // namespace rsort
