// nvcc -O3 -std=c++17 -arch=sm_86 radix_gpu.cu -o radix_gpu.exe
// TODOs refactor unsigned_of?
// Utils header
#pragma once

#include <cuda_runtime.h>
#include <type_traits>
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


// ---- memory helpers ----
static inline size_t align_up_size(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}


static inline size_t reserve_aligned(size_t* off, size_t bytes, size_t align) {
    *off = align_up_size(*off, align);
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


__device__ __forceinline__ int lane_id_u32() {
    int x;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(x));
    return x;
}


__device__ __forceinline__ int lane_mask_le_u32() {
    int x;
    asm volatile("mov.u32 %0, %%lanemask_le;" : "=r"(x));
    return x;
}


__device__ __host__ __forceinline__ uint32_t div_round_up_u32(uint32_t v, uint32_t d) {
    return v / d + (v % d != 0); // overflow safe
}


template<typename T>
__device__ __host__ __forceinline__ T div_round_up(T v, T d) {
    return v / d + (v % d != 0); // overflow safe
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

// ---- radix helpers ----
// for 32 bits
// 18, 38 (8)
// 12, 21 (16645)
// 10, 
// 16, 15
// 14, 17

// for 64 bits:
// 7, 12 / 6, 14
// 9, 10 / 18, 8 (worse)

// for 16 bits:
// 25, 10


struct radix_consts {
    static constexpr uint32_t RADIX_BITS = 8;
    static constexpr uint32_t RADIX_BIN_SIZE = 1u << RADIX_BITS;
    static constexpr uint32_t RADIX_MASK = RADIX_BIN_SIZE - 1u;
};


template<typename T>
struct radix_tuning : radix_consts {

    static constexpr uint32_t RADIX_PASSES = sizeof(T);

    static constexpr uint32_t CTA_MULTIPLIER = (sizeof(T) <= 4) ? 21 : 6;
    static constexpr uint32_t REORDER_WARPS = (sizeof(T) <= 4) ? 12 : 14;
    
    static constexpr uint32_t REORDER_THREADS = REORDER_WARPS * WARP_SIZE;  // 256
    static constexpr uint32_t SORT_BLOCK_SIZE = CTA_MULTIPLIER * REORDER_THREADS; // items per CTA
    static constexpr uint32_t REORDER_ITEMS_PER_THREAD = SORT_BLOCK_SIZE / REORDER_THREADS;
    static constexpr uint32_t REORDER_ITEMS_PER_WARP = REORDER_ITEMS_PER_THREAD * WARP_SIZE;
    static constexpr uint32_t REORDER_LOGICAL_BLOCK_SIZE = SORT_BLOCK_SIZE;

    static_assert(
        (SORT_BLOCK_SIZE % REORDER_THREADS) == 0,
        "SORT_BLOCK_SIZE must be divisible by REORDER_THREADS"
    );
};


template<typename U>
__device__ __forceinline__ uint32_t extract_byte(U x, U bit) {
    //if constexpr (Descending) x = ~x;
    return (x >> bit) & radix_consts::RADIX_MASK;
}


template<typename T>
struct radix_type_traits {


    // 128 bit support (for integer types in GCC and Clang)
#if defined(__SIZEOF_INT128__)
    using native_u128 = unsigned __int128;
    static constexpr bool has_native_u128 = true;
#else
    using native_u128 = uint64_t;
    static constexpr bool has_native_u128 = false;
#endif


    static_assert(std::is_arithmetic_v<T>, "Radix type must be a numeric.");
    static_assert(sizeof(T) <= 16, "Radix type too large.");
    static_assert(
        (sizeof(T) != 16) || has_native_u128,
        "128-bit keys require native same-width unsigned type support."
    );
    static_assert(
        (sizeof(T) != 16) || std::is_integral_v<T>,
        "Only 128-bit integer keys are supported!"
    );
    // Are you on Linux and want to sort long doubles on your GPU? Ask Stalin Sort!


    using unsigned_of =
        std::conditional_t<sizeof(T) == 1, uint8_t,
        std::conditional_t<sizeof(T) == 2, uint16_t,
        std::conditional_t<sizeof(T) == 4, uint32_t,
        std::conditional_t<sizeof(T) == 8, uint64_t,
        native_u128>>>>;
        
    static constexpr bool is_signed_integral = std::is_signed_v<T> && std::is_integral_v<T>;
    static constexpr unsigned_of sign_mask_of = unsigned_of(1) << (sizeof(T) * 8 - 1);


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


    //quick and dirty estimation
    static uint32_t __host__ max_array_bits(uint32_t vram_gb) {
        if (vram_gb == 0) {
            return 0;
        }

        uint32_t gib_bits = 0;
        uint32_t x = vram_gb;
        while (x > 1) {
            x >>= 1;
            ++gib_bits;
        }

        constexpr uint32_t elem_adjust =
            sizeof(T) == 16 ? -1 :
            sizeof(T) == 8 ? 0 :
            sizeof(T) == 4 ? 1 :
            sizeof(T) == 2 ? 2 :
            sizeof(T) == 1 ? 3 :
            0;

        constexpr uint32_t base_bits_u64 = 26; // 2 arrays of 64-bit elements fit in 1 GiB
        return base_bits_u64 + elem_adjust + gib_bits;
    }
};


template<typename T>
using get_unsigned_of = typename radix_type_traits<T>::unsigned_of;

template<typename T>
static constexpr bool is_signed_integral_v =
    radix_type_traits<T>::is_signed_integral;

template<typename T>
static constexpr get_unsigned_of<T> radix_sign_mask_v =
    radix_type_traits<T>::sign_mask_of;


template <bool Descending, typename T>
__device__ __forceinline__ get_unsigned_of<T> tail_filler_bits() {
    //constexpr U max_u = (std::numeric_limits<U>::max)();
    //return max_u;

    using U = get_unsigned_of<T>;
    return U(~U(0)); // always last in ascending ordered-bit space
}


template <bool Descending, typename T>
__host__ __device__ __forceinline__ get_unsigned_of<T> twiddle_in(T x) {
    using U = get_unsigned_of<T>;

    U bits = radix_type_traits<T>::type_to_bits(x);
    if constexpr (std::is_floating_point_v<T>) {
        constexpr U sign_mask = radix_sign_mask_v<T>;
        bits = (bits & sign_mask) ? ~bits : (bits ^ sign_mask);
    } else if constexpr (is_signed_integral_v<T>) {
        bits ^= radix_sign_mask_v<T>;
    }

    if constexpr (Descending) {
        bits = ~bits;
    }

    return bits;
}


template <bool Descending, typename T>
__host__ __device__ __forceinline__ T twiddle_out(get_unsigned_of<T> bits) {
    using U = get_unsigned_of<T>;

    if constexpr (Descending) {
        bits = ~bits;
    }

    if constexpr (std::is_floating_point_v<T>) {
        constexpr U sign_mask = radix_sign_mask_v<T>;
        bits = (bits & sign_mask) ? (bits ^ sign_mask) : ~bits;
        return radix_type_traits<T>::bits_to_type(bits);
    } else if constexpr (is_signed_integral_v<T>) {
        return T(bits ^ radix_sign_mask_v<T>);
    } else {
        return T(bits);
    }
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



#define RNG_FULL_LCG    0
__host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) {
    
#if RNG_FULL_LCG
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


template<typename Key_T, typename Len_T>
__global__ void init_keys(Key_T* a, Len_T n, uint32_t seed, Array_Modes arr_mode) {
    Len_T i = blockIdx.x * blockDim.x + threadIdx.x;
    using U = get_unsigned_of<Key_T>;

    if (i < n) {
        U x;

        if constexpr (sizeof(U) <= 4) {
            uint32_t r = mix32(i ^ seed);
            x = U(r);
        } else {
            uint64_t hi = (uint64_t)(mix32(i) ^ seed);
            uint64_t lo = (uint64_t)(mix32(i + 0x9e3779b9u) ^ seed);
            x = U((hi << 32) | lo);
        }

        if (arr_mode == Array_Modes::blank_bytes) {
            x = (x & U(0xFF00FF00FF00FF00ull)) | U(0x0055005500550055ull);
        }

        a[i] = radix_type_traits<Key_T>::bits_to_type(x);
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
template<typename T>
double stdev(const T* arr, double avg, size_t n) {
    if (n == 0) {
        return 0.0;
    }
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double temp = (double)arr[i] - avg;
        acc += temp * temp;
    }
    return std::sqrt(acc / (double)n);
}


#define MIN_WARMUP_MS 50.0f

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
