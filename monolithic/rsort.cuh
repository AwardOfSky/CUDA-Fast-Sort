// This is rsort's monolithic header.
// might not be up to date with delevelopment!
#pragma once

/* 
    TODOs:
    - change C-style pointers to C++ style
    - add suport for long doubles
    - key pair types
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <type_traits>
#include <string_view>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <time.h>
#endif


// Template heavy, only enable if you want to run validation
#define VALIDATION_TEST     0

// ===================== Validation Types =====================
#define U32_TYPE    uint32_t,
#define UINT_TYPES  U32_TYPE uint8_t, uint16_t, uint64_t,
#define INT_TYPES   UINT_TYPES int16_t, int8_t, int32_t, int64_t,
#define FP_TYPES    float, double,
#define ALL_TYPES   INT_TYPES FP_TYPES

// Change types to test HERE!!!
#define TYPE_SET_TEST ALL_TYPES
// ============================================================

// user knobs
#define USE_PSUM_SHARED         1
#define USE_CUDA_GRAPH_SORT     1
#define EXIT_EARLY_OPT          0

#define LOOKBACK_OVERRIDE               0
#define LOOKBACK_EPOCH_TAG              1
#define REUSE_LOOKBACK_PER_PASS         1

#define USE_INLINE_LOOKBACK             0
#define LOOKBACK_USE_NANOSLEEP_BACKOFF  1


namespace parse {

// Benchmark configuration members.
// Struct could be given as a parameter, but better to have
// individual explicit arguments
struct Global_Config {
    size_t n = size_t(1) << 27;
    uint32_t iterations = 30;
    uint32_t warmups = 10;
    uint32_t warm_ms = 250; // ms
    bool descending = false;
    bool validation = false;
};


void print_usage(const char* exe_name) {
    printf(
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --n <value>            Number of elements\n"
        "  --iterations <value>   Timed iterations\n"
        "  --warmup <value>       Warmup iterations\n"
        "  --warmup_ms <value>    Minimum warmup time (ms)\n"
        "  --descending           Sort descending\n"
        "  --validation           Validate algorithm\n"
        "  --help                 Show this help\n",
        exe_name
    );
}


// simple string to unsigned parser
template<typename T>
bool parse_u(const char* s, T* out_value) {
    if (!s || !*s) {
        return false;
    }

    char* end = nullptr;
    unsigned long long value = strtoull(s, &end, 10);

    if ((end == s) || (*end != '\0')) {
        return false;
    }

    if (value > 0xFFFFFFFFul) {
        return false;
    }

    *out_value = (T)value;
    return true;
}


// quick parser
bool args(int argc, char** argv, Global_Config* conf) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (!strcmp(arg, "--help")) {
            print_usage(argv[0]);
            return false; // caller can treat this specially if desired
        } else if (!strcmp(arg, "--descending")) {
            conf->descending = true;
        } else if (!strcmp(arg, "--validation")) {
            conf->validation = true;
        } else if (!strcmp(arg, "--n")) {
            if (i + 1 >= argc) {
                printf("Error: missing value after --n\n");
                return false;
            }
            if (!parse_u<size_t>(argv[++i], &conf->n)) {
                printf("Error: invalid value for --n: %s\n", argv[i]);
                return false;
            }
        } else if (!strcmp(arg, "--iterations")) {
            if (i + 1 >= argc) {
                printf("Error: missing value after --iterations\n");
                return false;
            }
            if (!parse_u<uint32_t>(argv[++i], &conf->iterations)) {
                printf("Error: invalid value for --iterations: %s\n", argv[i]);
                return false;
            }
        } else if (!strcmp(arg, "--warmup")) {
            if (i + 1 >= argc) {
                printf("Error: missing value after --warmup\n");
                return false;
            }
            if (!parse_u<uint32_t>(argv[++i], &conf->warmups)) {
                printf("Error: invalid value for --warmup: %s\n", argv[i]);
                return false;
            }
        } else if (!strcmp(arg, "--warmup_ms")) {
            if (i + 1 >= argc) {
                printf("Error: missing value after --warmup_ms\n");
                return false;
            }
            if (!parse_u<uint32_t>(argv[++i], &conf->warm_ms)) {
                printf("Error: invalid value for --warmup_ms: %s\n", argv[i]);
                return false;
            }
        } else {
            printf("Error: unknown argument: %s\n", arg);
            return false;
        }
    }

    return true;
}

} //namespace parse



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


// Tuning (extends constants)
struct radix_consts {
    static constexpr uint32_t RADIX_BITS = 8;
    static constexpr uint32_t RADIX_BIN_SIZE = 1u << RADIX_BITS;
    static constexpr uint32_t RADIX_MASK = RADIX_BIN_SIZE - 1u;
};


template<typename T>
struct radix_tuning : radix_consts {

    static constexpr uint32_t RADIX_PASSES = sizeof(T);

    // These CTA_MULTIPLIER and REORDER_WARPS are for sm_86
    // In the end, these two constants should be a lookup table
    // according to the sm_xx of the card, but I only have this one
    static constexpr uint32_t CTA_MULTIPLIER =  (sizeof(T) <= 4) ? 21 : 
                                                (sizeof(T) <= 8) ? 6  : 
                                                2;
    static constexpr uint32_t REORDER_WARPS =   (sizeof(T) <= 4) ? 12 :
                                                (sizeof(T) <= 8) ? 14 :
                                                20;
    
    static constexpr uint32_t REORDER_THREADS = REORDER_WARPS * WARP_SIZE;
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
    static_assert(std::is_arithmetic_v<T> || type_is_valid_128, "Radix type must be a numeric.");
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


    // quick and dirty array size estimation
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


// twiddling kernels
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


template<typename Key_T, typename Len_T>
__global__ void init_keys(Key_T* a, Len_T n, uint32_t seed, Array_Modes arr_mode) {
    using U = get_unsigned_of<Key_T>;
    using RT = radix_type_traits<Key_T>;
    
    Len_T i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        U x;

        // random init based on type
        if constexpr (sizeof(U) <= 4) {
            x = mix32(i ^ seed);
        } else if constexpr (sizeof(U) <= 8) {
            x = (uint64_t(mix32(i + 0x00000000u) ^ seed) << 32) |
                (uint64_t(mix32(i + 0x9e3779b9u) ^ seed) << 0);
        } else if constexpr (RT::has_native_u128) {
            using U128 = typename RT::native_u128;
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
            } else if constexpr (RT::has_native_u128) {
                using U128 = typename RT::native_u128;

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


// Main histogram kernel performaing a global count of all passes
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
    using U = get_unsigned_of<Key_T>;


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

    // work load definition
    auto histogram_chunk = [&](Len_T block_ind) {
        #pragma unroll
        for (int j = 0; j < GHIST_ITEMS_PER_THREAD; ++j) {
            Len_T idx = block_ind * GHIST_ITEM_PER_BLOCK + threadIdx.x * GHIST_ITEMS_PER_THREAD + (Len_T)j;
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
    };

    // Dynamic work stealing - faster but only safe if each block's bin count fit 32-bits
    // always true if using using 32-bit lookback
    Len_T num_blocks = div_round_up<Len_T>(n, GHIST_ITEM_PER_BLOCK);
    if constexpr (DYNAMIC_WORK_STEAL) {
        for (;;) {

            // global counter
            __shared__ uint32_t i_block;
            if (threadIdx.x == 0) {
                i_block = atomicInc(counter, 0xFFFFFFFFu);
            }
            __syncthreads();

            // stop condition
            if ((Len_T)i_block >= num_blocks) {
                break;
            }

            histogram_chunk(i_block);
            __syncthreads();
        }
    } else {
        // Fixed CTA striding. With 32 bit n, the maximum items seen by one CTA is
        // ceil(num_blocks / gridDim.x) * GHIST_ITEM_PER_BLOCK <= n, so uint32 locals stay safe.
        for (uint32_t i_block_fixed = (uint32_t)blockIdx.x; i_block_fixed < num_blocks; i_block_fixed += (uint32_t)gridDim.x) {
            histogram_chunk(i_block_fixed);
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


// user parameters
constexpr uint32_t LOOKBACK_NANOSLEEP_INITIAL = 4;
constexpr uint32_t LOOKBACK_NANOSLEEP_MAX = 256;
constexpr uint32_t LOOKBACK_SPINS = 1; // 2 for 18 CTAs, 1 for 38

static_assert(
    !(LOOKBACK_EPOCH_TAG && !REUSE_LOOKBACK_PER_PASS && LOOKBACK_OVERRIDE),
    "Epoch-tagged lookback requires reused per-pass lookback slices."
);


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


    // packing functions, always pack epoch bits along with publish state
    static __device__ __forceinline__ T pack_global(T val, T epoch_bits) {
        if constexpr(Epoch) {
            return (val & LB::VALUE_MASK) | LB::GLOBAL_MASK | epoch_bits;
        } else {
            return (val & LB::VALUE_MASK) | LB::GLOBAL_MASK;
        }
    }

    static __device__ __forceinline__ T pack_partial(T val, T epoch_bits) {
        if constexpr(Epoch) {
            return (val & LB::VALUE_MASK) | LB::PARTIAL_MASK | epoch_bits;
        } else {
            return (val & LB::VALUE_MASK) | LB::PARTIAL_MASK;
        }
    }


    // Rule #1 of high performance programming:
    // the more underscores your code has the faster it goes
    static __device__ __host__ __forceinline__ T pack_epoch(uint32_t epoch) {
        if constexpr(Epoch) {
            using LB = Lookback_Config<T, Epoch>;
            return ((T)epoch & LB::EPOCH_VALUE_MASK) << LB::EPOCH_SHIFT;
        } else {
            return 0;        
        }
    }


    // Note: No need to check for 0 if using Epoch bits because
    // lookback will be EPOCH_TAG filled for the first pass (and on epoch wrap)
    static __device__ __forceinline__ bool invalid_lookback_state(T raw, T lb) {
        if constexpr(Epoch) {
            //return ((raw & LB::EPOCH_MASK) != lb) || (raw == 0u);
            return (raw & LB::EPOCH_MASK) != lb;
        } else {
            return raw == 0u;
        }
    }


    // standard exponential backoff strategy
    static __device__ __forceinline__ void nano_wait(T* raw, const volatile T* ptr, T epoch_bits) {
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
    static __device__ __forceinline__ void wait_valid_lookback_state(T* raw, const volatile T* ptr, T epoch_bits) {
        do {
            *raw = *ptr;
            nano_wait(raw, ptr, epoch_bits);
        } while (invalid_lookback_state(*raw, epoch_bits));
    }


    static __device__ __forceinline__ bool valid_global_lookback_state(T raw, T epoch_bits) {
        if constexpr(Epoch) {
            return ((raw & LB::GLOBAL_MASK) != 0) && ((raw & LB::EPOCH_MASK) == epoch_bits) && (raw != 0);
        } else {
            return ((raw & LB::GLOBAL_MASK) != 0) && (raw != 0);
        }
    }


    // Main lookback function, decoupled, partial publishing
    // less branching (more register pressure?)
    static __device__ __forceinline__ typename Policy::T lookback_prefix(
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
        #pragma unroll
        for (int i = 0; i < LOOKBACK_SPINS; ++i) {
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
    static __device__ __forceinline__ typename Policy::T lookback_prefix_old(
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

        #pragma unroll
        for (int spin = 0; spin < LOOKBACK_SPINS; ++spin) {
            raw = *((const volatile T*)&lookback_bin[lb]);
            if (valid_global_lookback_state(raw, lb_epoch_bits)) {
                p = (raw & VALUE_MASK);
                got_global = true;
                break;
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

        // read radices
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
            digits[u] = (uint8_t)extract_byte<U>(keys[u], bit_location);
        }

        // init offsets
        int* warp_offsets = &temp_storage.warp_offsets[warp][0];
        #pragma unroll
        for (int bin = lane; bin < RADIX_BIN_SIZE; bin += WARP_SIZE) {
            warp_offsets[bin] = 0;
        }
        __syncwarp(0xFFFFFFFFu); // do not remove

        // Warp-private histogram
        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
            atomicAdd(&warp_offsets[(uint32_t)digits[u]], 1);
        }

        // Block-wide upsweep
        __syncthreads();
        int bins = 0;
        if (bin_owner) {
            const int bin = (int)threadIdx.x;
            int* warp_bin_ptr = &temp_storage.warp_offsets[0][bin];
            int* warp_bin_it = warp_bin_ptr;

            // warp partials scan
            #pragma unroll
            for (int j = 0; j < REORDER_WARPS; ++j) {
                int count = *warp_bin_it;
                *warp_bin_it = bins;
                bins += count;
                warp_bin_it += RADIX_BIN_SIZE;
            }

            // callback
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


// Portable workspace struct for post-sort checks
template <typename Key_T, typename Lookback_T, typename Len_T>
struct Sort_Workspace {
    Len_T num_blocks = 0;
    size_t lb_els = 0;
    size_t total_bytes = 0;
    size_t off_tmp = 0;
    size_t off_gp = 0;
    size_t off_counter = 0;
    size_t off_look_partial = 0;


    struct View {
        Key_T* tmp = nullptr;
        Lookback_T* gp = nullptr;
        uint32_t* counter = nullptr;
        Lookback_T* look_partial = nullptr;
    };


    // build workspace - set memory pointers according to policy 
    template <typename Lookback_Policy>
    static inline Sort_Workspace build(Len_T n) {
        using RT = radix_tuning<Key_T>;

        Sort_Workspace ws;
        ws.num_blocks = div_round_up<Len_T>(n, RT::SORT_BLOCK_SIZE);

        if constexpr (Lookback_Policy::reuse) {
            ws.lb_els = (size_t)ws.num_blocks * RT::RADIX_BIN_SIZE;
        } else {
            ws.lb_els = (size_t)RT::RADIX_PASSES * ws.num_blocks * RT::RADIX_BIN_SIZE;
        }

        size_t off = 0;
        ws.off_tmp = reserve_aligned(&off, (size_t)n * sizeof(Key_T), 256);
        ws.off_gp = reserve_aligned(&off, (size_t)RT::RADIX_PASSES * RT::RADIX_BIN_SIZE * sizeof(Lookback_T), 256);
        ws.off_counter = reserve_aligned(&off, sizeof(uint32_t), 64);
        ws.off_look_partial = reserve_aligned(&off, ws.lb_els * sizeof(Lookback_T), 256);
        ws.total_bytes = off;
        return ws;
    }


    View bind(uint8_t* base) const {
        View view;
        view.tmp = reinterpret_cast<Key_T*>(base + off_tmp);
        view.gp = reinterpret_cast<Lookback_T*>(base + off_gp);
        view.counter = reinterpret_cast<uint32_t*>(base + off_counter);
        view.look_partial = reinterpret_cast<Lookback_T*>(base + off_look_partial);
        return view;
    }
};


// scatter kernel with full block support and using simple storaging
template <bool Full_Block, bool Descending, typename Lookback_T, typename Key_T, typename Len_T, typename U>
__device__ __forceinline__ void scatter_staged(
    const U* __restrict__ staged,
    const Lookback_T* __restrict__ bin_offset,
    Key_T* __restrict__ out_keys,
    uint32_t actual_tile_items,
    uint32_t bit_location) {

        
    using RT = radix_tuning<Key_T>;
    constexpr uint32_t ITEMS_PER_THREAD = RT::REORDER_ITEMS_PER_THREAD;
    constexpr uint32_t REORDER_THREADS = RT::REORDER_THREADS;


    #pragma unroll
    for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
        Len_T idx = threadIdx.x + (Len_T)u * REORDER_THREADS;
        if constexpr (Full_Block) {
            U key = staged[idx];
            U digit = extract_byte<U>(key, bit_location);
            out_keys[idx + (Len_T)bin_offset[digit]] = twiddle_out<Descending, Key_T>(key);
        } else {
            if (idx < (Len_T)actual_tile_items) {
                U key = staged[idx];
                U digit = extract_byte<U>(key, bit_location);
                out_keys[idx + (Len_T)bin_offset[digit]] = twiddle_out<Descending, Key_T>(key);
            }
        }
        __syncwarp(0xFFFFFFFFu);
    }
}


// Non-decoupled lookback path with early partial publication via chained callback
// The lookback also includes epoch bits to avoid cudamemset on every pass.
// Standard CUB ranking, implicit full blocking
template<bool Descending, typename Lookback_Policy, typename Key_T, typename Len_T>
__global__ __launch_bounds__(radix_tuning<Key_T>::REORDER_THREADS)
void onesweep_byte(
    const Key_T* __restrict__ in_keys,
    Key_T* __restrict__ out_keys,
    Len_T n,
    const typename Lookback_Policy::T* __restrict__ gp_sum_buffer,      // [RADIX_PASSES][RADIX_BIN_SIZE]
    volatile typename Lookback_Policy::T* __restrict__ lookback_partial, // [numBlocks * RADIX_BIN_SIZE]
    uint32_t iteration,
    typename Lookback_Policy::T lookback_epoch_bits) {
        
    
    using Lookback_T = typename Lookback_Policy::T;
    using LB = typename Lookback_Policy::conf;
    using U = get_unsigned_of<Key_T>;
    using Ranker = Radix_Ranker<Key_T>;
    
    using RT = radix_tuning<Key_T>;
    constexpr uint32_t SORT_BLOCK_SIZE = RT::SORT_BLOCK_SIZE;
    constexpr uint32_t LOGICAL_BLOCK_SIZE = RT::REORDER_LOGICAL_BLOCK_SIZE;
    constexpr uint32_t ITEMS_PER_THREAD = RT::REORDER_ITEMS_PER_THREAD;
    constexpr uint32_t ITEMS_PER_WARP = RT::REORDER_ITEMS_PER_WARP;
    constexpr uint32_t RADIX_BIN_SIZE = RT::RADIX_BIN_SIZE;
    constexpr uint32_t RADIX_BITS = RT::RADIX_BITS;
    
    static_assert(RADIX_BIN_SIZE == 256, "CUB-like kernel expects 8-bit radix.");


#if USE_PSUM_SHARED
    __shared__ Lookback_T p_sum[RADIX_BIN_SIZE];
#endif

    struct SMem {
        typename Ranker::Temp_Storage rank_temp;
        uint16_t bin_count[RADIX_BIN_SIZE];
        Lookback_T bin_offset[RADIX_BIN_SIZE];
        U staged[LOGICAL_BLOCK_SIZE];
    };
    __shared__ SMem smem;

    uint32_t block_index = (uint32_t)blockIdx.x;
    const int bin_owner = (int)threadIdx.x < (int)RADIX_BIN_SIZE;

    uint32_t bit_location = (uint32_t)RADIX_BITS * iteration;
    Len_T block_base = block_index * (Len_T)SORT_BLOCK_SIZE;
    const int full_block = (((Len_T)block_base + (Len_T)SORT_BLOCK_SIZE) <= (Len_T)n);

    uint32_t actual_tile_items = full_block ? SORT_BLOCK_SIZE : (uint32_t)(n - block_base);
    uint32_t invalid_items = LOGICAL_BLOCK_SIZE - actual_tile_items;

    U keys[ITEMS_PER_THREAD];
    int ranks[ITEMS_PER_THREAD]; // TODOs: why is this int? 

    int warp = (int)threadIdx.x / WARP_SIZE;
    int lane = (int)lane_id_u32();

    int exclusive_digit_prefix = 0;
    if (full_block) {
        Len_T warp_base = block_base + (Len_T)warp * ITEMS_PER_WARP;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            keys[i] = twiddle_in<Descending, Key_T>(in_keys[warp_base + lane + i * WARP_SIZE]);
        }

        Ranker::template match_early_counts<true, Descending, Lookback_Policy>(
            smem.rank_temp,
            keys,
            ranks,
            bit_location,
            exclusive_digit_prefix,
            smem.bin_count,
            lookback_partial,
            block_index,
            0u,
            lookback_epoch_bits
        );

    } else {
        //const uint64_t warp_base64 = (uint64_t)block_base + (uint64_t)warp * ITEMS_PER_WARP;
        const uint64_t warp_base64 = (uint64_t)block_base + (uint64_t)warp * ITEMS_PER_WARP;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            //uint32_t idx = warp_base + lane + i * WARP_SIZE;
            uint64_t idx = warp_base64 + (uint64_t)lane + (uint64_t)i * WARP_SIZE;
            
            // The filler key must map to the last radix bin in the current order.
            //keys[i] = twiddle_in<Descending, Key_T>(
            //    (idx < n) ? in_keys[idx] : tail_filler_key<Descending, Key_T>());
            keys[i] = (idx < (uint64_t)n)
                ? twiddle_in<Descending, Key_T>(in_keys[(Len_T)idx])
                : tail_filler_bits<Descending, Key_T>();
        }

        Ranker::template match_early_counts<false, Descending, Lookback_Policy>(
            smem.rank_temp,
            keys,
            ranks,
            bit_location,
            exclusive_digit_prefix,
            smem.bin_count,
            lookback_partial,
            block_index,
            invalid_items,
            lookback_epoch_bits
        );

    }
    //__syncthreads(); // taken out to not mess with the next 2 __sync's performance

    Lookback_T p = 0;
    if (bin_owner) {
        const uint32_t b = (uint32_t)threadIdx.x;
        const int p_ind = (int)((block_index << RADIX_BITS) + b);

#if USE_INLINE_LOOKBACK
        p = Lookback<Lookback_Policy>::lookback_prefix_old(
            lookback_partial, block_index, b, lookback_epoch_bits
        );
#else
        p = Lookback<Lookback_Policy>::lookback_prefix(
            lookback_partial + b, block_index, lookback_epoch_bits
        );
#endif

        lookback_partial[p_ind] = Lookback<Lookback_Policy>::pack_global(
            p + (uint32_t)smem.bin_count[b], lookback_epoch_bits);


        const Lookback_T* gp_row = gp_sum_buffer + (size_t)iteration * RADIX_BIN_SIZE;
        
        // Note: Performance wise, no real difference in computing psum in shared or not
#if USE_PSUM_SHARED
        p_sum[b] = gp_row[b] + p;
        smem.bin_offset[b] = p_sum[b] - (Lookback_T)exclusive_digit_prefix;
#else
        Lookback_T base = gp_row[b] + p;
        smem.bin_offset[b] = base - (Lookback_T)exclusive_digit_prefix;
#endif
    }
    __syncthreads();

    // Stage in local rank order.
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        smem.staged[ranks[k]] = keys[k];
    }
    __syncthreads();

    if (full_block) {
        scatter_staged<true, Descending, Lookback_T, Key_T, Len_T, U>(
            smem.staged, smem.bin_offset, out_keys, actual_tile_items, bit_location);
    } else {
        scatter_staged<false, Descending, Lookback_T, Key_T, Len_T, U>(
            smem.staged, smem.bin_offset, out_keys, actual_tile_items, bit_location);
    }
    //scatter_staged<false, Descending, Lookback_T, Key_T, Len_T, U>(smem.staged, smem.bin_offset, out_keys, actual_tile_items, bit_location);
}


// Generates the skip mask for early-exiting
template <typename Lookback_T, typename Key_T, typename Len_T>
static uint32_t compute_uniform_pass_skip_mask(const Lookback_T* d_gp, Len_T n) {
    
    using RT = radix_tuning<Key_T>;
    constexpr uint32_t RADIX_PASSES = RT::RADIX_PASSES;
    constexpr uint32_t RADIX_BIN_SIZE = RT::RADIX_BIN_SIZE;
    constexpr size_t HIST_ELEMS = (size_t)RADIX_PASSES * RADIX_BIN_SIZE;
    Lookback_T h_gp[HIST_ELEMS];
    cudaMemcpy(h_gp, d_gp, HIST_ELEMS * sizeof(Lookback_T), cudaMemcpyDeviceToHost);

    uint32_t skip_mask = 0u;
    for(uint32_t p = 0; p < RADIX_PASSES; ++p) {
        const Lookback_T* row = h_gp + (size_t)p * RADIX_BIN_SIZE;
        for(uint32_t b = 0; b < RADIX_BIN_SIZE; ++b) {
            
            Lookback_T lo = row[b];
            Lookback_T hi = (b + 1u < RADIX_BIN_SIZE) ? row[b + 1u] : (Lookback_T)n;
            if ((hi - lo) == (Lookback_T)n) {
                skip_mask |= (1u << p);
                break;
            }
            /*
            if(row[b] == n) {
                skip_mask |= (1u << p);
                break;
            }
            */
        }
    }

    return skip_mask;
}


// Enrty point of the sorting kernel
template <bool Descending, typename Lookback_Policy, typename Key_T, typename Len_T>
static void onesweep_byte_sort_enqueue(
    cudaStream_t stream,
    Key_T* d_inout,
    Key_T* d_tmp,
    typename Lookback_Policy::T* d_gp,
    uint32_t* d_counter,
    typename Lookback_Policy::T* d_look_partial,
    Len_T n,
    Len_T num_blocks,
    size_t lb_els) {


    using LB = typename Lookback_Policy::conf;
    using Lookback_T = typename Lookback_Policy::T;
    constexpr bool lb_epoch = Lookback_Policy::epoch;
    constexpr bool lb_reuse = Lookback_Policy::reuse;

    using RT = radix_tuning<Key_T>;
    constexpr uint32_t REORDER_THREADS = RT::REORDER_THREADS;
    constexpr uint32_t RADIX_PASSES = RT::RADIX_PASSES;

    using H = histogram_tuning;
    constexpr uint32_t HIST_BLOCKS = H::HIST_BLOCKS;
    constexpr uint32_t GHIST_THREADS = H::GHIST_THREADS;

    // conditionally call memsets according to lookback policy
    if constexpr(lb_reuse) {
        size_t initial_bytes = (size_t)((uint8_t*)(d_counter + 1) - (uint8_t*)d_gp);
        cudaMemsetAsync(d_gp, 0, initial_bytes, stream);
    } else if constexpr(!lb_epoch) {
        size_t initial_bytes = (size_t)((uint8_t*)(d_look_partial + lb_els) - (uint8_t*)d_gp);
        cudaMemsetAsync(d_gp, 0, initial_bytes, stream);
    }
    if constexpr(lb_epoch) {
        cudaMemsetAsync(d_look_partial, LB::EPOCH_TAG, lb_els * sizeof(Lookback_T), stream);
    }

    Key_T* in  = d_inout;
    Key_T* out = d_tmp;

    // Histogram call - template on dynamic work stealing (using lookback type for compile time branching)
    if constexpr (sizeof(Lookback_T) <= sizeof(uint32_t)) {
        GHistogram_8bits<Descending, Lookback_T, Key_T, Len_T, true>
        <<<HIST_BLOCKS, GHIST_THREADS, 0, stream>>>(d_inout, n, d_gp, 0u, d_counter);
    } else {
        uint32_t h_blocks = HIST_BLOCKS;
        Len_T hist_tiles = div_round_up<Len_T>(n, H::GHIST_ITEM_PER_BLOCK);
        if (h_blocks > hist_tiles) {
            h_blocks = hist_tiles;
        }
        GHistogram_8bits<Descending, Lookback_T, Key_T, Len_T, false>
        <<<h_blocks, GHIST_THREADS, 0, stream>>>(d_inout, n, d_gp, 0u, d_counter);
    }

    // build skip mask if doing early exit
#if EXIT_EARLY_OPT
    uint32_t skip_mask = compute_uniform_pass_skip_mask<Lookback_T, Key_T, Len_T>(d_gp, n);
#endif

    // passes loop
    for (uint32_t it = 0; it < RADIX_PASSES;) {

        // probably not the best pratice to define the end of a loop at the beginning
        auto iteration_end = [&]() {
            ++it;
            if constexpr(Lookback_Policy::epoch) {
                if (((it & LB::EPOCH_VALUE_MASK) == 0) && (it < RADIX_PASSES)) {
                    cudaMemsetAsync(d_look_partial, LB::EPOCH_TAG, lb_els * sizeof(Lookback_T), stream);
                }
            }
        };


#if EXIT_EARLY_OPT
        if (skip_mask & (1u << it)) {
            iteration_end();
            continue;
        }
#endif
        volatile Lookback_T* look_partial_pass = nullptr;

        // if epoch disabled memset, else pack bits 
        if constexpr(lb_reuse) {
            if constexpr(!lb_epoch) {
                cudaMemsetAsync(d_look_partial, 0, lb_els * sizeof(Lookback_T), stream);
            }
            look_partial_pass = d_look_partial;
        } else {
            Len_T tileBase = it * (num_blocks * RT::RADIX_BIN_SIZE);
            look_partial_pass = d_look_partial ? (d_look_partial + (size_t)tileBase) : nullptr;
        }
        Lookback_T lb_bits = 0;
        if constexpr(lb_epoch) {
            lb_bits = Lookback<Lookback_Policy>::pack_epoch(it & LB::EPOCH_VALUE_MASK);
        }

        // onesweep entry
        onesweep_byte<Descending, Lookback_Policy>
        <<<num_blocks, REORDER_THREADS, 0, stream>>>(in, out, n, d_gp, look_partial_pass, it, lb_bits);

        // ping-pong buffers
        Key_T* tmp = in;
        in = out;
        out = tmp;

        iteration_end();
    }

    // copy if not in dest (odd number of passes)
    if (in != d_inout) {
        cudaMemcpyAsync(d_inout, in, (size_t)n * sizeof(Key_T), cudaMemcpyDeviceToDevice, stream);
    }
}


// Sorting 
template <bool Descending, typename Lookback_Policy, typename Key_T, typename Len_T>
static void onesweep_byte_sort_impl(
    Key_T* d_inout,
    size_t* temp_bytes,
    uint8_t* d_workspace, 
    Len_T n,
    cudaStream_t stream = 0) {


    using LB = typename Lookback_Policy::conf;
    using Lookback_T = typename Lookback_Policy::T;
    using Workspace = Sort_Workspace<Key_T, Lookback_T, Len_T>;


    Workspace ws = Workspace::template build<Lookback_Policy>(n);

    // return if just asking for size, like CUB
    if (d_workspace == nullptr) {
        *temp_bytes = ws.total_bytes;
        return;
    }

    // Continue to entry point
    auto view = ws.bind(d_workspace);
    onesweep_byte_sort_enqueue<Descending, Lookback_Policy, Key_T, Len_T>(
        stream, d_inout, view.tmp, view.gp, view.counter,
        view.look_partial, n, ws.num_blocks, ws.lb_els
    );
}


// Policy swtich
template<bool Descending, typename Key_T, typename Len_T>
static inline void looback_policy_enforcer(
    Key_T* d_inout, size_t* temp_bytes, uint8_t* d_workspace, Len_T n, cudaStream_t stream = 0) {
    
    Lookback_Modes mode = get_lookback_mode(n); // according to array size

    // template heavy
    switch(mode) {
        case Lookback_Modes::u32_epoch:
            onesweep_byte_sort_impl<Descending, Faster_LB_Policy, Key_T, Len_T>(
                d_inout, temp_bytes, d_workspace, n, stream);
            break;
        case Lookback_Modes::u32_plain:
            onesweep_byte_sort_impl<Descending, Fast_LB_Policy, Key_T, Len_T>(
                d_inout, temp_bytes, d_workspace, n, stream);
            break;
        case Lookback_Modes::u64_epoch:
            onesweep_byte_sort_impl<Descending, General_LB_Policy, Key_T, Len_T>(
                d_inout, temp_bytes, d_workspace, n, stream);
            break;
        default:
            break;
    }
}


// Wrap for the sorting call:
// Builds the CUDA graph for the sort if defined
// Performance gains are not significant 
template<bool Descending, typename Key_T, typename Len_T>
static inline void onesweep_byte_sort_wrap(
    Key_T* d_inout, size_t* temp_bytes, uint8_t* d_workspace, Len_T n) {
    cudaStream_t capture_stream = nullptr;
#if USE_CUDA_GRAPH_SORT && !EXIT_EARLY_OPT
    if (d_workspace != nullptr) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaStream_t stream = 0;

        cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);

        cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
        looback_policy_enforcer<Descending, Key_T, Len_T>(d_inout, temp_bytes, d_workspace, n, capture_stream);
        cudaStreamEndCapture(capture_stream, &graph);

        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        cudaGraphLaunch(exec, stream);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(capture_stream);
        return;
    }
#endif
    looback_policy_enforcer<Descending, Key_T, Len_T>(d_inout, temp_bytes, d_workspace, n, capture_stream);
}


// onesweep_byte_sort_wrap<false, Key_T, Len_T>(d_inout, temp_bytes, d_workspace, n);
// however, size_t is faster in benchmarking
template<typename Key_T, typename Len_T>
static inline void onesweep_byte_sort(
    Key_T* d_inout, size_t* temp_bytes, uint8_t* d_workspace, Len_T n) {
    onesweep_byte_sort_wrap<false, Key_T, size_t>(d_inout, temp_bytes, d_workspace, n);
}


template<typename Key_T, typename Len_T>
static inline void onesweep_byte_sort_descending(
    Key_T* d_inout, size_t* temp_bytes, uint8_t* d_workspace, Len_T n) {
    onesweep_byte_sort_wrap<true, Key_T, size_t>(d_inout, temp_bytes, d_workspace, n);
}


// Reconstructs the digit histogram for the last pass and checks count of each radix
// in the sorted array to assert if sorted output has the same counts.
template <bool Descending, typename Lookback_Policy, typename Key_T, typename Len_T>
static bool verify_digit_histograms_preserved(
    const Key_T* d_sorted,
    uint8_t* d_workspace,
    Len_T n,
    uint32_t seed,
    Array_Modes arr_mode) {


    using Lookback_T = typename Lookback_Policy::T;
    using RT = radix_tuning<Key_T>;
    constexpr uint32_t RADIX_PASSES = RT::RADIX_PASSES;
    constexpr uint32_t RADIX_BIN_SIZE = RT::RADIX_BIN_SIZE;
    constexpr size_t HIST_ELEMS = (size_t)RADIX_PASSES * RADIX_BIN_SIZE;
    using H = histogram_tuning;
    constexpr uint32_t HIST_BLOCKS = H::HIST_BLOCKS;
    constexpr uint32_t GHIST_THREADS = H::GHIST_THREADS;
    

    Lookback_T* d_hist_ref = nullptr;
    Lookback_T* d_hist_chk = nullptr;
    uint32_t* d_counter = nullptr;
    Lookback_T h_hist_ref[HIST_ELEMS];
    Lookback_T h_hist_chk[HIST_ELEMS];

    CHECK_CUDA(cudaMalloc(&d_hist_ref, HIST_ELEMS * sizeof(Lookback_T)));
    CHECK_CUDA(cudaMalloc(&d_hist_chk, HIST_ELEMS * sizeof(Lookback_T)));
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(uint32_t)));

    using Workspace = Sort_Workspace<Key_T, Lookback_T, Len_T>;
    Workspace ws = Workspace::template build<Lookback_Policy>(n);
    auto workspace_view = ws.bind(d_workspace);

    CHECK_CUDA(cudaMemset(d_hist_ref, 0, HIST_ELEMS * sizeof(Lookback_T)));
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(uint32_t)));
    init_keys<Key_T, Len_T><<<div_round_up<Len_T>(n, 256), 256>>>(workspace_view.tmp, n, seed, arr_mode);
    CHECK_CUDA(cudaGetLastError());
    GHistogram_8bits<Descending, Lookback_T, Key_T, Len_T>
    <<<HIST_BLOCKS, GHIST_THREADS>>>(workspace_view.tmp, n, d_hist_ref, 0u, d_counter);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemset(d_hist_chk, 0, HIST_ELEMS * sizeof(Lookback_T)));
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(uint32_t)));
    GHistogram_8bits<Descending, Lookback_T, Key_T, Len_T>
    <<<HIST_BLOCKS, GHIST_THREADS>>>(d_sorted, n, d_hist_chk, 0u, d_counter);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_hist_ref, d_hist_ref, HIST_ELEMS * sizeof(Lookback_T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_hist_chk, d_hist_chk, HIST_ELEMS * sizeof(Lookback_T), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_counter));
    CHECK_CUDA(cudaFree(d_hist_chk));
    CHECK_CUDA(cudaFree(d_hist_ref));

    // TODOs: change (b + 1u < RADIX_BIN_SIZE) ? ref[b + 1u] : (Lookback_T)n;
    // to be ref[b + 1u] and do RADIX_BIN_SIZE - 1
    for (uint32_t p = 0; p < RADIX_PASSES; ++p) {
        const Lookback_T* ref = h_hist_ref + (size_t)p * RADIX_BIN_SIZE;
        const Lookback_T* chk = h_hist_chk + (size_t)p * RADIX_BIN_SIZE;
        for (uint32_t b = 0; b < RADIX_BIN_SIZE; ++b) {
            Lookback_T ref_lo = ref[b];
            Lookback_T ref_hi = (b + 1u < RADIX_BIN_SIZE) ? ref[b + 1u] : (Lookback_T)n;
            Lookback_T chk_lo = chk[b];
            Lookback_T chk_hi = (b + 1u < RADIX_BIN_SIZE) ? chk[b + 1u] : (Lookback_T)n;
            Lookback_T ref_count = ref_hi - ref_lo;
            Lookback_T chk_count = chk_hi - chk_lo;
            if (ref_count != chk_count) {
                printf("digit-count mismatch at pass %u bin %u: input=%llu output=%llu\n",
                    p, b,
                    (unsigned long long)ref_count,
                    (unsigned long long)chk_count);
                return false;
            }
        }
    }

    return true;
}


// Verify histogram according to mode
template <bool Descending, typename Key_T, typename Len_T>
static bool verify_hist_by_mode(
    Lookback_Modes mode,
    Array_Modes arr_mode,
    const Key_T* d_keys,
    uint8_t* d_workspace,
    Len_T n,
    uint32_t seed)  {

    switch (mode) {
        case Lookback_Modes::u32_epoch:
            return verify_digit_histograms_preserved<Descending, Faster_LB_Policy, Key_T, Len_T>(
                d_keys, d_workspace, n, seed, arr_mode);
        case Lookback_Modes::u32_plain:
            return verify_digit_histograms_preserved<Descending, Fast_LB_Policy, Key_T, Len_T>(
                d_keys, d_workspace, n, seed, arr_mode);
        case Lookback_Modes::u64_epoch:
            return verify_digit_histograms_preserved<Descending, General_LB_Policy, Key_T, Len_T>(
                d_keys, d_workspace, n, seed, arr_mode);
    }
    return false;
}


// Benchmarking function (entry point of the sort)
template<bool Descending, typename T, typename Len_T>
int benchmark(
    Len_T n,
    uint32_t iters,
    uint32_t warmups,
    uint32_t warm_ms,
    Array_Modes arr_mode,
    bool validation = false) {


    using RT = radix_tuning<T>;


    // initialization
    size_t temp_bytes = 0;
    T* d_keys = nullptr;
    uint8_t* d_workspace = nullptr;

    auto launch_sorting_kernel = [&]() {
        if constexpr (Descending) {
            onesweep_byte_sort_descending<T>(d_keys, &temp_bytes, d_workspace, n);
        } else {
            onesweep_byte_sort<T>(d_keys, &temp_bytes, d_workspace, n);
        }
    };

    // do not time memory allocations
    CHECK_CUDA(cudaMalloc(&d_keys, sizeof(T) * n));
    launch_sorting_kernel();
    CHECK_CUDA(cudaMalloc(&d_workspace, temp_bytes));
    CHECK_CUDA(cudaGetLastError());

    // timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // define sorting iteration
    auto run_timed_iteration = [&](uint32_t seed) {
        init_keys<T, Len_T><<<div_round_up<Len_T>(n, 256), 256>>>(d_keys, n, seed, arr_mode);
        CHECK_CUDA(cudaGetLastError());

        cudaEventRecord(start);
        launch_sorting_kernel();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        return ms_total;
    };

    // warmup
    double warm_ms_passed = 0.0;
    uint32_t warmups_done = 0;
    uint32_t seed_counter = 0;
    while ((warmups_done < warmups) || (!validation && (warm_ms_passed < (double)warm_ms))) {
        warm_ms_passed += run_timed_iteration(set_seed_radix(seed_counter++));
        ++warmups_done;
    }

    // benchmark runs
    double ms_acc = 0.0;
    double* timings = (double *)malloc(sizeof(*timings) * iters);
    for (uint32_t i = 0; i < iters; ++i) {
        double temp = run_timed_iteration(set_seed_radix(seed_counter++));
        ms_acc += temp;
        timings[i] = temp;
    }
    
    // calculate stats
    double ms_avg = ms_acc / (double)iters;
    long long els = (long long)((1000.0 / ms_avg) * (double)n);
    double psel = 1'000'000'000'000.0 / (double)els;
    double us_std = stdev(timings, ms_avg, (size_t)iters) * 1000.0;
    free(timings);

    // Device properties
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::string_view sv_temp = type_name<T>();
    if (!validation) {
        printf("\nGPU: %s\n"
            "\nRADIX\t=\t%d\n"
            "N\t=\t%llu\n"
            "Type\t=\t%.*s\n"
            "Blocks\t=\t%d\n"
            "Threads\t=\t%d\n"
            "it/thd\t=\t%d\n"
            "Memory\t=\t%.3f MB\n"
            "Time\t=\t%.3f ms (+- %.3f us)\n"
            "\t\t(%d avg w/ %d wm)\n"
            "it/s\t=\t%lld\n"
            "ps/el\t=\t%.3f\n"
            "Mode\t=\t%s\n"
            "Order\t=\t%s\n",

            prop.name,
            RT::RADIX_BITS,
            (long long unsigned int)n,
            (int)sv_temp.size(), sv_temp.data(),
            (uint32_t)div_round_up<Len_T>(n, RT::SORT_BLOCK_SIZE),
            RT::REORDER_THREADS,
            RT::REORDER_ITEMS_PER_THREAD,
            (double)temp_bytes / (1024. * 1024.),
            ms_avg, us_std,
            iters, warmups_done,
            els,
            psel,
            arr_modes_to_string(arr_mode),
            Descending ? "DESC" : "ASC"
        );
    }
    
    Lookback_Modes mode = get_lookback_mode(n);
    if (!validation) {
        print_lookback_policy(mode);
    }

    // check if sorted
    cudaEventRecord(start);
    
    int* d_ok = nullptr;
    int temp = 1;
    CHECK_CUDA(cudaMalloc(&d_ok, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_ok, &temp, sizeof(int), cudaMemcpyHostToDevice));

    Len_T check_blocks = div_round_up<Len_T>(n, 256);
    check_sorted<Descending, T><<<check_blocks, 256>>>(d_keys, n, d_ok);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
    CHECK_CUDA(cudaGetLastError());

    int h_ok = 0;
    CHECK_CUDA(cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost));
    if (!validation) {
        printf("\nsorted ok? %s (%.3f ms)\n", h_ok ? "YES" : "NO", ms_total);
    }

    // check histograms 
    uint32_t last_seed = (uint32_t)(set_seed_radix(seed_counter - 1));
    bool hist_ok = false;
    if (Descending) {
        hist_ok = verify_hist_by_mode<true, T>(
            mode, arr_mode, d_keys, d_workspace, n, last_seed);
    } else {
        hist_ok = verify_hist_by_mode<false, T>(
            mode, arr_mode, d_keys, d_workspace, n, last_seed);
    }

    int bench_valid = hist_ok && h_ok;
    if (!validation) {
        printf("digit counts ok? %s\n", hist_ok ? "YES" : "NO");
    } else {
        printf("Sorting %llu els of type %.*s (%s), %s array - %.3f ms (%d avg + %d wm)... %s\n",
            (long long unsigned int)n,
            (int)sv_temp.size(), sv_temp.data(),
            Descending ? "DESC" : "ASC",
            arr_modes_to_string(arr_mode),
            ms_avg, iters, warmups_done,
            bench_valid ? "passed" : "failed"
        );
    }

    // cleanup
    CHECK_CUDA(cudaFree(d_ok));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_keys));

    return bench_valid;
}


// in ms 
#define COOL_BETWEEN_RUNS   0.0

template<typename... Ts> struct type_list {};

using radix_test_types = type_list<
#if VALIDATION_TEST
    TYPE_SET_TEST
#endif
>;


// Simple counter struct for validation tests
struct Validation_Result {
    uint32_t tests = 0;
    uint32_t passed = 0;

    constexpr void add(bool ok) {
        ++tests;
        passed += ok ? 1u : 0u;
    }

    constexpr void merge(const Validation_Result& other) {
        tests += other.tests;
        passed += other.passed;
    }

    constexpr bool all_passed() const {
        return tests == passed;
    }
};


// Benchmark entry point of the validation
template<typename T>
Validation_Result validate_radix_type(
    bool descending,
    int iter,
    int warm,
    Array_Modes mode,
    uint32_t vram_gb) {

    Validation_Result result;
    uint32_t bit_end = radix_type_traits<T>::max_array_bits(vram_gb);
    uint32_t bit_start = bit_end - 10;

    for (int i = (int)bit_start; i <= (int)bit_end; ++i) {
        
        uint64_t n = 1ull << i;
        int pass_ok;
        if (descending) {
            pass_ok = benchmark<true, T>(n, warm, iter, 0, mode, true);
        } else {
            pass_ok = benchmark<false, T>(n, warm, iter, 0, mode, true);
        }
        result.add(pass_ok);

        if (!pass_ok) {
            std::string_view sv = type_name<T>();
            printf("Failed test with: n = 2^%d, Order = %s, mode = %s, type = %.*s\n",
                i,
                descending ? "DESC" : "ASC",
                arr_modes_to_string(mode),
                (int)sv.size(), sv.data()
            );
        }

        sleep_ms(COOL_BETWEEN_RUNS);
    }

    return result;
}


// Validate accoring to list of types
template<typename... Ts>
Validation_Result validate_radix_wrap(
    bool desc,
    int iter,
    int warm,
    Array_Modes mode) {

    Validation_Result total;

    if constexpr (sizeof...(Ts) > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        uint64_t vram_bytes = prop.totalGlobalMem;
        uint32_t vram_gb = vram_bytes >> 30;

        (total.merge(
                validate_radix_type<Ts>(false, iter, warm, mode, vram_gb)
            ), ...
        );
        if (desc) {
            (total.merge(
                    validate_radix_type<Ts>(true, iter, warm, mode, vram_gb)
                ), ...
            );
        }
    }

    return total;
}


template<typename List> struct validate_radix_list;

template<typename... Ts>
struct validate_radix_list<type_list<Ts...>> {
    static Validation_Result run(
        bool desc,
        int iter,
        int warm,
        Array_Modes mode) {
        return validate_radix_wrap<Ts...>(desc, iter, warm, mode);
    }
};


// validation entry point
bool validate(bool all_modes, bool desc, int iter, int warm) {
    Validation_Result total;

    printf("\nStarting validation tests... (%d iterations, %d warmup runs each)\n\n", iter, warm);
    Array_Modes mode_stop = all_modes ? Array_Modes::end : Array_Modes::blank_bytes;

    for (uint32_t m = (uint32_t)Array_Modes::start + 1; m < (uint32_t)mode_stop; ++m) {
        Array_Modes mode = (Array_Modes)m;

        total.merge(
            validate_radix_list<radix_test_types>::run(desc, iter, warm, mode)
        );
    }

    printf("\nDone... passed %u/%u tests.\n", total.passed, total.tests);
    return total.all_passed();
}


} // namespace rsort
