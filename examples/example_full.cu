/* 
    Full example to test different sorting configurations. 

    Compile:
    nvcc -O3 -std=c++17 -arch=sm_86 -I../monolithic example_full.cu -o example_full(.exe)
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <rsort.cuh>


// Test different configurations and APIs
#define DESCENDING  0
#define KV_PAIRS    0
using Key_T         = uint32_t; // type of key
using Val_T         = uint32_t; // type of val
using Len_T         = size_t;   // type of n


namespace rd = rsort::detail;


int main(int argc, char** argv) {

    // API example adapted from rsort::benchmark function
    // Go check it if you want better insight.  
    using Value_T = std::conditional_t<KV_PAIRS, Val_T, rd::no_value_t>;

    // initialization
    size_t n = size_t(1) << 27; // size_t(1) << 27 
    size_t temp_bytes = 0;
    Key_T* d_keys = nullptr;
    [[maybe_unused]] Value_T* d_vals = nullptr;
    uint8_t* d_workspace = nullptr;
    rd::Array_Modes arr_mode = rd::Array_Modes::random;
    uint32_t seed = rd::set_seed_radix(0);
    
    // CUDA long double support
    using Key_T_Sort = rd::native_128bit_support::try_valid_long_double_t<Key_T>;
    static constexpr bool is_ld = rd::native_128bit_support::is_valid_long_double<Key_T>();
    Key_T_Sort* d_keys_sort = nullptr; // used to get long double identity

    // sorting launch
    auto launch_sorting_kernel = [&]() {
        if constexpr (KV_PAIRS) {
            if constexpr (DESCENDING) {
                rsort::sort_pairs_descending<Key_T, Len_T>(d_keys, d_vals, &temp_bytes, d_workspace, n);
            } else {
                rsort::sort_pairs<Key_T, Len_T>(d_keys, d_vals, &temp_bytes, d_workspace, n);
            }
        } else {
            if constexpr (DESCENDING) {
                rsort::sort_descending<Key_T, Len_T>(d_keys, &temp_bytes, d_workspace, n);
            } else {
                rsort::sort<Key_T, Len_T>(d_keys, &temp_bytes, d_workspace, n);
            }
        }
    };

    // memory allocations
    size_t vals_bytes = 0;
    CHECK_CUDA(cudaMalloc(&d_keys, sizeof(Key_T) * n));    
    if constexpr (KV_PAIRS) {
        // align values up n bytes 
        vals_bytes = rd::align_up_power<sizeof(uint32_t)>(sizeof(Value_T) * n);
        CHECK_CUDA(cudaMalloc(&d_vals, vals_bytes));
    }
    launch_sorting_kernel();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMalloc(&d_workspace, temp_bytes));

    // transform keys into unsigned identity to support long doubles
    d_keys_sort = reinterpret_cast<Key_T_Sort*>(d_keys);

    // init keys and values
    rd::init_keys<Key_T_Sort, Len_T, is_ld><<<rd::div_round_up<Len_T>(n, 256), 256>>>(
        d_keys_sort, n, seed, arr_mode
    );
    if constexpr (KV_PAIRS) {
        uint32_t* d_vals_sort = reinterpret_cast<uint32_t*>(d_vals);
        rd::init_keys<uint32_t, Len_T><<<rd::div_round_up<Len_T>(n, 256), 256>>>(
            d_vals_sort,
            vals_bytes / sizeof(uint32_t),
            seed,
            arr_mode
        );
    }
    CHECK_CUDA(cudaGetLastError());

    // timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // timed sorting (only 1 iteration!)
    cudaEventRecord(start);
    
    launch_sorting_kernel();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));

    // stats
    printf(
        "N\t=\t%llu\n"
        "Time\t=\t%.3f ms\n"
        "Mode\t=\t%s\n"
        "Order\t=\t%s\n",

        (long long unsigned int)n,
        ms_total,
        arr_modes_to_string(arr_mode),
        DESCENDING ? "DESC" : "ASC"
    );

    // cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_keys));
    if constexpr (KV_PAIRS) {
        CHECK_CUDA(cudaFree(d_vals));
    }

    return 0;
}
