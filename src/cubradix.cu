// nvcc -O3 -std=c++17 -arch=sm_86 cubradix.cu bench_parser.cpp -o cub_radix(.exe)
// - lineinfo optional

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <time.h>

#include <cub/device/device_radix_sort.cuh>

#include "radix/utils.cuh"
#include "bench_parser.h"


template<
    bool Descending,
    typename T,
    typename Len_T,
    typename Value_T = rsort::no_value_t
>
int benchmark_cub(
Len_T n,
uint32_t iters,
uint32_t warmups,
uint32_t warm_ms,
rsort::Array_Modes arr_mode,
bool validation = false);


int main(int argc, char** argv) {
    parse::Global_Config conf{};

    // simple arg parser
    if (!parse::args(argc, argv, &conf)) {
        printf("\n");
        parse::print_usage(argv[0]);
        return 1;
    }

    // Benchmark example
    bool ret = benchmark_cub<(bool)rsort::Order::ascending, uint32_t, size_t>(
        conf.n,
        conf.iterations,
        conf.warmups,
        conf.warm_ms,
        rsort::Array_Modes::random
    );

    return 0;
}


// based on rsort's benchmarking template
template<bool Descending, typename T, typename Len_T, typename Value_T>
int benchmark_cub(
    Len_T n,
    uint32_t iters,
    uint32_t warmups,
    uint32_t warm_ms,
    rsort::Array_Modes arr_mode,
    bool validation) {

    using RT = rsort::radix_tuning<T, Value_T>;
    static constexpr bool SORTING_PAIRS = RT::sorting_pairs;

    size_t temp_bytes = 0;
    T* d_keys = nullptr;
    T* d_out = nullptr;
    uint8_t* d_workspace = nullptr;
    [[maybe_unused]] Value_T* d_vals = nullptr;
    [[maybe_unused]] Value_T* d_vals_out = nullptr;
    
    auto launch_sorting_kernel = [&]() {
        if constexpr (SORTING_PAIRS) {
            if constexpr (Descending) {
                cub::DeviceRadixSort::SortPairsDescending(
                    d_workspace, temp_bytes, d_keys, d_out, d_vals, d_vals_out, n
                );
            } else {
                cub::DeviceRadixSort::SortPairs(
                    d_workspace, temp_bytes, d_keys, d_out, d_vals, d_vals_out, n
                );
            }
        } else {
            if constexpr (Descending) {
                cub::DeviceRadixSort::SortKeysDescending(d_workspace, temp_bytes, d_keys, d_out, n);
            } else {
                cub::DeviceRadixSort::SortKeys(d_workspace, temp_bytes, d_keys, d_out, n);
            }
        }
    };

    // do not time memory allocations
    CHECK_CUDA(cudaMalloc(&d_keys, sizeof(T) * n));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(T) * n));
    if constexpr (SORTING_PAIRS) {
        size_t vals_bytes = align_up_power<sizeof(uint32_t)>(size_t(sizeof(Value_T)) * size_t(n));
        CHECK_CUDA(cudaMalloc(&d_vals, vals_bytes));
        CHECK_CUDA(cudaMalloc(&d_vals_out, vals_bytes));
    }
    launch_sorting_kernel();
    CHECK_CUDA(cudaMalloc(&d_workspace, temp_bytes));
    CHECK_CUDA(cudaGetLastError());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto run_timed_iteration = [&](uint32_t seed) {
        rsort::init_keys<T, Len_T><<<div_round_up(n, (Len_T)256), 256>>>(
            d_keys, n, seed, arr_mode
        );
        if constexpr (SORTING_PAIRS) {
            rsort::init_keys<uint32_t, Len_T><<<div_round_up(n, (Len_T)256), 256>>>(
                (uint32_t*)d_vals,
                align_up_power<sizeof(uint32_t)>(size_t(sizeof(Value_T)) * size_t(n)) / sizeof(uint32_t),
                seed,
                arr_mode
            );
        }
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
        warm_ms_passed += run_timed_iteration(rsort::set_seed_radix(seed_counter++));
        ++warmups_done;
    }

    // benchmark runs
    double ms_acc = 0.0;
    double* timings = (double*)malloc(sizeof(*timings) * iters);
    for (uint32_t i = 0; i < iters; ++i) {
        double temp = run_timed_iteration(rsort::set_seed_radix(seed_counter++));
        ms_acc += temp;
        timings[i] = temp;
    }

    // calculate stats
    double ms_avg = ms_acc / (double)iters;
    long long els = (long long)((1000.0 / ms_avg) * (double)n);
    double psel = 1'000'000'000'000.0 / (double)els;
    double us_std = rsort::stdev(timings, ms_avg, (size_t)iters) * 1000.0;
    free(timings);

    // Device properties
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::string_view sv_temp = type_name<T>();
    std::string_view svt_temp = type_name<Value_T>();
    if (!validation) {
        printf("\nGPU: %s\n"
            "\nRADIX\t=\t%s\n"
            "N\t=\t%lld\n"
            "Type\t=\t%.*s\n"
            "Memory\t=\t%.3f MB\n"
            "Time\t=\t%.3f ms (+- %.3f us)\n"
            "\t\t(%d avg + %d wm)\n"
            "it/s\t=\t%lld\n"
            "ps/el\t=\t%.3f\n"
            "Mode\t=\t%s\n"
            "Order\t=\t%s\n"
            "Pairs\t=\t%.*s\n",

            prop.name,
            "CUB",
            (long long unsigned int)n,
            (int)sv_temp.size(), sv_temp.data(),
            (double)temp_bytes / (1024. * 1024.),
            ms_avg, us_std,
            iters, warmups_done,
            els,
            psel,
            arr_modes_to_string(arr_mode),
            Descending ? "DESC" : "ASC",
            SORTING_PAIRS ? (int)svt_temp.size() : 2,
            SORTING_PAIRS ? svt_temp.data() : "No"
        );
    } else {
        std::string pair_str = "Pair: " + std::string(svt_temp) + " ,";
        printf(
            "\nSorting %lld els of type %.*s, %.*s (%s), %s array - %.3f ms (%d avg + %d wm).\n",
            (long long unsigned int)n,
            (int)sv_temp.size(), sv_temp.data(),
            SORTING_PAIRS ? (int)pair_str.size() : 0,
            SORTING_PAIRS ? pair_str.data() : "",
            Descending ? "DESC" : "ASC",
            arr_modes_to_string(arr_mode),
            ms_avg, iters, warmups_done
        );
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_keys));
    if constexpr (SORTING_PAIRS) {
        CHECK_CUDA(cudaFree(d_vals_out));
        CHECK_CUDA(cudaFree(d_vals));
    }

    return 1;
}
