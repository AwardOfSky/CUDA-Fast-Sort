// Benchmark header
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "radix_kernel.cuh"


namespace rsort {

// Reconstructs the digit histogram for the last pass and checks count of each radix
// in the sorted array to assert if sorted output has the same counts.
template <bool Descending, typename Lookback_Policy, typename Key_T, typename Len_T, typename Value_T>
static bool verify_digit_histograms_preserved(
    const Key_T* d_sorted,
    uint8_t* d_workspace,
    Len_T n,
    uint32_t seed,
    Array_Modes arr_mode) {


    using Lookback_T = typename Lookback_Policy::T;
    using RT = radix_tuning<Key_T, Value_T>;
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

    using Workspace = Sort_Workspace<Key_T, Lookback_T, Len_T, Value_T>;
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
template <bool Descending, typename Key_T, typename Len_T, typename Value_T = no_value_t>
static bool verify_hist_by_mode(
    Lookback_Modes mode,
    Array_Modes arr_mode,
    const Key_T* d_keys,
    uint8_t* d_workspace,
    Len_T n,
    uint32_t seed)  {

    switch (mode) {
        case Lookback_Modes::u32_epoch:
            return verify_digit_histograms_preserved<Descending, Faster_LB_Policy, Key_T, Len_T, Value_T>(
                d_keys, d_workspace, n, seed, arr_mode);
        case Lookback_Modes::u32_plain:
            return verify_digit_histograms_preserved<Descending, Fast_LB_Policy, Key_T, Len_T, Value_T>(
                d_keys, d_workspace, n, seed, arr_mode);
        case Lookback_Modes::u64_epoch:
            return verify_digit_histograms_preserved<Descending, General_LB_Policy, Key_T, Len_T, Value_T>(
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
            onesweep_byte_sort_descending<T, Len_T>(d_keys, &temp_bytes, d_workspace, n);
        } else {
            onesweep_byte_sort<T, Len_T>(d_keys, &temp_bytes, d_workspace, n);
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
        hist_ok = verify_hist_by_mode<true, T, Len_T>(
            mode, arr_mode, d_keys, d_workspace, n, last_seed);
    } else {
        hist_ok = verify_hist_by_mode<false, T, Len_T>(
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

} // namespace rsort
