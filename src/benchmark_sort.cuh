// Benchmark header
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include "radix_kernel.cuh"


#define BENCH_DEBUG     0


namespace rsort {

template <typename Lookback_T, typename Len_T>
static bool check_hist_buffers(
    Len_T n,
    const Lookback_T* h_hist_ref,
    const Lookback_T* h_hist_chk,
    const uint32_t RADIX_PASSES,
    const uint32_t RADIX_BIN_SIZE
) {
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
                printf(
                    "digit-count mismatch at pass %u bin %u: input=%llu output=%llu\n",
                    p,
                    b,
                    (unsigned long long)ref_count,
                    (unsigned long long)chk_count
                );
                return false;
                break; // double
            }
        }
    }
    return true;
}


// Reconstructs the digit histogram for the last pass and checks count of each radix
// in the sorted array to assert if sorted output has the same counts.
template <
    bool Descending,
    typename Lookback_Policy,
    typename Workspace,
    typename Key_T,
    typename Len_T,
    typename Value_T = no_value_t,
    bool Is_Long_Double = false
>
static bool verify_digit_histograms(
    Key_T* d_sorted, // Key_T_tentative
    uint8_t* d_workspace,
    Len_T n,
    uint32_t seed,
    Array_Modes arr_mode,
    Workspace* ws,
    bool validation,
    bool timing = false
) {


    using Lookback_T = typename Lookback_Policy::T;
    //using Lookback_T = typename ws::LT;
    using RT = radix_tuning<Key_T, Value_T>;
    constexpr uint32_t RADIX_PASSES = RT::RADIX_PASSES;
    constexpr uint32_t RADIX_BIN_SIZE = RT::RADIX_BIN_SIZE;
    constexpr size_t HIST_ELEMS = (size_t)RADIX_PASSES * RADIX_BIN_SIZE;
    using H = histogram_tuning;
    constexpr uint32_t HIST_BLOCKS = H::HIST_BLOCKS;
    constexpr uint32_t GHIST_THREADS = H::GHIST_THREADS;
    

    // timers
    cudaEvent_t start, stop;
    if (timing) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    // Declare buffers
    Lookback_T* d_hist_ref = nullptr;
    Lookback_T* d_hist_chk = nullptr;
    uint32_t* d_counter = nullptr;
    Lookback_T h_hist_ref[HIST_ELEMS];
    Lookback_T h_hist_chk[HIST_ELEMS];

    // Initialize buffers
    CHECK_CUDA(cudaMalloc(&d_hist_ref, HIST_ELEMS * sizeof(Lookback_T)));
    CHECK_CUDA(cudaMalloc(&d_hist_chk, HIST_ELEMS * sizeof(Lookback_T)));
    CHECK_CUDA(cudaMalloc(&d_counter, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_hist_ref, 0, HIST_ELEMS * sizeof(Lookback_T)));
    CHECK_CUDA(cudaMemset(d_hist_chk, 0, HIST_ELEMS * sizeof(Lookback_T)));
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(uint32_t)));

    auto workspace_view = ws->bind(d_workspace);
    if (!ws->init_keys) {
        init_keys<Key_T, Len_T, Is_Long_Double><<<div_round_up<Len_T>(n, 256), 256>>>(
            workspace_view.tmp, n, seed, arr_mode
        );
        CHECK_CUDA(cudaGetLastError());
        ws->init_keys = true;
    }
    
    // Count digits from both arrays (original and sorted)
    GHistogram_8bits<Descending, Lookback_T, Key_T, Len_T, Is_Long_Double>
    <<<HIST_BLOCKS, GHIST_THREADS>>>(workspace_view.tmp, n, d_hist_ref, 0u, d_counter);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemset(d_counter, 0, sizeof(uint32_t)));
    GHistogram_8bits<Descending, Lookback_T, Key_T, Len_T, Is_Long_Double>
    <<<HIST_BLOCKS, GHIST_THREADS>>>(d_sorted, n, d_hist_chk, 0u, d_counter);
    CHECK_CUDA(cudaGetLastError());

    // Copy to CPU
    CHECK_CUDA(cudaMemcpy(
        h_hist_ref, d_hist_ref, HIST_ELEMS * sizeof(Lookback_T), cudaMemcpyDeviceToHost)
    );
    CHECK_CUDA(cudaMemcpy(
        h_hist_chk, d_hist_chk, HIST_ELEMS * sizeof(Lookback_T), cudaMemcpyDeviceToHost)
    );

    // Free CUDA buffers
    CHECK_CUDA(cudaFree(d_counter));
    CHECK_CUDA(cudaFree(d_hist_chk));
    CHECK_CUDA(cudaFree(d_hist_ref));

    // Check if both histogram count buffers are equal
    bool h_ok = check_hist_buffers<Lookback_T>(
        n, h_hist_ref, h_hist_chk, RADIX_PASSES, RADIX_BIN_SIZE
    );

    // stop timing
    float ms_total;
    if (timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        ms_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        CHECK_CUDA(cudaGetLastError());
    }

    // print stats
    if (!validation) {
        if (timing) {
            printf("digit counts ok? %s (%.3f ms)\n", h_ok ? "YES" : "NO", ms_total);
        } else {
            printf("digit counts ok? %s\n", h_ok ? "YES" : "NO");
        }
    }

    return h_ok;
}


template <
    bool Descending,
    typename Workspace,
    typename Key_T,
    typename Len_T,
    typename Value_T,
    bool Is_Long_Double = false
>
static bool verify_kv_pairings(
    Key_T* d_sorted_keys,
    Value_T* d_sorted_vals,
    uint8_t* d_workspace,
    Len_T n,
    uint32_t seed,
    Array_Modes arr_mode,
    Workspace *ws,
    bool validation,
    bool timing = false
) {

    // timers
    cudaEvent_t start, stop;
    if (timing) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    // init original keys and vals
    auto workspace_view = ws->bind(d_workspace);
    if (!ws->init_keys) {
        init_keys<Key_T, Len_T, Is_Long_Double><<<div_round_up<Len_T>(n, 256), 256>>>(
            workspace_view.tmp, n, seed, arr_mode
        );
        CHECK_CUDA(cudaGetLastError());
        ws->init_keys = true;
    }
    if (!ws->init_vals) {
        init_keys<Value_T, Len_T><<<div_round_up<Len_T>(n, 256), 256>>>(
            workspace_view.tmp_vals,
            n,
            seed,
            Array_Modes::asc
        );
        CHECK_CUDA(cudaGetLastError());
        ws->init_vals = true;
    }

    // check actual pairings
    int* d_ok = nullptr;
    int temp = 1;
    CHECK_CUDA(cudaMalloc(&d_ok, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_ok, &temp, sizeof(int), cudaMemcpyHostToDevice));
    Len_T check_blocks = div_round_up<Len_T>(n, 256);
    check_pairings<Descending, Key_T, Len_T, Value_T, Is_Long_Double><<<check_blocks, 256>>>(
        workspace_view.tmp,
        workspace_view.tmp_vals,
        d_sorted_keys,
        d_sorted_vals,
        n, 
        d_ok);
    int h_ok = 0;
    CHECK_CUDA(cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_ok));

    // stop timing
    float ms_total;
    if (timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        ms_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        CHECK_CUDA(cudaGetLastError());
    }

    // print stats
    if (!validation) {
        if (timing) {
            printf("pairs counts ok? %s (%.3f ms)\n", h_ok ? "YES" : "NO", ms_total);
        } else {
            printf("pairs counts ok? %s\n", h_ok ? "YES" : "NO");
        }
    }

    return h_ok; 
}


template<
    bool Descending,
    typename Key_T,
    typename Len_T,
    bool Is_Long_Double = false
>
static bool verify_order(
    Key_T* d_keys,
    Len_T n,
    bool validation,
    bool timing = false
) {

    // timers
    cudaEvent_t start, stop;
    if (timing) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    // initialize data
    int* d_ok = nullptr;
    int temp = 1;
    CHECK_CUDA(cudaMalloc(&d_ok, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_ok, &temp, sizeof(int), cudaMemcpyHostToDevice));

    // call verify kernel
    Len_T check_blocks = div_round_up<Len_T>(n, 256);
    check_sorted<Descending, Key_T, Is_Long_Double><<<check_blocks, 256>>>(
        d_keys, n, d_ok
    );
    CHECK_CUDA(cudaGetLastError());
    
    // result
    int h_ok = 0;
    CHECK_CUDA(cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_ok));

    // stop timing
    float ms_total;
    if (timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        ms_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        CHECK_CUDA(cudaGetLastError());
    }

    // print stats
    if (!validation) {
        if (timing) {
            printf("\nsorted ok? %s (%.3f ms)\n", h_ok ? "YES" : "NO", ms_total);
        } else {
            printf("\nsorted ok? %s\n", h_ok ? "YES" : "NO");
        }
    }

    return h_ok;
}


template <typename T, typename Len_T>
static bool can_validate_pair(bool validation, Len_T n) {
    return validation && 
        ((sizeof(T) == 8) || 
        ((sizeof(T) == 4) && (n < std::numeric_limits<uint32_t>::max())));
}


template<
    bool Descending,
    typename Lookback_Policy,
    typename Key_T,
    typename Len_T,
    typename Value_T,
    bool Short_Mode,
    bool Is_Long_Double = false
>
static bool verify_sorted_policy(
    Key_T* d_keys,
    Value_T* d_vals,
    uint8_t* d_workspace,
    Len_T n,
    uint32_t seed,
    Array_Modes arr_mode,
    bool validation
) {


    constexpr bool SORTING_PAIRS = !std::is_same_v<Value_T, no_value_t>;
    using Lookback_T = typename Lookback_Policy::T;
    using Workspace = Sort_Workspace<Key_T, Lookback_T, Len_T, Value_T, Short_Mode>;

    
    // build workspace
    Workspace ws = Workspace::template build<Lookback_Policy>(n);

    // verify order
    bool order_ok = verify_order<
        Descending, Key_T, Len_T, Is_Long_Double
    >(d_keys, n, validation, true);

    // verify histogram
    bool hist_ok = verify_digit_histograms<
        Descending, Lookback_Policy, Workspace, Key_T, Len_T, Value_T, Is_Long_Double
    >(d_keys, d_workspace, n, seed, arr_mode, &ws, validation, false);

    // verify pairings
    bool pairs_ok = true;
    // stupid, but the second check is redeundant, only there for compilation purposes 
    if constexpr (SORTING_PAIRS && (sizeof(Value_T) <= 8)) {
        if (can_validate_pair<Value_T, Len_T>(validation, n)) {
            pairs_ok = verify_kv_pairings<
                Descending, Workspace, Key_T, Len_T, Value_T, Is_Long_Double
            >(d_keys, d_vals, d_workspace, n, seed, arr_mode, &ws, validation, false);
        }
    }

    bool ret = order_ok && hist_ok && pairs_ok;
    if(!ret) {
        printf("Post sort checks [order, hist, pair]: [%d, %d, %d]\n", order_ok, hist_ok, pairs_ok);
    }

    return ret;
}


template<
    bool Descending,
    typename Key_T,
    typename Len_T,
    typename Value_T = no_value_t,
    bool Is_Long_Double = false
>
static bool verify_sorted(
    Key_T* d_keys,
    Value_T* d_vals,
    uint8_t* d_workspace,
    Len_T n,
    uint32_t seed,
    Array_Modes arr_mode,
    bool validation
) {

    if (n <= LOW_N) {
        return verify_sorted_policy<
            Descending,
            Faster_LB_Policy,
            Key_T,
            Len_T,
            Value_T,
            true,
            Is_Long_Double
        >(d_keys, d_vals, d_workspace, n, seed, arr_mode, validation);
    }

    Lookback_Modes mode = get_lookback_mode(n); 

    switch (mode) {
        case Lookback_Modes::u32_epoch:
            return verify_sorted_policy<
                Descending,
                Faster_LB_Policy,
                Key_T,
                Len_T,
                Value_T,
                false,
                Is_Long_Double
            >(d_keys, d_vals, d_workspace, n, seed, arr_mode, validation);

        case Lookback_Modes::u32_plain:
            return verify_sorted_policy<
                Descending,
                Fast_LB_Policy,
                Key_T,
                Len_T,
                Value_T,
                false,
                Is_Long_Double
            >(d_keys, d_vals, d_workspace, n, seed, arr_mode, validation);

        case Lookback_Modes::u64_epoch: default:
            return verify_sorted_policy<
                Descending,
                General_LB_Policy,
                Key_T,
                Len_T,
                Value_T,
                false,
                Is_Long_Double
            >(d_keys, d_vals, d_workspace, n, seed, arr_mode, validation);
    }
}


template<
    bool Descending,
    typename Key_T,
    typename Len_T,
    typename Value_T
>
struct Benchmark_Context {


    // aliases
    using ull_t = long long unsigned int;
    using RT_default = radix_tuning<Key_T, Value_T>;
    static constexpr uint32_t RADIX_BITS = RT_default::RADIX_BITS;
    static constexpr bool SORTING_PAIRS = RT_default::sorting_pairs;


    // benchmark tuning
    uint32_t SORT_BLOCK_SIZE;
    uint32_t REORDER_THREADS;
    uint32_t REORDER_ITEMS_PER_THREAD;

    // CUDA long double support
    using Key_T_Sort = native_128bit_support::try_valid_long_double_t<Key_T>;
    static constexpr bool is_ld = native_128bit_support::is_valid_long_double<Key_T>();

    // var declaration - config
    Len_T n;
    uint32_t iters;
    uint32_t warmups;
    uint32_t warm_ms;
    bool validation;
    Array_Modes arr_mode;
    uint32_t warmups_done = 0;

    // sort buffers
    Key_T* d_keys = nullptr;
    Value_T* d_vals = nullptr;
    uint8_t* d_workspace = nullptr;
    Key_T_Sort* d_keys_sort = nullptr; // used to get long double identity
    
    // helpers
    size_t temp_bytes = 0;
    size_t vals_bytes = 0;
    std::string key_name;
    std::string_view val_name;

    // CUDA specific vars
    cudaEvent_t start{};
    cudaEvent_t stop{};
    int device = 0;
    cudaDeviceProp prop;


    struct Timings {
        double  ms_avg = 0.0;
        double  ms_acc = 0.0;
        long long els = 0;
        double psel = 0.0;
        double us_std = 0.0;

        // don't feel like including another header for this
        double *d = nullptr;

        void init(uint32_t iters) {
            d = (double *)malloc(sizeof(double) * iters);
        }

        void calculate(uint32_t iters, Len_T n) {
            ms_avg = ms_acc / (double)iters;
            els = (long long)((1000.0 / ms_avg) * (double)n);
            psel = 1'000'000'000'000.0 / (double)els;
            us_std = stdev(d, ms_avg, (size_t)iters) * 1000.0;
        }

        void cleanup() {
            free(d);
        }

        ~Timings() {
            cleanup();
        }

    } timings;


    void init_bench_tuning() {
        if (n <= LOW_N) {
            using RT = radix_tuning<Key_T, Value_T, true>;
            SORT_BLOCK_SIZE = RT::SORT_BLOCK_SIZE;
            REORDER_THREADS = RT::REORDER_THREADS;
            REORDER_ITEMS_PER_THREAD = RT::REORDER_ITEMS_PER_THREAD;
        } else {
            using RT = radix_tuning<Key_T, Value_T, false>;
            SORT_BLOCK_SIZE = RT::SORT_BLOCK_SIZE;
            REORDER_THREADS = RT::REORDER_THREADS;
            REORDER_ITEMS_PER_THREAD = RT::REORDER_ITEMS_PER_THREAD;
        }    
    }


    void launch_sorting_kernel() {
        if constexpr (SORTING_PAIRS) {
            if constexpr (Descending) {
                onesweep_byte_sort_pairs_descending<Key_T, Len_T, Value_T>(
                    d_keys, d_vals, &temp_bytes, d_workspace, n
                );
            } else {
                onesweep_byte_sort_pairs<Key_T, Len_T, Value_T>(
                    d_keys, d_vals, &temp_bytes, d_workspace, n
                );
            }
        } else {
            if constexpr (Descending) {
                onesweep_byte_sort_descending<Key_T, Len_T>(
                    d_keys, &temp_bytes, d_workspace, n
                );
            } else {
                onesweep_byte_sort<Key_T, Len_T>(
                    d_keys, &temp_bytes, d_workspace, n
                );
            }
        }
    }


    bool allocate_buffers() {
        // alloc keys and vals
        vals_bytes = 0;
        CHECK_CUDA(cudaMalloc(&d_keys, sizeof(Key_T) * n));    
        if constexpr (SORTING_PAIRS) {
            // align values up n bytes 
            vals_bytes = align_up_power<sizeof(uint32_t)>(sizeof(Value_T) * n);
            CHECK_CUDA(cudaMalloc(&d_vals, vals_bytes));
        }
        launch_sorting_kernel();

        // optional debug
        if constexpr (BENCH_DEBUG) {
            printf(
                "n: %llu, key size: %llu, val size: %llu, temp_bytes requested: %llu\n",
                (ull_t)n,
                (ull_t)sizeof(Key_T) * n,
                (ull_t)(SORTING_PAIRS ? vals_bytes : 0),
                (ull_t)temp_bytes
            );
        }

        // alloc workspace
        CHECK_CUDA(cudaMalloc(&d_workspace, temp_bytes));
        CHECK_CUDA(cudaGetLastError());

        // transform keys into unsigned identity to support long doubles
        d_keys_sort = reinterpret_cast<Key_T_Sort*>(d_keys);

        return true;
    }


    void create_timer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }


    void timed_sort() {
        cudaEventRecord(start);
        launch_sorting_kernel();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }


    // define sorting iteration
    float run_timed_iteration(uint32_t seed) {
        
        // init keys and values
        init_keys<Key_T_Sort, Len_T, is_ld><<<div_round_up<Len_T>(n, 256), 256>>>(
            d_keys_sort, n, seed, arr_mode
        );
        if constexpr (SORTING_PAIRS) {

            if (can_validate_pair<Value_T, Len_T>(validation, n)) {
                init_keys<Value_T, Len_T><<<div_round_up<Len_T>(n, 256), 256>>>(
                    d_vals,
                    n,
                    seed,
                    Array_Modes::asc
                );
            } else {
                uint32_t* d_vals_sort = reinterpret_cast<uint32_t*>(d_vals);
                init_keys<uint32_t, Len_T><<<div_round_up<Len_T>(n, 256), 256>>>(
                    d_vals_sort,
                    vals_bytes / sizeof(uint32_t),
                    seed,
                    arr_mode
                );
            }

        }
        CHECK_CUDA(cudaGetLastError());

        // run actual sort
        timed_sort();

        // return time elpased
        float ms_total = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        return ms_total;
    };

    
    // warmup
    uint32_t warmup(uint32_t init_seed = 0) {
        double warm_ms_passed = 0.0;
        warmups_done = 0;
        uint32_t seed_counter = init_seed;
        while (
            (warmups_done < warmups + init_seed) || 
            (!validation && (warm_ms_passed < (double)warm_ms))
        ) {
            warm_ms_passed += run_timed_iteration(set_seed_radix(seed_counter++));
            ++warmups_done;
        }
        return seed_counter;
    }


    // benchmark runs
    double bench(uint32_t seed_counter) {
        timings.init(iters);
        timings.ms_acc = 0.0;
        for (uint32_t i = 0; i < iters; ++i) {
            double temp = run_timed_iteration(set_seed_radix(seed_counter++));
            timings.ms_acc += temp;
            timings.d[i] = temp;
        }
        return seed_counter;
    }

    
    // Device properties
    void init_dev_properties() {
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
    }


    void init_strings() {
        key_name = std::string(type_name<Key_T>());
        if constexpr (std::is_same_v<Key_T, long double>) {
            key_name += " (" + std::to_string(sizeof(Key_T) * 8) + " bits)";
        }
        val_name = type_name<Value_T>();
    }


    // print main stats
    void print_stats() {
        
        if (!validation) {
            printf(
                "\nGPU: %s\n"
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
                "Order\t=\t%s\n"
                "Pairs\t=\t%.*s\n",

                prop.name,
                RADIX_BITS,
                (ull_t)n,
                (int)key_name.size(), key_name.data(),
                (uint32_t)div_round_up<Len_T>(n, SORT_BLOCK_SIZE),
                REORDER_THREADS,
                REORDER_ITEMS_PER_THREAD,
                (double)temp_bytes / (1024. * 1024.),
                timings.ms_avg, timings.us_std,
                iters, warmups_done,
                timings.els,
                timings.psel,
                arr_modes_to_string(arr_mode),
                Descending ? "DESC" : "ASC",
                SORTING_PAIRS ? (int)val_name.size() : 2,
                SORTING_PAIRS ? val_name.data() : "No"
            );
            print_lookback_policy(get_lookback_mode(n));
        }
    }
    

    void print_validation_stats(bool test_valid) {
        if(validation) {
            // verification stats
            std::string pair_str = "Pair: " + std::string(val_name) + ",";
            printf(
                "Sorting %llu els "
                "of type %.*s, %.*s (%.3f MB)"
                "(%s), %s array - %.3f ms"
                "(%d avg + %d wm)... %s\n",

                (ull_t)n,
                (int)key_name.size(), key_name.data(),
                SORTING_PAIRS ? (int)pair_str.size() : 0,
                SORTING_PAIRS ? pair_str.data() : "",
                (double)temp_bytes / (1024. * 1024.),
                Descending ? "DESC" : "ASC",
                arr_modes_to_string(arr_mode),
                timings.ms_avg, iters, warmups_done,
                test_valid ? "passed" : "failed"
            );
        }
    }


    // post sort verification
    bool bench_valid(uint32_t seed_counter) {
        uint32_t last_seed = (uint32_t)(set_seed_radix(seed_counter - 1));
        bool valid = verify_sorted<Descending, Key_T_Sort, Len_T, Value_T, is_ld>(
            d_keys_sort, d_vals, d_workspace, n, last_seed, arr_mode, validation
        );
        return valid;
    }


    void cleanup () {
        // cleanup
        if (start)  {
            CHECK_CUDA(cudaEventDestroy(start));
        }
        if (stop)  {
            CHECK_CUDA(cudaEventDestroy(stop));
        }
        if (d_workspace) {
            CHECK_CUDA(cudaFree(d_workspace));
        }
        if (d_keys) {
            CHECK_CUDA(cudaFree(d_keys));
        }
        if constexpr (SORTING_PAIRS) {
            if (d_vals) {
                CHECK_CUDA(cudaFree(d_vals));
            }
        }
    }


    ~Benchmark_Context() {
        cleanup();
    }
};


template<
    bool Descending,
    typename Key_T,
    typename Len_T,
    typename Value_T = no_value_t
>
int benchmark(
    Len_T n,
    uint32_t iters,
    uint32_t warmups,
    uint32_t warm_ms,
    Array_Modes arr_mode,
    bool validation = false
) {

    Benchmark_Context<Descending, Key_T, Len_T, Value_T> ctx{
        .n = n,
        .iters = iters,
        .warmups = warmups,
        .warm_ms = warm_ms,
        .validation = validation,
        .arr_mode = arr_mode,
    };
    
    ctx.init_bench_tuning();
    ctx.init_dev_properties();
    ctx.create_timer();
    ctx.init_strings();
    
    ctx.allocate_buffers();
    
    uint32_t seed = ctx.warmup();
    seed = ctx.bench(seed);

    ctx.timings.calculate(iters, n);
    ctx.print_stats();
    bool passed = ctx.bench_valid(seed);
    ctx.print_validation_stats(passed);

    return passed;
};

} // namespace rsort
