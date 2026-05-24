/* 
    This example contains the command line parser
    and options to interact with the program outside the API
    These are not included in the monolithic header.
    Feel free to use them in your tests/examples.
*/

// Compile:
// nvcc -O3 -std=c++17 -arch=sm_86 example.cu -o rsort(.exe)
#include "rsort.cuh"


// define API examples
#define API_EXAMPLE     1
#define API_KV_SORT     0
#define API_KV_DESC     0


// Benchmark configuration struct
// Struct could be given as a parameter, but better to have
// individual explicit arguments
struct Global_Config {
    size_t n = size_t(1) << 27;
    uint32_t iterations = 30; // 30
    uint32_t warmups = 10; // 20
    uint32_t warm_ms = 250; // 250 ms
    bool descending = false;
    bool validation = false;
};

void print_usage(const char* exe_name);
template<typename T> bool parse_u(const char* s, T* out_value);
bool args(int argc, char** argv, Global_Config* conf);


int main(int argc, char** argv) {
    // bench_parser.h for default values
    Global_Config conf{};

    // simple arg parser
    if (!args(argc, argv, &conf)) {
        printf("\n");
        print_usage(argv[0]);
        return 1;
    }

    // Benchmark example
    if (!API_EXAMPLE) {
        if (!conf.validation) {
            bool ret = rsort::benchmark<false, uint32_t>(
                conf.n,
                conf.iterations,
                conf.warmups,
                conf.warm_ms,
                rsort::Array_Modes::random
            );
        // Validation example
        } else {
            bool ret = rsort::validate(true, true, conf.iterations, conf.warmups);
        }
    } else { 

        // API example adapted from rsort::benchmark function
        // Go check it if you want better insight.  

        using Key_T = uint32_t;
        using Len_T = size_t;
        using Value_T = std::conditional_t<API_KV_SORT, uint32_t, rsort::no_value_t>;


        // initialization
        size_t n = conf.n;
        size_t temp_bytes = 0;
        Key_T* d_keys = nullptr;
        [[maybe_unused]] Value_T* d_vals = nullptr;
        uint8_t* d_workspace = nullptr;
        rsort::Array_Modes arr_mode = rsort::Array_Modes::random;
        uint32_t seed = rsort::set_seed_radix(0);

        // sorting launch
        auto launch_sorting_kernel = [&]() {
            if constexpr (API_KV_SORT) {
                if constexpr (API_KV_DESC) {
                    rsort::onesweep_byte_sort_pairs_descending<Key_T, Len_T>(d_keys, d_vals, &temp_bytes, d_workspace, n);
                } else {
                    rsort::onesweep_byte_sort_pairs<Key_T, Len_T>(d_keys, d_vals, &temp_bytes, d_workspace, n);
                }
            } else {
                if constexpr (API_KV_DESC) {
                    rsort::onesweep_byte_sort_descending<Key_T, Len_T>(d_keys, &temp_bytes, d_workspace, n);
                } else {
                    rsort::onesweep_byte_sort<Key_T, Len_T>(d_keys, &temp_bytes, d_workspace, n);
                }
            }
        };

        // memory allocations
        CHECK_CUDA(cudaMalloc(&d_keys, sizeof(Key_T) * n));    
        if constexpr (API_KV_SORT) {
            CHECK_CUDA(cudaMalloc(&d_vals, align_up_power<sizeof(uint32_t)>(sizeof(Value_T) * n))); // 
        }
        launch_sorting_kernel();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMalloc(&d_workspace, temp_bytes));

        // array initialization
        rsort::init_keys<Key_T, Len_T><<<div_round_up<Len_T>(n, 256), 256>>>(d_keys, n, seed, arr_mode);
        if constexpr (API_KV_SORT) {
            rsort::init_keys<uint32_t, Len_T><<<div_round_up<Len_T>(n, 256), 256>>>(
                (uint32_t *)d_vals,
                align_up_power<sizeof(uint32_t)>(sizeof(Value_T) * n) / sizeof(uint32_t),
                seed,
                arr_mode
            );
        }
        CHECK_CUDA(cudaGetLastError());

        // timers
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // sorting
        CHECK_CUDA(cudaEventRecord(start));
        launch_sorting_kernel();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

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
            API_KV_DESC ? "DESC" : "ASC"
        );

        // cleanup
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_workspace));
        CHECK_CUDA(cudaFree(d_keys));
        if constexpr (API_KV_SORT) {
            CHECK_CUDA(cudaFree(d_vals));
        }
    }

    return 0;
}


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
