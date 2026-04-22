// nvcc -O3 -std=c++17 -arch=sm_86 radix_gpu.cu bench_parser.cpp -o radix_gpu.exe
#include "bench_parser.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>


namespace parse {

void print_usage(const char* exe_name) {
    printf(
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --n <value>            Number of elements\n"
        "  --iterations <value>   Timed iterations\n"
        "  --warmup <value>       Warmup iterations\n"
        "  --descending           Sort descending\n"
        "  --validation           Validate algorithm\n"
        "  --help                 Show this help\n",
        exe_name
    );
}


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
        }
        else {
            printf("Error: unknown argument: %s\n", arg);
            return false;
        }
    }

    return true;
}

} //namespace bench
