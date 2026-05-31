/*
    Validation suite was run both in Windows (MSVC) and Linux (gcc)
        Linux: 6552/6552 tests passed (signed and unsigned 128-bit integers)
        Windows: N/N tests passed
    Compilation tested with both -std=c++17 and -std=c++20 standards
    Validation is template HEAVY! Only enable if you want to run validation.

    Compile:
        nvcc -O3 -std=c++17 -arch=sm_86 radix_gpu.cu bench_parser.cpp -o radix_gpu.exe
    Example: 
        ./radix_gpu --n 10000000 --iterations 30
    Validation:
        ./radix_gpu --validation --iterations 1 --warmup 0

    TODOs:
    - Change C-style pointers to C++ style
    - AoS API (when nvcc gets reflection)
*/


#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "radix_kernel.cuh"
#include "benchmark_sort.cuh"
#include "bench_parser.h"

// vvv HEAVY! Only enable if you want to run validation.
#define VALIDATION_TEST     0
#include "validate_sort.cuh"


int main(int argc, char** argv) {

    // bench_parser.h for default values
    parse::Global_Config conf{};

    // simple arg parser
    if (!parse::args(argc, argv, &conf)) {
        printf("\n");
        parse::print_usage(argv[0]);
        return 1;
    }

    // Benchmark example
    if (!conf.validation) {
        bool ret = rsort::benchmark<false, uint32_t, size_t>(
            conf.n,
            conf.iterations,
            conf.warmups,
            conf.warm_ms,
            rsort::Array_Modes::random // blank_bytes
        );
    // Validation example
    } else {
        bool ret = rsort::validate(true, true, conf.iterations, conf.warmups);
    }

    return 0;
}
