/*
    Validation suite was run both in Windows (MSVC) and Linux (gcc)
        Linux: 6552/6552 tests passed (signed and unsigned 128-bit integers)
        Windows: 4200/4200 tests passed
    Compilation tested with both -std=c++17 and -std=c++20 standards
    Validation is template HEAVY! Only enable if you want to run validation.

    Compile:
        nvcc -O3 -std=c++17 -arch=sm_86 -I../include radix_gpu.cu bench_parser.cpp -o rsort(.exe)
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
        bool ret = rsort::benchmark<rsort::Order::ascending, uint64_t, size_t, uint64_t>(
            conf.n,
            conf.iterations,
            conf.warmups,
            conf.warm_ms,
            rsort::Array_Modes::random // byte_skip
        );
    // Validation example
    } else {
        bool ret = rsort::validate(true, true, conf.iterations, conf.warmups);
    }

    return 0;
}
