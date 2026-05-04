// Compile:
// nvcc -O3 -std=c++17 -arch=sm_86 radix_gpu.cu bench_parser.cpp -o radix_gpu.exe

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

#include "radix_kernel.cuh"
#include "benchmark_sort.cuh"
#include "bench_parser.h"

// Template heavy, only enable if you want to run validation
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
        bool ret = rsort::benchmark<false, uint64_t>(
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

    return 0;
}
