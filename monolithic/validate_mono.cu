// Run validation for monolithic library
// ./validate_mono --validation --iterations 1 --warmup 0
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../benchmark/bench_parser.h"

#define RSORT_USE_MONOLITHIC_HEADER
#define VALIDATION_TEST                 1
#include "../benchmark/validate_sort.cuh"


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
        bool ret = rsort::benchmark<rsort::Order::ascending, uint32_t, size_t>(
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
