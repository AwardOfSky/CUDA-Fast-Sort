// Compile:
// nvcc -O3 -std=c++17 -arch=sm_86 example.cu -o rsort.exe
#include "rsort.cuh"

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

    return 0;
}
