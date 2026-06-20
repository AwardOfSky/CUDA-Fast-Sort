/* 
    Tests sort with simple random array of 2 ^ 27
    unsigned integer elements in ascending order

    Compile:
    nvcc -O3 -std=c++17 -arch=sm_86 -I../monolithic example_basic.cu -o example_basic(.exe)
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <rsort.cuh>

namespace rd = rsort::detail;

int main(int argc, char** argv) {

    // vars and buffers
    size_t n = size_t(1) << 27;
    size_t temp_bytes = 0;
    uint32_t* d_keys = nullptr;
    uint32_t seed = rd::set_seed_radix(0);
    uint8_t* d_workspace = nullptr;

    // memory allocations
    CHECK_CUDA(cudaMalloc(&d_keys, sizeof(uint32_t) * n));
    rsort::sort(d_keys, &temp_bytes, d_workspace, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMalloc(&d_workspace, temp_bytes));
    rd::init_keys<<<rd::div_round_up<size_t>(n, 256), 256>>>(
        d_keys, n, seed, rd::Array_Modes::random
    );
    CHECK_CUDA(cudaGetLastError());

    // timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // timed sorting (only 1 iteration)
    cudaEventRecord(start);
    rsort::sort(d_keys, &temp_bytes, d_workspace, n);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
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
        "random",
        "ASC"
    );

    // cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_keys));
    return 0;
}
