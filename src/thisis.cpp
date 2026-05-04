// g++ -Wall -O3 thisis.cpp -o thisis

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

#define WARMUP 0

using Clock = std::chrono::high_resolution_clock;

int main() {
    const size_t N = 1 << 20; // ~67M elements (adjust if needed)

    std::vector<uint32_t> a(N);

    // Random generator
    std::mt19937 rng(123);
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

    for (size_t i = 0; i < N; i++) {
        a[i] = dist(rng);
    }

    // Warmup (optional)
    if (WARMUP) {
        auto tmp = a;
        std::sort(tmp.begin(), tmp.end());
    }

    // Timing
    auto start = Clock::now();
    std::sort(a.begin(), a.end());
    auto end = Clock::now();

    std::chrono::duration<double> elapsed = end - start;

    double elems_per_sec = N / elapsed.count();

    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Time:\t\t" << elapsed.count() * 1000.0 << " ms\n";
    std::cout << "Throughput:\t" << elems_per_sec << " el/s\n";

    return 0;
}