// nvcc -O3 -std=c++17 -arch=sm_86 radix_gpu.cu bench_parser.cpp -o radix_gpu.exe
#pragma once

#include <cstdint>
#include <cstddef>


namespace parse {

struct Global_Config {
    size_t n = 1 << 27;
    uint32_t iterations = 30;
    uint32_t warmups = 10;
    bool descending = false;
    bool validation = false;
};

void print_usage(const char* exe_name);
template<typename T> bool parse_u(const char* s, T* out_value);
bool args(int argc, char** argv, Global_Config* conf);

} // namespace parse

