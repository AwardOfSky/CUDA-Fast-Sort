// Parser header
#pragma once

#include <cstdint>
#include <cstddef>


namespace parse {

// Benchmark configuration members.
// Struct could be given as a parameter, but better to have
// individual explicit arguments
struct Global_Config {
    size_t n = size_t(1) << 27;
    uint32_t iterations = 30; // 30
    uint32_t warmups = 10; // 10
    uint32_t warm_ms = 250; // 250 ms
    bool descending = false;
    bool validation = false;
};

void print_usage(const char* exe_name);
template<typename T> bool parse_u(const char* s, T* out_value);
bool args(int argc, char** argv, Global_Config* conf);

} // namespace parse

