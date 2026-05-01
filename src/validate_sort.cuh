// Validation header
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "benchmark_sort.cuh"


#ifndef VALIDATION_TEST
#define VALIDATION_TEST     1
#endif 


// ===================== Validation Types =====================
#define U32_TYPE    uint32_t,
#define UINT_TYPES  U32_TYPE uint8_t, uint16_t, uint64_t,
#define INT_TYPES   UINT_TYPES int16_t, int8_t, int32_t, int64_t,
#define FP_TYPES    float, double,
#define ALL_TYPES   INT_TYPES FP_TYPES

// Change types to test HERE!!!
#define TYPE_SET_TEST ALL_TYPES
// ============================================================


namespace rsort {

// in ms 
#define COOL_BETWEEN_RUNS   0.0

template<typename... Ts> struct type_list {};

using radix_test_types = type_list<
#if VALIDATION_TEST
    TYPE_SET_TEST
#endif
>;


// Simple counter struct for validation tests
struct Validation_Result {
    uint32_t tests = 0;
    uint32_t passed = 0;

    constexpr void add(bool ok) {
        ++tests;
        passed += ok ? 1u : 0u;
    }

    constexpr void merge(const Validation_Result& other) {
        tests += other.tests;
        passed += other.passed;
    }

    constexpr bool all_passed() const {
        return tests == passed;
    }
};


// Benchmark entry point of the validation
template<typename T>
Validation_Result validate_radix_type(
    bool descending,
    int iter,
    int warm,
    Array_Modes mode,
    uint32_t vram_gb) {

    Validation_Result result;
    uint32_t bit_end = radix_type_traits<T>::max_array_bits(vram_gb);
    uint32_t bit_start = bit_end - 10;

    for (int i = (int)bit_start; i <= (int)bit_end; ++i) {
        
        uint64_t n = 1ull << i;
        int pass_ok;
        if (descending) {
            pass_ok = benchmark<true, T>(n, iter, warm, 0, mode, true);
        } else {
            pass_ok = benchmark<false, T>(n, iter, warm, 0, mode, true);
        }
        result.add(pass_ok);

        if (!pass_ok) {
            std::string_view sv = type_name<T>();
            printf("Failed test with: n = 2^%d, Order = %s, mode = %s, type = %.*s\n",
                i,
                descending ? "DESC" : "ASC",
                arr_modes_to_string(mode),
                (int)sv.size(), sv.data()
            );
        }

        sleep_ms(COOL_BETWEEN_RUNS);
    }

    return result;
}


// Validate accoring to list of types
template<typename... Ts>
Validation_Result validate_radix_wrap(
    bool desc,
    int iter,
    int warm,
    Array_Modes mode) {

    Validation_Result total;

    if constexpr (sizeof...(Ts) > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        uint64_t vram_bytes = prop.totalGlobalMem;
        uint32_t vram_gb = vram_bytes >> 30;

        (total.merge(
                validate_radix_type<Ts>(false, iter, warm, mode, vram_gb)
            ), ...
        );
        if (desc) {
            (total.merge(
                    validate_radix_type<Ts>(true, iter, warm, mode, vram_gb)
                ), ...
            );
        }
    }

    return total;
}


template<typename List> struct validate_radix_list;

template<typename... Ts>
struct validate_radix_list<type_list<Ts...>> {
    static Validation_Result run(
        bool desc,
        int iter,
        int warm,
        Array_Modes mode) {
        return validate_radix_wrap<Ts...>(desc, iter, warm, mode);
    }
};


// validation entry point
bool validate(bool all_modes, bool desc, int iter, int warm) {
    Validation_Result total;

    printf("\nStarting validation tests... (%d iterations, %d warmup runs each)\n\n", iter, warm);
    Array_Modes mode_stop = all_modes ? Array_Modes::end : Array_Modes::blank_bytes;

    for (uint32_t m = (uint32_t)Array_Modes::start + 1; m < (uint32_t)mode_stop; ++m) {
        Array_Modes mode = (Array_Modes)m;

        total.merge(
            validate_radix_list<radix_test_types>::run(desc, iter, warm, mode)
        );
    }

    printf("\nDone... passed %u/%u tests.\n", total.passed, total.tests);
    return total.all_passed();
}

} // namespace rsort