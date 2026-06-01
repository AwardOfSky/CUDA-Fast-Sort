// Validation header
// example: ./radix_gpu --validation --iterations 1 --warmup 0
#pragma once


#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <string>

#include "benchmark_sort.cuh"

// User knobs (check validation types below)
#define PAIR_VALIDATION         1
#define LONG_DOUBLE_VALIDATION  1
#define U128_BIT_VALIDATION     1
// in ms 
#define COOL_BETWEEN_RUNS       0.0

#if defined(__SIZEOF_INT128__) && U128_BIT_VALIDATION 
    #define NATIVE_U128_TOKEN   native_128bit_support::native_u128,
    #define NATIVE_I128_TOKEN   native_128bit_support::native_i128,
#else
    #define NATIVE_U128_TOKEN
    #define NATIVE_I128_TOKEN
#endif

#if defined(__SIZEOF_INT128__) && LONG_DOUBLE_VALIDATION
    #define LONG_DOUBLE_TOKEN   long double, 
#else
    #define LONG_DOUBLE_TOKEN
#endif

// ======================= Validation Types =======================
#define U32_TYPE    uint32_t,
#define UINT_TYPES  U32_TYPE uint8_t, uint16_t, uint64_t, NATIVE_U128_TOKEN
#define INT_TYPES   UINT_TYPES int16_t, int8_t, int32_t, int64_t, NATIVE_I128_TOKEN
#define FP_TYPES    float, double, LONG_DOUBLE_TOKEN
#define ALL_TYPES   INT_TYPES FP_TYPES

// Change types to test HERE!!!
#define TYPE_SET_TEST ALL_TYPES
// ================================================================

// TODOs: #ifdef VALIDATION_TEST include lib
#ifndef VALIDATION_TEST
#define VALIDATION_TEST     1
#endif 


namespace rsort {

template<typename... Ts> struct type_list {};

// Types to validate (keys)
using radix_test_types = type_list<
#if VALIDATION_TEST
    TYPE_SET_TEST
#endif
>;

// Validate different sized values for each key type
using kv_types = type_list<
    no_value_t,
#if PAIR_VALIDATION
    UINT_TYPES
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
template<typename Key_T, typename Value_T = no_value_t>
Validation_Result validate_radix_type(
    bool descending,
    int iter,
    int warm,
    Array_Modes mode,
    uint32_t vram_gb) {

    Validation_Result result;
    constexpr bool sorting_pairs = !std::is_same_v<Value_T, no_value_t>;
    uint32_t size_el = sizeof(Key_T) + (sorting_pairs ? sizeof(Value_T) : 0); 
    uint32_t bit_end = max_array_bits(vram_gb, size_el);
    uint32_t bit_start = bit_end - 10;

    // small array tests indices [1, small_tests]
    const int small_tests = 3;
    static const uint64_t pow10[] = {1ull, 10ull, 100ull, 1000ull};

    for (int i = (int)bit_start - small_tests; i <= (int)bit_end; ++i) {
        
        uint64_t n = (i < bit_start) ? (pow10[bit_start - i]) : (1ull << i);
        int pass_ok;

        if (descending) {
            pass_ok = benchmark<true, Key_T, uint64_t, Value_T>(n, iter, warm, 0, mode, true);
        } else {
            pass_ok = benchmark<false, Key_T, uint64_t, Value_T>(n, iter, warm, 0, mode, true);
        }

        result.add(pass_ok);

        if (!pass_ok) {
            std::string_view sv = type_name<Key_T>();
            std::string_view svt_temp = type_name<Value_T>();
            std::string pair_str = "Pair: " + std::string(svt_temp);

            printf(
                "Failed test with: n = 2^%d, Order = %s, mode = %s, type = %.*s, %.*s\n",
                i,
                descending ? "DESC" : "ASC",
                arr_modes_to_string(mode),
                (int)sv.size(), sv.data(),
                sorting_pairs ? (int)pair_str.size() : 0,
                sorting_pairs ? pair_str.data() : ""
            );
        }

        sleep_ms(COOL_BETWEEN_RUNS);
    }

    return result;
}


template<typename Key_T, typename Value_List>
struct validate_kv_pair_list;

template<typename Key_T, typename... Value_Ts>
struct validate_kv_pair_list<Key_T, type_list<Value_Ts...>> {
    static Validation_Result run(
        bool descending,
        int iter,
        int warm,
        Array_Modes mode,
        uint32_t vram_gb)
    {
        Validation_Result total;
        (total.merge(
            validate_radix_type<Key_T, Value_Ts>(descending, iter, warm, mode, vram_gb)
        ), ...);
        return total;
    }
};


// Validate accoring to list of types
template<typename... Key_Ts>
Validation_Result validate_radix_wrap(
    bool desc,
    int iter,
    int warm,
    Array_Modes mode) {

    Validation_Result total;

    if constexpr (sizeof...(Key_Ts) > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        uint32_t vram_gb = prop.totalGlobalMem >> 30;

        (total.merge(
            validate_kv_pair_list<Key_Ts, kv_types>::run(
                false, iter, warm, mode, vram_gb
            )
        ), ...);
        if (desc) {
            (total.merge(
                validate_kv_pair_list<Key_Ts, kv_types>::run(
                    true, iter, warm, mode, vram_gb
                )
            ), ...);
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
