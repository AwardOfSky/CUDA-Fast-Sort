// Public API header
#pragma once

#include <cstddef>
#include <cstdint>
#include "detail/radix_kernel.cuh"


namespace rsort {

// onesweep_byte_sort_wrap<false, Key_T, Len_T>
// however, size_t is faster in benchmarking
template<typename Key_T, typename Len_T>
inline void sort(
    Key_T* d_inout,
    size_t* temp_bytes,
    uint8_t* d_workspace,
    Len_T n,
    cudaStream_t capture_stream = nullptr
) {
    
    detail::onesweep_byte_sort_wrap<false, Key_T, size_t, detail::no_value_t>(
        d_inout,
        temp_bytes,
        d_workspace,
        n,
        nullptr,
        capture_stream
    );
}


template<typename Key_T, typename Len_T>
inline void sort_descending(
    Key_T* d_inout,
    size_t* temp_bytes,
    uint8_t* d_workspace,
    Len_T n,
    cudaStream_t capture_stream = nullptr
) {
    
    detail::onesweep_byte_sort_wrap<true, Key_T, size_t, detail::no_value_t>(
        d_inout,
        temp_bytes,
        d_workspace,
        n,
        nullptr,
        capture_stream
    );
}


template<typename Key_T, typename Len_T, typename Value_T>
inline void sort_pairs(
    Key_T* d_inout,
    Value_T* d_inout_values,
    size_t* temp_bytes,
    uint8_t* d_workspace,
    Len_T n,
    cudaStream_t capture_stream = nullptr
) {
    
    detail::onesweep_byte_sort_wrap<false, Key_T, size_t, Value_T>(
        d_inout,
        temp_bytes,
        d_workspace,
        n,
        d_inout_values,
        capture_stream
    );
}


template<typename Key_T, typename Len_T, typename Value_T>
inline void sort_pairs_descending(
    Key_T* d_inout,
    Value_T* d_inout_values,
    size_t* temp_bytes,
    uint8_t* d_workspace,
    Len_T n,
    cudaStream_t capture_stream = nullptr
) {
    
    detail::onesweep_byte_sort_wrap<true, Key_T, size_t, Value_T>(
        d_inout,
        temp_bytes, 
        d_workspace,
        n,
        d_inout_values,
        capture_stream
    );
}


/*
    Note:

    It would be interesting to have API support for AoS input,
    and do the conversion to SoA automatically.
    But will have to wait for reflection to get to NVCC.
    Check the "experimental.h" header in radix directory.
*/


} // namespace rsort