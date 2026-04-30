// nvcc -O3 -std=c++17 -arch=sm_86 radix_gpu.cu bench_parser.cpp -o radix_gpu.exe
// Reorder kernel header

// Compare SASS agaisnt "sass.txt" baseline:
// (use fc instead of diff if on Windows)
//
// nvcc -O3 -std=c++17 -arch=sm_86 radix_gpu.cu bench_parser.cpp -o radix_gpu.exe &&
// cuobjdump --dump-sass radix_gpu.exe > sass1.txt &&
// diff sass.txt sass1.txt


#pragma once

// user knobs
#define USE_PSUM_SHARED         1
#define USE_CUDA_GRAPH_SORT     1
#define EXIT_EARLY_OPT          0

#include <cuda_runtime.h>
#include "radix/utils.cuh"
#include "radix/lookback.cuh"
#include "radix/histogram.cuh"
#include "radix/ranker.cuh"


namespace rsort {

// Portable workspace struct for post-sort checks
template <typename Key_T, typename Lookback_T, typename Len_T>
struct Sort_Workspace {
    Len_T num_blocks = 0;
    size_t lb_els = 0;
    size_t total_bytes = 0;
    size_t off_tmp = 0;
    size_t off_gp = 0;
    size_t off_counter = 0;
    size_t off_look_partial = 0;


    struct View {
        Key_T* tmp = nullptr;
        Lookback_T* gp = nullptr;
        uint32_t* counter = nullptr;
        Lookback_T* look_partial = nullptr;
    };


    // build workspace - set memory pointers according to policy 
    template <typename Lookback_Policy>
    static inline Sort_Workspace build(Len_T n) {
        using RT = radix_tuning<Key_T>;

        Sort_Workspace ws;
        ws.num_blocks = div_round_up<Len_T>(n, RT::SORT_BLOCK_SIZE);

        if constexpr (Lookback_Policy::reuse) {
            ws.lb_els = (size_t)ws.num_blocks * RT::RADIX_BIN_SIZE;
        } else {
            ws.lb_els = (size_t)RT::RADIX_PASSES * ws.num_blocks * RT::RADIX_BIN_SIZE;
        }

        size_t off = 0;
        ws.off_tmp = reserve_aligned(&off, (size_t)n * sizeof(Key_T), 256);
        ws.off_gp = reserve_aligned(&off, (size_t)RT::RADIX_PASSES * RT::RADIX_BIN_SIZE * sizeof(Lookback_T), 256);
        ws.off_counter = reserve_aligned(&off, sizeof(uint32_t), 64);
        ws.off_look_partial = reserve_aligned(&off, ws.lb_els * sizeof(Lookback_T), 256);
        ws.total_bytes = off;
        return ws;
    }


    View bind(uint8_t* base) const {
        View view;
        view.tmp = reinterpret_cast<Key_T*>(base + off_tmp);
        view.gp = reinterpret_cast<Lookback_T*>(base + off_gp);
        view.counter = reinterpret_cast<uint32_t*>(base + off_counter);
        view.look_partial = reinterpret_cast<Lookback_T*>(base + off_look_partial);
        return view;
    }
};


// scatter kernel with full block support and using simple storaging
template <bool Full_Block, bool Descending, typename Lookback_T, typename Key_T, typename Len_T, typename U>
__device__ __forceinline__ void scatter_staged(
    const U* __restrict__ staged,
    const Lookback_T* __restrict__ bin_offset,
    Key_T* __restrict__ out_keys,
    uint32_t actual_tile_items,
    uint32_t bit_location) {

        
    using RT = radix_tuning<Key_T>;
    constexpr uint32_t ITEMS_PER_THREAD = RT::REORDER_ITEMS_PER_THREAD;
    constexpr uint32_t REORDER_THREADS = RT::REORDER_THREADS;


    #pragma unroll
    for (int u = 0; u < ITEMS_PER_THREAD; ++u) {
        Len_T idx = threadIdx.x + (Len_T)u * REORDER_THREADS;
        if constexpr (Full_Block) {
            U key = staged[idx];
            U digit = extract_byte<U>(key, bit_location);
            out_keys[idx + (Len_T)bin_offset[digit]] = twiddle_out<Descending, Key_T>(key);
        } else {
            if (idx < (Len_T)actual_tile_items) {
                U key = staged[idx];
                U digit = extract_byte<U>(key, bit_location);
                out_keys[idx + (Len_T)bin_offset[digit]] = twiddle_out<Descending, Key_T>(key);
            }
        }
        __syncwarp(0xFFFFFFFFu);
    }
}


// Non-decoupled lookback path with early partial publication via chained callback
// The lookback also includes epoch bits to avoid cudamemset on every pass.
// Standard CUB ranking, implicit full blocking
template<bool Descending, typename Lookback_Policy, typename Key_T, typename Len_T>
__global__ __launch_bounds__(radix_tuning<Key_T>::REORDER_THREADS)
void onesweep_byte(
    const Key_T* __restrict__ in_keys,
    Key_T* __restrict__ out_keys,
    Len_T n,
    const typename Lookback_Policy::T* __restrict__ gp_sum_buffer,      // [RADIX_PASSES][RADIX_BIN_SIZE]
    volatile typename Lookback_Policy::T* __restrict__ lookback_partial, // [numBlocks * RADIX_BIN_SIZE]
    uint32_t iteration,
    typename Lookback_Policy::T lookback_epoch_bits) {
        
    
    using Lookback_T = typename Lookback_Policy::T;
    using LB = typename Lookback_Policy::conf;
    using U = get_unsigned_of<Key_T>;
    using Ranker = Radix_Ranker<Key_T>;
    
    using RT = radix_tuning<Key_T>;
    constexpr uint32_t SORT_BLOCK_SIZE = RT::SORT_BLOCK_SIZE;
    constexpr uint32_t LOGICAL_BLOCK_SIZE = RT::REORDER_LOGICAL_BLOCK_SIZE;
    constexpr uint32_t ITEMS_PER_THREAD = RT::REORDER_ITEMS_PER_THREAD;
    constexpr uint32_t ITEMS_PER_WARP = RT::REORDER_ITEMS_PER_WARP;
    constexpr uint32_t RADIX_BIN_SIZE = RT::RADIX_BIN_SIZE;
    constexpr uint32_t RADIX_BITS = RT::RADIX_BITS;
    
    static_assert(RADIX_BIN_SIZE == 256, "CUB-like kernel expects 8-bit radix.");


#if USE_PSUM_SHARED
    __shared__ Lookback_T p_sum[RADIX_BIN_SIZE];
#endif

    struct SMem {
        typename Ranker::Temp_Storage rank_temp;
        uint16_t bin_count[RADIX_BIN_SIZE];
        Lookback_T bin_offset[RADIX_BIN_SIZE];
        U staged[LOGICAL_BLOCK_SIZE];
    };
    __shared__ SMem smem;

    uint32_t block_index = (uint32_t)blockIdx.x;
    const int bin_owner = (int)threadIdx.x < (int)RADIX_BIN_SIZE;

    uint32_t bit_location = (uint32_t)RADIX_BITS * iteration;
    Len_T block_base = block_index * (Len_T)SORT_BLOCK_SIZE;
    const int full_block = (((Len_T)block_base + (Len_T)SORT_BLOCK_SIZE) <= (Len_T)n);

    uint32_t actual_tile_items = full_block ? SORT_BLOCK_SIZE : (uint32_t)(n - block_base);
    uint32_t invalid_items = LOGICAL_BLOCK_SIZE - actual_tile_items;

    U keys[ITEMS_PER_THREAD];
    int ranks[ITEMS_PER_THREAD]; // TODOs: why is this int? 

    int warp = (int)threadIdx.x / WARP_SIZE;
    int lane = (int)lane_id_u32();

    int exclusive_digit_prefix = 0;
    if (full_block) {
        Len_T warp_base = block_base + (Len_T)warp * ITEMS_PER_WARP;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            keys[i] = twiddle_in<Descending, Key_T>(in_keys[warp_base + lane + i * WARP_SIZE]);
        }

        Ranker::template match_early_counts<true, Descending, Lookback_Policy>(
            smem.rank_temp,
            keys,
            ranks,
            bit_location,
            exclusive_digit_prefix,
            smem.bin_count,
            lookback_partial,
            block_index,
            0u,
            lookback_epoch_bits
        );

    } else {
        //const uint64_t warp_base64 = (uint64_t)block_base + (uint64_t)warp * ITEMS_PER_WARP;
        const uint64_t warp_base64 = (uint64_t)block_base + (uint64_t)warp * ITEMS_PER_WARP;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            //uint32_t idx = warp_base + lane + i * WARP_SIZE;
            uint64_t idx = warp_base64 + (uint64_t)lane + (uint64_t)i * WARP_SIZE;
            
            // The filler key must map to the last radix bin in the current order.
            //keys[i] = twiddle_in<Descending, Key_T>(
            //    (idx < n) ? in_keys[idx] : tail_filler_key<Descending, Key_T>());
            keys[i] = (idx < (uint64_t)n)
                ? twiddle_in<Descending, Key_T>(in_keys[(Len_T)idx])
                : tail_filler_bits<Descending, Key_T>();
        }

        Ranker::template match_early_counts<false, Descending, Lookback_Policy>(
            smem.rank_temp,
            keys,
            ranks,
            bit_location,
            exclusive_digit_prefix,
            smem.bin_count,
            lookback_partial,
            block_index,
            invalid_items,
            lookback_epoch_bits
        );

    }
    //__syncthreads(); // taken out to not mess with the next 2 __sync's performance

    Lookback_T p = 0;
    if (bin_owner) {
        const uint32_t b = (uint32_t)threadIdx.x;
        const int p_ind = (int)((block_index << RADIX_BITS) + b);

#if USE_INLINE_LOOKBACK
        p = Lookback<Lookback_Policy>::lookback_prefix_old(
            lookback_partial, block_index, b, lookback_epoch_bits
        );
#else
        p = Lookback<Lookback_Policy>::lookback_prefix(
            lookback_partial + b, block_index, lookback_epoch_bits
        );
#endif

        lookback_partial[p_ind] = Lookback<Lookback_Policy>::pack_global(
            p + (uint32_t)smem.bin_count[b], lookback_epoch_bits);


        const Lookback_T* gp_row = gp_sum_buffer + (size_t)iteration * RADIX_BIN_SIZE;
        
        // Note: Performance wise, no real difference in computing psum in shared or not
#if USE_PSUM_SHARED
        p_sum[b] = gp_row[b] + p;
        smem.bin_offset[b] = p_sum[b] - (Lookback_T)exclusive_digit_prefix;
#else
        Lookback_T base = gp_row[b] + p;
        smem.bin_offset[b] = base - (Lookback_T)exclusive_digit_prefix;
#endif
    }
    __syncthreads();

    // Stage in local rank order.
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        smem.staged[ranks[k]] = keys[k];
    }
    __syncthreads();

    if (full_block) {
        scatter_staged<true, Descending, Lookback_T, Key_T, Len_T, U>(
            smem.staged, smem.bin_offset, out_keys, actual_tile_items, bit_location);
    } else {
        scatter_staged<false, Descending, Lookback_T, Key_T, Len_T, U>(
            smem.staged, smem.bin_offset, out_keys, actual_tile_items, bit_location);
    }
    //scatter_staged<false, Descending, Lookback_T, Key_T, Len_T, U>(smem.staged, smem.bin_offset, out_keys, actual_tile_items, bit_location);
}


// Generates the skip mask for early-exiting
template <typename Lookback_T, typename Key_T, typename Len_T>
static uint32_t compute_uniform_pass_skip_mask(const Lookback_T* d_gp, Len_T n) {
    
    using RT = radix_tuning<Key_T>;
    constexpr uint32_t RADIX_PASSES = RT::RADIX_PASSES;
    constexpr uint32_t RADIX_BIN_SIZE = RT::RADIX_BIN_SIZE;
    constexpr size_t HIST_ELEMS = (size_t)RADIX_PASSES * RADIX_BIN_SIZE;
    Lookback_T h_gp[HIST_ELEMS];
    cudaMemcpy(h_gp, d_gp, HIST_ELEMS * sizeof(Lookback_T), cudaMemcpyDeviceToHost);

    uint32_t skip_mask = 0u;
    for(uint32_t p = 0; p < RADIX_PASSES; ++p) {
        const Lookback_T* row = h_gp + (size_t)p * RADIX_BIN_SIZE;
        for(uint32_t b = 0; b < RADIX_BIN_SIZE; ++b) {
            
            Lookback_T lo = row[b];
            Lookback_T hi = (b + 1u < RADIX_BIN_SIZE) ? row[b + 1u] : (Lookback_T)n;
            if ((hi - lo) == (Lookback_T)n) {
                skip_mask |= (1u << p);
                break;
            }
            /*
            if(row[b] == n) {
                skip_mask |= (1u << p);
                break;
            }
            */
        }
    }

    return skip_mask;
}


// Enrty point of the sorting kernel
template <bool Descending, typename Lookback_Policy, typename Key_T, typename Len_T>
static void onesweep_byte_sort_enqueue(
    cudaStream_t stream,
    Key_T* d_inout,
    Key_T* d_tmp,
    typename Lookback_Policy::T* d_gp,
    uint32_t* d_counter,
    typename Lookback_Policy::T* d_look_partial,
    Len_T n,
    Len_T num_blocks,
    size_t lb_els) {


    using LB = typename Lookback_Policy::conf;
    using Lookback_T = typename Lookback_Policy::T;
    constexpr bool lb_epoch = Lookback_Policy::epoch;
    constexpr bool lb_reuse = Lookback_Policy::reuse;

    using RT = radix_tuning<Key_T>;
    constexpr uint32_t REORDER_THREADS = RT::REORDER_THREADS;
    constexpr uint32_t RADIX_PASSES = RT::RADIX_PASSES;

    using H = histogram_tuning;
    constexpr uint32_t HIST_BLOCKS = H::HIST_BLOCKS;
    constexpr uint32_t GHIST_THREADS = H::GHIST_THREADS;

    // conditionally call memsets according to lookback policy
    if constexpr(lb_reuse) {
        size_t initial_bytes = (size_t)((uint8_t*)(d_counter + 1) - (uint8_t*)d_gp);
        cudaMemsetAsync(d_gp, 0, initial_bytes, stream);
    } else if constexpr(!lb_epoch) {
        size_t initial_bytes = (size_t)((uint8_t*)(d_look_partial + lb_els) - (uint8_t*)d_gp);
        cudaMemsetAsync(d_gp, 0, initial_bytes, stream);
    }
    if constexpr(lb_epoch) {
        cudaMemsetAsync(d_look_partial, LB::EPOCH_TAG, lb_els * sizeof(Lookback_T), stream);
    }

    Key_T* in  = d_inout;
    Key_T* out = d_tmp;

    // Histogram call - template on dynamic work stealing (using lookback type for compile time branching)
    if constexpr (sizeof(Lookback_T) <= sizeof(uint32_t)) {
        GHistogram_8bits<Descending, Lookback_T, Key_T, Len_T, true>
        <<<HIST_BLOCKS, GHIST_THREADS, 0, stream>>>(d_inout, n, d_gp, 0u, d_counter);
    } else {
        uint32_t h_blocks = HIST_BLOCKS;
        Len_T hist_tiles = div_round_up<Len_T>(n, H::GHIST_ITEM_PER_BLOCK);
        if (h_blocks > hist_tiles) {
            h_blocks = hist_tiles;
        }
        GHistogram_8bits<Descending, Lookback_T, Key_T, Len_T, false>
        <<<h_blocks, GHIST_THREADS, 0, stream>>>(d_inout, n, d_gp, 0u, d_counter);
    }

    // build skip mask if doing early exit
#if EXIT_EARLY_OPT
    uint32_t skip_mask = compute_uniform_pass_skip_mask<Lookback_T, Key_T, Len_T>(d_gp, n);
#endif

    // passes loop
    for (uint32_t it = 0; it < RADIX_PASSES;) {

        // probably not the best pratice to define the end of a loop at the beginning
        auto iteration_end = [&]() {
            ++it;
            if constexpr(Lookback_Policy::epoch) {
                if (((it & LB::EPOCH_VALUE_MASK) == 0) && (it < RADIX_PASSES)) {
                    cudaMemsetAsync(d_look_partial, LB::EPOCH_TAG, lb_els * sizeof(Lookback_T), stream);
                }
            }
        };


#if EXIT_EARLY_OPT
        if (skip_mask & (1u << it)) {
            iteration_end();
            continue;
        }
#endif
        volatile Lookback_T* look_partial_pass = nullptr;

        // if epoch disabled memset, else pack bits 
        if constexpr(lb_reuse) {
            if constexpr(!lb_epoch) {
                cudaMemsetAsync(d_look_partial, 0, lb_els * sizeof(Lookback_T), stream);
            }
            look_partial_pass = d_look_partial;
        } else {
            Len_T tileBase = it * (num_blocks * RT::RADIX_BIN_SIZE);
            look_partial_pass = d_look_partial ? (d_look_partial + (size_t)tileBase) : nullptr;
        }
        Lookback_T lb_bits = 0;
        if constexpr(lb_epoch) {
            lb_bits = Lookback<Lookback_Policy>::pack_epoch(it & LB::EPOCH_VALUE_MASK);
        }

        // onesweep entry
        onesweep_byte<Descending, Lookback_Policy>
        <<<num_blocks, REORDER_THREADS, 0, stream>>>(in, out, n, d_gp, look_partial_pass, it, lb_bits);

        // ping-pong buffers
        Key_T* tmp = in;
        in = out;
        out = tmp;

        iteration_end();
    }

    // copy if not in dest (odd number of passes)
    if (in != d_inout) {
        cudaMemcpyAsync(d_inout, in, (size_t)n * sizeof(Key_T), cudaMemcpyDeviceToDevice, stream);
    }
}


// Sorting 
template <bool Descending, typename Lookback_Policy, typename Key_T, typename Len_T>
static void onesweep_byte_sort_impl(
    Key_T* d_inout,
    size_t* temp_bytes,
    uint8_t* d_workspace, 
    Len_T n,
    cudaStream_t stream = 0) {


    using LB = typename Lookback_Policy::conf;
    using Lookback_T = typename Lookback_Policy::T;
    using Workspace = Sort_Workspace<Key_T, Lookback_T, Len_T>;


    Workspace ws = Workspace::template build<Lookback_Policy>(n);

    // return if just asking for size, like CUB
    if (d_workspace == nullptr) {
        *temp_bytes = ws.total_bytes;
        return;
    }

    // Continue to entry point
    auto view = ws.bind(d_workspace);
    onesweep_byte_sort_enqueue<Descending, Lookback_Policy, Key_T, Len_T>(
        stream, d_inout, view.tmp, view.gp, view.counter,
        view.look_partial, n, ws.num_blocks, ws.lb_els
    );
}


// Policy swtich
template<bool Descending, typename Key_T, typename Len_T>
static inline void looback_policy_enforcer(
    Key_T* d_inout, size_t* temp_bytes, uint8_t* d_workspace, Len_T n, cudaStream_t stream = 0) {
    
    Lookback_Modes mode = get_lookback_mode(n); // according to array size

    // template heavy
    switch(mode) {
        case Lookback_Modes::u32_epoch:
            onesweep_byte_sort_impl<Descending, Faster_LB_Policy, Key_T, Len_T>(
                d_inout, temp_bytes, d_workspace, n, stream);
            break;
        case Lookback_Modes::u32_plain:
            onesweep_byte_sort_impl<Descending, Fast_LB_Policy, Key_T, Len_T>(
                d_inout, temp_bytes, d_workspace, n, stream);
            break;
        case Lookback_Modes::u64_epoch:
            onesweep_byte_sort_impl<Descending, General_LB_Policy, Key_T, Len_T>(
                d_inout, temp_bytes, d_workspace, n, stream);
            break;
        default:
            break;
    }
}


// Wrap for the sorting call:
// Builds the CUDA graph for the sort if defined
// Performance gains are not significant 
template<bool Descending, typename Key_T, typename Len_T>
static inline void onesweep_byte_sort_wrap(
    Key_T* d_inout, size_t* temp_bytes, uint8_t* d_workspace, Len_T n) {
    cudaStream_t capture_stream = nullptr;
#if USE_CUDA_GRAPH_SORT && !EXIT_EARLY_OPT
    if (d_workspace != nullptr) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaStream_t stream = 0;

        cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);

        cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
        looback_policy_enforcer<Descending, Key_T, Len_T>(d_inout, temp_bytes, d_workspace, n, capture_stream);
        cudaStreamEndCapture(capture_stream, &graph);

        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        cudaGraphLaunch(exec, stream);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(capture_stream);
        return;
    }
#endif
    looback_policy_enforcer<Descending, Key_T, Len_T>(d_inout, temp_bytes, d_workspace, n, capture_stream);
}


// onesweep_byte_sort_wrap<false, Key_T, Len_T>(d_inout, temp_bytes, d_workspace, n);
// however, size_t is faster in benchmarking
template<typename Key_T, typename Len_T>
static inline void onesweep_byte_sort(
    Key_T* d_inout, size_t* temp_bytes, uint8_t* d_workspace, Len_T n) {
    onesweep_byte_sort_wrap<false, Key_T, size_t>(d_inout, temp_bytes, d_workspace, n);
}


template<typename Key_T, typename Len_T>
static inline void onesweep_byte_sort_descending(
    Key_T* d_inout, size_t* temp_bytes, uint8_t* d_workspace, Len_T n) {
    onesweep_byte_sort_wrap<true, Key_T, size_t>(d_inout, temp_bytes, d_workspace, n);
}

} //namespace rsort
