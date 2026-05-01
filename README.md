# Fast CUDA Radix Sort


A re-implementation of NVIDIA's CUB onesweep Radix Sort algorithm.

Initially motivated by [this article](https://gpuopen.com/learn/boosting_gpu_radix_sort/), this project started as a small educational endeavour to get familiar with CUDA and delve deeper into C++.
The original goal was simply to build a radix sort for the GPU from scratch, without relying on third-party kernels.

However, as my understanding of CUDA and parallel programming improved, I began exploring ways to make the implementation more efficient, eventually leading me to CUB’s own implementation as a better reference point.

As it stands, this project serves both as a re-implementation and as a proof of concept for additional optimizations such as lookback epoch bits and early-exiting, without loss of generality.

Validation tests and benchmarks are included to corroborate correctness and performance against CUB, but the primary focus of the project remains on expressing the algorithms and optimizations implemented, as well as their trade-offs.


## Overview

The implementation is largely based on CUB's DeviceRadixSort:

- Stable
- 8-bit radix
- A priori global histogram counts
- Onesweep design (per pass: read once, write once)
- Coupled lookback chain with early publication and epoch bits
- Warp-level design

Therefore, most of the implementation is, effectively, a simplified translation of these techniques/principles.

## Key Features

### The algorithm currently implements:

- keys of all the standard types <sup>1</sup>
- "semi"-arbitrary array sizes <sup>2</sup>
- ascend, descend
- limited support for 128-bit keys (only integrals for now)

<sup>1</sup> Support for unsigned types is done through the twiddling in and out of unsigned types of the same size.
This is the same idea CUB uses. Support for descending sorting is achieved using the same principle: by simply inverting the bit representation during reordering.

<sup>2</sup> Arrays with a number of elements larger than 32 bits are supported. However, in practice, each local histogram counter of each block can only count up to 32 bits. In my testing, making the atomic counter 64-bit hurt performance significantly. Still, it would take trillions of elements with the same exact key for overflow to happen, even at 32 bits.

### Assumptions and missing features (different from CUB):

From a **proof-of-concept** perspective, some features were intentionally left out:

- No key-value pair support <sup>1</sup>
- No short circuiting (see section below)
- Input array is replaced with sorted output <sup>2</sup>

<sup>1</sup> Key-value support would follow a SoA approach, preserving the current key-array layout. Conceptually, this extends the scatter stage so that key moves are mirrored over the corresponding values. While this feature should be required for a production-ready sort, it is mostly independent from the optimizations already performed, so it's left for future work.

<sup>2</sup> This was mainly done to test larger arrays. Unlike CUB, which preserves the input and writes to a separate output buffer, this implementation uses a standard two-buffer ping-pong scheme. This simplifies the pipeline and avoids additional buffer management, at the performance cost of potentially overwriting the input (if an odd number of passes is performed, a final extra array copy is required, as far as I understand, CUB performs no copies).


## Optimizations:

### Lookback epoch bits


As far as improving performance is concerned, this optimization does a lot of the heavy lifting because it removes a full lookback buffer clear on most passes and for most scenarios.

The idea is to tag lookback entries with epoch/pass bits so the same lookback buffer can be reused across passes without having to be cleared every time. cudaMemset is cheap, but clearing, sometimes MBs of data every pass, quickly adds up, and the clear must complete before we do any more sorting work.
The lookback path is a great place to pay this extra bookkeeping cost as it is already latency-tolerant: each entry may have to wait for previous blocks to publish, which has the potential to amortize some bit logic.

In addition to the partial/global publish bits, this implementation stores epoch bits in each lookback entry. Instead of testing if ```entry == 0``` to check for an invalid state, the epoch-tagged path treats an entry as invalid when its epoch bits do not match the current pass: ```(entry & EPOCH_MASK) != pass```.  When we write both partial and global publications, we also pack the current pass in the epoch bits, so later passes ignore stale entries from earlier passes.

The epoch check might be slightly more expensive than a zero check in isolation, but in practice, the chained waiting pattern of the lookback hides most of that cost. In testing, the main downside was that the extra bookkeeping seems to make the ```__nanosleep()```'ing backoff slightly less effective, _i.e._ threads seem to sleep for fewer cycles.

We still need to initialize the lookback buffer before the first use of a given epoch value, but we can fill it with a dedicated epoch-tag pattern with 0 for the publication bits and non-zero for the epoch bits (as that is the number of the pass). That way, a matching epoch with no partial/global bits set is immediately recognized as not yet published.
The pseudocode (for the lookback) is as follows:

```python
memset lookback buffer with epoch-tag pattern
for each pass:

    ...
    publish partial state with current epoch
    ...
    wait for valid lookback state by checking epoch bits
    while lookback state is partial:
        go back a block
        wait for valid lookback state by checking epoch bits
        accumulate partial lookback
    publish global state with current epoch
    ...

    if epoch wraps:
        memset lookback buffer with epoch-tag pattern
```

With the current 32-bit epoch-tagged policy, 2 bits are reserved for epoch state (this is easily modifiable), leaving 28 value bits in the least significant part of the representation. This is the main trade-off for this optimization. In the current policy, epoch tagging is disabled for array sizes above 2<sup>28</sup>-1, and plain 32-bit lookback is used until 2<sup>30</sup>-1.
Moreover, for key sizes larger than 4 bytes, the buffer is cleared every 4th pass (excluding the last one).

For larger arrays (more than 2<sup>30</sup> elements), the implementation switches to a 64-bit lookback, with 6 epoch bits and 2 publish bits, leaving 56 bits for the actual counter. Although this doubles the lookback memory usage, sorting at this scale already requires gigabytes of VRAM, even for 8-bit key types, so the additional memory footprint isn't great in comparison. Besides, saving a couple of MBs and switching to a multi-group lookback design with worse performance didn't look like a good trade-off, especially since we are not clearing this buffer every pass anymore.

### Early-exiting

It makes sense to skip sorting work if all radices for a pass fall into the same bin.

CUB implements this idea at the block/CTA level using _short-circuiting_. Instead of checking for all radices, each block independently detects uniformity and effectively fast-forwards to the scatter stage, copying keys directly to the destination buffer.
 
The _early-exiting_ approach implemented here is much simpler and naive.
A pass is skipped entirely if all keys fall into a single radix bin, similar to what would be done in a serial radix sort (I didn't know better at the time 😄). On the GPU, detecting this condition efficiently is less straightforward.
However, we have already built a histogram for all passes before sorting. The idea is to take advantage of this computation and transfer this data to the CPU, where a compact _skip mask_ is generated (one bit per pass). Albeit simple, this approach proved to be a good value relative to its low computational cost (see benchmarking section).

The main drawback of this technique is probabilistic: it is significantly less likely for all radices in the entire array to fall into a single bin than for keys within a single block to do so. This means that, on average, _early-exiting_ triggers less frequently than block-level _short-circuiting_. 
However, when it does trigger, it skips an entire pass, leading to more substantial speedups (the onesweep kernel is not called, so no GPU work is performed for that pass, aside from the copy to host). In contrast, short-circuiting occurs more often but yields smaller gains per occurrence.

It is worth noting that this optimization is experimental and is not enabled by default (check EXIT_EARLY_OPT in the user knobs of [radix_kernel.cuh](radix_kernel.cuh)). Still, given the low overhead of generating the skip mask (_e.g._, copying and evaluating ~1 KB of histogram data for 32-bit keys assuming an 8-bit radix), I'd argue a fully-fledged GPU radix sort would ideally use both approaches together, since they operate at different levels.

### Fantastic GPU micro-optimizations and where to find them

Honestly, I don't know.

Micro-optimizations are a tricky topic in the era of modern compilers, and I'm no voodoo ~~wizard~~ child. Still, the goal of this project was to learn the underlying concepts, so almost all of the code was written from scratch rather than copied.
Because of that, each component ended up being tailored specifically for this sort. Many of the smaller changes were checked by diffing SASS dumps, which helped keep the hot path lean, even if it doesn’t exactly match what CUB or other sources do.

Some noteworthy micro-optimizations:

- **n forced to size_t** - Even though the entire kernel was written and validated to accept array sizes in any integer data type,  at some point I realized the algorithm was ever so consistently faster if _n_ was always _size_t_. I assume it has to do with casting choices improving codegen (_e.g._, fewer implicit conversions and better register usage), but I did not investigate this further as the SASS diff proved too extensive and terse.

- **__sync primitives** - Not every ```__syncwarp()``` or ```__syncthreads()``` written is needed. However, they often help with performance. In fact, in testing, I found that different combinations of sync primitives along a given kernel, even if not strictly required for correctness, can sometimes yield better performance gains, mainly due to subtle changes in warp scheduling and thread-pacing (the main reorder kernel is a good example).

- **PTX instructions** - In some areas, it is worth getting your hands dirty and writing some raw assembly. As a matter of fact, in the case of ```__match_any_sync()``` this is more a performance requirement rather than a micro-optimization, as the radix label is only 8 bits (in our case), so there is no need to match on a whole 32-bit register, and this falls right in the ballot-counting hot path of the ranker. Other small lane helpers also use PTX, though mostly to avoid including extra CUB libraries (also helps reduce compilation times).


## Compilation and Usage 

#### Compile with:
```bash
nvcc -O3 -std=c++17 -arch=sm_86 radix_gpu.cu bench_parser.cpp -o radix_gpu.exe
```

#### Run:
```bash
./radix_gpu
```

#### Command-line flags

-   `--n <value>`: Number of elements `[default = 2^27]`
-   `--iterations <value>`: # of timed iterations `[default = 30]`
-   `--warmup <value>`: # of warmup iterations `[default = 10]`
-  `--warmup_ms <value>`: Minimum warmup time (in milliseconds) `[default = 10]`
-   `--descending`: Sort in descending order `[default = false]`
-   `--validation`: Run validation instead of benchmark `[default = false]`
-   `--help`: Show available flags


#### Example benchmark run:
```bash
./radix_gpu --n 10000000 --iterations 30 --warmup 10
```
Performs a benchmark run, sorting an array of 10M random elements, showing the results of an average of 30 runs, with at least 10 runs of warmup.


## Benchmarks

### Setup

Configuration used for all  experiments:

- **Hardware:**
	- **GPU** - NVIDIA GeForce RTX 3080 Ti <sup>1</sup>
		- Ampere (sm_86)
		- 12 GB GDDR6X (stock clocks)
	- **CPU** - 13th Gen Intel i7-13700K
- **Software:**
	- **CUDA** - V13.1.115
	- **Driver** - 580.126.09
	- **OS** - Ubuntu 24.04.4 LTS x86_64
	- **Kernel** - 6.17.0-22-generic
- **Compilation:**
	- **nvcc flags** - -O3 -arch=sm_86

<sup>1</sup> No background processes running; headless (no displays attached).
	
### Data collection

Unless expressed otherwise, all collected data points represent an average of 30 runs, with a warmup of at least 250 ms (or 10 runs, whichever takes longer). This will be referred to as the _steady-state_ configuration.
The warmup metric was chosen with the goal of tightening the standard deviation between runs as much as possible, not to bias any of the approaches.
In addition, all experiments correspond to sorting unsigned keys in ascending order, randomly initialized with a standard avalanche bit mixer (check _init_keys_ kernel in [utils.cuh](radix/utils.cuh)).  All graphs show throughput instead of timings in number of elements sorted per second - _el/s_). This project's implementation is referred to as _rsort_.

### Scaling Behavior

For 32-bit keys, rsort is faster across all tested array sizes, from 2<sup>20</sup> to 2<sup>29</sup> elements. Speedups are more pronounced for smaller arrays, starting in the double digits and eventually stabilizing around 3% as peak throughput is reached. Peak throughput for rsort is reached around 22.4B el/s for 2<sup>27</sup> elements. CUB peaks slightly lower, at around 21.9B el/s reached at 2<sup>28</sup> elements.
It's interesting to note the drop in throughput from 2<sup>28</sup> elements onwards for rsort, as it coincides with the policy change that disables epoch bits in the lookback path. Surprisingly, rsort still leads by around 2% with epoch bits turned off.  For 32 bits, rsort uses 21 items per thread and 384 threads per block, which, as far as I can tell, is the same CTA geometry CUB uses for this scenario in sm_86. Rsort pipeline being optimized for an 8-bit radix onesweep makes a strong case against CUB's more general policies, at least for high-throughput scenarios. Lookback epoch bits seem to provide the largest relative advantage for smaller arrays.

<p align="center">
  <img src="graphs/scaling_uint32_t.png" width="48%" />
  <img src="graphs/scaling_uint64_t.png" width="48%" />
</p>

For 64-bit keys, the story changes slightly. Whereas rsort had the biggest advantage for 32-bit keys of smaller sizes, CUB now stays ahead for arrays of 2<sup>20</sup>, 64-bit elements. It is unclear to me why this change happens. rsort uses 448 threads per block and 6 items per thread to sort 64-bit keys; changing the policy to match CUB's geometry seems to yield very similar results. Changing other knobs in Rsort's reorder and lookback policy does not seem to move performance significantly either. As the array size increases, CUB gradually loses the performance advantage in the next two size jumps. Eventually, rsort stabilizes at about a 4% performance gain with a peak throughput of around 5.92B el/s, before falling to a 2.6% gain when turning off the lookback epoch policy at an array size of 2<sup>28</sup>. CUB's throughput peaks at around 5.69B el/s.

Overall, standard deviations seem to be tighter for rsort across most sizes, although for 64-bit keys, CUB seems to present less jitter on average (check [benchmark_data](graphs/benchmark_data.py)). Worth noting that, in my testing, even using an average of 30 runs along with a hefty warmup configuration, standard deviation still varies quite a bit from run to run, so it is better to focus on the global picture and not read too much into specific values.

<p align="center">
  <img src="graphs/scaling_uint8_t.png" width="48%" />
  <img src="graphs/scaling_uint16_t.png" width="48%" />
</p>

For 8 and 16 bits, rsort wins more clearly all around. The CTA geometry used by rsort for both cases is the same as for 32 bits. As far as I could tell, CUB does not use onesweep for keys smaller than 32 bits on sm_86, and the radix is also smaller than 8 bits. I'm not sure of the reasons behind this choice. Perhaps memory management, although looking at memory consumption, the two algorithms do not seem significantly different. Still, this means these two graphs are not comparing implementations of the same algorithm at all anymore.

For 8-bit keys, initial gains for smaller arrays range from 40% to 50%, eventually dropping down to a 21% performance lead at a peak throughput of around 89.1B el/s reached for an array of 2<sup>29</sup> elements. Performance then drops again to 17%-18% from sizes 2<sup>30</sup> onwards, as rsort's lookback policy is forced into 64 bits. Interestingly, we don't see the same behaviour of dropping throughput when disabling lookback from sizes 2<sup>28</sup> to 2<sup>30</sup> that we saw for 32 and 64 bits. This is possibly explained by the smaller keys requiring larger array sizes to achieve peak throughput numbers. In fact, even at N = 2<sup>31</sup>, CUB is still peaking throughput at around 73.7B el/s.

It is also worth mentioning the peak in throughput that happens at N = 2<sup>21</sup> for both approaches. This appears to be an artefact of memory and how data fits in the GPU rather than an algorithmic quirk. In fact, at 1 byte per element and only 1-2M elements, I'd argue the most likely explanation is that the whole array and workspace memory fits in the GPU's L2 6MB cache. This is further corroborated by the fact that timings are almost identical for arrays of 2<sup>20</sup> and 2<sup>21</sup> 8-bit elements.

For 16 bits, rsort's performance gains are more pronounced, reaching a 112% speedup for N = 2<sup>22</sup>, eventually stabilizing at around a 70% at a peak throughput of 59.4B el/s for an array of 2<sup>29</sup> elements. The same situation seen in the 8-bit scenario then happens when the lookback is forced into 64 bits for array sizes of 2<sup>30</sup> onwards. In turn, CUB reaches a peak throughput of around 35.1B el/s for N = 2<sup>30</sup>.
It is interesting to note that if we think about throughput in sorted bytes per second instead of sorted elements per second, we see that, for 8 bits, 89.1B bytes/s is almost the same as the 22.4 x 4 = 89.6B bytes/s figure we saw for 32-bit (4 bytes) peak throughput. However, the throughput for 16-bit keys is considerably higher, reaching a peak of 59.4 x 2 = 118.8B bytes/s, or 110.6GB/s, or (setting aside the parallelism of things for a minute) an even crazier 8.42 picoseconds per sorted byte! Enough for light to travel a bit more than 2mm. Remember this the next time your favourite AAA title dips below 60 fps 😀  maybe the hardware isn’t the only thing to blame!
 
### Cold vs. Steady state

These benchmarks were carried out in _steady-state_ with significant warmup. This means the code caches and paths are pre-loaded and hot. Even though this is typically how most performance benchmarks are performed (especially for GPUs), there is value in testing algorithmic performance on more realistic conditions. A user calling a sort algorithm is most likely to do some other work after with the sorted output, possibly even outside the GPU. The above bar graph shows the results of running both states for an unsigned 32-bit array of size 2<sup>27</sup>. The _cold-state_ consists of 1 single run with no warmup of any kind. Results still correspond to an average of 30 runs performed individually and spaced out in time sufficiently to cool the GPU.

<p align="center">
  <img src="graphs/cold_vs_steady_uint32_n27.png" width="55%" />
</p>

In this experiment, CUB achieves a throughput of 21.7B el/s on _steady-state_ vs. 21.5B el/ for  _cold-state_, corresponding to a performance drop of around 1%. In turn, rsort manages 22.4B el/s for _steady-state_ vs 22.2B el/s on _cold-state_, which translates to a 0.7% drop. This corresponds to an advantage of around 3.1% for rsort in _steady-state_ and 3.3% in _cold-state_. 
Standard deviations are just slightly tighter for rsort in both states. However, to establish a pattern for _cold-state_ would necessitate more extensive scaling benchmarking, as done for _steady-state_.

### Short-Circuiting vs. Early-Exiting


For this experiment, we are again using _steady-state_ with 32-bit arrays of 2<sup>27</sup> elements. Here, instead of randomly initializing the entire value, only every other byte is randomly initialized, the remaining ones being filled with the same patterns for all keys. This scenario is beneficial for rsort's early-exiting approach, as 2 of the bytes will be marked for skipping. However, CUB's CTA-level short-circuiting should also fast-forward the work of every block that handles the constant-valued bytes to the scatter stage for copying. The idea is to make both mechanisms trigger half of the time in order to compare speedups. Besides, many real-world applications tag bit information in a specific location or bytes of the data (_i.e._ the lookback mechanism of both these algorithms, for instance), making this scenario not so far-fetched.

<p align="center">
  <img src="graphs/byte_skip_uint32_n27.png" width="55%" />
</p>

Results show a throughput increase from 21.7B to 22.1B el/s (+1.6%) for CUB when switching from random to byte-skipped arrays. This was somewhat surprising to profile, as I assumed the work skipped would be much closer to 50%. Still, not performing any GPU work for 2 of the 4 bytes should have rsort decrease timings to about half, giving it an advantage. Indeed, results show that rsort's throughput jumps to about 40.4B el/s when sorting byte-skipped arrays in this scenario.
As for array mode variance, results show a 2.8% performance gain for random arrays (with early-exiting enabled) against CUB, and 82.6% for short-circuiting versus early-exiting in byte-skipped arrays.

Paying attention to the baseline random initialization results for rsort might raise the question as to why the timings are not 6.003 ms instead of the 5.990 ms seen in the last test with the same setup. This is because early-exiting is not enabled by default in rsort, and in this scenario, we are paying the cost of this mechanism explicitly. Still, in testing, the measured cost for early-exit stayed within the 10-20 µs range (13 µs here), so it's safe to say that it's not an overtly expensive optimization to perform. Also note that the performance cost of early-sxiting is solely linearly dependent on the number of passes, and constant otherwise.

I think it is worth noting that the comparison between these mechanisms should not invalidate either of them, as they are mutually exclusive and operate at different levels (globally and CTA). Therefore, any implementation could definitely, and probably should, benefit from implementing both techniques. 

### 128-bit results

This scenario consists of a preliminary experiment for sorting 128-bit arrays (again, unsigned keys, _steady-state_). Ultimately, this should be a scaling graph like the first four, but to avoid cluttering this benchmarking section and sparing my poor GPU of further torture, let's stick with the sizes with the highest potential for peak throughput that fit in GPU VRAM (max 12GB). For array sizes of 2<sup>27</sup> elements, rsort achieves a throughput of around 1.49B el/s when compared to CUB's 1.45B el/s, representing a speedup of around 2.4%. This throughput figure reduces slightly to 1.46B el/s for N = 2<sup>28</sup>.

<p align="center">
  <img src="graphs/benchmark_128bit.png" width="55%" />
</p>

Because of CUB's input preserving requirement, sorting this many 128-bit elements is not doable, as workspace memory exceeds available VRAM, resulting in OOM. Standard deviations for CUB seem to hint towards less jitter than rsort for this scenario. Again, further scaling tests are needed in order to assert this.


### TL;DR

- Strong performance gains for small key sizes (especially 16-bit).
- Competitive performance for 32 and 64-bit keys (~2–4% gains at peak throughput).
- Scaling behaviour identical to CUB, with minor differences due to lookback policies.
- Optimizations described reduce runtime (significantly in more favourable scenarios).
- Performance gains in both _steady_ and _cold_ states.
- Low run-to-run variance, also identical to CUB, overall.


## Validation

To ensure the correctness of the implementation, a validation suite is provided to test sorting across a range of different scenarios. A full validation run consists of sorting arrays of all data types, for every array mode (random and byte-skipped initialization) and sort orders (ascending and descending). Array sizes span from the largest power of 2 that fits the GPU VRAM (including workspace memory), down through 11 successive halvings. The full validation includes the follwing data types: ```uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double```, for a total of 440 tests. Each test performs order checks on the elements, but also radix count checks over the whole array.

### Example validation run:
```bash
./radix_gpu --validation --iterations 1 --warmup 0
```
Performs a validation run, calling a single benchmark with no warmup for all defined array modes, for all defined sizes, including descending, and for every defined data type.

### Notes (READ!):

- Due to heavy template instantiation, validation is disabled by default. The toggle is the single "VALIDATION_TEST" macro at the start of the main file.
- You can use the macros at the start of [validate_sort.cuh](validate_sort.cuh) to change the set of data types to check validation for. Some other standard subsets are provided, or you can define your own.

## Failed Experiments

 - **Multi-lookback** - Learning about the chained lookback, at first, does not seem like the most efficient structure. Basically, block 0 must publish something, even if partially, before any lookback work is done. I experimented with a multi-group lookback where every group gathers information independently, reducing contention on a single chain and paying a small prefix sum cost at the end. Sounded good... doesn't work. Or at least I didn't manage to reduce performance with it. 
 
 - **Ranker** - A lot of time was spent trying to optimize the ranker to no avail. I eventually decided to keep my sanity and checked CUB’s approach 😀 This is probably the component that more closely matches it. 

 -  **Circular buffer** - The [GPUOpen article](https://gpuopen.com/learn/boosting_gpu_radix_sort/) that motivated this project describes a decoupled lookback using a circular buffer. After much effort to reproduce this approach, I could not achieve consistent performance comparable to the coupled variant, so I abandoned the idea.


## Notes and Disclaimers

- This implementation is not intended to be production-ready (in its current state). It is a proof of concept and is not a drop-in replacement for CUB.
- For convenience, a monolithic single-header version is provided in [monolithic/rsort.cuh](monolithic/rsort.cuh). The main implementation remains split across files for readability. This monolithic version might not be up to date with development!
- The implementation was tested on a single GPU (mine). Cross-device results may vary.
- The optimizations explored here are, to the best of my knowledge, not widely documented in existing GPU radix sort literature, hence this project.
- Part of the point of this implementation was to prove the optimizations can be carried out without loss of generality and without getting in the way of other features.


## TODOs

- Implmenting pairs and other features for completeness.
- Investigate integrating or aligning with benchmarking approaches used in NVIDIA/cccl to improve consistency. I wasn't aware of the bench suite when writing this benchmark.
- **Optimizations**: Some ideas for ranker work distribution still to test; lookback configurations; improve memory alignment overall; etc. 
  

## Sources

- GPUOpen article - https://gpuopen.com/learn/boosting_gpu_radix_sort/
- Onesweep Radix Sort - https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21572-a-faster-radix-sort-implementation.pdf
- NVIDIA CCCL - https://github.com/NVIDIA/cccl/tree/main/cub
- CUDA docs - https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
- Several CUDA YouTube talks - https://youtu.be/GmNkYayuaA4?t=430 - I'm available if the offer still stands 🙂
<!--
<pre>
																								🫳
																									🎤
</pre>
-->
