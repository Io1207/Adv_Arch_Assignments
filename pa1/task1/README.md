# CS683 PA-1, Task 1: Hardware-Conscious 2D Convolution

In this task you take a correct-but-slow 2D convolution and make it fast by
**thinking about the hardware**: the cache hierarchy, instruction-level parallelism,
and the SIMD (vector) units of a modern x86 core. You will apply four classic
techniques, **loop reordering, loop unrolling, cache tiling, and SIMD (AVX2)**,
one per stage, and measure the benefit of each.

The goal is not only to make the code fast, but also to understand *why* each
transformation helps (or does not):
the same arithmetic, reorganized to match the machine, runs an order of magnitude faster.

---

## 1. The computation

A single-channel 2D convolution of a `float32` image with a `K×K` kernel, in zero-padded
**"same"** mode so the output is the same size as the input. This operation is technically
cross-correlation because the kernel is **not** flipped.

**Data layout (given to you; every stage uses the same signature):**

```cpp
// in : PADDED input, (H+2p) rows × (W+2p) cols, row-major, row stride = W+2p, p = K/2
//      The p-wide border is zero, so every access inside the loops is in bounds.
//      No boundary check is required.
// out: output, H × W, row-major, row stride = W
// ker: kernel, K × K, row-major
void conv_<stage>(const float* in, float* out, const float* ker,
                  int H, int W, int K);
```

`W` is always a multiple of 8, so an 8-wide (AVX2) inner loop over columns has **no
remainder tail** to special-case. `K` is odd (3 and 5 are tested).

---

## 2. What you implement

Edit **only** these five files in `src/`. Each already contains a detailed header
comment describing the technique and a hint. `conv_naive` is provided as the
correctness reference. **Do not modify it.**

| File | Technique | The idea in one line |
|------|-----------|----------------------|
| `src/conv_reorder.cpp`   | Loop reordering | Hoist the kernel loops out so the inner loop is a unit-stride stream over output columns. |
| `src/conv_unroll.cpp`    | Loop unrolling  | Unroll that inner loop to expose independent work (ILP) and hide FP latency. |
| `src/conv_tile.cpp`      | Cache tiling    | Block the output into cache-resident strips so the K·K passes hit cache, not DRAM. |
| `src/conv_simd.cpp`      | SIMD (AVX2)     | Do 8 columns per instruction with `_mm256_fmadd_ps` and a broadcast weight. |
| `src/conv_optimized.cpp` | **All combined** | Reorder + unroll + tile + AVX2, then tune. **This one is graded on speed.** |

The stages build on each other. A key teaching point lives in the first two stages:
the "obvious" reorder makes `K·K` passes over the whole (16 MB) output image, which does
**not** fit in cache. At `K=3`, it can therefore run *as slowly as or more slowly than*
the naive implementation, and
unrolling a memory-bound loop barely helps. **Cache tiling (Stage 3) is what unlocks the
payoff.** The speedup column should demonstrate this behavior.

---

## 3. Build and run

```bash
make        # builds ./bin/conv from your src/*.cpp
./bin/conv  # or: make run
```

`./bin/conv` (no arguments) runs a correctness smoke test, then prints a per-stage table
(correctness, time, GFLOP/s, speedup vs naive) for the graded `2048×2048, K=3` workload
and your autograder score, followed by an ungraded `K=5` table for reference.

### Checking one stage while you develop it

You do **not** have to finish everything before you can test. Each stub compiles from
the start because it calls the naive implementation, so the project always builds.
You can check a single stage against the naive reference using a workload of your choice:

```bash
./bin/conv reorder              # naive vs reorder only, default 2048x2048 K=3
./bin/conv reorder 512 512 3    # custom H W K (W must be a multiple of 8, K odd)
./bin/conv reorder 512 512 3 42 # ... and a custom RNG seed (try several inputs)
./bin/conv all 1024 1024 5      # every stage on a custom workload (no score)
./bin/conv help                 # usage
```

The `correct` column compares your kernel's output to `conv_naive` on that exact input
(maximum absolute error < `1e-3`), and the `speedup` column is measured against the naive
implementation on the same workload. This provides a fast edit-build-check cycle. Changing
the seed generates a different random image/kernel, which is an easy way to gain confidence
your kernel is correct on more than one input. The final score always comes from the
no-argument run.

### Compiler flags are pinned; do not change them

The `Makefile` compiles **every** stage with the identical flags:

```
-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma
```

Do **not** add `-march=native`, `-O3`, or `-ftree-vectorize`. The autograder uses
exactly the flags above.

---

## 4. Submission

Submit your five edited files:
`src/conv_reorder.cpp`, `src/conv_unroll.cpp`, `src/conv_tile.cpp`,
`src/conv_simd.cpp`, `src/conv_optimized.cpp`. Do not modify any other file.

Additionally, plot the speedup of each optimized version (`conv_reorder`,
`conv_unroll`, `conv_tile`, `conv_simd`, `conv_optimized`) over `conv_naive`,
varying both the input image dimensions and the kernel size. Submit a report
(PDF) that includes these plots and justifies the observed performance
gains or losses for each optimization technique.
