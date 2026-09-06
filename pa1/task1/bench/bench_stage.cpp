// bench_stage.cpp -- standalone analysis tool, NOT part of the graded build.
//
// Times and perf-profiles ONE named stage in isolation so the scripts in
// task1/bench/ can gather timing/speedup and perf-counter data (instructions,
// L1D misses -> MPKI) per technique/variant across a grid of (H, W, K),
// without touching main.cpp, convolution.h, or any file the Makefile treats
// as graded (STUDENT_SRC). It links directly against the same src/*.cpp
// stage implementations the graded ./bin/conv uses.
//
// Two modes:
//   time <stage> H W K [warmup] [reps] [seed]
//       Runs the stage under pa1::time_median_ms and prints "ms,gflops" to
//       stdout -- one line, easy for a driver script to parse.
//
//   loop <stage> H W K reps [seed] [--verify]
//       Runs the stage `reps` times back to back with NO internal timing, so
//       a `perf stat` wrapper around this process reflects (almost only) the
//       target stage's instructions/misses -- setup (alloc + fill_random,
//       done once) is amortized over `reps` calls instead of re-measured
//       every time. --verify additionally checks correctness against
//       conv_naive once, before the timed loop, and prints the result to
//       stderr; leave it off for the actual perf-wrapped invocation so the
//       one-off naive reference call doesn't pollute the counters.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "convolution.h"
#include "timer.h"
#include "utils.h"

// Reorder variants live in src/conv_reorder.cpp but are intentionally not
// part of convolution.h (only conv_reorder is graded/interface); declare them
// here so this analysis tool can name and compare them directly.
extern void conv_reorder_ky_kx_oy_ox(const float*, float*, const float*, int, int, int);
extern void conv_reorder_oy_ky_kx_ox(const float*, float*, const float*, int, int, int);

// Unroll variants live in src/conv_unroll.cpp; only conv_unroll is part of
// the graded interface (convolution.h). Declared here for the same reason.
extern void conv_unroll_ox8(const float*, float*, const float*, int, int, int);
extern void conv_unroll_kx4(const float*, float*, const float*, int, int, int);

// SIMD width variants live in src/conv_simd.cpp; only conv_simd is graded.
extern void conv_simd128(const float*, float*, const float*, int, int, int);
extern void conv_simd256(const float*, float*, const float*, int, int, int);

// Multi-accumulator variants live in src/conv_optimized.cpp; only
// conv_optimized is graded.
extern void conv_optimized_acc1(const float*, float*, const float*, int, int, int);
extern void conv_optimized_acc2(const float*, float*, const float*, int, int, int);
extern void conv_optimized_acc4(const float*, float*, const float*, int, int, int);
extern void conv_optimized_acc8(const float*, float*, const float*, int, int, int);
extern void conv_optimized_acc16(const float*, float*, const float*, int, int, int);

namespace {

struct StageEntry {
    const char* key;
    ConvFn fn;
};

const StageEntry kStages[] = {
    {"naive", conv_naive},
    {"reorder_a", conv_reorder_ky_kx_oy_ox},  // ky, kx, oy, ox (suggested order)
    {"reorder_b", conv_reorder_oy_ky_kx_ox},  // oy, ky, kx, ox
    {"reorder", conv_reorder},                // whichever variant conv_reorder currently wraps
    {"unroll_ox8", conv_unroll_ox8},          // oy, ox(x8 accumulators), ky, kx
    {"unroll_kx4", conv_unroll_kx4},          // oy, ox, ky, kx(x4, single accumulator)
    {"unroll", conv_unroll},                  // whichever variant conv_unroll currently wraps
    {"tile", conv_tile},
    {"simd128", conv_simd128},                // SSE,  4-wide
    {"simd256", conv_simd256},                // AVX2, 8-wide
    {"simd", conv_simd},                      // whichever width conv_simd currently wraps
    {"opt_acc1", conv_optimized_acc1},        // 1 x __m256  (8-wide)
    {"opt_acc2", conv_optimized_acc2},        // 2 x __m256  (16-wide)
    {"opt_acc4", conv_optimized_acc4},        // 4 x __m256  (32-wide)
    {"opt_acc8", conv_optimized_acc8},        // 8 x __m256  (64-wide)
    {"opt_acc16", conv_optimized_acc16},      // 16 x __m256 (128-wide)
    {"optimized", conv_optimized},            // whichever Nacc conv_optimized currently wraps
};
constexpr int kNumStages = sizeof(kStages) / sizeof(kStages[0]);

ConvFn find_stage(const char* key) {
    for (int i = 0; i < kNumStages; ++i)
        if (std::strcmp(kStages[i].key, key) == 0) return kStages[i].fn;
    return nullptr;
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s time <stage> H W K [warmup] [reps] [seed]\n"
        "  %s loop <stage> H W K reps [seed] [--verify]\n"
        "stages: naive | reorder_a | reorder_b | reorder | unroll_ox8 | unroll_kx4 | unroll | "
        "tile | simd128 | simd256 | simd | opt_acc1 | opt_acc2 | opt_acc4 | opt_acc8 | "
        "opt_acc16 | optimized\n",
        prog, prog);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) { usage(argv[0]); return 1; }

    const char* mode = argv[1];
    const char* stage_key = argv[2];
    const int H = std::atoi(argv[3]);
    const int W = std::atoi(argv[4]);
    const int K = std::atoi(argv[5]);

    const ConvFn fn = find_stage(stage_key);
    if (!fn) {
        std::fprintf(stderr, "error: unknown stage '%s'\n", stage_key);
        usage(argv[0]);
        return 1;
    }
    if (H <= 0 || W <= 0 || K <= 0 || W % 8 != 0 || K % 2 == 0) {
        std::fprintf(stderr, "error: need H,W>0, W%%8==0, K odd (got H=%d W=%d K=%d)\n", H, W, K);
        return 1;
    }

    unsigned seed = 1234u;
    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);

    if (std::strcmp(mode, "time") == 0) {
        const int warmup = argc > 6 ? std::atoi(argv[6]) : 2;
        const int reps   = argc > 7 ? std::atoi(argv[7]) : 7;
        if (argc > 8) seed = static_cast<unsigned>(std::strtoul(argv[8], nullptr, 10));

        auto run = [&]() { fn(in, out, ker, H, W, K); };
        const double ms = pa1::time_median_ms(run, warmup, reps);
        const double gflops = pa1::conv_flops(H, W, K) / (ms * 1e6);
        std::printf("%.6f,%.6f\n", ms, gflops);

    } else if (std::strcmp(mode, "loop") == 0) {
        if (argc < 7) { usage(argv[0]); return 1; }
        const int reps = std::atoi(argv[6]);
        bool verify = false;
        for (int i = 7; i < argc; ++i) {
            if (std::strcmp(argv[i], "--verify") == 0) verify = true;
            else seed = static_cast<unsigned>(std::strtoul(argv[i], nullptr, 10));
        }

        if (verify) {
            float* ref = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
            conv_naive(in, ref, ker, H, W, K);
            fn(in, out, ker, H, W, K);
            const float err = pa1::max_abs_diff(out, ref, H, W);
            std::fprintf(stderr, "verify %s H=%d W=%d K=%d: max_abs_diff=%.6g (%s)\n",
                         stage_key, H, W, K, err, err <= 1e-3f ? "OK" : "FAIL");
            pa1::free_floats(ref);
        }

        // Perf-measured region: ONLY the target stage, `reps` times, nothing else.
        for (int r = 0; r < reps; ++r) fn(in, out, ker, H, W, K);

    } else {
        usage(argv[0]);
        return 1;
    }

    pa1::free_floats(in);
    pa1::free_floats(img);
    pa1::free_floats(ker);
    pa1::free_floats(out);
    return 0;
}
