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

namespace {

struct StageEntry {
    const char* key;
    ConvFn fn;
};

const StageEntry kStages[] = {
    {"naive", conv_naive},
    {"reorder", conv_reorder},
    {"unroll", conv_unroll},
    {"tile", conv_tile},
    {"simd", conv_simd},
    {"optimized", conv_optimized},
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
        "stages: naive | reorder | unroll | tile | simd | optimized\n",
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
