// bench_stage.cpp -- standalone analysis tool, NOT part of the graded build.
//
// Times and perf-profiles ONE named stage in isolation, for the CSV-logging
// scripts in task2/bench/. Links directly against the same src/matmul_*.cpp
// implementations the graded ./bin/matmul uses.
//
//   time <stage> M N K [warmup] [reps] [seed]
//       Runs the stage under pa1::time_median_ms, prints "ms,gflops".
//   loop <stage> M N K reps [seed] [--verify]
//       Runs the stage `reps` times back to back with no internal timing, so
//       a `perf stat` wrapper reflects only the target stage's counters.
//       --verify checks correctness against matmul_naive first (stderr).

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"
#include "timer.h"
#include "utils.h"

// SIMD width variants live in src/matmul_simd.cpp; only matmul_simd is graded.
extern void matmul_simd128(const float*, const float*, float*, int, int, int, int, int, int);
extern void matmul_simd256(const float*, const float*, float*, int, int, int, int, int, int);

// Accumulator-count variants live in src/matmul_optimized.cpp; only
// matmul_optimized is graded.
extern void matmul_optimized_acc1(const float*, const float*, float*, int, int, int, int, int, int);
extern void matmul_optimized_acc2(const float*, const float*, float*, int, int, int, int, int, int);
extern void matmul_optimized_acc4(const float*, const float*, float*, int, int, int, int, int, int);

namespace {

struct StageEntry {
    const char* key;
    MatMulFn fn;
};

const StageEntry kStages[] = {
    {"naive", matmul_naive},
    {"simd128", matmul_simd128},
    {"simd256", matmul_simd256},
    {"simd", matmul_simd},
    {"prefetch", matmul_prefetch},
    {"opt_acc1", matmul_optimized_acc1},
    {"opt_acc2", matmul_optimized_acc2},
    {"opt_acc4", matmul_optimized_acc4},
    {"optimized", matmul_optimized},
};
constexpr int kNumStages = sizeof(kStages) / sizeof(kStages[0]);

MatMulFn find_stage(const char* key) {
    for (int i = 0; i < kNumStages; ++i)
        if (std::strcmp(kStages[i].key, key) == 0) return kStages[i].fn;
    return nullptr;
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s time <stage> M N K [warmup] [reps] [seed]\n"
        "  %s loop <stage> M N K reps [seed] [--verify]\n"
        "stages: naive | simd128 | simd256 | simd | prefetch | "
        "opt_acc1 | opt_acc2 | opt_acc4 | optimized\n",
        prog, prog);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) { usage(argv[0]); return 1; }

    const char* mode = argv[1];
    const char* stage_key = argv[2];
    const int M = std::atoi(argv[3]);
    const int N = std::atoi(argv[4]);
    const int K = std::atoi(argv[5]);

    const MatMulFn fn = find_stage(stage_key);
    if (!fn) {
        std::fprintf(stderr, "error: unknown stage '%s'\n", stage_key);
        usage(argv[0]);
        return 1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        std::fprintf(stderr, "error: need M,N,K>0 (got M=%d N=%d K=%d)\n", M, N, K);
        return 1;
    }

    unsigned seed = 1234u;
    const int lda = K, ldb = K, ldc = N;
    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    pa1::fill_random(A, static_cast<std::size_t>(M) * K, seed);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, seed + 1u);

    if (std::strcmp(mode, "time") == 0) {
        const int warmup = argc > 6 ? std::atoi(argv[6]) : 1;
        const int reps   = argc > 7 ? std::atoi(argv[7]) : 3;
        if (argc > 8) seed = static_cast<unsigned>(std::strtoul(argv[8], nullptr, 10));

        auto run = [&]() { fn(A, B, C, M, N, K, lda, ldb, ldc); };
        const double ms = pa1::time_median_ms(run, warmup, reps);
        const double gflops = pa1::matmul_flops(M, N, K) / (ms * 1e6);
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
            float* ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
            matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
            fn(A, B, C, M, N, K, lda, ldb, ldc);
            const float ref_mag = pa1::max_abs(ref, static_cast<std::size_t>(M) * N);
            const float tol = 1e-4f * (ref_mag + 1e-30f);
            const float err = pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N);
            std::fprintf(stderr, "verify %s M=%d N=%d K=%d: max_abs_diff=%.6g (%s)\n",
                         stage_key, M, N, K, err, err <= tol ? "OK" : "FAIL");
            pa1::free_floats(ref);
        }

        for (int r = 0; r < reps; ++r) fn(A, B, C, M, N, K, lda, ldb, ldc);

    } else {
        usage(argv[0]);
        return 1;
    }

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    return 0;
}
