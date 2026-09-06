// bench_prefetch.cpp -- standalone analysis tool, NOT part of the graded
// build. Exercises matmul_prefetch_tuned(..., distance, hint) directly since
// its two extra parameters don't fit the plain MatMulFn signature every
// other stage shares. hint: 0=NTA, 1=T2, 2=T1, 3=T0 (matches
// src/matmul_prefetch.cpp's switch).
//
//   time M N K distance hint [warmup] [reps] [seed]
//   loop M N K distance hint reps [seed] [--verify]

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"
#include "timer.h"
#include "utils.h"

extern void matmul_prefetch_tuned(const float*, const float*, float*,
                                   int, int, int, int, int, int, int, int);

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
            "usage:\n"
            "  %s time M N K distance hint [warmup] [reps] [seed]\n"
            "  %s loop M N K distance hint reps [seed] [--verify]\n",
            argv[0], argv[0]);
        return 1;
    }

    const char* mode = argv[1];
    const int M = std::atoi(argv[2]);
    const int N = std::atoi(argv[3]);
    const int K = std::atoi(argv[4]);
    const int distance = std::atoi(argv[5]);
    const int hint = std::atoi(argv[6]);

    unsigned seed = 1234u;
    const int lda = K, ldb = K, ldc = N;
    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    pa1::fill_random(A, static_cast<std::size_t>(M) * K, seed);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, seed + 1u);

    auto run = [&]() { matmul_prefetch_tuned(A, B, C, M, N, K, lda, ldb, ldc, distance, hint); };

    if (std::strcmp(mode, "time") == 0) {
        const int warmup = argc > 7 ? std::atoi(argv[7]) : 1;
        const int reps   = argc > 8 ? std::atoi(argv[8]) : 3;
        if (argc > 9) seed = static_cast<unsigned>(std::strtoul(argv[9], nullptr, 10));
        const double ms = pa1::time_median_ms(run, warmup, reps);
        const double gflops = pa1::matmul_flops(M, N, K) / (ms * 1e6);
        std::printf("%.6f,%.6f\n", ms, gflops);

    } else if (std::strcmp(mode, "loop") == 0) {
        if (argc < 8) return 1;
        const int reps = std::atoi(argv[7]);
        bool verify = false;
        for (int i = 8; i < argc; ++i) {
            if (std::strcmp(argv[i], "--verify") == 0) verify = true;
            else seed = static_cast<unsigned>(std::strtoul(argv[i], nullptr, 10));
        }
        if (verify) {
            float* ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
            matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
            run();
            const float ref_mag = pa1::max_abs(ref, static_cast<std::size_t>(M) * N);
            const float tol = 1e-4f * (ref_mag + 1e-30f);
            const float err = pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N);
            std::fprintf(stderr, "verify prefetch d=%d h=%d M=%d N=%d K=%d: max_abs_diff=%.6g (%s)\n",
                         distance, hint, M, N, K, err, err <= tol ? "OK" : "FAIL");
            pa1::free_floats(ref);
        }
        for (int r = 0; r < reps; ++r) run();
    }

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    return 0;
}
