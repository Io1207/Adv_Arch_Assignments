// prefetch_tuned.cpp -- bench-only tunable software-prefetch kernel (NOT
// graded, NOT part of src/). Used by task2/bench/bench_prefetch.cpp /
// collect_prefetch.py for the "speedup vs. prefetch distance" and
// "speedup vs. prefetch level" plots (Task 2A / README plots 2 & 3).
//
// The ONLY optimisation over matmul_naive is an _mm_prefetch of the A and B
// streams issued `distance` floats ahead of the current K position, at one of
// the four locality hints. Naive (i, j, p) loop order, scalar FMA, single
// accumulator -- so the sweep isolates the prefetch-distance and cache-fill
// -level effects with nothing else changing.
//
//   distance : look-ahead in floats (0 == no prefetch == matmul_naive).
//              One _mm_prefetch pair is issued per 64-byte line (every 16
//              floats of p), covering `a + p + distance` and `b + p + distance`.
//   hint     : 0=_MM_HINT_NTA  1=_MM_HINT_T2  2=_MM_HINT_T1  3=_MM_HINT_T0
//              (matches bench_prefetch.cpp's documented mapping)

#include <immintrin.h>

namespace {

constexpr int kLineFloats = 64 / sizeof(float);   // 16

template <_mm_hint HINT>
void run(const float* A, const float* B, float* C,
         int M, int N, int K, int lda, int ldb, int ldc, int dist) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        float* c = C + static_cast<long>(i) * ldc;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            float s = 0.0f;
            int p = 0;
            for (; p < K; ++p) {
                if ((p % kLineFloats) == 0) {
                    _mm_prefetch(reinterpret_cast<const char*>(a + p + dist), HINT);
                    _mm_prefetch(reinterpret_cast<const char*>(b + p + dist), HINT);
                }
                s += a[p] * b[p];
            }
            c[j] = s;
        }
    }
}

// distance == 0: identical to matmul_naive, no prefetch instructions at all.
void run_plain(const float* A, const float* B, float* C,
               int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        float* c = C + static_cast<long>(i) * ldc;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            float s = 0.0f;
            for (int p = 0; p < K; ++p) s += a[p] * b[p];
            c[j] = s;
        }
    }
}

}  // namespace

void matmul_prefetch_tuned(const float* A, const float* B, float* C,
                           int M, int N, int K, int lda, int ldb, int ldc,
                           int distance, int hint) {
    if (distance <= 0) {
        run_plain(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }
    switch (hint) {
        case 0:  run<_MM_HINT_NTA>(A, B, C, M, N, K, lda, ldb, ldc, distance); break;
        case 1:  run<_MM_HINT_T2 >(A, B, C, M, N, K, lda, ldb, ldc, distance); break;
        case 2:  run<_MM_HINT_T1 >(A, B, C, M, N, K, lda, ldb, ldc, distance); break;
        default: run<_MM_HINT_T0 >(A, B, C, M, N, K, lda, ldb, ldc, distance); break;
    }
}
