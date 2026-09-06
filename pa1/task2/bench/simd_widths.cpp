// simd_widths.cpp -- bench-only SIMD register-width variants (NOT graded, NOT
// part of src/). Used by task2/bench/collect_simd.py for the
// "speedup vs. SIMD width" plot (Task 2B / README plot 4).
//
// Every variant keeps the naive (i, j, p) loop order with a single accumulator
// and vectorises ONLY the K-dimension dot product, so the plot isolates the
// effect of the SIMD register width itself -- no register blocking, no tiling,
// no prefetch (that is Task 2C / src/matmul_optimized.cpp).
//
//   matmul_simd128 : 128-bit SSE     (__m128,  4 lanes)
//   matmul_simd256 : 256-bit AVX2    (__m256,  8 lanes)
//   matmul_simd512 : 512-bit AVX-512 (__m512, 16 lanes) -- compiled only when
//                    the translation unit is built with -mavx512f; otherwise a
//                    stub that falls back to the 256-bit path so the symbol
//                    still links. This CPU (i7-12650H) exposes no avx512 flags.
//
// Built with the pinned flags (-mavx2 -mfma) for 128/256; the 512 file, if
// used, is a separate ungraded build per README section 5.

#include <immintrin.h>

namespace {

inline float hsum128(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);          // [1 1 3 3]
    __m128 sums = _mm_add_ps(v, shuf);         // [0+1 . 2+3 .]
    shuf = _mm_movehl_ps(shuf, sums);          // [2+3 ...]
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hsum128(_mm_add_ps(lo, hi));
}

}  // namespace

// ---- 128-bit (SSE) ---------------------------------------------------------
void matmul_simd128(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        float* c = C + static_cast<long>(i) * ldc;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m128 acc = _mm_setzero_ps();
            int p = 0;
            for (; p + 4 <= K; p += 4)
                acc = _mm_fmadd_ps(_mm_loadu_ps(a + p), _mm_loadu_ps(b + p), acc);
            float s = hsum128(acc);
            for (; p < K; ++p) s += a[p] * b[p];
            c[j] = s;
        }
    }
}

// ---- 256-bit (AVX2) ------------------------------------------------------- -
void matmul_simd256(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        float* c = C + static_cast<long>(i) * ldc;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= K; p += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
            float s = hsum256(acc);
            for (; p < K; ++p) s += a[p] * b[p];
            c[j] = s;
        }
    }
}

// ---- 512-bit (AVX-512) --------------------------------------------------- --
#ifdef __AVX512F__
void matmul_simd512(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        float* c = C + static_cast<long>(i) * ldc;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m512 acc = _mm512_setzero_ps();
            int p = 0;
            for (; p + 16 <= K; p += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + p), _mm512_loadu_ps(b + p), acc);
            float s = _mm512_reduce_add_ps(acc);
            for (; p < K; ++p) s += a[p] * b[p];
            c[j] = s;
        }
    }
}
#else
// No AVX-512 on the pinned build / this CPU: keep the symbol so bench_stage
// links, but route it to the 256-bit implementation. collect_simd.py skips
// "simd512" unless `lscpu` shows an avx512 flag.
void matmul_simd512(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
    matmul_simd256(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif
