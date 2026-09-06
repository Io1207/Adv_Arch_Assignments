// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
//
// Register-blocked 4x2 micro-kernel (no cache blocking):
//   - 8 YMM accumulators (c00..c31), one per output element of the 4x2 tile
//   - each B vector is loaded once and reused across the 4 rows
//   - each A vector is loaded once and reused across the 2 columns
//   - 8 independent FMA chains -> FMA latency is fully hidden
//   - K loop hinted for x2 unroll so loads of step p+8 overlap FMAs of step p
#include <immintrin.h>
#include "matmul.h"

namespace {

    constexpr int kIR = 4; // rows per micro-kernel (register tile height)
    constexpr int kJR = 2; // cols per micro-kernel (register tile width)

    inline float hsum256(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    // Vectorized dot product of length K between a and b (both length >= K).
    // Used only for the M/N remainder paths.
    inline float dot(const float* a, const float* b, int K) {
        __m256 acc = _mm256_setzero_ps();
        int p = 0;
        for (; p + 8 <= K; p += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
        }
        float sum = hsum256(acc);
        for (; p < K; ++p) sum += a[p] * b[p];
        return sum;
    }

    // C[i0..i0+3][j0..j0+1] = A[i0..i0+3][:] . B[j0..j0+1][:]   (B is transposed)
    inline void micro_kernel_4x2(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int i0, int j0,
                                 int K, int lda, int ldb, int ldc) {
        const float* a0 = A + static_cast<long>(i0 + 0) * lda;
        const float* a1 = A + static_cast<long>(i0 + 1) * lda;
        const float* a2 = A + static_cast<long>(i0 + 2) * lda;
        const float* a3 = A + static_cast<long>(i0 + 3) * lda;
        const float* b0 = B + static_cast<long>(j0 + 0) * ldb;
        const float* b1 = B + static_cast<long>(j0 + 1) * ldb;

        __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

        int p = 0;
        #pragma GCC unroll 2
        for (; p + 8 <= K; p += 8) {
            const __m256 bv0 = _mm256_loadu_ps(b0 + p);
            const __m256 bv1 = _mm256_loadu_ps(b1 + p);
            __m256 av;
            av = _mm256_loadu_ps(a0 + p); c00 = _mm256_fmadd_ps(av, bv0, c00); c01 = _mm256_fmadd_ps(av, bv1, c01);
            av = _mm256_loadu_ps(a1 + p); c10 = _mm256_fmadd_ps(av, bv0, c10); c11 = _mm256_fmadd_ps(av, bv1, c11);
            av = _mm256_loadu_ps(a2 + p); c20 = _mm256_fmadd_ps(av, bv0, c20); c21 = _mm256_fmadd_ps(av, bv1, c21);
            av = _mm256_loadu_ps(a3 + p); c30 = _mm256_fmadd_ps(av, bv0, c30); c31 = _mm256_fmadd_ps(av, bv1, c31);
        }

        float s00 = hsum256(c00), s01 = hsum256(c01);
        float s10 = hsum256(c10), s11 = hsum256(c11);
        float s20 = hsum256(c20), s21 = hsum256(c21);
        float s30 = hsum256(c30), s31 = hsum256(c31);

        // K remainder (K % 8): scalar, all 8 outputs at once
        for (; p < K; ++p) {
            const float av0 = a0[p], av1 = a1[p], av2 = a2[p], av3 = a3[p];
            const float bp0 = b0[p], bp1 = b1[p];
            s00 += av0 * bp0; s01 += av0 * bp1;
            s10 += av1 * bp0; s11 += av1 * bp1;
            s20 += av2 * bp0; s21 += av2 * bp1;
            s30 += av3 * bp0; s31 += av3 * bp1;
        }

        float* c0 = C + static_cast<long>(i0 + 0) * ldc;
        float* c1 = C + static_cast<long>(i0 + 1) * ldc;
        float* c2 = C + static_cast<long>(i0 + 2) * ldc;
        float* c3 = C + static_cast<long>(i0 + 3) * ldc;
        c0[j0 + 0] = s00; c0[j0 + 1] = s01;
        c1[j0 + 0] = s10; c1[j0 + 1] = s11;
        c2[j0 + 0] = s20; c2[j0 + 1] = s21;
        c3[j0 + 0] = s30; c3[j0 + 1] = s31;
    }
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    int i = 0;
    for (; i + kIR <= M; i += kIR) {
        int j = 0;
        for (; j + kJR <= N; j += kJR) {
            micro_kernel_4x2(A, B, C, i, j, K, lda, ldb, ldc);
        }
        // N remainder (N % kJR): one column, still vectorized over K
        for (; j < N; ++j) {
            const float* b_row = B + static_cast<long>(j) * ldb;
            for (int r = 0; r < kIR; ++r) {
                const float* a_row = A + static_cast<long>(i + r) * lda;
                C[static_cast<long>(i + r) * ldc + j] = dot(a_row, b_row, K);
            }
        }
    }

    // M remainder (M % kIR): leftover rows, vectorized over K per column
    for (; i < M; ++i) {
        const float* a_row = A + static_cast<long>(i) * lda;
        float* c_row = C + static_cast<long>(i) * ldc;
        for (int j = 0; j < N; ++j) {
            const float* b_row = B + static_cast<long>(j) * ldb;
            c_row[j] = dot(a_row, b_row, K);
        }
    }
}
