// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

namespace {

    constexpr int kJR = 4; // number of B columns processed per A load - register tile width

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

    // Vectorized dot product of length K between a and b arbitrary base both with length >= K
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
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a_row = A + static_cast<long>(i) * lda;
        float* c_row = C + static_cast<long>(i) * ldc;

        int j = 0;
        for (; j + kJR <= N; j += kJR) {
            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int p = 0;
            for (; p + 8 <= K; p += 8) {
                const __m256 avec = _mm256_loadu_ps(a_row + p);
                acc0 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(b0 + p), acc0);
                acc1 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(b1 + p), acc1);
                acc2 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(b2 + p), acc2);
                acc3 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(b3 + p), acc3);
            }

            float s0 = hsum256(acc0), s1 = hsum256(acc1);
            float s2 = hsum256(acc2), s3 = hsum256(acc3);
            for (; p < K; ++p) {
                const float av = a_row[p];
                s0 += av * b0[p];
                s1 += av * b1[p];
                s2 += av * b2[p];
                s3 += av * b3[p];
            }

            c_row[j + 0] = s0;
            c_row[j + 1] = s1;
            c_row[j + 2] = s2;
            c_row[j + 3] = s3;
        }

        // Remainder columns (N % kJR != 0): still vectorized over K
        for (; j < N; ++j) {
            const float* b_row = B + static_cast<long>(j) * ldb;
            c_row[j] = dot(a_row, b_row, K);
        }
    }
}