// matmul_prefetch.cpp — STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>
#include <algorithm>
#include <cstring>

#include "matmul.h"

// const int MTILE = 32;
// const int NTILE = 32;
// const int KTILE = 32;

// const int PREFETCH_DISTANCE = 3;
// const int CACHE_LINE_SIZE = 64;
// const int CACHE_FILL_LEVEL = 3; // 0 -> PREFETCHNTA, 1 -> PREFETCH2, 2 -> PREFETCH1, 3 (default) -> PREFETCH0
// // PREFETCHNTA -> Fetched into a location that won't evict your regularly-reused working set
// // PREFETCH2 -> Fetch relatively far out, towards L3
// // PREFETCH1 -> Fetch into L2 and beyond
// // PREFETCH0 -> Fetch into L1 (For reusing soon)

// void matmul_prefetch(const float* A, const float* B, float* C,
//                      int M, int N, int K, int lda, int ldb, int ldc) {

//     const int ELEMENTS_PER_LINE = CACHE_LINE_SIZE / sizeof(float);

//     // Zero C once upfront, since we accumulate into it across pp blocks
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             C[static_cast<long>(i) * ldc + j] = 0.0f;
//         }
//     }

//     for (int ii = 0; ii < M; ii += MTILE) {
//         const int i_end = std::min(ii + MTILE, M);
//         for (int jj = 0; jj < N; jj += NTILE) {
//             const int j_end = std::min(jj + NTILE, N);
//             for (int pp = 0; pp < K; pp += KTILE) {
//                 const int p_end = std::min(pp + KTILE, K);
//                 // ---- Block-level prefetch: bring in the next pp-tile ----
//                 const int next_pp = pp + KTILE * PREFETCH_DISTANCE;
//                 if (next_pp < K) {
//                     const int next_p_end = std::min(next_pp + KTILE, K);
//                     // A tile: rows ii..i_end, columns next_pp..next_p_end
//                     for (int i = ii; i < i_end; ++i) {
//                         for (int p = next_pp; p < next_p_end; p += ELEMENTS_PER_LINE) {
//                             __builtin_prefetch(&A[static_cast<long>(i) * lda + p], 0, CACHE_FILL_LEVEL);
//                         }
//                     }
//                     // B tile: rows jj..j_end, columns next_pp..next_p_end (B is N x K, row-major)
//                     for (int j = jj; j < j_end; ++j) {
//                         for (int p = next_pp; p < next_p_end; p += ELEMENTS_PER_LINE) {
//                             __builtin_prefetch(&B[static_cast<long>(j) * ldb + p], 0, CACHE_FILL_LEVEL);
//                         }
//                     }
//                 }

//                 // ---- Compute the current block ----
//                 for (int i = ii; i < i_end; ++i) {
//                     const float* a_row = &A[static_cast<long>(i) * lda];

//                     for (int j = jj; j < j_end; ++j) {
//                         const float* b_row = &B[static_cast<long>(j) * ldb];
//                         float sum = 0.0f;

//                         for (int p = pp; p < p_end; ++p) {
//                             sum += a_row[p] * b_row[p];
//                         }

//                         C[static_cast<long>(i) * ldc + j] += sum;
//                     }
//                 }
//             }
//         }
//     }
// }

namespace {

    constexpr int kMTile = 64;
    constexpr int kNTile = 64;
    constexpr int kKTile = 256;

    constexpr int kJR = 4;
    constexpr int kPrefetchDistance = 1;
    constexpr int kCacheLineFloats = 64 / sizeof(float);

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

    inline float partial_dot(const float* a, const float* b, int p_start, int p_end) {
        __m256 acc = _mm256_setzero_ps();
        int p = p_start;
        for (; p + 8 <= p_end; p += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
        }
        float sum = hsum256(acc);
        for (; p < p_end; ++p) sum += a[p] * b[p];
        return sum;
    }
}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    
    for (int i = 0; i < M; ++i) {
        std::memset(C + static_cast<long>(i) * ldc, 0, sizeof(float) * static_cast<std::size_t>(N));
    }

    for (int ii = 0; ii < M; ii += kMTile) {
        const int i_end = std::min(ii + kMTile, M);
        for (int jj = 0; jj < N; jj += kNTile) {
            const int j_end = std::min(jj + kNTile, N);

            for (int pp = 0; pp < K; pp += kKTile) {
                const int p_end = std::min(pp + kKTile, K);

                // Software prefetching: bring in the next pptile of A and B
                const int next_pp = pp + kKTile * kPrefetchDistance;
                if (next_pp < K) {
                    const int next_p_end = std::min(next_pp + kKTile, K);
                    for (int i = ii; i < i_end; ++i) {
                        const float* a_row = A + static_cast<long>(i) * lda;
                        for (int p = next_pp; p < next_p_end; p += kCacheLineFloats) {
                            _mm_prefetch(reinterpret_cast<const char*>(a_row + p), _MM_HINT_T0);
                        }
                    }
                    for (int j = jj; j < j_end; ++j) {
                        const float* b_row = B + static_cast<long>(j) * ldb;
                        for (int p = next_pp; p < next_p_end; p += kCacheLineFloats) {
                            _mm_prefetch(reinterpret_cast<const char*>(b_row + p), _MM_HINT_T0);
                        }
                    }
                }

                for (int i = ii; i < i_end; ++i) {
                    const float* a_row = A + static_cast<long>(i) * lda;
                    float* c_row = C + static_cast<long>(i) * ldc;

                    int j = jj;
                    for (; j + kJR <= j_end; j += kJR) {
                        const float* b0 = B + static_cast<long>(j + 0) * ldb;
                        const float* b1 = B + static_cast<long>(j + 1) * ldb;
                        const float* b2 = B + static_cast<long>(j + 2) * ldb;
                        const float* b3 = B + static_cast<long>(j + 3) * ldb;

                        __m256 acc0 = _mm256_setzero_ps();
                        __m256 acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps();
                        __m256 acc3 = _mm256_setzero_ps();

                        int p = pp;
                        for (; p + 8 <= p_end; p += 8) {
                            const __m256 avec = _mm256_loadu_ps(a_row + p);
                            acc0 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(b0 + p), acc0);
                            acc1 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(b1 + p), acc1);
                            acc2 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(b2 + p), acc2);
                            acc3 = _mm256_fmadd_ps(avec, _mm256_loadu_ps(b3 + p), acc3);
                        }
                        float s0 = hsum256(acc0), s1 = hsum256(acc1);
                        float s2 = hsum256(acc2), s3 = hsum256(acc3);
                        for (; p < p_end; ++p) {
                            const float av = a_row[p];
                            s0 += av * b0[p];
                            s1 += av * b1[p];
                            s2 += av * b2[p];
                            s3 += av * b3[p];
                        }
                        c_row[j + 0] += s0;
                        c_row[j + 1] += s1;
                        c_row[j + 2] += s2;
                        c_row[j + 3] += s3;
                    }

                    for (; j < j_end; ++j) {
                        const float* b_row = B + static_cast<long>(j) * ldb;
                        c_row[j] += partial_dot(a_row, b_row, pp, p_end);
                    }
                }
            }
        }
    }
}