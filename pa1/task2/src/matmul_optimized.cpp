// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <array>

#include "matmul.h"

constexpr int kIR = 4;
constexpr int kJR = 2;
constexpr int kNumAcc = 1;

namespace {

    constexpr int kMTile = 128;
    constexpr int kNTile = 128;
    constexpr int kKTile = 512;
    constexpr int kPrefetchDistance = 1;
    constexpr int kCacheLineFloats = 64 / sizeof(float);
    constexpr int kSuperStep = 8 * kNumAcc;

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

    inline void micro_kernel_full(const float* A, const float* B, float* C, int i0, int j0,
                                  int pp, int p_end, int lda, int ldb, int ldc) {
        std::array<const float*, kIR> a_rows;
        std::array<const float*, kJR> b_rows;
        #pragma GCC unroll 16
        for (int r = 0; r < kIR; ++r) a_rows[r] = A + static_cast<long>(i0 + r) * lda;
        #pragma GCC unroll 16
        for (int c = 0; c < kJR; ++c) b_rows[c] = B + static_cast<long>(j0 + c) * ldb;

        // acc[r][c][a]: kNumAcc independent accumulators per (r, c) position
        std::array<std::array<std::array<__m256, kNumAcc>, kJR>, kIR> acc{};
        #pragma GCC unroll 16
        for (int r = 0; r < kIR; ++r)
            #pragma GCC unroll 16
            for (int c = 0; c < kJR; ++c)
                #pragma GCC unroll 16
                for (int a = 0; a < kNumAcc; ++a) acc[r][c][a] = _mm256_setzero_ps();

        int p = pp;
        for (; p + kSuperStep <= p_end; p += kSuperStep) {
            std::array<std::array<__m256, kNumAcc>, kJR> bv;
            #pragma GCC unroll 16
            for (int c = 0; c < kJR; ++c)
                #pragma GCC unroll 16
                for (int a = 0; a < kNumAcc; ++a) bv[c][a] = _mm256_loadu_ps(b_rows[c] + p + a * 8);

            #pragma GCC unroll 16
            for (int r = 0; r < kIR; ++r) {
                std::array<__m256, kNumAcc> av;
                #pragma GCC unroll 16
                for (int a = 0; a < kNumAcc; ++a) av[a] = _mm256_loadu_ps(a_rows[r] + p + a * 8);
                #pragma GCC unroll 16
                for (int c = 0; c < kJR; ++c) {
                    #pragma GCC unroll 16
                    for (int a = 0; a < kNumAcc; ++a) {
                        acc[r][c][a] = _mm256_fmadd_ps(av[a], bv[c][a], acc[r][c][a]);
                    }
                }
            }
        }

        
        float sums[kIR][kJR];
        #pragma GCC unroll 16
        for (int r = 0; r < kIR; ++r) {
            #pragma GCC unroll 16
            for (int c = 0; c < kJR; ++c) {
                __m256 combined = acc[r][c][0];
                #pragma GCC unroll 16
                for (int a = 1; a < kNumAcc; ++a) combined = _mm256_add_ps(combined, acc[r][c][a]);
                sums[r][c] = hsum256(combined);
            }
        }

        for (; p + 8 <= p_end; p += 8) {
            std::array<__m256, kJR> bv;
            #pragma GCC unroll 16
            for (int c = 0; c < kJR; ++c) bv[c] = _mm256_loadu_ps(b_rows[c] + p);
            #pragma GCC unroll 16
            for (int r = 0; r < kIR; ++r) {
                __m256 av = _mm256_loadu_ps(a_rows[r] + p);
                #pragma GCC unroll 16
                for (int c = 0; c < kJR; ++c) sums[r][c] += hsum256(_mm256_mul_ps(av, bv[c]));
            }
        }

        for (; p < p_end; ++p) {
            #pragma GCC unroll 16
            for (int r = 0; r < kIR; ++r) {
                const float av = a_rows[r][p];
                #pragma GCC unroll 16
                for (int c = 0; c < kJR; ++c) sums[r][c] += av * b_rows[c][p];
            }
        }

        #pragma GCC unroll 16
        for (int r = 0; r < kIR; ++r) {
            float* c_row = C + static_cast<long>(i0 + r) * ldc;
            #pragma GCC unroll 16
            for (int c = 0; c < kJR; ++c) c_row[j0 + c] += sums[r][c];
        }
    }

}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        std::memset(C + static_cast<long>(i) * ldc, 0,
                    sizeof(float) * static_cast<std::size_t>(N));
    }

    for (int ii = 0; ii < M; ii += kMTile) {
        const int i_end = std::min(ii + kMTile, M);
        for (int jj = 0; jj < N; jj += kNTile) {
            const int j_end = std::min(jj + kNTile, N);

            for (int pp = 0; pp < K; pp += kKTile) {
                const int p_end = std::min(pp + kKTile, K);

                const int next_pp = pp + kKTile * kPrefetchDistance;
                if (next_pp < K) {
                    const int next_p_end = std::min(next_pp + kKTile, K);
                    for (int i = ii; i < i_end; ++i) {
                        const float* a_row = A + static_cast<long>(i) * lda;
                        for (int p = next_pp; p < next_p_end; p += kCacheLineFloats)
                            _mm_prefetch(reinterpret_cast<const char*>(a_row + p), _MM_HINT_T0);
                    }
                    for (int j = jj; j < j_end; ++j) {
                        const float* b_row = B + static_cast<long>(j) * ldb;
                        for (int p = next_pp; p < next_p_end; p += kCacheLineFloats)
                            _mm_prefetch(reinterpret_cast<const char*>(b_row + p), _MM_HINT_T0);
                    }
                }

                int i = ii;
                for (; i + kIR <= i_end; i += kIR) {
                    int j = jj;
                    for (; j + kJR <= j_end; j += kJR) {
                        micro_kernel_full(A, B, C, i, j, pp, p_end, lda, ldb, ldc);
                    }
                    for (; j < j_end; ++j) {  // N remainder within tile
                        const float* b_row = B + static_cast<long>(j) * ldb;
                        for (int r = 0; r < kIR; ++r) {
                            const float* a_row = A + static_cast<long>(i + r) * lda;
                            C[static_cast<long>(i + r) * ldc + j] += partial_dot(a_row, b_row, pp, p_end);
                        }
                    }
                }
                for (; i < i_end; ++i) {  // M remainder within tile
                    const float* a_row = A + static_cast<long>(i) * lda;
                    float* c_row = C + static_cast<long>(i) * ldc;
                    for (int j = jj; j < j_end; ++j) {
                        const float* b_row = B + static_cast<long>(j) * ldb;
                        c_row[j] += partial_dot(a_row, b_row, pp, p_end);
                    }
                }
            }
        }
    }
}