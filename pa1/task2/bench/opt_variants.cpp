// opt_variants.cpp -- bench-only accumulator-count variants of the combined
// kernel (NOT graded, NOT part of src/). Used by task2/bench/collect_optimized.py
// to show the effect of the number of independent SIMD accumulator chains
// ("unroll the vector register itself") on FMA-latency hiding.
//
// Fixed 2x2 register tile (MR rows x NR output columns), AVX2/FMA, naive tile
// iteration order -- the ONLY variable is NACC, the number of __m256
// accumulators per tile position, each covering a strided slice of K
// (p, p+8, ... for accumulator 0; p+8*a for accumulator a). They are summed
// before the final horizontal reduction.
//
//   matmul_optimized_acc1 : NACC = 1   (4  __m256 accumulators total)
//   matmul_optimized_acc2 : NACC = 2   (8  __m256 accumulators total)
//   matmul_optimized_acc4 : NACC = 4   (16 __m256 accumulators total)
//
// The graded, fully-tuned kernel is src/matmul_optimized.cpp; this file only
// feeds the supporting "accumulators vs. speedup" analysis.

#include <immintrin.h>

namespace {

constexpr int MR = 2;
constexpr int NR = 2;

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

inline float dot_tail(const float* a, const float* b, int p, int K) {
    float s = 0.0f;
    for (; p < K; ++p) s += a[p] * b[p];
    return s;
}

template <int NACC>
void kernel(const float* A, const float* B, float* C,
            int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int STEP = 8 * NACC;
    int i = 0;
    for (; i + MR <= M; i += MR) {
        int j = 0;
        for (; j + NR <= N; j += NR) {
            const float* arow[MR];
            const float* brow[NR];
            for (int r = 0; r < MR; ++r) arow[r] = A + static_cast<long>(i + r) * lda;
            for (int c = 0; c < NR; ++c) brow[c] = B + static_cast<long>(j + c) * ldb;

            __m256 acc[MR][NR][NACC];
            for (int r = 0; r < MR; ++r)
                for (int c = 0; c < NR; ++c)
                    for (int a = 0; a < NACC; ++a) acc[r][c][a] = _mm256_setzero_ps();

            int p = 0;
            for (; p + STEP <= K; p += STEP) {
                for (int a = 0; a < NACC; ++a) {
                    const int off = p + a * 8;
                    __m256 bv[NR];
                    for (int c = 0; c < NR; ++c) bv[c] = _mm256_loadu_ps(brow[c] + off);
                    for (int r = 0; r < MR; ++r) {
                        const __m256 av = _mm256_loadu_ps(arow[r] + off);
                        for (int c = 0; c < NR; ++c)
                            acc[r][c][a] = _mm256_fmadd_ps(av, bv[c], acc[r][c][a]);
                    }
                }
            }

            for (int r = 0; r < MR; ++r) {
                float* crow = C + static_cast<long>(i + r) * ldc;
                for (int c = 0; c < NR; ++c) {
                    __m256 sum = acc[r][c][0];
                    for (int a = 1; a < NACC; ++a) sum = _mm256_add_ps(sum, acc[r][c][a]);
                    float s = hsum256(sum);
                    s += dot_tail(arow[r], brow[c], p, K);
                    crow[j + c] = s;
                }
            }
        }
        // N remainder
        for (; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            for (int r = 0; r < MR; ++r) {
                const float* a = A + static_cast<long>(i + r) * lda;
                __m256 acc = _mm256_setzero_ps();
                int p = 0;
                for (; p + 8 <= K; p += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
                C[static_cast<long>(i + r) * ldc + j] = hsum256(acc) + dot_tail(a, b, p, K);
            }
        }
    }
    // M remainder
    for (; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        float* crow = C + static_cast<long>(i) * ldc;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= K; p += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
            crow[j] = hsum256(acc) + dot_tail(a, b, p, K);
        }
    }
}

}  // namespace

void matmul_optimized_acc1(const float* A, const float* B, float* C,
                           int M, int N, int K, int lda, int ldb, int ldc) {
    kernel<1>(A, B, C, M, N, K, lda, ldb, ldc);
}
void matmul_optimized_acc2(const float* A, const float* B, float* C,
                           int M, int N, int K, int lda, int ldb, int ldc) {
    kernel<2>(A, B, C, M, N, K, lda, ldb, ldc);
}
void matmul_optimized_acc4(const float* A, const float* B, float* C,
                           int M, int N, int K, int lda, int ldb, int ldc) {
    kernel<4>(A, B, C, M, N, K, lda, ldb, ldc);
}
