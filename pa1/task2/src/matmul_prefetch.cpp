// matmul_prefetch.cpp — STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>
#include <algorithm>

#include "matmul.h"

const int MTILE = 32;
const int NTILE = 32;
const int KTILE = 32;

const int PREFETCH_DISTANCE = 3;
const int CACHE_LINE_SIZE = 64;
const int CACHE_FILL_LEVEL = 3; // 0 -> PREFETCHNTA, 1 -> PREFETCH2, 2 -> PREFETCH1, 3 (default) -> PREFETCH0
// PREFETCHNTA -> Fetched into a location that won't evict your regularly-reused working set
// PREFETCH2 -> Fetch relatively far out, towards L3
// PREFETCH1 -> Fetch into L2 and beyond
// PREFETCH0 -> Fetch into L1 (For reusing soon)

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {

    const int ELEMENTS_PER_LINE = CACHE_LINE_SIZE / sizeof(float);

    // Zero C once upfront, since we accumulate into it across pp blocks
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[static_cast<long>(i) * ldc + j] = 0.0f;
        }
    }

    for (int ii = 0; ii < M; ii += MTILE) {
        const int i_end = std::min(ii + MTILE, M);
        for (int jj = 0; jj < N; jj += NTILE) {
            const int j_end = std::min(jj + NTILE, N);
            for (int pp = 0; pp < K; pp += KTILE) {
                const int p_end = std::min(pp + KTILE, K);
                // ---- Block-level prefetch: bring in the next pp-tile ----
                const int next_pp = pp + KTILE * PREFETCH_DISTANCE;
                if (next_pp < K) {
                    const int next_p_end = std::min(next_pp + KTILE, K);
                    // A tile: rows ii..i_end, columns next_pp..next_p_end
                    for (int i = ii; i < i_end; ++i) {
                        for (int p = next_pp; p < next_p_end; p += ELEMENTS_PER_LINE) {
                            __builtin_prefetch(&A[static_cast<long>(i) * lda + p], 0, CACHE_FILL_LEVEL);
                        }
                    }
                    // B tile: rows jj..j_end, columns next_pp..next_p_end (B is N x K, row-major)
                    for (int j = jj; j < j_end; ++j) {
                        for (int p = next_pp; p < next_p_end; p += ELEMENTS_PER_LINE) {
                            __builtin_prefetch(&B[static_cast<long>(j) * ldb + p], 0, CACHE_FILL_LEVEL);
                        }
                    }
                }

                // ---- Compute the current block ----
                for (int i = ii; i < i_end; ++i) {
                    const float* a_row = &A[static_cast<long>(i) * lda];

                    for (int j = jj; j < j_end; ++j) {
                        const float* b_row = &B[static_cast<long>(j) * ldb];
                        float sum = 0.0f;

                        for (int p = pp; p < p_end; ++p) {
                            sum += a_row[p] * b_row[p];
                        }

                        C[static_cast<long>(i) * ldc + j] += sum;
                    }
                }
            }
        }
    }
}
