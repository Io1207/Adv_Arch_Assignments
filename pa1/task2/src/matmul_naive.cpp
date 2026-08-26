// matmul_naive.cpp  PROVIDED reference implementation.
//
// The correctness baseline AND the performance baseline that all speedups are measured
// against. Do NOT modify this file.

#include "matmul.h"

void matmul_naive(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; ++p) {
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
