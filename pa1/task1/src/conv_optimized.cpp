// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include "convolution.h"

namespace {

inline void conv_optimized_tail8(const float* in, float* out, const float* ker,
                                  int W, int K, int in_stride, int oy, int ox0) {
    for (int ox = ox0; ox < W; ox += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int ky = 0; ky < K; ++ky) {
            const float* in_row  = in + (oy + ky) * in_stride + ox;
            const float* ker_row = ker + ky * K;
            for (int kx = 0; kx < K; ++kx) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx),
                                       _mm256_set1_ps(ker_row[kx]), acc);
            }
        }
        _mm256_storeu_ps(out + oy * W + ox, acc);
    }
}

}  // namespace

void conv_optimized(const float* in, float* out, const float* ker,
                     int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;
        for (; ox + 64 <= W; ox += 64) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row  = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker_row[kx]);
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 0 * 8 + kx), w, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 1 * 8 + kx), w, acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 2 * 8 + kx), w, acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 3 * 8 + kx), w, acc3);
                    acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 4 * 8 + kx), w, acc4);
                    acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 5 * 8 + kx), w, acc5);
                    acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 6 * 8 + kx), w, acc6);
                    acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 7 * 8 + kx), w, acc7);
                }
            }
            _mm256_storeu_ps(out_row + ox + 0 * 8, acc0);
            _mm256_storeu_ps(out_row + ox + 1 * 8, acc1);
            _mm256_storeu_ps(out_row + ox + 2 * 8, acc2);
            _mm256_storeu_ps(out_row + ox + 3 * 8, acc3);
            _mm256_storeu_ps(out_row + ox + 4 * 8, acc4);
            _mm256_storeu_ps(out_row + ox + 5 * 8, acc5);
            _mm256_storeu_ps(out_row + ox + 6 * 8, acc6);
            _mm256_storeu_ps(out_row + ox + 7 * 8, acc7);
        }
        conv_optimized_tail8(in, out, ker, W, K, in_stride, oy, ox);
    }
}
