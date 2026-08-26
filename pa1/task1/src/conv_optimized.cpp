// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
 
    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;
        // Process two 8-wide (16 column) blocks per iteration with two
        // independent accumulators: this exposes enough independent FMA
        // chains to hide the FMA pipeline latency (the single-accumulator
        // version above stalls waiting on `acc`'s dependency chain).
        for (; ox + 16 <= W; ox += 16) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx), w, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 8 + kx), w, acc1);
                }
            }
            _mm256_storeu_ps(out_row + ox, acc0);
            _mm256_storeu_ps(out_row + ox + 8, acc1);
        }
        for (; ox < W; ox += 8) {  // remainder (never taken: W % 16 handled below)
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx), w, acc);
                }
            }
            _mm256_storeu_ps(out_row + ox, acc);
        }
    }
}
