// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        for (int ox = 0; ox < W; ox += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_base = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w   = _mm256_set1_ps(ker_row[kx]);
                    const __m256 inv = _mm256_loadu_ps(in_base + kx);
                    acc = _mm256_fmadd_ps(inv, w, acc);
                }
            }
            _mm256_storeu_ps(out_row + ox, acc);
        }
    }
}
