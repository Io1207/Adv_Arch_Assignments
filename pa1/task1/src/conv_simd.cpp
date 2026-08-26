// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"
#include <cstring>

namespace {
constexpr int kTileRows = 32;
}
 
void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
 
    std::memset(out, 0, sizeof(float) * static_cast<std::size_t>(H) * W);
 
    for (int oy0 = 0; oy0 < H; oy0 += kTileRows) {
        const int oy_end = (oy0 + kTileRows < H) ? (oy0 + kTileRows) : H;
 
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                const __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                for (int oy = oy0; oy < oy_end; ++oy) {
                    const float* in_row = in + (oy + ky) * in_stride + kx;
                    float* out_row = out + oy * W;
                    for (int ox = 0; ox < W; ox += 8) {
                        __m256 acc = _mm256_loadu_ps(out_row + ox);
                        __m256 inv = _mm256_loadu_ps(in_row + ox);
                        acc = _mm256_fmadd_ps(inv, w, acc);
                        _mm256_storeu_ps(out_row + ox, acc);
                    }
                }
            }
        }
    }
}
