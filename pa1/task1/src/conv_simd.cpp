// conv_simd.cpp  STAGE 4: SIMD
//
// Both widths keep the NAIVE loop nest order (oy, ox, ky, kx) -- no
// reordering, no cache tiling. They're literally the SIMD version of
// conv_unroll.cpp's conv_unroll_ox8: instead of 8 independent SCALAR
// accumulators covering 8 output columns, one VECTOR register holds all of
// them, and the per-tap broadcast weight * load becomes one _mm*_fmadd_ps.
//
//   conv_simd128 : __m128  (SSE,  4 floats/register) -- ox unrolled by 4.
//   conv_simd256 : __m256  (AVX2, 8 floats/register) -- ox unrolled by 8.
//
// 512-bit (AVX-512) is intentionally not implemented: `lscpu` reports no
// avx512* flags on this machine (12th-gen client part, E-cores disable
// AVX-512 on the P-cores too), so there is no register to target.
//
// W is guaranteed a multiple of 8 (hence also of 4), so neither width needs
// a scalar remainder tail.

#include <immintrin.h>
#include "convolution.h"

void conv_simd128(const float* in, float* out, const float* ker,
                   int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        for (int ox = 0; ox < W; ox += 4) {
            __m128 acc = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_base = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m128 w   = _mm_set1_ps(ker_row[kx]);
                    const __m128 inv = _mm_loadu_ps(in_base + kx);
                    acc = _mm_fmadd_ps(inv, w, acc);
                }
            }
            _mm_storeu_ps(out_row + ox, acc);
        }
    }
}

void conv_simd256(const float* in, float* out, const float* ker,
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

// conv_simd: the symbol main.cpp calls and the one that's graded/submitted.
// 256-bit (AVX2) -- the widest register actually available on this machine,
// and the one the pinned Makefile flags (-mavx2 -mfma) target.
void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    conv_simd256(in, out, ker, H, W, K);
}
