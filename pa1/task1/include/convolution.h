// convolution.h  shared interface for all convolution stages (CS683 PA-1, Task 1)
//
// Every stage implements the SAME signature so the harness can call them uniformly
// and compare their results and timings.
//
// Data layout (2D single-channel float32 convolution, zero-padded "same" mode):
//
//   in  : PADDED input image, (H + 2p) rows x (W + 2p) cols, row-major.
//         Row stride is (W + 2p) elements, where p = K/2 (K is odd).
//         The p-wide border is zero, so every read inside the loops is in-bounds
//         and NO boundary checks are needed.
//   out : output image, H rows x W cols, row-major, row stride = W.
//   ker : convolution kernel, K x K, row-major (row stride = K).
//
// Definition (cross-correlation, i.e. the ML convention  the kernel is NOT flipped):
//
//   out[y*W + x] = sum_{ky=0..K-1} sum_{kx=0..K-1}
//                      in[(y+ky)*(W+2p) + (x+kx)] * ker[ky*K + kx]
//
// W is guaranteed to be a multiple of 8 so the AVX2 (8-wide float) inner loop over
// x has no scalar remainder tail.

#ifndef CS683_PA1_CONVOLUTION_H
#define CS683_PA1_CONVOLUTION_H

// Function-pointer type used by the harness to iterate over the stages.
using ConvFn = void (*)(const float* in, float* out, const float* ker,
                        int H, int W, int K);

// ---- The six stages -------------------------------------------------------
// conv_naive is PROVIDED and serves as the correctness reference.
// The rest are student TODOs (see src/*.cpp).

void conv_naive(const float* in, float* out, const float* ker, int H, int W, int K);
void conv_reorder(const float* in, float* out, const float* ker, int H, int W, int K);
void conv_unroll(const float* in, float* out, const float* ker, int H, int W, int K);
void conv_tile(const float* in, float* out, const float* ker, int H, int W, int K);
void conv_simd(const float* in, float* out, const float* ker, int H, int W, int K);
void conv_optimized(const float* in, float* out, const float* ker, int H, int W, int K);

#endif  // CS683_PA1_CONVOLUTION_H
