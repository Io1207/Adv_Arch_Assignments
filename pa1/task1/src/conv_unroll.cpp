// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
//
// Both variants keep the NAIVE loop nest order (oy, ox, ky, kx) unchanged --
// no loop reordering here, that's conv_reorder.cpp's job. They only differ in
// WHICH loop is unrolled:
//
//   conv_unroll_ox8 : unrolls ox (the outer, large loop -- W, up to thousands)
//                     with 8 INDEPENDENT accumulators, one per output column.
//   conv_unroll_kx4 : unrolls kx (the inner, tiny loop -- K, only 3/5/7/9)
//                     into a single accumulator, i.e. still one dependency
//                     chain per output pixel.
//
// Why ox and not kx/ky: K is small (3-9) and every kx/ky iteration feeds the
// SAME scalar accumulator, so unrolling it only trims loop-branch overhead --
// the FMA dependency chain length (and therefore the latency the CPU has to
// hide) is unchanged. Unrolling ox instead spreads the K*K taps across 8
// independent accumulator chains that the out-of-order engine can execute in
// parallel, which is where the actual speedup comes from. conv_unroll_kx4 is
// kept here purely as the losing comparison point for that argument (see
// task1/bench/logs/task1a_unroll.csv).

#include "convolution.h"

// oy, ox(unrolled x8), ky, kx -- 8 independent __m-free scalar accumulators
// covering 8 output columns at once. Each accumulator is written to out[]
// exactly once (no read-modify-write, no upfront memset needed). W is
// guaranteed a multiple of 8, so there's no scalar remainder tail.
void conv_unroll_ox8(const float* in, float* out, const float* ker,
                      int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        for (int ox = 0; ox + 8 <= W; ox += 8) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const float* in_base = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const float w = ker_row[kx];
                    const float* in_row = in_base + kx;
                    acc0 += in_row[0] * w;
                    acc1 += in_row[1] * w;
                    acc2 += in_row[2] * w;
                    acc3 += in_row[3] * w;
                    acc4 += in_row[4] * w;
                    acc5 += in_row[5] * w;
                    acc6 += in_row[6] * w;
                    acc7 += in_row[7] * w;
                }
            }
            out_row[ox + 0] = acc0;
            out_row[ox + 1] = acc1;
            out_row[ox + 2] = acc2;
            out_row[ox + 3] = acc3;
            out_row[ox + 4] = acc4;
            out_row[ox + 5] = acc5;
            out_row[ox + 6] = acc6;
            out_row[ox + 7] = acc7;
        }
    }
}

// oy, ox, ky, kx(unrolled x4) -- naive nest, single accumulator per output
// pixel, only the kx loop's body is unrolled (with a scalar remainder for
// K % 4 != 0, since K is a runtime value: 3, 5, 7, 9 are all tested). Every
// unrolled term still feeds the SAME `acc`, so this is one long serial FMA
// chain per pixel, same as naive -- just fewer taken branches.
void conv_unroll_kx4(const float* in, float* out, const float* ker,
                      int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                int kx = 0;
                for (; kx + 4 <= K; kx += 4) {
                    acc += in_row[kx + 0] * ker_row[kx + 0];
                    acc += in_row[kx + 1] * ker_row[kx + 1];
                    acc += in_row[kx + 2] * ker_row[kx + 2];
                    acc += in_row[kx + 3] * ker_row[kx + 3];
                }
                for (; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}

// conv_unroll: the symbol main.cpp calls and the one that's graded/submitted.
void conv_unroll(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    conv_unroll_ox8(in, out, ker, H, W, K);
}
