// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
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
