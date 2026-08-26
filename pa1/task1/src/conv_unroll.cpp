// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"
#include <cstring>

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
 
    std::memset(out, 0, sizeof(float) * static_cast<std::size_t>(H) * W);
 
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            const float w = ker[ky * K + kx];
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row = in + (oy + ky) * in_stride + kx;
                float* out_row = out + oy * W;
 
                int ox = 0;
                for (; ox + 8 <= W; ox += 8) {
                    out_row[ox + 0] += in_row[ox + 0] * w;
                    out_row[ox + 1] += in_row[ox + 1] * w;
                    out_row[ox + 2] += in_row[ox + 2] * w;
                    out_row[ox + 3] += in_row[ox + 3] * w;
                    out_row[ox + 4] += in_row[ox + 4] * w;
                    out_row[ox + 5] += in_row[ox + 5] * w;
                    out_row[ox + 6] += in_row[ox + 6] * w;
                    out_row[ox + 7] += in_row[ox + 7] * w;
                }
                for (; ox < W; ++ox) {
                    out_row[ox] += in_row[ox] * w;
                }
            }
        }
    }
}