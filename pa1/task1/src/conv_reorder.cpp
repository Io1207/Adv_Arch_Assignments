// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"
#include <cstring>

void conv_reorder(const float* in, float* out, const float* ker,
                   int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        std::memset(out_row, 0, sizeof(float) * static_cast<std::size_t>(W));
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                const float w = ker[ky * K + kx];
                const float* in_row = in + (oy + ky) * in_stride + kx;
                for (int ox = 0; ox < W; ++ox) {
                    out_row[ox] += in_row[ox] * w;
                }
            }
        }
    }
}
