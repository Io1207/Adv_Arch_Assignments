// conv_tile.cpp  STAGE 3: CACHE TILING
#include "convolution.h"

namespace {
    constexpr int kTileRows = 16;
}

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy0 = 0; oy0 < H; oy0 += kTileRows) {
        const int oy_end = (oy0 + kTileRows < H) ? (oy0 + kTileRows) : H;
        for (int oy = oy0; oy < oy_end; ++oy) {
            for (int ox = 0; ox < W; ++ox) {
                float acc = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    }
                }
                out[oy * W + ox] = acc;
            }
        }
    }
}
