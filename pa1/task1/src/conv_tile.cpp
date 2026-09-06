// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

namespace {
constexpr int kTileDim = 16;
}

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy0 = 0; oy0 < H; oy0 += kTileDim) {
        const int oy_end = (oy0 + kTileDim < H) ? (oy0 + kTileDim) : H;
        for (int ox0 = 0; ox0 < W; ox0 += kTileDim) {
            const int ox_end = (ox0 + kTileDim < W) ? (ox0 + kTileDim) : W;
            for (int oy = oy0; oy < oy_end; ++oy) {
                for (int ox = ox0; ox < ox_end; ++ox) {
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
}
