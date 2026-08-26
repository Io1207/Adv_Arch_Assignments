// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include <cstring>
#include <cmath>

namespace {
constexpr int KiB = 1024;
constexpr int L1D = 32 * KiB; 
}
 
void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int unit_sz = int(sizeof(*in));
    const int kTileRows = (L1D - unit_sz * K * K) / (2 * unit_sz * W); 
 
    std::memset(out, 0, sizeof(float) * static_cast<std::size_t>(H) * W);
 
    for (int oy0 = 0; oy0 < H; oy0 += kTileRows) {
        const int oy_end = (oy0 + kTileRows < H) ? (oy0 + kTileRows) : H;
 
        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                const float w = ker[ky * K + kx];
                for (int oy = oy0; oy < oy_end; ++oy) {
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
//                    for (; ox < W; ++ox) out_row[ox] += in_row[ox] * w;
                }
            }
        }
    }
}

void conv_tile_sq(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int unit_sz = int(sizeof(*in));
    const int kTileRows = std::sqrt((L1D - unit_sz * K * K) / (2 * unit_sz)); 
 
    std::memset(out, 0, sizeof(float) * static_cast<std::size_t>(H) * W);
 
    for (int oy0 = 0; oy0 < H; oy0 += kTileRows) {
        const int oy_end = (oy0 + kTileRows < H) ? (oy0 + kTileRows) : H;
	for (int ox0 = 0; ox0 < W; ox0 += kTileRows) {
		const int ox_end = (ox0 + kTileRows < W) ? (ox0 + kTileRows) : W;
        	for (int ky = 0; ky < K; ++ky) {
            		for (int kx = 0; kx < K; ++kx) {
                		const float w = ker[ky * K + kx];
                		for (int oy = oy0; oy < oy_end; ++oy) {
                    			const float* in_row = in + (oy + ky) * in_stride + kx;
                    			float* out_row = out + oy * W;
                    			int ox = ox0;
                    			for (; ox + 8 <= ox_end; ox += 8) {
                        			out_row[ox + 0] += in_row[ox + 0] * w;
                        			out_row[ox + 1] += in_row[ox + 1] * w;
                        			out_row[ox + 2] += in_row[ox + 2] * w;
                        			out_row[ox + 3] += in_row[ox + 3] * w;
                        			out_row[ox + 4] += in_row[ox + 4] * w;
                        			out_row[ox + 5] += in_row[ox + 5] * w;
                        			out_row[ox + 6] += in_row[ox + 6] * w;
                        			out_row[ox + 7] += in_row[ox + 7] * w;
                    			}
                    			for (; ox < ox_end; ++ox) out_row[ox] += in_row[ox] * w;
                		}
            		}
        	}
    	}
    }
}
