// conv_tile.cpp  STAGE 3: CACHE TILING
//
// Both shapes keep the NAIVE loop nest order (oy, ox, ky, kx) INSIDE a tile
// -- no reordering, no unrolling, that's conv_reorder.cpp / conv_unroll.cpp's
// job. Only an outer blocking loop over oy (row tiling) or oy AND ox (square
// tiling) is added, so the K input rows (row tiling) or K input rows x K
// input columns (square tiling) a tile needs stay resident in L1D across
// every pixel's K*K taps in that tile.
//
//   conv_tile_row_ts(..., tileRows) : block oy only,      tile is tileRows x W.
//   conv_tile_sq_ts (..., tileDim ) : block oy AND ox,    tile is tileDim x tileDim.
//
// Both take an EXPLICIT tile size so task1/bench/collect_tile.py can sweep
// tile size x matrix size x kernel size and measure which size (and which
// shape) actually minimizes L1-D MPKI / execution time on this machine,
// instead of trusting the capacity formula below blindly.
//
// conv_tile (graded/submitted) and conv_tile_sq (kept for the shape
// comparison, not called by main.cpp) default that formula: size the tile so
// the working set -- one tile's input rows/columns plus its output --
// divides evenly into L1D. L1D = 48 KiB is this machine's PER-CORE L1 data
// cache (confirmed via /sys/devices/system/cpu/cpu2/cache/index1/size on a
// P-core; the E-cores report 32 KiB, but task1/bench/ pins to a P-core).

#include "convolution.h"
#include <algorithm>
#include <cmath>

namespace {
constexpr int KiB = 1024;
constexpr int L1D = 48 * KiB;
}

void conv_tile_row_ts(const float* in, float* out, const float* ker,
                       int H, int W, int K, int tileRows) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    tileRows = std::max(tileRows, 1);

    for (int oy0 = 0; oy0 < H; oy0 += tileRows) {
        const int oy_end = std::min(oy0 + tileRows, H);
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

void conv_tile_sq_ts(const float* in, float* out, const float* ker,
                      int H, int W, int K, int tileDim) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    tileDim = std::max(tileDim, 1);

    for (int oy0 = 0; oy0 < H; oy0 += tileDim) {
        const int oy_end = std::min(oy0 + tileDim, H);
        for (int ox0 = 0; ox0 < W; ox0 += tileDim) {
            const int ox_end = std::min(ox0 + tileDim, W);
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

// conv_tile: the symbol main.cpp calls and the one that's graded/submitted.
//
// Fixed tileRows=16, NOT the L1D-capacity formula below: task1/bench/
// collect_tile.py swept fixed row-tile heights (16/32/64/128/256) and the
// formula's own output against naive across H,W in {128..4096} and K in
// {3,5,7,9} (logs/task1b_tile.csv). At large W the formula collapses to a
// degenerate 1-2 row tile (e.g. tileRows=2 at W=2048,K=3) that measured
// *slower* than naive (0.97x); tileRows=16 measured 1.11x at that same
// config and was competitive across the whole swept grid, so it's used as a
// fixed constant instead.
void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    conv_tile_row_ts(in, out, ker, H, W, K, 16);
}

// Square tiling, same L1D-capacity budget solved for a square footprint.
// Not called by main.cpp; kept for the row-vs-square comparison in the report.
void conv_tile_sq(const float* in, float* out, const float* ker,
                   int H, int W, int K) {
    const int unit_sz = static_cast<int>(sizeof(float));
    const int tileDim = static_cast<int>(
        std::sqrt(static_cast<double>(L1D - unit_sz * K * K) / (2 * unit_sz)));
    conv_tile_sq_ts(in, out, ker, H, W, K, tileDim);
}
