// conv_reorder.cpp  STAGE 1: LOOP REORDERING
//
// Two loop orders, kept as separate named functions so the bench harness
// (task1/bench/) can time and perf-profile both before `conv_reorder` -- the
// symbol the graded main.cpp actually calls -- is fixed to whichever one the
// data supports for the report:
//
//   conv_reorder_ky_kx_oy_ox : ky, kx, oy, ox   (the assignment's suggested order)
//   conv_reorder_oy_ky_kx_ox : oy, ky, kx, ox   (kernel loops moved inside the row loop)
//
// Both compute out[oy,ox] = sum_{ky,kx} in[oy+ky,ox+kx] * ker[ky,kx] via
// read-modify-write accumulation into out[] (no per-pixel register
// accumulator -- that's the unrolling stage's job).

#include "convolution.h"
#include <cstring>

// ky, kx, oy, ox -- kernel-tap loops hoisted outermost. Each of the K*K
// passes sweeps the ENTIRE output image, adding one shifted-and-scaled copy
// of the input to it. The inner ox loop is unit-stride, but the working set
// of each pass is the whole H*W output (plus one input row), not just the K
// input rows a row needs -- so for large images this does not fit L1D/L2 and
// gives little to no benefit over the naive order on its own (see
// conv_tile.cpp for the fix).
void conv_reorder_ky_kx_oy_ox(const float* in, float* out, const float* ker,
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
                for (int ox = 0; ox < W; ++ox) {
                    out_row[ox] += in_row[ox] * w;
                }
            }
        }
    }
}

// oy, ky, kx, ox -- output-row-major order. Every output row is finished (all
// K*K taps applied) before moving to the next row, so the only working set
// that needs to stay resident for a row is the K overlapping input rows plus
// that one output row -- independent of H. This is the same access pattern
// conv_tile.cpp later blocks explicitly; here it falls out of the loop order
// alone, with no explicit tile-size bookkeeping.
void conv_reorder_oy_ky_kx_ox(const float* in, float* out, const float* ker,
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

// conv_reorder: the symbol main.cpp calls and the one that's graded/submitted.
// Defaults to the assignment's suggested order; repoint at
// conv_reorder_oy_ky_kx_ox once task1/bench/logs/ data says which is faster.
void conv_reorder(const float* in, float* out, const float* ker,
                   int H, int W, int K) {
    conv_reorder_ky_kx_oy_ox(in, out, ker, H, W, K);
}
