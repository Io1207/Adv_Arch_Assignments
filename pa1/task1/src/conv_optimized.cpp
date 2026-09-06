// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps -- let the
// numbers, not intuition, decide.
//
// This "combines" SIMD with a deeper form of unrolling: instead of one
// __m256 accumulator per 8-wide chunk (conv_simd256), UNROLL THE VECTOR
// REGISTER ITSELF -- N independent __m256 accumulators processing N*8
// columns per outer-loop iteration. Naive nest order throughout (oy, ox, ky,
// kx); no explicit reordering or cache tiling -- task1/bench/logs/
// task1b_tile.csv already showed tiling barely helps here because naive's
// own L1D MPKI is tiny (0.02-0.20) at every size/kernel tested: this
// workload is compute/ILP-bound, not memory-bound, on this machine. Each
// N*8-wide chunk is finished (all K*K taps, one store) before the next
// chunk starts, which is already the same cache-friendly access pattern
// conv_reorder_oy_ky_kx_ox exploits, just realized implicitly by the chunk
// granularity instead of an explicit outer reorder.
//
// Why unroll the accumulator at all: a single __m256 FMA chain (conv_simd256)
// has ~4 cycles of latency but the port throughput is ~2 FMAs/cycle, so one
// chain leaves the FMA units stalled waiting on their own last result most
// of the time. N independent chains give the out-of-order engine N
// independent chains to interleave, hiding that latency -- up to the point
// where N*(registers per chain) exceeds the 16 architectural YMM registers
// and the compiler starts spilling accumulators to the stack, which is
// *worse* than not unrolling at all. task1/bench/collect_optimized.py sweeps
// N in {1,2,4,8,16} (chunk width 8..128) against naive to find where that
// crossover actually is on this CPU instead of assuming it.
//
// Every Nacc variant is hand-written with named acc0..accN-1 registers
// (never a `__m256 acc[N]` array indexed by a runtime loop variable): an
// array isn't guaranteed to be promoted to registers, and if it isn't, every
// FMA becomes a load-fmadd-store round trip through memory -- worse than a
// single scalar accumulator. A shared 8-wide, single-accumulator remainder
// loop (conv_optimized_tail8) handles whatever's left after the last full
// N*8-wide chunk -- for this assignment's W (always a multiple of 8), that
// remainder is itself always a multiple of 8, so no scalar tail is needed
// anywhere.

#include <immintrin.h>
#include "convolution.h"

namespace {

// Shared remainder handler: processes ox..W-1 (a multiple of 8, since W
// always is) 8 columns at a time with one accumulator. Used by every Nacc
// variant below for whatever doesn't divide evenly into its N*8-wide chunk.
inline void conv_optimized_tail8(const float* in, float* out, const float* ker,
                                  int W, int K, int in_stride, int oy, int ox0) {
    for (int ox = ox0; ox < W; ox += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int ky = 0; ky < K; ++ky) {
            const float* in_row  = in + (oy + ky) * in_stride + ox;
            const float* ker_row = ker + ky * K;
            for (int kx = 0; kx < K; ++kx) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx),
                                       _mm256_set1_ps(ker_row[kx]), acc);
            }
        }
        _mm256_storeu_ps(out + oy * W + ox, acc);
    }
}

}  // namespace

// N=1: identical to conv_simd256 -- the zero-unroll reference point for the
// sweep.
void conv_optimized_acc1(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    for (int oy = 0; oy < H; ++oy) {
        conv_optimized_tail8(in, out, ker, W, K, in_stride, oy, 0);
    }
}

// N=2: 16 columns/iteration.
void conv_optimized_acc2(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;
        for (; ox + 16 <= W; ox += 16) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row  = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker_row[kx]);
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 0 * 8 + kx), w, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 1 * 8 + kx), w, acc1);
                }
            }
            _mm256_storeu_ps(out_row + ox + 0 * 8, acc0);
            _mm256_storeu_ps(out_row + ox + 1 * 8, acc1);
        }
        conv_optimized_tail8(in, out, ker, W, K, in_stride, oy, ox);
    }
}

// N=4: 32 columns/iteration.
void conv_optimized_acc4(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;
        for (; ox + 32 <= W; ox += 32) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row  = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker_row[kx]);
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 0 * 8 + kx), w, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 1 * 8 + kx), w, acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 2 * 8 + kx), w, acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 3 * 8 + kx), w, acc3);
                }
            }
            _mm256_storeu_ps(out_row + ox + 0 * 8, acc0);
            _mm256_storeu_ps(out_row + ox + 1 * 8, acc1);
            _mm256_storeu_ps(out_row + ox + 2 * 8, acc2);
            _mm256_storeu_ps(out_row + ox + 3 * 8, acc3);
        }
        conv_optimized_tail8(in, out, ker, W, K, in_stride, oy, ox);
    }
}

// N=8: 64 columns/iteration. Expected sweet spot: AVX2 FMA has ~4 cycles
// latency and ~2/cycle throughput, so latency / (1/throughput) = 4/0.5 = 8
// independent chains are needed to keep the FMA units saturated.
void conv_optimized_acc8(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;
        for (; ox + 64 <= W; ox += 64) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row  = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker_row[kx]);
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 0 * 8 + kx), w, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 1 * 8 + kx), w, acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 2 * 8 + kx), w, acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 3 * 8 + kx), w, acc3);
                    acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 4 * 8 + kx), w, acc4);
                    acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 5 * 8 + kx), w, acc5);
                    acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 6 * 8 + kx), w, acc6);
                    acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 7 * 8 + kx), w, acc7);
                }
            }
            _mm256_storeu_ps(out_row + ox + 0 * 8, acc0);
            _mm256_storeu_ps(out_row + ox + 1 * 8, acc1);
            _mm256_storeu_ps(out_row + ox + 2 * 8, acc2);
            _mm256_storeu_ps(out_row + ox + 3 * 8, acc3);
            _mm256_storeu_ps(out_row + ox + 4 * 8, acc4);
            _mm256_storeu_ps(out_row + ox + 5 * 8, acc5);
            _mm256_storeu_ps(out_row + ox + 6 * 8, acc6);
            _mm256_storeu_ps(out_row + ox + 7 * 8, acc7);
        }
        conv_optimized_tail8(in, out, ker, W, K, in_stride, oy, ox);
    }
}

// N=16: 128 columns/iteration. Expected to REGRESS vs N=8: 16 accumulators
// alone exhaust all 16 YMM registers, leaving none for `w`/`inv`, forcing
// spills -- kept in the sweep specifically to demonstrate that crossover.
void conv_optimized_acc16(const float* in, float* out, const float* ker,
                           int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;
        for (; ox + 128 <= W; ox += 128) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
            __m256 acc8 = _mm256_setzero_ps(), acc9 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps();
            __m256 acc12 = _mm256_setzero_ps(), acc13 = _mm256_setzero_ps();
            __m256 acc14 = _mm256_setzero_ps(), acc15 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row  = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker_row[kx]);
                    acc0  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  0 * 8 + kx), w, acc0);
                    acc1  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  1 * 8 + kx), w, acc1);
                    acc2  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  2 * 8 + kx), w, acc2);
                    acc3  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  3 * 8 + kx), w, acc3);
                    acc4  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  4 * 8 + kx), w, acc4);
                    acc5  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  5 * 8 + kx), w, acc5);
                    acc6  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  6 * 8 + kx), w, acc6);
                    acc7  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  7 * 8 + kx), w, acc7);
                    acc8  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  8 * 8 + kx), w, acc8);
                    acc9  = _mm256_fmadd_ps(_mm256_loadu_ps(in_row +  9 * 8 + kx), w, acc9);
                    acc10 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 10 * 8 + kx), w, acc10);
                    acc11 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 11 * 8 + kx), w, acc11);
                    acc12 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 12 * 8 + kx), w, acc12);
                    acc13 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 13 * 8 + kx), w, acc13);
                    acc14 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 14 * 8 + kx), w, acc14);
                    acc15 = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + 15 * 8 + kx), w, acc15);
                }
            }
            _mm256_storeu_ps(out_row + ox +  0 * 8, acc0);
            _mm256_storeu_ps(out_row + ox +  1 * 8, acc1);
            _mm256_storeu_ps(out_row + ox +  2 * 8, acc2);
            _mm256_storeu_ps(out_row + ox +  3 * 8, acc3);
            _mm256_storeu_ps(out_row + ox +  4 * 8, acc4);
            _mm256_storeu_ps(out_row + ox +  5 * 8, acc5);
            _mm256_storeu_ps(out_row + ox +  6 * 8, acc6);
            _mm256_storeu_ps(out_row + ox +  7 * 8, acc7);
            _mm256_storeu_ps(out_row + ox +  8 * 8, acc8);
            _mm256_storeu_ps(out_row + ox +  9 * 8, acc9);
            _mm256_storeu_ps(out_row + ox + 10 * 8, acc10);
            _mm256_storeu_ps(out_row + ox + 11 * 8, acc11);
            _mm256_storeu_ps(out_row + ox + 12 * 8, acc12);
            _mm256_storeu_ps(out_row + ox + 13 * 8, acc13);
            _mm256_storeu_ps(out_row + ox + 14 * 8, acc14);
            _mm256_storeu_ps(out_row + ox + 15 * 8, acc15);
        }
        conv_optimized_tail8(in, out, ker, W, K, in_stride, oy, ox);
    }
}

// conv_optimized: the symbol main.cpp calls and the one that's graded/
// submitted. Set to conv_optimized_acc8 pending task1/bench/collect_optimized.py
// confirming N=8 wins on this machine (see the file header for why 8 is
// predicted); swap here if the data says otherwise.
void conv_optimized(const float* in, float* out, const float* ker,
                     int H, int W, int K) {
    conv_optimized_acc8(in, out, ker, H, W, K);
}
