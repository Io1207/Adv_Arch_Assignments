// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// GEBP (Goto / BLIS) kernel: AVX2+FMA micro-kernel, cache blocking, on-the-fly packing,
// and software prefetch. Tuned for the i7-12650H P-core (L1d 48 KB, L2 1.25 MB, L3 24 MB).
//
//   packed A panel   Mc x Kc          -> resident in L2
//   packed B panel   Kc x N           -> resident in L3
//     its Kc x Nr sliver              -> resident in L1
//   C micro-tile     Mr x Nr = 6 x 16 -> 12 YMM accumulators, held across the whole Kc
//
// B arrives transposed (row j contiguous over the contraction index). pack_B gathers it
// into k-major, Nr-contiguous panels so the micro-kernel is a pure broadcast+FMA stream
// with NO horizontal reductions and an _mm256 add/store straight back into C.

#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

#include "matmul.h"

namespace {

    constexpr int kMr  = 6;              // register-tile rows
    constexpr int kNr  = 16;             // register-tile cols  (2 x __m256)
    constexpr int kNrV = kNr / 8;        // vector columns per tile

    constexpr int kKc = 256;             // K block
    constexpr int kMc = 192;             // M block (multiple of kMr) -> packed A stays in L2
    // Nc == N : the packed B panel lives in L3

    constexpr int kPfDist = 16;          // k look-ahead for the in-kernel prefetch

    inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

    inline float* alloc_aligned(std::size_t nfloats) {
        std::size_t bytes = ((nfloats * sizeof(float)) + 63) & ~std::size_t(63);
        return static_cast<float*>(std::aligned_alloc(64, bytes));
    }

    // A[ic:ic+mc, pc:pc+kc] (row-major, stride lda)  ->  row-contiguous micro-panels
    //   Apack[panel*(kMr*kc) + r*kc + k]   (ragged last panel zero-padded)
    // The micro-kernel only broadcasts from A, so no transpose is needed: each row is a
    // straight memcpy (vectorized libc, unaffected by -fno-tree-vectorize).
    void pack_A(const float* __restrict__ A, int lda, int ic, int mc, int pc, int kc,
                float* __restrict__ Apack) {
        const int panels = round_up(mc, kMr) / kMr;
        const std::size_t rb = sizeof(float) * static_cast<std::size_t>(kc);
        for (int panel = 0; panel < panels; ++panel) {
            float* dst = Apack + static_cast<long>(panel) * kMr * kc;
            const int r0 = panel * kMr;
            const int rn = std::min(kMr, mc - r0);
            for (int r = 0; r < rn; ++r)
                std::memcpy(dst + r * kc, A + static_cast<long>(ic + r0 + r) * lda + pc, rb);
            for (int r = rn; r < kMr; ++r)
                std::memset(dst + r * kc, 0, rb);
        }
    }

    // B (transposed: row j contiguous over p, stride ldb), cols [0:n), slice [pc:pc+kc]
    //   ->  Bpack[panel*(kc*kNr) + k*kNr + c]   (ragged last panel zero-padded)
    void pack_B(const float* __restrict__ B, int ldb, int n, int pc, int kc,
                float* __restrict__ Bpack) {
        const int panels = round_up(n, kNr) / kNr;
        for (int panel = 0; panel < panels; ++panel) {
            float* dst = Bpack + static_cast<long>(panel) * kc * kNr;
            const int c0 = panel * kNr;
            const int cn = std::min(kNr, n - c0);
            for (int c = 0; c < cn; ++c) {
                const float* s = B + static_cast<long>(c0 + c) * ldb + pc;
                for (int k = 0; k < kc; ++k) dst[k * kNr + c] = s[k];
            }
            for (int c = cn; c < kNr; ++c)
                for (int k = 0; k < kc; ++k) dst[k * kNr + c] = 0.0f;
        }
    }

    // C[i0:i0+mr, j0:j0+nr] += Apanel(kMr x kc) . Bpanel(kc x kNr)   over kc.
    // Apanel / Bpanel are contiguous and 32-byte aligned.
    inline void micro_kernel(const float* __restrict__ Ap, const float* __restrict__ Bp,
                             float* __restrict__ C, int i0, int j0, int kc,
                             int mr, int nr, int ldc) {
        std::array<std::array<__m256, kNrV>, kMr> acc;
        #pragma GCC unroll 8
        for (int r = 0; r < kMr; ++r)
            #pragma GCC unroll 8
            for (int v = 0; v < kNrV; ++v) acc[r][v] = _mm256_setzero_ps();

        for (int k = 0; k < kc; ++k) {
            std::array<__m256, kNrV> bv;
            #pragma GCC unroll 8
            for (int v = 0; v < kNrV; ++v) bv[v] = _mm256_load_ps(Bp + k * kNr + v * 8);

            _mm_prefetch(reinterpret_cast<const char*>(Bp + (k + kPfDist) * kNr), _MM_HINT_T0);

            #pragma GCC unroll 8
            for (int r = 0; r < kMr; ++r) {
                const __m256 av = _mm256_broadcast_ss(Ap + r * kc + k);
                #pragma GCC unroll 8
                for (int v = 0; v < kNrV; ++v)
                    acc[r][v] = _mm256_fmadd_ps(av, bv[v], acc[r][v]);
            }
        }

        if (mr == kMr && nr == kNr) {
            #pragma GCC unroll 8
            for (int r = 0; r < kMr; ++r) {
                float* cr = C + static_cast<long>(i0 + r) * ldc + j0;
                #pragma GCC unroll 8
                for (int v = 0; v < kNrV; ++v)
                    _mm256_storeu_ps(cr + v * 8,
                                     _mm256_add_ps(_mm256_loadu_ps(cr + v * 8), acc[r][v]));
            }
        } else {
            alignas(32) float buf[kMr][kNr];
            #pragma GCC unroll 8
            for (int r = 0; r < kMr; ++r)
                #pragma GCC unroll 8
                for (int v = 0; v < kNrV; ++v) _mm256_store_ps(buf[r] + v * 8, acc[r][v]);
            for (int r = 0; r < mr; ++r) {
                float* cr = C + static_cast<long>(i0 + r) * ldc + j0;
                for (int c = 0; c < nr; ++c) cr[c] += buf[r][c];
            }
        }
    }

    // Scalar fallback used if the pack buffers can't be allocated (never expected).
    void matmul_ref(const float* A, const float* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float s = 0.0f;
                for (int p = 0; p < K; ++p)
                    s += A[static_cast<long>(i) * lda + p] * B[static_cast<long>(j) * ldb + p];
                C[static_cast<long>(i) * ldc + j] = s;
            }
    }
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i)
        std::memset(C + static_cast<long>(i) * ldc, 0,
                    sizeof(float) * static_cast<std::size_t>(N));

    const int Mc_pad = round_up(kMc, kMr);
    const int N_pad  = round_up(N, kNr);
    float* Apack = alloc_aligned(static_cast<std::size_t>(Mc_pad) * kKc);
    float* Bpack = alloc_aligned(static_cast<std::size_t>(std::max(N_pad, kNr)) * kKc);
    if (!Apack || !Bpack) {
        std::free(Apack);
        std::free(Bpack);
        matmul_ref(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }

    for (int pc = 0; pc < K; pc += kKc) {              // K panels — C accumulates
        const int kc = std::min(kKc, K - pc);
        pack_B(B, ldb, N, pc, kc, Bpack);              // -> L3

        for (int ic = 0; ic < M; ic += kMc) {          // M panels
            const int mc = std::min(kMc, M - ic);
            pack_A(A, lda, ic, mc, pc, kc, Apack);      // -> L2
            const int mpanels = round_up(mc, kMr) / kMr;

            for (int jr = 0; jr < N; jr += kNr) {       // B micro-panels
                const int nr = std::min(kNr, N - jr);
                const float* Bp = Bpack + static_cast<long>(jr / kNr) * kc * kNr;
                _mm_prefetch(reinterpret_cast<const char*>(Bp + kc * kNr), _MM_HINT_T0);

                for (int q = 0; q < mpanels; ++q) {     // A micro-panels
                    const int i0 = ic + q * kMr;
                    const int mr = std::min(kMr, mc - q * kMr);
                    const float* Ap = Apack + static_cast<long>(q) * kMr * kc;
                    _mm_prefetch(reinterpret_cast<const char*>(Ap + kMr * kc), _MM_HINT_T1);
                    for (int r = 0; r < mr; ++r)
                        _mm_prefetch(reinterpret_cast<const char*>(
                                         C + static_cast<long>(i0 + r) * ldc + jr),
                                     _MM_HINT_T0);
                    micro_kernel(Ap, Bp, C, i0, jr, kc, mr, nr, ldc);
                }
            }
        }
    }

    std::free(Apack);
    std::free(Bpack);
}
