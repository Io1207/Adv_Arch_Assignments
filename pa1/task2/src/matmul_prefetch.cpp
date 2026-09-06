// matmul_prefetch.cpp — STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCH (scalar, no SIMD)
//
// Goto/BLIS-style layering, sized for the i7-12650H P-core
// (L1d 48 KB / 12-way, L2 1.25 MB, L3 24 MB):
//
//   packed A macro-panel   Mc x Kc  (128 x 256 -> 128 KB)  ....  resident in L2
//   packed B macro-panel   Kc x N                          ....  resident in L3
//     its  Kc x Nr  sliver  (256 x 3 -> 3 KB)              ....  resident in L1
//   C micro-tile           Mr x Nr  (4 x 3)                ....  held in registers over all Kc
//
// Packing makes A and B contiguous so the panels actually stay resident instead of
// conflict-missing (L1 has only 64 sets). _mm_prefetch then pulls the *next* micro-panel
// a short distance ahead. The micro-kernel is 12 independent scalar FMA chains, enough
// to hide scalar FMA latency (~4 cyc, 2/cyc issue) with no vector instructions.

#include <immintrin.h>   // _mm_prefetch only
#include <algorithm>
#include <cstring>
#include <vector>
#include "matmul.h"

namespace {

    constexpr int kKc = 256;   // K block
    constexpr int kMc = 128;   // M block  -> packed A + C block both fit L2
    constexpr int kMr = 4;     // register tile rows
    constexpr int kNr = 3;     // register tile cols   (kMr*kNr = 12 scalar accumulators)

    constexpr int kLine = 64 / sizeof(float);   // floats per cache line
    constexpr int kPfK  = 8;                     // prefetch look-ahead inside the k-loop

    inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

    // A[ic:ic+mc, pc:pc+kc] (row-major, stride lda) -> micro-panel order:
    //   Apack[panel*(kMr*kc) + k*kMr + r],  ragged last panel zero-padded.
    void pack_A(const float* __restrict__ A, int lda, int ic, int mc, int pc, int kc,
                float* __restrict__ Apack) {
        const int panels = round_up(mc, kMr) / kMr;
        for (int panel = 0; panel < panels; ++panel) {
            float* dst = Apack + static_cast<long>(panel) * kMr * kc;
            const int r0 = panel * kMr;
            const int rn = std::min(kMr, mc - r0);
            for (int r = 0; r < rn; ++r) {
                const float* src = A + static_cast<long>(ic + r0 + r) * lda + pc;
                _mm_prefetch(reinterpret_cast<const char*>(src + kLine), _MM_HINT_T0);
                for (int k = 0; k < kc; ++k) dst[k * kMr + r] = src[k];
            }
            for (int r = rn; r < kMr; ++r)
                for (int k = 0; k < kc; ++k) dst[k * kMr + r] = 0.0f;
        }
    }

    // B is transposed: row j is contiguous over p (stride ldb). Slice [pc:pc+kc] of
    // rows [0:n) -> micro-panel order:
    //   Bpack[panel*(kNr*kc) + k*kNr + c],  ragged last panel zero-padded.
    void pack_B(const float* __restrict__ B, int ldb, int n, int pc, int kc,
                float* __restrict__ Bpack) {
        const int panels = round_up(n, kNr) / kNr;
        for (int panel = 0; panel < panels; ++panel) {
            float* dst = Bpack + static_cast<long>(panel) * kNr * kc;
            const int c0 = panel * kNr;
            const int cn = std::min(kNr, n - c0);
            for (int c = 0; c < cn; ++c) {
                const float* src = B + static_cast<long>(c0 + c) * ldb + pc;
                _mm_prefetch(reinterpret_cast<const char*>(src + kLine), _MM_HINT_T0);
                for (int k = 0; k < kc; ++k) dst[k * kNr + c] = src[k];
            }
            for (int c = cn; c < kNr; ++c)
                for (int k = 0; k < kc; ++k) dst[k * kNr + c] = 0.0f;
        }
    }

    // C[i0:i0+mr, j0:j0+nr] += Apanel(kMr x kc) . Bpanel(kc x kNr) over kc.
    // 12 scalar accumulators live in registers for the whole k-loop.
    inline void micro_kernel(const float* __restrict__ Apanel, const float* __restrict__ Bpanel,
                             float* __restrict__ C, int i0, int j0, int kc,
                             int mr, int nr, int ldc) {
        float c00 = 0, c01 = 0, c02 = 0;
        float c10 = 0, c11 = 0, c12 = 0;
        float c20 = 0, c21 = 0, c22 = 0;
        float c30 = 0, c31 = 0, c32 = 0;

        #pragma GCC unroll 4
        for (int k = 0; k < kc; ++k) {
            const float* ap = Apanel + k * kMr;
            const float* bp = Bpanel + k * kNr;
            _mm_prefetch(reinterpret_cast<const char*>(ap + kPfK * kMr), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(bp + kPfK * kNr), _MM_HINT_T0);
            const float a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
            const float b0 = bp[0], b1 = bp[1], b2 = bp[2];
            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2;
        }

        const float res[kMr][kNr] = {{c00, c01, c02}, {c10, c11, c12},
                                     {c20, c21, c22}, {c30, c31, c32}};
        for (int r = 0; r < mr; ++r) {
            float* c_row = C + static_cast<long>(i0 + r) * ldc + j0;
            for (int c = 0; c < nr; ++c) c_row[c] += res[r][c];
        }
    }
}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i)
        std::memset(C + static_cast<long>(i) * ldc, 0,
                    sizeof(float) * static_cast<std::size_t>(N));

    std::vector<float> Apack(static_cast<std::size_t>(round_up(kMc, kMr)) * kKc);
    std::vector<float> Bpack(static_cast<std::size_t>(round_up(N, kNr)) * kKc);

    for (int pc = 0; pc < K; pc += kKc) {             // K panels — accumulate into C
        const int kc = std::min(kKc, K - pc);

        pack_B(B, ldb, N, pc, kc, Bpack.data());      // -> L3

        for (int ic = 0; ic < M; ic += kMc) {         // M panels
            const int mc = std::min(kMc, M - ic);
            pack_A(A, lda, ic, mc, pc, kc, Apack.data());   // -> L2

            const int mpanels = round_up(mc, kMr) / kMr;
            for (int jr = 0; jr < N; jr += kNr) {      // B micro-panels
                const int nr = std::min(kNr, N - jr);
                const float* Bpanel =
                    Bpack.data() + static_cast<long>(jr / kNr) * kNr * kc;
                _mm_prefetch(reinterpret_cast<const char*>(Bpanel + kNr * kc), _MM_HINT_T0);

                for (int p = 0; p < mpanels; ++p) {    // A micro-panels
                    const int i0 = ic + p * kMr;
                    const int mr = std::min(kMr, mc - p * kMr);
                    const float* Apanel =
                        Apack.data() + static_cast<long>(p) * kMr * kc;
                    _mm_prefetch(reinterpret_cast<const char*>(Apanel + kMr * kc), _MM_HINT_T1);
                    micro_kernel(Apanel, Bpanel, C, i0, jr, kc, mr, nr, ldc);
                }
            }
        }
    }
}
