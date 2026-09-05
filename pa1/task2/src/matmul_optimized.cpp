// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>
#include <algorithm>
#include <cstring>

#include "matmul.h"

namespace {
    constexpr int kMTile = 128;
    constexpr int kNTile = 128;
    constexpr int kKTile = 512;
    constexpr int kPrefetchDistance = 1;
    constexpr int kCacheLineFloats = 64 / sizeof(float);

    inline float hsum256(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }
    inline float partial_dot(const float* a, const float* b, int p_start, int p_end) {
        __m256 acc = _mm256_setzero_ps();
        int p = p_start;
        for (; p + 8 <= p_end; p += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+p), _mm256_loadu_ps(b+p), acc);
        float sum = hsum256(acc);
        for (; p < p_end; ++p) sum += a[p]*b[p];
        return sum;
    }

    constexpr int kIR = 4;
    constexpr int kJR = 2;
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    
    for (int i = 0; i < M; ++i)
        std::memset(C + static_cast<long>(i)*ldc, 0, sizeof(float)*static_cast<std::size_t>(N));

    for (int ii = 0; ii < M; ii += kMTile) {
        const int i_end = std::min(ii + kMTile, M);
        for (int jj = 0; jj < N; jj += kNTile) {
            const int j_end = std::min(jj + kNTile, N);
            for (int pp = 0; pp < K; pp += kKTile) {
                const int p_end = std::min(pp + kKTile, K);
                const int next_pp = pp + kKTile * kPrefetchDistance;
                if (next_pp < K) {
                    const int next_p_end = std::min(next_pp + kKTile, K);
                    for (int i = ii; i < i_end; ++i) {
                        const float* a_row = A + static_cast<long>(i)*lda;
                        for (int p = next_pp; p < next_p_end; p += kCacheLineFloats)
                            _mm_prefetch(reinterpret_cast<const char*>(a_row+p), _MM_HINT_T0);
                    }
                    for (int j = jj; j < j_end; ++j) {
                        const float* b_row = B + static_cast<long>(j)*ldb;
                        for (int p = next_pp; p < next_p_end; p += kCacheLineFloats)
                            _mm_prefetch(reinterpret_cast<const char*>(b_row+p), _MM_HINT_T0);
                    }
                }
                int i = ii;
                for (; i + kIR <= i_end; i += kIR) {
                    const float* a0 = A+(long)(i+0)*lda; const float* a1 = A+(long)(i+1)*lda;
                    const float* a2 = A+(long)(i+2)*lda; const float* a3 = A+(long)(i+3)*lda;
                    float* c0 = C+(long)(i+0)*ldc; float* c1 = C+(long)(i+1)*ldc;
                    float* c2 = C+(long)(i+2)*ldc; float* c3 = C+(long)(i+3)*ldc;
                    int j = jj;
                    for (; j + kJR <= j_end; j += kJR) {
                        const float* b0 = B+(long)(j+0)*ldb; const float* b1 = B+(long)(j+1)*ldb;
                        __m256 a00=_mm256_setzero_ps(),a01=_mm256_setzero_ps();
                        __m256 a10=_mm256_setzero_ps(),a11=_mm256_setzero_ps();
                        __m256 a20=_mm256_setzero_ps(),a21=_mm256_setzero_ps();
                        __m256 a30=_mm256_setzero_ps(),a31=_mm256_setzero_ps();
                        int p = pp;
                        for (; p+8 <= p_end; p += 8) {
                            const __m256 bv0=_mm256_loadu_ps(b0+p), bv1=_mm256_loadu_ps(b1+p);
                            __m256 av=_mm256_loadu_ps(a0+p); a00=_mm256_fmadd_ps(av,bv0,a00); a01=_mm256_fmadd_ps(av,bv1,a01);
                            av=_mm256_loadu_ps(a1+p); a10=_mm256_fmadd_ps(av,bv0,a10); a11=_mm256_fmadd_ps(av,bv1,a11);
                            av=_mm256_loadu_ps(a2+p); a20=_mm256_fmadd_ps(av,bv0,a20); a21=_mm256_fmadd_ps(av,bv1,a21);
                            av=_mm256_loadu_ps(a3+p); a30=_mm256_fmadd_ps(av,bv0,a30); a31=_mm256_fmadd_ps(av,bv1,a31);
                        }
                        float s00=hsum256(a00),s01=hsum256(a01),s10=hsum256(a10),s11=hsum256(a11);
                        float s20=hsum256(a20),s21=hsum256(a21),s30=hsum256(a30),s31=hsum256(a31);
                        for (; p < p_end; ++p) {
                            s00+=a0[p]*b0[p]; s01+=a0[p]*b1[p];
                            s10+=a1[p]*b0[p]; s11+=a1[p]*b1[p];
                            s20+=a2[p]*b0[p]; s21+=a2[p]*b1[p];
                            s30+=a3[p]*b0[p]; s31+=a3[p]*b1[p];
                        }
                        c0[j+0]+=s00; c0[j+1]+=s01; c1[j+0]+=s10; c1[j+1]+=s11;
                        c2[j+0]+=s20; c2[j+1]+=s21; c3[j+0]+=s30; c3[j+1]+=s31;
                    }
                    for (; j < j_end; ++j) {
                        const float* b_row = B+(long)j*ldb;
                        c0[j]+=partial_dot(a0,b_row,pp,p_end); c1[j]+=partial_dot(a1,b_row,pp,p_end);
                        c2[j]+=partial_dot(a2,b_row,pp,p_end); c3[j]+=partial_dot(a3,b_row,pp,p_end);
                    }
                }
                for (; i < i_end; ++i) {
                    const float* a_row = A+(long)i*lda; float* c_row = C+(long)i*ldc;
                    for (int j = jj; j < j_end; ++j) {
                        const float* b_row = B+(long)j*ldb;
                        c_row[j]+=partial_dot(a_row,b_row,pp,p_end);
                    }
                }
            }
        }
    }
}