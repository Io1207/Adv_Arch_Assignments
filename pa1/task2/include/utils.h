// utils.h  allocation, initialization, and correctness helpers (CS683 PA-1, Task 2)
//
// Header-only. Students generally do not need to touch this file; the harness and the
// reference implementation use it to build buffers and check results consistently.

#ifndef CS683_PA1_UTILS_H
#define CS683_PA1_UTILS_H

#include <immintrin.h>  // _mm_malloc / _mm_free (64-byte aligned allocation)

#include <cmath>
#include <cstddef>
#include <random>

namespace pa1 {

// 64-byte aligned float allocation (AVX-512 cache line; also fine for AVX2).
inline float* alloc_floats(std::size_t n) {
    return static_cast<float*>(_mm_malloc(n * sizeof(float), 64));
}

inline void free_floats(float* p) { _mm_free(p); }

// Fill a buffer with deterministic pseudo-random values in [-1, 1].
inline void fill_random(float* buf, std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

// Maximum absolute value in a buffer (used to form a relative tolerance).
inline float max_abs(const float* a, std::size_t n) {
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float v = std::fabs(a[i]);
        if (v > m) m = v;
    }
    return m;
}

// Maximum absolute elementwise difference between two buffers (the correctness metric).
inline float max_abs_diff(const float* a, const float* b, std::size_t n) {
    float m = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Floating-point operations for one M x N x K matmul: 1 multiply + 1 add per MAC.
inline double matmul_flops(int M, int N, int K) {
    return 2.0 * static_cast<double>(M) * static_cast<double>(N) *
           static_cast<double>(K);
}

}  // namespace pa1

#endif  // CS683_PA1_UTILS_H
