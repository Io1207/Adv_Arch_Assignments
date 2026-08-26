// utils.h  allocation, initialization, and correctness helpers (CS683 PA-1, Task 1)
// Header-only. Students generally do not need to touch this file
#ifndef CS683_PA1_UTILS_H
#define CS683_PA1_UTILS_H

#include <immintrin.h>  // _mm_malloc / _mm_free (64-byte aligned allocation)

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>

namespace pa1 {

// 64-byte aligned float allocation (matches an AVX-512 cache line; also fine for AVX2).
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

// Build a zero-padded halo buffer from a logical H x W image.
// Returns a freshly allocated (H + 2p) x (W + 2p) buffer (p = K/2); caller frees it.
inline float* make_padded(const float* img, int H, int W, int K) {
    const int p = K / 2;
    const int PH = H + 2 * p;
    const int PW = W + 2 * p;
    float* padded = alloc_floats(static_cast<std::size_t>(PH) * PW);
    for (int i = 0; i < PH * PW; ++i) padded[i] = 0.0f;
    for (int y = 0; y < H; ++y) {
        const float* src = img + static_cast<std::size_t>(y) * W;
        float* dst = padded + static_cast<std::size_t>(y + p) * PW + p;
        for (int x = 0; x < W; ++x) dst[x] = src[x];
    }
    return padded;
}

// Maximum absolute difference between two H x W images (the correctness metric).
inline float max_abs_diff(const float* a, const float* b, int H, int W) {
    float m = 0.0f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(H) * W; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Floating-point operations for one convolution: 1 multiply + 1 add per kernel tap.
inline double conv_flops(int H, int W, int K) {
    return 2.0 * K * K * static_cast<double>(H) * static_cast<double>(W);
}

}  // namespace pa1

#endif  // CS683_PA1_UTILS_H
