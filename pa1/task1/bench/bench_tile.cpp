// bench_tile.cpp -- standalone analysis tool, NOT part of the graded build.
//
// Times and perf-profiles ONE tiling shape (row or square) at an EXPLICIT
// tile size, so collect_tile.py can sweep tile size x matrix size x kernel
// size for the Task 1B report. Kept separate from bench_stage.cpp because
// tile size is an extra parameter that doesn't fit ConvFn's fixed signature.
//
// Two modes (mirroring bench_stage.cpp):
//   time <row|square> H W K tileSize [warmup] [reps] [seed]
//   loop <row|square> H W K tileSize reps [seed] [--verify]

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "convolution.h"
#include "timer.h"
#include "utils.h"

extern void conv_tile_row_ts(const float*, float*, const float*, int, int, int, int);
extern void conv_tile_sq_ts(const float*, float*, const float*, int, int, int, int);

namespace {
void usage(const char* prog) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s time <row|square> H W K tileSize [warmup] [reps] [seed]\n"
        "  %s loop <row|square> H W K tileSize reps [seed] [--verify]\n",
        prog, prog);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) { usage(argv[0]); return 1; }

    const char* mode  = argv[1];
    const char* shape = argv[2];
    const int H = std::atoi(argv[3]);
    const int W = std::atoi(argv[4]);
    const int K = std::atoi(argv[5]);
    const int tileSize = std::atoi(argv[6]);

    bool is_row;
    if (std::strcmp(shape, "row") == 0) is_row = true;
    else if (std::strcmp(shape, "square") == 0) is_row = false;
    else { std::fprintf(stderr, "error: shape must be 'row' or 'square'\n"); return 1; }

    if (H <= 0 || W <= 0 || K <= 0 || W % 8 != 0 || K % 2 == 0 || tileSize <= 0) {
        std::fprintf(stderr, "error: need H,W>0, W%%8==0, K odd, tileSize>0 "
                              "(got H=%d W=%d K=%d tileSize=%d)\n", H, W, K, tileSize);
        return 1;
    }

    unsigned seed = 1234u;
    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);

    auto call = [&](const float* in_, float* out_) {
        if (is_row) conv_tile_row_ts(in_, out_, ker, H, W, K, tileSize);
        else        conv_tile_sq_ts(in_, out_, ker, H, W, K, tileSize);
    };

    if (std::strcmp(mode, "time") == 0) {
        const int warmup = argc > 7 ? std::atoi(argv[7]) : 2;
        const int reps   = argc > 8 ? std::atoi(argv[8]) : 7;
        if (argc > 9) seed = static_cast<unsigned>(std::strtoul(argv[9], nullptr, 10));

        auto run = [&]() { call(in, out); };
        const double ms = pa1::time_median_ms(run, warmup, reps);
        const double gflops = pa1::conv_flops(H, W, K) / (ms * 1e6);
        std::printf("%.6f,%.6f\n", ms, gflops);

    } else if (std::strcmp(mode, "loop") == 0) {
        if (argc < 8) { usage(argv[0]); return 1; }
        const int reps = std::atoi(argv[7]);
        bool verify = false;
        for (int i = 8; i < argc; ++i) {
            if (std::strcmp(argv[i], "--verify") == 0) verify = true;
            else seed = static_cast<unsigned>(std::strtoul(argv[i], nullptr, 10));
        }

        if (verify) {
            float* ref = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
            conv_naive(in, ref, ker, H, W, K);
            call(in, out);
            const float err = pa1::max_abs_diff(out, ref, H, W);
            std::fprintf(stderr, "verify %s(ts=%d) H=%d W=%d K=%d: max_abs_diff=%.6g (%s)\n",
                         shape, tileSize, H, W, K, err, err <= 1e-3f ? "OK" : "FAIL");
            pa1::free_floats(ref);
        }

        for (int r = 0; r < reps; ++r) call(in, out);

    } else {
        usage(argv[0]);
        return 1;
    }

    pa1::free_floats(in);
    pa1::free_floats(img);
    pa1::free_floats(ker);
    pa1::free_floats(out);
    return 0;
}
