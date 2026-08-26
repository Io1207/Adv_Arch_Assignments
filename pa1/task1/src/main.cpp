// main.cpp  PROVIDED test / grading harness for Task 1.  Do NOT modify.
//
// No arguments -> full graded run (correctness smoke test, graded K=3 workload with a
// score, and a reference K=5 table).
//
// You can also check ONE stage in isolation on a workload of your choice while you are
// developing it, e.g. right after you finish conv_reorder:
//
//     ./conv reorder                 # naive vs reorder, default 2048x2048 K=3
//     ./conv reorder 512 512 3       # custom H W K
//     ./conv reorder 512 512 3 42    # custom H W K and RNG seed
//     ./conv all 1024 1024 5         # every stage on a custom workload (no score)
//     ./conv help                    # usage
//
// The result of every stage is always checked against the naive reference (max abs
// error < 1e-3), so a passing "correct" column means your kernel is numerically right
// on that input; the random seed lets you try many different inputs.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "convolution.h"
#include "timer.h"
#include "utils.h"

// ------------------------------- configuration -----------------------------
// Correctness tolerance (max absolute difference vs the naive reference).
static constexpr float kTol = 1e-3f;

// Timing: warmups (untimed) then timed repetitions; the median is reported.
static constexpr int kWarmup = 2;
static constexpr int kReps = 7;

// Default workload and seed.
static constexpr int kDefH = 2048, kDefW = 2048, kDefK = 3;
static constexpr unsigned kDefSeed = 1234u;

// Scoring knobs (calibrated against solution/; see README rubric).
// Scored stages are everything except naive: reorder, unroll, tile, simd, optimized.
static constexpr double kCorrectPtsPerStage = 8.0;  // -> 5 * 8 = 40 correctness points

struct Tier {
    double speedup;
    double points;
};
// Cumulative tiers for conv_optimized speedup vs naive (highest reached wins).
// CALIBRATION: these are tuned so the reference solution (~8-9x at K=3 on a Skylake/
// Alder Lake class core) reaches the top tier. Re-run `make solution && ./conv_solution`
// on your lab machines and adjust these thresholds so a correct reference lands at 60.
static constexpr Tier kTiers[] = {
    {2.0, 15.0},
    {4.0, 30.0},
    {6.0, 45.0},
    {8.0, 60.0},
};
static constexpr double kMaxTierPts = 60.0;  // = kTiers[last].points

struct Stage {
    const char* name;  // display name
    const char* key;   // short name for the command line
    ConvFn fn;
    bool scored;  // false only for the naive reference
};

static const Stage kStages[] = {
    {"naive (ref)", "naive", conv_naive, false},
    {"reorder", "reorder", conv_reorder, true},
    {"unroll", "unroll", conv_unroll, true},
    {"tile", "tile", conv_tile, true},
    {"simd", "simd", conv_simd, true},
    {"optimized", "optimized", conv_optimized, true},
};
static constexpr int kNumStages = sizeof(kStages) / sizeof(kStages[0]);

// ------------------------------- one workload ------------------------------
// Runs the stages on an (H x W, K) workload.
//   only   : nullptr / "all" -> every stage; otherwise just naive + the named stage.
//   show   : print the per-stage timing table.
//   scored : also compute and print the autograder score (returned).
// When !scored the function returns the number of INCORRECT scored stages that ran.
static double run_config(int H, int W, int K, unsigned seed, const char* only,
                         bool show, bool scored) {
    float* img = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ker = pa1::alloc_floats(static_cast<std::size_t>(K) * K);
    float* out = pa1::alloc_floats(static_cast<std::size_t>(H) * W);
    float* ref = pa1::alloc_floats(static_cast<std::size_t>(H) * W);

    pa1::fill_random(img, static_cast<std::size_t>(H) * W, seed);
    pa1::fill_random(ker, static_cast<std::size_t>(K) * K, seed + 1u);
    float* in = pa1::make_padded(img, H, W, K);  // zero-padded halo buffer, stride W+2p

    const bool all = (only == nullptr) || (std::strcmp(only, "all") == 0);

    // Reference output from the naive kernel.
    conv_naive(in, ref, ker, H, W, K);

    const double flops = pa1::conv_flops(H, W, K);
    double naive_ms = 0.0;
    double optimized_speedup = 0.0;
    bool optimized_ok = false;
    int correct_scored = 0;
    int ran_scored = 0;
    int incorrect_scored = 0;

    if (show) {
        std::printf("\n=== Workload  H=%d W=%d K=%d seed=%u %s ===\n", H, W, K, seed,
                    scored ? "(GRADED)" : "(not graded)");
        std::printf("%-14s  %-7s  %10s  %10s  %9s\n", "stage", "correct", "time(ms)",
                    "GFLOP/s", "speedup");
        std::printf("--------------------------------------------------------------\n");
    }

    for (int s = 0; s < kNumStages; ++s) {
        const bool is_naive = (kStages[s].fn == conv_naive);
        // In single-stage mode, always run naive (baseline/reference) + the chosen stage.
        if (!all && !is_naive && std::strcmp(kStages[s].key, only) != 0) continue;

        auto run = [&]() { kStages[s].fn(in, out, ker, H, W, K); };
        run();  // one untimed run to produce the result we check
        const bool ok = (pa1::max_abs_diff(out, ref, H, W) <= kTol);
        if (kStages[s].scored) {
            ++ran_scored;
            if (ok) ++correct_scored; else ++incorrect_scored;
        }

        const double ms = pa1::time_median_ms(run, kWarmup, kReps);
        const double gflops = flops / (ms * 1e6);  // ms->s and FLOP->GFLOP
        if (is_naive) naive_ms = ms;
        const double speedup = (ms > 0.0) ? naive_ms / ms : 0.0;
        if (kStages[s].fn == conv_optimized) {
            optimized_speedup = speedup;
            optimized_ok = ok;
        }

        if (show) {
            std::printf("%-14s  %-7s  %10.3f  %10.2f  %8.2fx\n", kStages[s].name,
                        ok ? "yes" : "NO", ms, gflops, speedup);
        }
    }

    pa1::free_floats(in);
    pa1::free_floats(img);
    pa1::free_floats(ker);
    pa1::free_floats(out);
    pa1::free_floats(ref);

    if (!scored) return static_cast<double>(incorrect_scored);

    // ---- scoring (full graded run only) ------------------------------------
    const double correct_pts = correct_scored * kCorrectPtsPerStage;

    double tier_pts = 0.0;
    if (optimized_ok) {  // no speedup credit for an incorrect optimized kernel
        for (const Tier& t : kTiers) {
            if (optimized_speedup >= t.speedup) tier_pts = t.points;
        }
    }

    const double max_correct_pts = 5 * kCorrectPtsPerStage;
    const double total = correct_pts + tier_pts;
    std::printf("--------------------------------------------------------------\n");
    std::printf("correctness : %.0f / %.0f   (%d of 5 stages correct)\n", correct_pts,
                max_correct_pts, correct_scored);
    std::printf("optimized   : %.2fx%s  ->  %.0f / %.0f  speedup points\n",
                optimized_speedup, optimized_ok ? "" : " (INCORRECT)", tier_pts,
                kMaxTierPts);
    std::printf("TOTAL       : %.0f / %.0f\n", total, max_correct_pts + kMaxTierPts);
    return total;
}

static void usage(const char* prog) {
    std::printf("Usage:\n");
    std::printf("  %s                      full graded run (score printed)\n", prog);
    std::printf("  %s <stage> [H W K] [seed]   check one stage on a workload\n", prog);
    std::printf("  %s all [H W K] [seed]       run every stage (no score)\n", prog);
    std::printf("\nstage: naive | reorder | unroll | tile | simd | optimized | all\n");
    std::printf("H, W, K default to %d %d %d; seed defaults to %u.\n", kDefH, kDefW,
                kDefK, kDefSeed);
    std::printf("Constraints: W must be a multiple of 8, K must be odd and >= 1.\n");
    std::printf("\nExamples:\n");
    std::printf("  %s reorder            %s reorder 512 512 3            %s all 1024 1024 5\n",
                prog, prog, prog);
}

// Returns the matching stage key from kStages (or "all"), else nullptr.
static const char* match_stage(const char* s) {
    if (std::strcmp(s, "all") == 0) return "all";
    for (int i = 0; i < kNumStages; ++i)
        if (std::strcmp(s, kStages[i].key) == 0) return kStages[i].key;
    return nullptr;
}

int main(int argc, char** argv) {
    std::printf("CS683 PA-1 Task 1  2D convolution optimization harness\n");

    // ---- full graded run (no arguments) ------------------------------------
    if (argc == 1) {
        double bad = 0.0;
        bad += run_config(256, 256, 3, kDefSeed, nullptr, /*show=*/false, /*scored=*/false);
        bad += run_config(256, 256, 5, kDefSeed, nullptr, /*show=*/false, /*scored=*/false);
        if (bad > 0.0)
            std::printf("[smoke] %d scored stage(s) differ from reference on small inputs.\n",
                        static_cast<int>(bad));
        else
            std::printf("[smoke] all stages match the reference on small inputs.\n");

        run_config(kDefH, kDefW, kDefK, kDefSeed, nullptr, /*show=*/true, /*scored=*/true);
        // Reference K=5 table (not graded): tiling and SIMD help more as the kernel grows.
        run_config(2048, 2048, 5, kDefSeed, nullptr, /*show=*/true, /*scored=*/false);
        return 0;
    }

    // ---- single-stage / custom-workload mode -------------------------------
    if (std::strcmp(argv[1], "help") == 0 || std::strcmp(argv[1], "-h") == 0 ||
        std::strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }

    const char* stage = match_stage(argv[1]);
    if (!stage) {
        std::printf("error: unknown stage '%s'\n\n", argv[1]);
        usage(argv[0]);
        return 1;
    }

    int H = kDefH, W = kDefW, K = kDefK;
    unsigned seed = kDefSeed;
    if (argc >= 5) {
        H = std::atoi(argv[2]);
        W = std::atoi(argv[3]);
        K = std::atoi(argv[4]);
    } else if (argc != 2) {
        std::printf("error: give either no dimensions, or all three (H W K).\n\n");
        usage(argv[0]);
        return 1;
    }
    if (argc >= 6) seed = static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));

    if (H <= 0 || W <= 0 || K <= 0) {
        std::printf("error: H, W, K must be positive.\n");
        return 1;
    }
    if (W % 8 != 0) {
        std::printf("error: W (%d) must be a multiple of 8.\n", W);
        return 1;
    }
    if (K % 2 == 0) {
        std::printf("error: K (%d) must be odd.\n", K);
        return 1;
    }

    run_config(H, W, K, seed, stage, /*show=*/true, /*scored=*/false);
    return 0;
}
