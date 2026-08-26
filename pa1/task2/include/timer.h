// timer.h  median wall-clock timing helper (CS683 PA-1, Task 2)
//
// Header-only. Runs a callable `warmup` times (to warm caches / branch predictors),
// then `reps` timed times, and returns the MEDIAN elapsed time in milliseconds.
// The median is robust to occasional scheduling/turbo noise on shared lab machines.

#ifndef CS683_PA1_TIMER_H
#define CS683_PA1_TIMER_H

#include <algorithm>
#include <chrono>
#include <vector>

namespace pa1 {

template <class Fn>
double time_median_ms(Fn&& fn, int warmup, int reps) {
    using clock = std::chrono::steady_clock;
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        auto t0 = clock::now();
        fn();
        auto t1 = clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace pa1

#endif  // CS683_PA1_TIMER_H
