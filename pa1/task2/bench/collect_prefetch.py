#!/usr/bin/env python3
"""collect_prefetch.py -- Task 2A driver (README plots 2 & 3).

Sweeps software-prefetch DISTANCE (floats, issued one pair per 64-byte line)
x locality HINT (NTA / T2 / T1 / T0) x matrix SIZE for
matmul_prefetch_tuned (bench/prefetch_tuned.cpp): a naive (i,j,p) scalar
kernel whose ONLY optimisation is the _mm_prefetch, so the sweep isolates
the distance and cache-fill-level effects. distance 0 == matmul_naive.

One CSV row per (distance, hint, N) to logs/task2a_prefetch.csv:
time / gflops / speedup_vs_naive, plus -- when perf is readable --
instructions, L1D / L2 / LLC misses + MPKI, and sw_prefetch_access.any.

Sizes stop at 2048: matmul_prefetch_tuned is scalar, so a 4096^3 call is
~30 s and the full distance x hint grid there would take ~25 min.

Usage:   ./collect_prefetch.py       (needs ./bench_prefetch built)
perf columns need `sudo sysctl kernel.perf_event_paranoid=1`; without it the
timing/speedup columns are still collected.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "bench_prefetch"
LOG = HERE / "logs" / "task2a_prefetch.csv"

CPU_PIN = 2
SEED = 1234
SIZES = [512, 1024, 2048]
DISTANCES = [0, 16, 64, 256]                    # floats; 0 = no prefetch (== naive)
HINTS = {"NTA": 0, "T2": 1, "T1": 2, "T0": 3}   # matches prefetch_tuned.cpp

PERF_EVENTS = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/",
    "cpu_core/l2_rqsts.miss/",
    "cpu_core/l2_rqsts.references/",
    "cpu_core/longest_lat_cache.miss/",
    "cpu_core/longest_lat_cache.reference/",
    "cpu_core/sw_prefetch_access.any/",
]

TARGET_TIME_MS = 400.0
TARGET_PERF_MS = 300.0


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def perf_available():
    r = run(["perf", "stat", "-x,", "-e", "cpu_core/instructions/", "--", "true"])
    bad = ("not supported", "No supported events", "Access to performance monitoring")
    return not any(b in r.stderr for b in bad)


def verify(M, N, K, dist, hint):
    r = run([str(BENCH), "loop", str(M), str(N), str(K), str(dist), str(hint),
             "1", str(SEED), "--verify"])
    m = re.search(r"max_abs_diff=([\d.eE+-]+) \((OK|FAIL)\)", r.stderr)
    return (m.group(2), float(m.group(1))) if m else ("ERR", r.stderr.strip()[:80])


def probe_ms(M, N, K, dist, hint):
    r = run([str(BENCH), "time", str(M), str(N), str(K), str(dist), str(hint),
             "0", "1", str(SEED)])
    try:
        return float(r.stdout.strip().split(",")[0])
    except Exception:
        return None


def pick_reps(t_ms, target):
    if t_ms is None or t_ms <= 0:
        return 1, 3
    return (1 if t_ms < 100.0 else 0), max(1, min(9, round(target / t_ms)))


def time_run(M, N, K, dist, hint, t_probe):
    warmup, reps = pick_reps(t_probe, TARGET_TIME_MS)
    r = run([str(BENCH), "time", str(M), str(N), str(K), str(dist), str(hint),
             str(warmup), str(reps), str(SEED)])
    ms, gflops = r.stdout.strip().split(",")
    return float(ms), float(gflops)


def perf_run(M, N, K, dist, hint, reps):
    cmd = ["taskset", "-c", str(CPU_PIN), "perf", "stat", "-x,",
           "-e", ",".join(PERF_EVENTS), "--",
           str(BENCH), "loop", str(M), str(N), str(K), str(dist), str(hint),
           str(reps), str(SEED)]
    r = run(cmd)
    vals = {}
    for line in r.stderr.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        value, event = parts[0], parts[2]
        vals[event] = None if value.startswith("<") else float(value)
    return vals


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    have_perf = perf_available()
    if not have_perf:
        print("note: perf counters unavailable; timing/speedup only.\n")

    new_file = True  # always write a fresh, complete log
    with open(LOG, "w", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "distance", "hint", "M", "N", "K", "seed",
                "verify_status", "verify_max_abs_diff",
                "time_ms", "gflops", "speedup_vs_naive", "perf_reps",
                "instructions", "cycles", "ipc",
                "l1d_loads", "l1d_load_misses", "l1d_mpki",
                "l2_misses", "l2_refs", "l2_mpki",
                "llc_misses", "llc_refs", "llc_mpki", "sw_prefetch_access",
            ])

        for N in SIZES:
            M = K = N
            base_ms, _ = time_run(M, N, K, 0, 0, probe_ms(M, N, K, 0, 0))

            for dist in DISTANCES:
                hint_items = [("T0", 3)] if dist == 0 else list(HINTS.items())
                for hname, hint in hint_items:
                    status, err = verify(M, N, K, dist, hint)
                    tp = probe_ms(M, N, K, dist, hint)
                    ms, gflops = time_run(M, N, K, dist, hint, tp)
                    speedup = base_ms / ms if ms > 0 else 0.0

                    vals = {}
                    reps = 0
                    if have_perf:
                        reps = max(1, min(300, round(TARGET_PERF_MS / max(ms, 1e-6))))
                        vals = perf_run(M, N, K, dist, hint, reps)
                    instr = vals.get("cpu_core/instructions/")
                    cycles = vals.get("cpu_core/cycles/")
                    l1l = vals.get("cpu_core/L1-dcache-loads/")
                    l1m = vals.get("cpu_core/L1-dcache-load-misses/")
                    l2m = vals.get("cpu_core/l2_rqsts.miss/")
                    l2r = vals.get("cpu_core/l2_rqsts.references/")
                    llcm = vals.get("cpu_core/longest_lat_cache.miss/")
                    llcr = vals.get("cpu_core/longest_lat_cache.reference/")
                    swpf = vals.get("cpu_core/sw_prefetch_access.any/")
                    ipc = (instr / cycles) if (instr and cycles) else None
                    mpki = (l1m / instr * 1000.0) if (instr and l1m is not None) else None
                    l2mpki = (l2m / instr * 1000.0) if (instr and l2m is not None) else None
                    llcmpki = (llcm / instr * 1000.0) if (instr and llcm is not None) else None

                    def s(x, fmt="{}"):
                        return "" if x is None else fmt.format(x)

                    w.writerow([
                        dist, hname, M, N, K, SEED, status, err,
                        f"{ms:.6f}", f"{gflops:.6f}", f"{speedup:.6f}", reps,
                        s(instr), s(cycles), s(ipc, "{:.6f}"),
                        s(l1l), s(l1m), s(mpki, "{:.6f}"),
                        s(l2m), s(l2r), s(l2mpki, "{:.6f}"),
                        s(llcm), s(llcr), s(llcmpki, "{:.6f}"), s(swpf),
                    ])
                    f.flush()
                    print(f"N={N:<5} dist={dist:<4} hint={hname:<3} "
                          f"{ms:10.3f} ms  {speedup:6.3f}x  verify={status}")
            print()

    print(f"logged to {LOG}")


if __name__ == "__main__":
    if not BENCH.exists():
        sys.exit(f"error: {BENCH} not built -- run `make` in {HERE} first")
    main()
