#!/usr/bin/env python3
"""collect_prefetch.py -- Task 2A benchmark driver.

Sweeps software-prefetch distance (in floats, rounded to a cache-line
boundary internally) x locality hint (NTA/T2/T1/T0) x matrix size for the
scalar (no-SIMD, unblocked) matmul_prefetch_tuned kernel in
src/matmul_prefetch.cpp, against matmul_naive. Logs L1-D/L2/LLC misses,
sw_prefetch_access count, instructions, execution time and speedup for every
(distance, hint, M) to logs/task2a_prefetch.csv.

Usage:   ./collect_prefetch.py
Requires ./bench_prefetch already built (`make`) and perf readable as a
non-root user (see collect_simd.py's docstring for the CPU_PIN rationale).
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
SIZES = [128, 256, 512, 1024, 2048]
DISTANCES = [0, 16, 32, 64, 128, 256]          # floats; 0 = no prefetch (== naive)
HINTS = {"NTA": 0, "T2": 1, "T1": 2, "T0": 3}  # matches matmul_prefetch.cpp's switch

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

TARGET_PERF_MS = 300.0
MIN_REPS, MAX_REPS = 3, 200


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def verify(M, N, K, distance, hint):
    r = run([str(BENCH), "loop", str(M), str(N), str(K), str(distance), str(hint),
             "1", str(SEED), "--verify"])
    m = re.search(r"max_abs_diff=([\d.eE+-]+) \((OK|FAIL)\)", r.stderr)
    if not m:
        return "ERR", r.stderr.strip()
    return m.group(2), float(m.group(1))


def time_run(M, N, K, distance, hint, warmup=1, reps=3):
    r = run([str(BENCH), "time", str(M), str(N), str(K), str(distance), str(hint),
             str(warmup), str(reps), str(SEED)])
    ms, gflops = r.stdout.strip().split(",")
    return float(ms), float(gflops)


def perf_run(M, N, K, distance, hint, reps):
    cmd = ["taskset", "-c", str(CPU_PIN), "perf", "stat", "-x,",
           "-e", ",".join(PERF_EVENTS), "--",
           str(BENCH), "loop", str(M), str(N), str(K), str(distance), str(hint),
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
    new_file = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "distance", "hint", "M", "N", "K", "seed",
                "verify_status", "verify_max_abs_diff",
                "time_ms", "gflops", "speedup_vs_naive", "perf_reps",
                "instructions", "cycles", "l1d_loads", "l1d_load_misses", "l1d_mpki",
                "l2_misses", "l2_refs", "l2_mpki", "llc_misses", "llc_refs", "llc_mpki",
                "sw_prefetch_access", "ipc",
            ])

        for N in SIZES:
            M = K = N
            naive_ms, _ = time_run(M, N, K, 0, 0)

            for distance in DISTANCES:
                hint_list = [("T0", 3)] if distance == 0 else list(HINTS.items())
                for hint_name, hint in hint_list:
                    status, err = verify(M, N, K, distance, hint)
                    ms, gflops = time_run(M, N, K, distance, hint)
                    speedup = naive_ms / ms if ms > 0 else 0.0

                    reps = max(MIN_REPS, min(MAX_REPS, round(TARGET_PERF_MS / max(ms, 1e-6))))
                    pv = perf_run(M, N, K, distance, hint, reps)
                    instr = pv.get("cpu_core/instructions/")
                    cycles = pv.get("cpu_core/cycles/")
                    l1_loads = pv.get("cpu_core/L1-dcache-loads/")
                    l1_misses = pv.get("cpu_core/L1-dcache-load-misses/")
                    l2_misses = pv.get("cpu_core/l2_rqsts.miss/")
                    l2_refs = pv.get("cpu_core/l2_rqsts.references/")
                    llc_misses = pv.get("cpu_core/longest_lat_cache.miss/")
                    llc_refs = pv.get("cpu_core/longest_lat_cache.reference/")
                    swpf = pv.get("cpu_core/sw_prefetch_access.any/")

                    mpki = (l1_misses / instr * 1000.0) if (instr and l1_misses is not None) else None
                    l2_mpki = (l2_misses / instr * 1000.0) if (instr and l2_misses is not None) else None
                    llc_mpki = (llc_misses / instr * 1000.0) if (instr and llc_misses is not None) else None
                    ipc = (instr / cycles) if (instr and cycles) else None

                    writer.writerow([
                        distance, hint_name, M, N, K, SEED, status, err,
                        f"{ms:.6f}", f"{gflops:.6f}", f"{speedup:.6f}", reps,
                        instr, cycles, l1_loads, l1_misses,
                        f"{mpki:.6f}" if mpki is not None else "",
                        l2_misses, l2_refs,
                        f"{l2_mpki:.6f}" if l2_mpki is not None else "",
                        llc_misses, llc_refs,
                        f"{llc_mpki:.6f}" if llc_mpki is not None else "",
                        swpf,
                        f"{ipc:.6f}" if ipc is not None else "",
                    ])
                    f.flush()
                    print(f"N={N:<5} dist={distance:<4} hint={hint_name:<3} "
                          f"{ms:9.3f} ms  {speedup:6.2f}x  verify={status}")

    print(f"\nlogged to {LOG}")


if __name__ == "__main__":
    if not BENCH.exists():
        sys.exit(f"error: {BENCH} not built -- run `make` in {HERE} first")
    main()
