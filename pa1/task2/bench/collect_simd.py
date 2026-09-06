#!/usr/bin/env python3
"""collect_simd.py -- Task 2B benchmark driver.

Times and perf-profiles the two SIMD widths (simd128 = SSE/__m128, 4-wide;
simd256 = AVX2/__m256, 8-wide) against matmul_naive, across a grid of square
M=N=K matrix sizes. Both variants keep the naive (i,j,p) loop order -- no
register blocking, no tiling, no prefetch (that combination is Task 2C).
512-bit (AVX-512) is not tested: this machine has no avx512* flags. Appends
one CSV row per (stage, M, N, K) to logs/task2b_simd.csv.

Usage:   ./collect_simd.py
Requires ./bench_stage already built (`make`) and perf readable as a
non-root user. CPU_PIN follows the same P-core-pinning rationale as Task 1
(this hybrid CPU's E-core PMU does not support L1-dcache-load-misses).
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "bench_stage"
LOG = HERE / "logs" / "task2b_simd.csv"

CPU_PIN = 2
SEED = 1234
SIZES = [128, 256, 512, 1024, 2048]

STAGES = [("naive", "naive"), ("simd128", "simd128"), ("simd256", "simd256")]

PERF_EVENTS = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/",
]

TARGET_PERF_MS = 300.0
MIN_REPS, MAX_REPS = 3, 200


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def verify(stage_key, M, N, K):
    r = run([str(BENCH), "loop", stage_key, str(M), str(N), str(K), "1", str(SEED), "--verify"])
    m = re.search(r"max_abs_diff=([\d.eE+-]+) \((OK|FAIL)\)", r.stderr)
    if not m:
        return "ERR", r.stderr.strip()
    return m.group(2), float(m.group(1))


def time_stage(stage_key, M, N, K, warmup=1, reps=3):
    r = run([str(BENCH), "time", stage_key, str(M), str(N), str(K), str(warmup), str(reps), str(SEED)])
    ms, gflops = r.stdout.strip().split(",")
    return float(ms), float(gflops)


def perf_stage(stage_key, M, N, K, reps):
    cmd = ["taskset", "-c", str(CPU_PIN), "perf", "stat", "-x,",
           "-e", ",".join(PERF_EVENTS), "--",
           str(BENCH), "loop", stage_key, str(M), str(N), str(K), str(reps), str(SEED)]
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
                "stage", "M", "N", "K", "seed", "verify_status", "verify_max_abs_diff",
                "time_ms", "gflops", "speedup_vs_naive", "perf_reps",
                "instructions", "cycles", "l1d_loads", "l1d_load_misses", "l1d_mpki", "ipc",
            ])

        for N in SIZES:
            M = K = N
            naive_ms = None
            for label, stage_key in STAGES:
                status, err = verify(stage_key, M, N, K)
                ms, gflops = time_stage(stage_key, M, N, K)
                if label == "naive":
                    naive_ms = ms
                speedup = (naive_ms / ms) if naive_ms else 0.0

                reps = max(MIN_REPS, min(MAX_REPS, round(TARGET_PERF_MS / max(ms, 1e-6))))
                perf_vals = perf_stage(stage_key, M, N, K, reps)
                instr = perf_vals.get("cpu_core/instructions/")
                cycles = perf_vals.get("cpu_core/cycles/")
                l1_loads = perf_vals.get("cpu_core/L1-dcache-loads/")
                l1_misses = perf_vals.get("cpu_core/L1-dcache-load-misses/")
                mpki = (l1_misses / instr * 1000.0) if (instr and l1_misses is not None) else None
                ipc = (instr / cycles) if (instr and cycles) else None

                writer.writerow([
                    label, M, N, K, SEED, status, err,
                    f"{ms:.6f}", f"{gflops:.6f}", f"{speedup:.6f}",
                    reps, instr, cycles, l1_loads, l1_misses,
                    f"{mpki:.6f}" if mpki is not None else "",
                    f"{ipc:.6f}" if ipc is not None else "",
                ])
                f.flush()
                instr_s = f"{instr/1e6:8.1f}M" if instr else "     n/a"
                print(f"N={N:<5} {label:<8} {ms:9.3f} ms  {speedup:6.2f}x  instr={instr_s}  verify={status}")

    print(f"\nlogged to {LOG}")


if __name__ == "__main__":
    if not BENCH.exists():
        sys.exit(f"error: {BENCH} not built -- run `make` in {HERE} first")
    main()
