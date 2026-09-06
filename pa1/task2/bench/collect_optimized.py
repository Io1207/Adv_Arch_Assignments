#!/usr/bin/env python3
"""collect_optimized.py -- Task 2C benchmark driver.

Sweeps the accumulator count N in {1,2,4} (the register tile's own SIMD
chains, "unroll the vector register itself") for the combined
reorder+register-blocking+SIMD+tiling+prefetch kernel in
src/matmul_optimized.cpp, against matmul_naive, across a grid of square
M=N=K sizes. Logs instructions, cycles, IPC, L1-D MPKI, execution time and
speedup for every (Nacc, M) to logs/task2c_optimized.csv.

Usage:   ./collect_optimized.py
Requires ./bench_stage already built (`make`) and perf readable as a
non-root user.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "bench_stage"
LOG = HERE / "logs" / "task2c_optimized.csv"

CPU_PIN = 2
SEED = 1234
SIZES = [128, 256, 512, 1024, 2048]

STAGES = [("naive", "naive"), ("acc1", "opt_acc1"), ("acc2", "opt_acc2"), ("acc4", "opt_acc4")]

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
                pv = perf_stage(stage_key, M, N, K, reps)
                instr = pv.get("cpu_core/instructions/")
                cycles = pv.get("cpu_core/cycles/")
                l1_loads = pv.get("cpu_core/L1-dcache-loads/")
                l1_misses = pv.get("cpu_core/L1-dcache-load-misses/")
                mpki = (l1_misses / instr * 1000.0) if (instr and l1_misses is not None) else None
                ipc = (instr / cycles) if (instr and cycles) else None

                writer.writerow([
                    label, M, N, K, SEED, status, err,
                    f"{ms:.6f}", f"{gflops:.6f}", f"{speedup:.6f}", reps,
                    instr, cycles, l1_loads, l1_misses,
                    f"{mpki:.6f}" if mpki is not None else "",
                    f"{ipc:.6f}" if ipc is not None else "",
                ])
                f.flush()
                ipc_s = f"{ipc:5.2f}" if ipc is not None else "  n/a"
                print(f"N={N:<5} {label:<6} {ms:9.3f} ms  {speedup:6.2f}x  IPC={ipc_s}  verify={status}")

    print(f"\nlogged to {LOG}")


if __name__ == "__main__":
    if not BENCH.exists():
        sys.exit(f"error: {BENCH} not built -- run `make` in {HERE} first")
    main()
