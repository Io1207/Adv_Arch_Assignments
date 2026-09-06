#!/usr/bin/env python3
"""collect_tile.py -- Task 1B benchmark driver.

Sweeps tile size x matrix size x kernel size for both tiling shapes (row =
block oy only; square = block oy AND ox), against the naive baseline. Both
shapes keep the naive loop nest order (oy, ox, ky, kx) inside a tile -- no
reordering, no unrolling, that's collect_reorder.py / collect_unroll.py's
job. Logs L1-D MPKI, instructions, execution time and speedup for every
(shape, tile_size, H, W, K) to logs/task1b_tile.csv, including an "auto" tile
size per config computed from the same L1D-capacity formula src/conv_tile.cpp
uses for the graded conv_tile/conv_tile_sq, so the formula's own choice can be
checked against the swept fixed sizes.

Usage:   ./collect_tile.py
Requires ./bench_stage and ./bench_tile already built (`make` in this
directory) and perf readable as a non-root user. See collect_reorder.py's
docstring for the CPU_PIN / P-core-vs-E-core rationale.
"""
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_STAGE = HERE / "bench_stage"
BENCH_TILE = HERE / "bench_tile"
LOG = HERE / "logs" / "task1b_tile.csv"

CPU_PIN = 2
SEED = 1234
KiB = 1024
L1D = 48 * KiB  # per P-core (cpu2) -- /sys/devices/system/cpu/cpu2/cache/index1/size

SIZES = [128, 256, 512, 1024, 2048, 4096]
KERNELS = [3, 5, 7, 9]
FIXED_TILE_SIZES = [16, 32, 64, 128, 256]

PERF_EVENTS = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/",
]

TARGET_PERF_MS = 300.0
MIN_REPS, MAX_REPS = 3, 2000


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def auto_tile_size(shape, W, K):
    budget = L1D - 4 * K * K
    if shape == "row":
        return max(1, budget // (2 * 4 * W))
    return max(1, int(math.sqrt(max(budget, 1) / (2 * 4))))


def verify_tile(shape, H, W, K, tile_size):
    r = run([str(BENCH_TILE), "loop", shape, str(H), str(W), str(K), str(tile_size),
             "1", str(SEED), "--verify"])
    m = re.search(r"max_abs_diff=([\d.eE+-]+) \((OK|FAIL)\)", r.stderr)
    if not m:
        return "ERR", r.stderr.strip()
    return m.group(2), float(m.group(1))


def time_naive(H, W, K, warmup=2, reps=7):
    r = run([str(BENCH_STAGE), "time", "naive", str(H), str(W), str(K),
             str(warmup), str(reps), str(SEED)])
    ms, gflops = r.stdout.strip().split(",")
    return float(ms), float(gflops)


def time_tile(shape, H, W, K, tile_size, warmup=2, reps=7):
    r = run([str(BENCH_TILE), "time", shape, str(H), str(W), str(K), str(tile_size),
             str(warmup), str(reps), str(SEED)])
    ms, gflops = r.stdout.strip().split(",")
    return float(ms), float(gflops)


def _parse_perf(r):
    vals = {}
    for line in r.stderr.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        value, event = parts[0], parts[2]
        vals[event] = None if value.startswith("<") else float(value)
    return vals


def perf_naive(H, W, K, reps):
    cmd = ["taskset", "-c", str(CPU_PIN), "perf", "stat", "-x,",
           "-e", ",".join(PERF_EVENTS), "--",
           str(BENCH_STAGE), "loop", "naive", str(H), str(W), str(K), str(reps), str(SEED)]
    return _parse_perf(run(cmd))


def perf_tile(shape, H, W, K, tile_size, reps):
    cmd = ["taskset", "-c", str(CPU_PIN), "perf", "stat", "-x,",
           "-e", ",".join(PERF_EVENTS), "--",
           str(BENCH_TILE), "loop", shape, str(H), str(W), str(K), str(tile_size),
           str(reps), str(SEED)]
    return _parse_perf(run(cmd))


def derive(perf_vals):
    instr = perf_vals.get("cpu_core/instructions/")
    cycles = perf_vals.get("cpu_core/cycles/")
    l1_loads = perf_vals.get("cpu_core/L1-dcache-loads/")
    l1_misses = perf_vals.get("cpu_core/L1-dcache-load-misses/")
    mpki = (l1_misses / instr * 1000.0) if (instr and l1_misses is not None) else None
    ipc = (instr / cycles) if (instr and cycles) else None
    return instr, cycles, l1_loads, l1_misses, mpki, ipc


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "stage", "shape", "tile_size", "H", "W", "K", "seed",
                "verify_status", "verify_max_abs_diff",
                "time_ms", "gflops", "speedup_vs_naive",
                "perf_reps", "instructions", "cycles",
                "l1d_loads", "l1d_load_misses", "l1d_mpki", "ipc",
            ])

        for K in KERNELS:
            for N in SIZES:
                H = W = N

                # naive baseline, once per (N, K) -- shared denominator for speedup.
                naive_ms, _ = time_naive(H, W, K)
                naive_reps = max(MIN_REPS, min(MAX_REPS, round(TARGET_PERF_MS / max(naive_ms, 1e-6))))
                instr, cycles, l1_loads, l1_misses, mpki, ipc = derive(perf_naive(H, W, K, naive_reps))
                writer.writerow([
                    "naive", "-", 0, H, W, K, SEED, "OK", 0.0,
                    f"{naive_ms:.6f}", "", "1.000000",
                    naive_reps, instr, cycles, l1_loads, l1_misses,
                    f"{mpki:.6f}" if mpki is not None else "",
                    f"{ipc:.6f}" if ipc is not None else "",
                ])
                f.flush()
                mpki_s = f"{mpki:6.2f}" if mpki is not None else "  n/a"
                print(f"K={K:<2} N={N:<5} naive               "
                      f"{naive_ms:9.3f} ms   1.00x  MPKI={mpki_s}")

                for shape in ("row", "square"):
                    tile_candidates = [(str(ts), ts) for ts in FIXED_TILE_SIZES]
                    tile_candidates.append(("auto", auto_tile_size(shape, W, K)))

                    for label, tile_size in tile_candidates:
                        status, err = verify_tile(shape, H, W, K, tile_size)
                        ms, gflops = time_tile(shape, H, W, K, tile_size)
                        speedup = naive_ms / ms if ms > 0 else 0.0

                        reps = max(MIN_REPS, min(MAX_REPS, round(TARGET_PERF_MS / max(ms, 1e-6))))
                        instr, cycles, l1_loads, l1_misses, mpki, ipc = derive(
                            perf_tile(shape, H, W, K, tile_size, reps))

                        writer.writerow([
                            f"tile_{shape}", shape, tile_size, H, W, K, SEED, status, err,
                            f"{ms:.6f}", f"{gflops:.6f}", f"{speedup:.6f}",
                            reps, instr, cycles, l1_loads, l1_misses,
                            f"{mpki:.6f}" if mpki is not None else "",
                            f"{ipc:.6f}" if ipc is not None else "",
                        ])
                        f.flush()
                        mpki_s = f"{mpki:6.2f}" if mpki is not None else "  n/a"
                        print(f"K={K:<2} N={N:<5} {shape:<6} ts={label:<5}({tile_size:<4}) "
                              f"{ms:9.3f} ms  {speedup:6.2f}x  MPKI={mpki_s}  verify={status}")

    print(f"\nlogged to {LOG}")


if __name__ == "__main__":
    if not BENCH_STAGE.exists():
        sys.exit(f"error: {BENCH_STAGE} not built -- run `make` in {HERE} first")
    if not BENCH_TILE.exists():
        sys.exit(f"error: {BENCH_TILE} not built -- run `make` in {HERE} first")
    main()
