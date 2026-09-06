#!/usr/bin/env python3
"""collect_optimized.py -- Task 2C supporting driver ("accumulators vs. speedup").

Sweeps the accumulator count NACC in {1, 2, 4} for the fixed 2x2 AVX2
register tile in bench/opt_variants.cpp -- the number of independent __m256
FMA chains per tile position, splitting K. This isolates how much
FMA-latency hiding the extra accumulator chains buy, the register-blocking
axis of the combined kernel. Run against matmul_naive across square M=N=K
sizes. One CSV row per (variant, N) to logs/task2c_optimized.csv.

(The final "best of each technique vs. size" comparison for Task 2C is built
by make_plots.py from task2_size.csv + task2a_prefetch.csv + task2b_simd.csv.)
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
SIZES = [512, 1024, 2048, 4096]
STAGES = [("naive", "naive"), ("acc1", "opt_acc1"),
          ("acc2", "opt_acc2"), ("acc4", "opt_acc4")]

PERF_EVENTS = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/",
]

TARGET_TIME_MS = 400.0
TARGET_PERF_MS = 300.0


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def perf_available():
    r = run(["perf", "stat", "-x,", "-e", "cpu_core/instructions/", "--", "true"])
    bad = ("not supported", "No supported events", "Access to performance monitoring")
    return not any(b in r.stderr for b in bad)


def verify(key, M, N, K):
    r = run([str(BENCH), "loop", key, str(M), str(N), str(K), "1", str(SEED), "--verify"])
    m = re.search(r"max_abs_diff=([\d.eE+-]+) \((OK|FAIL)\)", r.stderr)
    return (m.group(2), float(m.group(1))) if m else ("ERR", r.stderr.strip()[:80])


def probe_ms(key, M, N, K):
    r = run([str(BENCH), "time", key, str(M), str(N), str(K), "0", "1", str(SEED)])
    try:
        return float(r.stdout.strip().split(",")[0])
    except Exception:
        return None


def pick_reps(t_ms, target):
    if t_ms is None or t_ms <= 0:
        return 1, 3
    return (1 if t_ms < 100.0 else 0), max(1, min(9, round(target / t_ms)))


def time_stage(key, M, N, K, t_probe):
    warmup, reps = pick_reps(t_probe, TARGET_TIME_MS)
    r = run([str(BENCH), "time", key, str(M), str(N), str(K),
             str(warmup), str(reps), str(SEED)])
    ms, gflops = r.stdout.strip().split(",")
    return float(ms), float(gflops)


def perf_stage(key, M, N, K, reps):
    cmd = ["taskset", "-c", str(CPU_PIN), "perf", "stat", "-x,",
           "-e", ",".join(PERF_EVENTS), "--",
           str(BENCH), "loop", key, str(M), str(N), str(K), str(reps), str(SEED)]
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
                "stage", "M", "N", "K", "seed", "verify_status", "verify_max_abs_diff",
                "time_ms", "gflops", "speedup_vs_naive", "perf_reps",
                "instructions", "cycles", "ipc",
                "l1d_loads", "l1d_load_misses", "l1d_mpki",
            ])

        for N in SIZES:
            M = K = N
            naive_ms = None
            for label, key in STAGES:
                status, err = verify(key, M, N, K)
                tp = probe_ms(key, M, N, K)
                ms, gflops = time_stage(key, M, N, K, tp)
                if label == "naive":
                    naive_ms = ms
                speedup = (naive_ms / ms) if naive_ms else 0.0

                instr = cycles = ipc = l1l = l1m = mpki = None
                reps = 0
                if have_perf:
                    reps = max(1, min(300, round(TARGET_PERF_MS / max(ms, 1e-6))))
                    pv = perf_stage(key, M, N, K, reps)
                    instr = pv.get("cpu_core/instructions/")
                    cycles = pv.get("cpu_core/cycles/")
                    l1l = pv.get("cpu_core/L1-dcache-loads/")
                    l1m = pv.get("cpu_core/L1-dcache-load-misses/")
                    ipc = (instr / cycles) if (instr and cycles) else None
                    mpki = (l1m / instr * 1000.0) if (instr and l1m is not None) else None

                def s(x, fmt="{}"):
                    return "" if x is None else fmt.format(x)

                w.writerow([
                    label, M, N, K, SEED, status, err,
                    f"{ms:.6f}", f"{gflops:.6f}", f"{speedup:.6f}", reps,
                    s(instr), s(cycles), s(ipc, "{:.6f}"),
                    s(l1l), s(l1m), s(mpki, "{:.6f}"),
                ])
                f.flush()
                ipc_s = f"{ipc:5.2f}" if ipc is not None else "  -  "
                print(f"N={N:<5} {label:<6} {ms:10.3f} ms  {gflops:8.2f} GF/s  "
                      f"{speedup:6.2f}x  IPC={ipc_s}  verify={status}")
            print()

    print(f"logged to {LOG}")


if __name__ == "__main__":
    if not BENCH.exists():
        sys.exit(f"error: {BENCH} not built -- run `make` in {HERE} first")
    main()
