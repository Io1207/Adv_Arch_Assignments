#!/usr/bin/env python3
"""collect_size.py -- Task 2 "speedup vs. matrix size" driver (README plot 1).

Times and perf-profiles the three graded stages (matmul_simd,
matmul_prefetch, matmul_optimized) against matmul_naive across square
M=N=K sizes that span from cache-resident to well beyond LLC. One CSV row
per (stage, N) to logs/task2_size.csv:

    time / gflops / speedup_vs_naive, and -- when perf counters are
    readable -- instructions, cycles, IPC, and L1D / L2 / LLC MPKI.

Usage:   ./collect_size.py
Requires ./bench_stage built (`make` here). perf counters need
kernel.perf_event_paranoid <= 1 (`sudo sysctl kernel.perf_event_paranoid=1`);
without it the timing columns are still collected and the perf columns are
left blank.

CPU_PIN must be a P-core ("cpu_core" PMU) on this hybrid 12th-gen part; the
E-core PMU does not expose L1-dcache-load-misses. `lscpu -e` shows CPUs
0-11 are the 6 P-cores (2 threads each); CPU 2 is used here as in Task 1.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "bench_stage"
LOG = HERE / "logs" / "task2_size.csv"

CPU_PIN = 2
SEED = 1234
SIZES = [512, 1024, 2048, 4096]                 # M = N = K
STAGES = [("naive", "naive"), ("simd", "simd"),
          ("prefetch", "prefetch"), ("optimized", "optimized")]

PERF_EVENTS = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/",
    "cpu_core/l2_rqsts.miss/",
    "cpu_core/l2_rqsts.references/",
    "cpu_core/longest_lat_cache.miss/",
    "cpu_core/longest_lat_cache.reference/",
]

TARGET_TIME_MS = 400.0     # aim for this much wall time per timed measurement
TARGET_PERF_MS = 300.0     # ... and inside the perf-wrapped loop


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def perf_available():
    r = run(["perf", "stat", "-x,", "-e", "cpu_core/instructions/", "--", "true"])
    return "not supported" not in r.stderr and "No supported events" not in r.stderr \
        and "Access to performance monitoring" not in r.stderr


def verify(stage_key, M, N, K):
    r = run([str(BENCH), "loop", stage_key, str(M), str(N), str(K), "1", str(SEED), "--verify"])
    m = re.search(r"max_abs_diff=([\d.eE+-]+) \((OK|FAIL)\)", r.stderr)
    return (m.group(2), float(m.group(1))) if m else ("ERR", r.stderr.strip()[:80])


def probe_ms(stage_key, M, N, K):
    r = run([str(BENCH), "time", stage_key, str(M), str(N), str(K), "0", "1", str(SEED)])
    try:
        return float(r.stdout.strip().split(",")[0])
    except Exception:
        return None


def pick_reps(t_ms, target):
    if t_ms is None or t_ms <= 0:
        return 1, 3
    reps = max(1, min(9, round(target / t_ms)))
    warmup = 1 if t_ms < 100.0 else 0
    return warmup, reps


def time_stage(stage_key, M, N, K, t_probe):
    warmup, reps = pick_reps(t_probe, TARGET_TIME_MS)
    r = run([str(BENCH), "time", stage_key, str(M), str(N), str(K),
             str(warmup), str(reps), str(SEED)])
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
    have_perf = perf_available()
    if not have_perf:
        print("note: perf counters unavailable (perf_event_paranoid); "
              "collecting timing/speedup only.\n")

    new_file = True  # always write a fresh, complete log
    with open(LOG, "w", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "stage", "M", "N", "K", "seed", "verify_status", "verify_max_abs_diff",
                "time_ms", "gflops", "speedup_vs_naive", "perf_reps",
                "instructions", "cycles", "ipc",
                "l1d_loads", "l1d_load_misses", "l1d_mpki",
                "l2_misses", "l2_refs", "l2_mpki",
                "llc_misses", "llc_refs", "llc_mpki",
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

                instr = cycles = ipc = None
                l1l = l1m = mpki = l2m = l2r = l2mpki = llcm = llcr = llcmpki = None
                reps = 0
                if have_perf:
                    reps = max(1, min(300, round(TARGET_PERF_MS / max(ms, 1e-6))))
                    pv = perf_stage(key, M, N, K, reps)
                    instr = pv.get("cpu_core/instructions/")
                    cycles = pv.get("cpu_core/cycles/")
                    l1l = pv.get("cpu_core/L1-dcache-loads/")
                    l1m = pv.get("cpu_core/L1-dcache-load-misses/")
                    l2m = pv.get("cpu_core/l2_rqsts.miss/")
                    l2r = pv.get("cpu_core/l2_rqsts.references/")
                    llcm = pv.get("cpu_core/longest_lat_cache.miss/")
                    llcr = pv.get("cpu_core/longest_lat_cache.reference/")
                    ipc = (instr / cycles) if (instr and cycles) else None
                    mpki = (l1m / instr * 1000.0) if (instr and l1m is not None) else None
                    l2mpki = (l2m / instr * 1000.0) if (instr and l2m is not None) else None
                    llcmpki = (llcm / instr * 1000.0) if (instr and llcm is not None) else None

                def s(x, fmt="{}"):
                    return "" if x is None else fmt.format(x)

                w.writerow([
                    label, M, N, K, SEED, status, err,
                    f"{ms:.6f}", f"{gflops:.6f}", f"{speedup:.6f}", reps,
                    s(instr), s(cycles), s(ipc, "{:.6f}"),
                    s(l1l), s(l1m), s(mpki, "{:.6f}"),
                    s(l2m), s(l2r), s(l2mpki, "{:.6f}"),
                    s(llcm), s(llcr), s(llcmpki, "{:.6f}"),
                ])
                f.flush()
                mp = f"{mpki:6.2f}" if mpki is not None else "   -  "
                print(f"N={N:<5} {label:<10} {ms:11.3f} ms  {gflops:8.2f} GF/s  "
                      f"{speedup:7.2f}x  L1D-MPKI={mp}  verify={status}")
            print()

    print(f"logged to {LOG}")


if __name__ == "__main__":
    if not BENCH.exists():
        sys.exit(f"error: {BENCH} not built -- run `make` in {HERE} first")
    main()
