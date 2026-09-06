#!/usr/bin/env python3
"""collect_hwprefetch.py -- Task 2A item 5: effect of the HARDWARE prefetchers.

Toggles the four core HW prefetchers via MSR 0x1A4 and re-measures the
graded stages, so the report can separate hardware from software prefetching.

    MSR 0x1A4 bits (1 = DISABLE):
      bit0  L2 hardware prefetcher
      bit1  L2 adjacent-cache-line prefetcher
      bit2  L1 DCU prefetcher
      bit3  L1 DCU IP prefetcher
    mask 0x0 = all HW prefetchers ON (default);  0xF = all OFF.

Requires root (writes MSRs) and the `msr` kernel module + `wrmsr`
(msr-tools):
      sudo modprobe msr
      sudo -v            # cache credentials, then:
      sudo ./collect_hwprefetch.py
If wrmsr is unavailable / not root, the script prints how to run it and
exits without touching the log.

One CSV row per (hw_pf, stage, N) to logs/task2a_hwprefetch.csv.
Perf columns need kernel.perf_event_paranoid <= 1 as well.
"""
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "bench_stage"
LOG = HERE / "logs" / "task2a_hwprefetch.csv"

CPU_PIN = 2
SEED = 1234
SIZES = [512, 1024, 2048]
STAGES = [("naive", "naive"), ("prefetch", "prefetch"), ("optimized", "optimized")]
PF_STATES = [("hw_on", 0x0), ("hw_off", 0xF)]

PERF_EVENTS = [
    "cpu_core/instructions/",
    "cpu_core/cycles/",
    "cpu_core/L1-dcache-loads/",
    "cpu_core/L1-dcache-load-misses/",
    "cpu_core/l2_rqsts.miss/",
    "cpu_core/longest_lat_cache.miss/",
    "cpu_core/sw_prefetch_access.any/",
]

TARGET_TIME_MS = 400.0
TARGET_PERF_MS = 300.0
NUM_CPUS = os.cpu_count() or 16


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def perf_available():
    r = run(["perf", "stat", "-x,", "-e", "cpu_core/instructions/", "--", "true"])
    bad = ("not supported", "No supported events", "Access to performance monitoring")
    return not any(b in r.stderr for b in bad)


def set_hw_prefetch(mask):
    """Write MSR 0x1A4 = mask on every logical CPU. Returns True on success."""
    if not shutil.which("wrmsr"):
        return False
    for c in range(NUM_CPUS):
        r = run(["wrmsr", "-p", str(c), "0x1a4", hex(mask)])
        if r.returncode != 0:
            # first failure: report and bail (likely all CPUs will fail)
            sys.stderr.write(r.stderr)
            return False
    return True


def read_hw_prefetch():
    if not shutil.which("rdmsr"):
        return None
    r = run(["rdmsr", "-p", str(CPU_PIN), "0x1a4"])
    return r.stdout.strip() if r.returncode == 0 else None


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
    if os.geteuid() != 0 or not shutil.which("wrmsr"):
        sys.exit("need root + msr-tools. Run:\n"
                 "  sudo modprobe msr && sudo ./collect_hwprefetch.py")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    have_perf = perf_available()
    if not have_perf:
        print("note: perf counters unavailable; timing/speedup only.\n")

    new_file = True  # always write a fresh, complete log
    try:
        with open(LOG, "w", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow([
                    "hw_pf", "msr_1a4", "stage", "M", "N", "K", "seed",
                    "verify_status", "time_ms", "gflops", "speedup_vs_naive",
                    "speedup_vs_hw_on", "perf_reps",
                    "instructions", "ipc", "l1d_mpki", "l2_misses",
                    "llc_misses", "sw_prefetch_access",
                ])

            base_hw_on = {}   # (stage, N) -> ms with HW pf ON, for the on/off ratio
            for pf_name, mask in PF_STATES:
                if not set_hw_prefetch(mask):
                    sys.exit("wrmsr failed -- MSR 0x1A4 not writable on this system.")
                msr_val = read_hw_prefetch()
                print(f"=== {pf_name}  MSR 0x1A4 = {msr_val} ===")

                for N in SIZES:
                    M = K = N
                    naive_ms = None
                    for label, key in STAGES:
                        status, _ = verify(key, M, N, K)
                        tp = probe_ms(key, M, N, K)
                        ms, gflops = time_stage(key, M, N, K, tp)
                        if label == "naive":
                            naive_ms = ms
                        spd = (naive_ms / ms) if naive_ms else 0.0
                        if pf_name == "hw_on":
                            base_hw_on[(label, N)] = ms
                            spd_onoff = 1.0
                        else:
                            b = base_hw_on.get((label, N))
                            spd_onoff = (b / ms) if b else 0.0

                        instr = ipc = mpki = l2m = llcm = swpf = None
                        reps = 0
                        if have_perf:
                            reps = max(1, min(300, round(TARGET_PERF_MS / max(ms, 1e-6))))
                            pv = perf_stage(key, M, N, K, reps)
                            instr = pv.get("cpu_core/instructions/")
                            cyc = pv.get("cpu_core/cycles/")
                            l1m = pv.get("cpu_core/L1-dcache-load-misses/")
                            l2m = pv.get("cpu_core/l2_rqsts.miss/")
                            llcm = pv.get("cpu_core/longest_lat_cache.miss/")
                            swpf = pv.get("cpu_core/sw_prefetch_access.any/")
                            ipc = (instr / cyc) if (instr and cyc) else None
                            mpki = (l1m / instr * 1000.0) if (instr and l1m is not None) else None

                        def s(x, fmt="{}"):
                            return "" if x is None else fmt.format(x)

                        w.writerow([
                            pf_name, msr_val, label, M, N, K, SEED, status,
                            f"{ms:.6f}", f"{gflops:.6f}", f"{spd:.6f}",
                            f"{spd_onoff:.6f}", reps,
                            s(instr), s(ipc, "{:.6f}"), s(mpki, "{:.6f}"),
                            s(l2m), s(llcm), s(swpf),
                        ])
                        f.flush()
                        print(f"  N={N:<5} {label:<10} {ms:10.3f} ms  "
                              f"{spd:6.2f}x vs naive  {spd_onoff:6.3f}x vs hw_on")
                print()
    finally:
        set_hw_prefetch(0x0)   # always restore HW prefetchers ON
        print("restored MSR 0x1A4 = 0x0 (all HW prefetchers ON)")

    print(f"logged to {LOG}")


if __name__ == "__main__":
    if not BENCH.exists():
        sys.exit(f"error: {BENCH} not built -- run `make` in {HERE} first")
    main()
