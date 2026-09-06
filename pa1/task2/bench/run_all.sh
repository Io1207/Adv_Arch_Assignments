#!/usr/bin/env bash
# Runs every task2 collect sweep in sequence -> logs/*.csv + logs/*.run.log.
# Perf-counter columns are populated only if kernel.perf_event_paranoid <= 1.
# collect_hwprefetch is skipped unless run as root (it self-guards too).
set -u
cd "$(dirname "$0")"
make -s >/dev/null || { echo "build failed"; exit 1; }

SWEEPS=(collect_size collect_simd collect_prefetch collect_optimized)
[ "$(id -u)" -eq 0 ] && SWEEPS+=(collect_hwprefetch)

for s in "${SWEEPS[@]}"; do
  echo "======== $s ($(date +%H:%M:%S)) ========"
  ./"$s".py 2>&1 | tee "logs/$s.run.log"
  echo
  sleep 20          # brief thermal cooldown between sweeps
done
echo "ALL SWEEPS DONE ($(date +%H:%M:%S))"
