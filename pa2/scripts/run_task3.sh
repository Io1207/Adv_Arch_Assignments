#!/usr/bin/env bash
# Task 3: SPP + prefetch-aware dynamic I-RRIP.
# Results go to output/task3/<config>/<trace>.txt
# Usage (from pa2/):  ./run_task3.sh        Env: JOBS (default 6), WARMUP / SIM (default 25M)

set -u
cd "$(dirname "$0")"

WARMUP=${WARMUP:-25000000}
SIM=${SIM:-25000000}
JOBS=${JOBS:-6}
TRACE_DIR=traces
OUT_DIR=output/task3

# name        l1d_pref  l2c_repl
CONFIGS="
baseline      no        lru
spp_lru       spp       lru
dyn           no        dyn_irrip
spp_dyn       spp       dyn_irrip
spp_dynpf3    spp       dyn_irrip_pf
"

# ChampSim needs a .gz/.xz extension; our traces have none
mkdir -p "$TRACE_DIR/gz"
TRACES=""
for t in "$TRACE_DIR"/*; do
    [ -f "$t" ] || continue
    name=$(basename "$t")
    ln -sf "../$name" "$TRACE_DIR/gz/$name.gz"
    TRACES="$TRACES $name"
done
[ -n "$TRACES" ] || { echo "No traces found in $TRACE_DIR/"; exit 1; }

# Build one binary per config (sequential: builds share src/ and obj/)
NAMES=""
while read -r name pref repl; do
    [ -n "$name" ] || continue
    echo "=== Building $name (L1D: $pref, L2C: $repl) ==="
    ./build_champsim.sh "$pref" "$repl" "$name" > "build_$name.log" 2>&1
    bin=$(ls bin/*-1core-"$name" 2>/dev/null | head -1)
    if [ -z "$bin" ]; then
        # macOS cannot link with -static (crt0.o missing): relink without it
        make LDFlags= >> "build_$name.log" 2>&1
        [ -f bin/champsim ] || { echo "BUILD FAILED, see build_$name.log"; exit 1; }
        bin="bin/$name"
        mv -f bin/champsim "$bin"
    fi
    rm -f "build_$name.log"
    eval "BIN_$name=\$bin"
    NAMES="$NAMES $name"
done <<< "$CONFIGS"

# Run every config on every trace, at most $JOBS at a time
for name in $NAMES; do
    mkdir -p "$OUT_DIR/$name"
    eval "bin=\$BIN_$name"
    for t in $TRACES; do
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
        echo "=== Running $name on $t ==="
        "./$bin" -warmup_instructions "$WARMUP" -simulation_instructions "$SIM" \
            -traces "$TRACE_DIR/gz/$t.gz" > "$OUT_DIR/$name/$t.txt" 2>&1 &
    done
done
wait

echo ""
echo "Done. IPC:"
for name in $NAMES; do
    for t in $TRACES; do
        ipc=$(grep -m1 "CPU 0 cumulative IPC" "$OUT_DIR/$name/$t.txt" | awk '{print $5}')
        printf "  %-12s %-12s %s\n" "$name" "$t" "${ipc:-<missing>}"
    done
done
