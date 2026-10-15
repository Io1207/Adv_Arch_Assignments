#!/usr/bin/env bash
# Task 2a: build and run RRIP and I-RRIP (DATA_HIT_RRPV = 0 and 1) on every trace.
# Results go to output/task2a/<config>/<trace>.txt
#
# Usage:  ./run_task2a.sh            (run from the pa2/ directory)
# Env:    JOBS=4 ./run_task2a.sh     max simulations in parallel (default 3)
#         WARMUP / SIM               instruction counts (default 25M each)

set -u
cd "$(dirname "$0")"

WARMUP=${WARMUP:-25000000}
SIM=${SIM:-25000000}
JOBS=${JOBS:-3}
L1D_PREF=no
TRACE_DIR=traces
OUT_DIR=${OUT_DIR:-output/task2a}

# config name -> l2c replacement file (replacement/<file>.l2c_repl)
CONFIGS="rrip irrip_hit0 irrip_hit1"

# ---------------------------------------------------------------------------
# ChampSim picks the decompressor from the trace's extension (.gz / .xz).
# Our traces have no extension, so expose them through .gz symlinks.
mkdir -p "$TRACE_DIR/gz"
TRACES=""
for t in "$TRACE_DIR"/*; do
    [ -f "$t" ] || continue
    name=$(basename "$t")
    ln -sf "../$name" "$TRACE_DIR/gz/$name.gz"
    TRACES="$TRACES $name"
done
[ -n "$TRACES" ] || { echo "No traces found in $TRACE_DIR/"; exit 1; }

# ---------------------------------------------------------------------------
# Build one binary per config. Builds are sequential because they share src/obj.
build() {
    local cfg=$1 repl=$2
    echo "=== Building $cfg (L2C repl: $repl) ==="
    ./build_champsim.sh "$L1D_PREF" "$repl" "$cfg" > "build_$cfg.log" 2>&1

    local bin
    bin=$(ls bin/*-"$repl"-*-1core-"$cfg" 2>/dev/null | head -1)
    if [ -z "$bin" ]; then
        # macOS cannot link with -static (crt0.o missing): relink without it
        echo "    build script failed to link, retrying without -static"
        make LDFlags= >> "build_$cfg.log" 2>&1
        [ -f bin/champsim ] || { echo "    BUILD FAILED, see build_$cfg.log"; exit 1; }
        bin="bin/hashed_perceptron-no-$L1D_PREF-no-no-no-no-no-lru-lru-lru-$repl-lru-lru-lru-lru-1core-$cfg"
        mv bin/champsim "$bin"
    fi
    rm -f "build_$cfg.log"
    echo "    -> $bin"
    eval "BIN_$cfg=\$bin"
}

for cfg in $CONFIGS; do
    case $cfg in
        rrip)
            build "$cfg" rrip ;;
        irrip_hit*)
            hit=${cfg#irrip_hit}
            # temporary copy of irrip with DATA_HIT_RRPV fixed to $hit
            sed "s/^#define DATA_HIT_RRPV .*/#define DATA_HIT_RRPV $hit/" \
                replacement/irrip.l2c_repl > "replacement/$cfg.l2c_repl"
            build "$cfg" "$cfg"
            rm -f "replacement/$cfg.l2c_repl" ;;
    esac
done

# ---------------------------------------------------------------------------
# Run every config on every trace, at most $JOBS at a time.
for cfg in $CONFIGS; do
    mkdir -p "$OUT_DIR/$cfg"
    eval "bin=\$BIN_$cfg"
    for t in $TRACES; do
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
        out="$OUT_DIR/$cfg/$t.txt"
        echo "=== Running $cfg on $t -> $out ==="
        "./$bin" -warmup_instructions "$WARMUP" -simulation_instructions "$SIM" \
            -traces "$TRACE_DIR/gz/$t.gz" > "$out" 2>&1 &
    done
done
wait

echo ""
echo "Done. IPC summary:"
for cfg in $CONFIGS; do
    for t in $TRACES; do
        ipc=$(grep -m1 "CPU 0 cumulative IPC" "$OUT_DIR/$cfg/$t.txt" | awk '{print $5}')
        printf "  %-12s %-12s IPC %s\n" "$cfg" "$t" "${ipc:-<missing - check log>}"
    done
done
