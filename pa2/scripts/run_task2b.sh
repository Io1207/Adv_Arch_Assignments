#!/usr/bin/env bash
# Task 2b: sweep one #define of a dynamic I-RRIP policy on every trace.
# Results go to output/task2b/<prefix>_<value>/<trace>.txt
# The Task 2a RRIP and I-RRIP (data hit -> 1) results are copied in as reference points.
#
# Usage:  ./run_task2b.sh                      (run from the pa2/ directory)
#         default: threshold sweep of replacement/dyn_irrip.l2c_repl -> thr_0.25, thr_0.35, ...
#
#         POLICY=psel_irrip PARAM=INSTR_MISS_WEIGHT VALUES="1 2 4" PREFIX=psel_w ./run_task2b.sh
#                                              any #define of any policy file
# Env:    JOBS=6                              max simulations in parallel (default 6)
#         WARMUP / SIM                        instruction counts (default 25M each)

set -u
cd "$(dirname "$0")"

WARMUP=${WARMUP:-25000000}
SIM=${SIM:-25000000}
JOBS=${JOBS:-6}
POLICY=${POLICY:-dyn_irrip}
PARAM=${PARAM:-THRESHOLD}
VALUES=${VALUES:-${THRESHOLDS:-"0.25 0.35 0.40 0.45 0.50"}}
PREFIX=${PREFIX:-thr}
L1D_PREF=no
TRACE_DIR=traces
OUT_DIR=${OUT_DIR:-output/task2b}
REF_DIR=output/task2a

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
# Build one binary per value. Builds are sequential because they share src/obj.
[ -f "replacement/$POLICY.l2c_repl" ] || { echo "No replacement/$POLICY.l2c_repl"; exit 1; }
grep -q "^#define $PARAM " "replacement/$POLICY.l2c_repl" || { echo "No '#define $PARAM' in $POLICY"; exit 1; }
CONFIGS=""
for val in $VALUES; do
    cfg="${PREFIX}_$val"
    repl="${POLICY}_${PREFIX}${val/./}"   # e.g. dyn_irrip_thr025 (no dot in the file name)
    sed "s/^#define $PARAM .*/#define $PARAM $val/" \
        "replacement/$POLICY.l2c_repl" > "replacement/$repl.l2c_repl"

    echo "=== Building $cfg (L2C repl: $repl) ==="
    ./build_champsim.sh "$L1D_PREF" "$repl" "$repl" > "build_$repl.log" 2>&1
    bin=$(ls bin/*-"$repl"-*-1core-"$repl" 2>/dev/null | head -1)
    if [ -z "$bin" ]; then
        # macOS cannot link with -static (crt0.o missing): relink without it
        echo "    build script failed to link, retrying without -static"
        make LDFlags= >> "build_$repl.log" 2>&1
        [ -f bin/champsim ] || { echo "    BUILD FAILED, see build_$repl.log"; exit 1; }
        bin="bin/hashed_perceptron-no-$L1D_PREF-no-no-no-no-no-lru-lru-lru-$repl-lru-lru-lru-lru-1core-$repl"
        mv -f bin/champsim "$bin"
    fi
    rm -f "build_$repl.log" "replacement/$repl.l2c_repl"
    echo "    -> $bin"
    eval "BIN_${repl}=\$bin"
    CONFIGS="$CONFIGS $cfg:$repl"
done

# ---------------------------------------------------------------------------
# Run every threshold on every trace, at most $JOBS at a time.
for entry in $CONFIGS; do
    cfg=${entry%%:*}; repl=${entry##*:}
    mkdir -p "$OUT_DIR/$cfg"
    eval "bin=\$BIN_$repl"
    for t in $TRACES; do
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
        out="$OUT_DIR/$cfg/$t.txt"
        echo "=== Running $cfg on $t -> $out ==="
        "./$bin" -warmup_instructions "$WARMUP" -simulation_instructions "$SIM" \
            -traces "$TRACE_DIR/gz/$t.gz" > "$out" 2>&1 &
    done
done
wait

# Reference points from Task 2a
for ref in rrip irrip_hit1; do
    if [ -d "$REF_DIR/$ref" ]; then
        mkdir -p "$OUT_DIR/$ref" && cp -f "$REF_DIR/$ref"/*.txt "$OUT_DIR/$ref/"
    else
        echo "note: $REF_DIR/$ref not found, run ./run_task2a.sh for the reference points"
    fi
done

echo ""
echo "Done. IPC / share of follower decisions in I-RRIP mode:"
for entry in $CONFIGS; do
    cfg=${entry%%:*}
    for t in $TRACES; do
        f="$OUT_DIR/$cfg/$t.txt"
        ipc=$(grep -m1 "CPU 0 cumulative IPC" "$f" | awk '{print $5}')
        pct=$(grep -m1 "follower updates in I-RRIP mode" "$f" | sed 's/.*(\(.*%\)).*/\1/')
        printf "  %-10s %-12s IPC %-10s I-RRIP %s\n" "$cfg" "$t" "${ipc:-<missing>}" "${pct:--}"
    done
done
