#!/usr/bin/env bash
# run_demo.sh  clone llama.cpp, build it, inject the student's matmul_optimized into ggml's
# CPU F32 matmul, and run a tiny LLM inference comparing the student kernel against llama.cpp's
# native CPU matmul (single-threaded).
#
# Invoked by `make llama-demo`. Overridable via environment variables:
#   LLAMA_TAG   git tag to build            (default b5731)
#   LLAMA_SRC   where to clone/build         (default <task2>/llama/llama.cpp)
#   MODEL_URL   tiny F32 GGUF to fetch       (default stories260K.gguf  F32, ~1.2MB)
#   PROMPT / NGEN                            (default "Once upon a time" / 64)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../task2/llama
TASK2="$(cd "$HERE/.." && pwd)"
TAG="${LLAMA_TAG:-b5731}"
SRC="${LLAMA_SRC:-$HERE/llama.cpp}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf}"
MODEL="$HERE/models/$(basename "$MODEL_URL")"
OPT_SRC="${OPT_SRC:-$TASK2/src/matmul_optimized.cpp}"   # instructor may point at solution/
PROMPT="${PROMPT:-Once upon a time}"
NGEN="${NGEN:-64}"
JOBS="$(nproc 2>/dev/null || echo 4)"
CMAKE_COMMON=(-DGGML_CUDA=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)

echo ">>> llama.cpp matmul demo (tag $TAG, 1 thread)"

# 1) clone pinned llama.cpp
if [ ! -d "$SRC/.git" ]; then
    echo ">>> cloning llama.cpp @ $TAG"
    git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi

# 1b) restore a pristine tree so re-runs are idempotent and the STOCK build is truly stock
echo ">>> resetting llama.cpp to pristine"
git -C "$SRC" checkout -- ggml/src/ggml-cpu/ggml-cpu.c ggml/src/ggml-cpu/CMakeLists.txt 2>/dev/null || true
rm -f "$SRC"/ggml/src/ggml-cpu/matmul.h \
      "$SRC"/ggml/src/ggml-cpu/matmul_naive.cpp \
      "$SRC"/ggml/src/ggml-cpu/matmul_optimized.cpp \
      "$SRC"/ggml/src/ggml-cpu/ggml_student_sgemm.cpp

# 2) STOCK build (native llama.cpp CPU matmul), before any injection
echo ">>> building STOCK llama-cli"
cmake -S "$SRC" -B "$SRC/build-stock" "${CMAKE_COMMON[@]}" >/dev/null
cmake --build "$SRC/build-stock" -j"$JOBS" --target llama-cli >/dev/null

# 3) inject the student's kernel and (re)build
echo ">>> injecting student matmul_optimized"
cp "$TASK2/include/matmul.h" "$TASK2/src/matmul_naive.cpp" \
   "$HERE/ggml_student_sgemm.cpp" "$SRC/ggml/src/ggml-cpu/"
cp "$OPT_SRC" "$SRC/ggml/src/ggml-cpu/matmul_optimized.cpp"
python3 "$HERE/inject.py" "$SRC"
echo ">>> building STUDENT llama-cli"
cmake -S "$SRC" -B "$SRC/build-student" "${CMAKE_COMMON[@]}" >/dev/null
cmake --build "$SRC/build-student" -j"$JOBS" --target llama-cli >/dev/null

# 4) tiny F32 model (F32 so the F32 mul_mat path  the one we injected  actually runs)
mkdir -p "$HERE/models"
[ -f "$MODEL" ] || { echo ">>> downloading tiny model"; curl -fL -o "$MODEL" "$MODEL_URL"; }

# 5) run both, single-threaded, deterministic
run() {  # <build-dir> <stdout-file> <stderr-file>
    "$1/bin/llama-cli" -m "$MODEL" -p "$PROMPT" -n "$NGEN" -t 1 -c 512 \
        --seed 0 -no-cnv --no-warmup >"$2" 2>"$3" || true
}
echo ">>> running STOCK inference";   run "$SRC/build-stock"   /tmp/mm_stock.out   /tmp/mm_stock.err
echo ">>> running STUDENT inference"; run "$SRC/build-student" /tmp/mm_student.out /tmp/mm_student.err

# 6) report
echo
echo "================ RESULT ================"
if grep -q "matmul_optimized is now serving" /tmp/mm_student.err; then
    echo "[ok] student matmul_optimized was actually used by ggml"
else
    echo "[WARN] student kernel marker not seen  the F32 path may not have fired"
    echo "       (is the model F32? is -t 1 set?)"
fi
if diff -q /tmp/mm_stock.out /tmp/mm_student.out >/dev/null; then
    echo "[ok] student output matches stock output token-for-token (kernel is correct)"
else
    echo "[FAIL] student output differs from stock  kernel is numerically wrong:"
    diff /tmp/mm_stock.out /tmp/mm_student.out | head
fi
echo
echo "--- timings (llama.cpp native) ---"; grep -E "eval time|tokens per second" /tmp/mm_stock.err   || true
echo "--- timings (student kernel)  ---"; grep -E "eval time|tokens per second" /tmp/mm_student.err || true
echo "======================================="
