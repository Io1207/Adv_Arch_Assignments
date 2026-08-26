#!/usr/bin/env python3
# inject.py <llama.cpp-root>
#
# Robustly (anchor-based, not line-based) injects the student's matmul into ggml's CPU
# matmul, so a single-threaded F32 x F32 mul_mat calls student_mul_mat_f32 (defined in the
# ggml_student_sgemm.cpp adapter that run_demo.sh copies into ggml/src/ggml-cpu/).
#
# Two edits, both idempotent:
#   1) ggml/src/ggml-cpu/ggml-cpu.c    -- insert the dispatch block inside ggml_compute_forward_mul_mat
#   2) ggml/src/ggml-cpu/CMakeLists.txt -- add the adapter to the CPU backend sources

import sys, os

MARKER = "CS683 PA-1 student matmul injection"

CALL_BLOCK = r'''
    // ==== CS683 PA-1 student matmul injection ====
    {
        extern void student_mul_mat_f32(int64_t, int64_t, int64_t,
                                        const float *, int64_t,
                                        const float *, int64_t,
                                        float *, int64_t);
        if (nth == 1 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
            const int64_t sr2 = ne12 / ne02;
            const int64_t sr3 = ne13 / ne03;
            for (int64_t i13 = 0; i13 < ne13; i13++) {
                for (int64_t i12 = 0; i12 < ne12; i12++) {
                    student_mul_mat_f32(ne01, ne11, ne00,
                        (const float *)((const char *)src0->data + i12/sr2*nb02 + i13/sr3*nb03), (int64_t)(nb01/sizeof(float)),
                        (const float *)((const char *)src1->data + i12*nb12 + i13*nb13),         (int64_t)(nb11/sizeof(float)),
                        (float *)((char *)dst->data + i12*nb2 + i13*nb3),                         (int64_t)(nb1/sizeof(float)));
                }
            }
            return;
        }
    }
    // ==== end injection ====
'''

ANCHOR = '    // TODO: extract to "extra_op"'


def patch_cpu_c(root):
    path = os.path.join(root, "ggml/src/ggml-cpu/ggml-cpu.c")
    src = open(path).read()
    if MARKER in src:
        print("  ggml-cpu.c already injected  skipping")
        return
    if ANCHOR not in src:
        sys.exit(f"ERROR: anchor not found in {path} (llama.cpp version mismatch?)")
    # Insert the dispatch block right before the anchor comment (which sits at the top of
    # ggml_compute_forward_mul_mat, after the ith/nth locals are in scope).
    src = src.replace(ANCHOR, CALL_BLOCK + "\n" + ANCHOR, 1)
    open(path, "w").write(src)
    print("  patched ggml-cpu.c")


def patch_cmake(root):
    path = os.path.join(root, "ggml/src/ggml-cpu/CMakeLists.txt")
    src = open(path).read()
    if "ggml_student_sgemm.cpp" in src:
        print("  CMakeLists.txt already injected  skipping")
        return
    key = "ggml-cpu/ggml-cpu.cpp"
    if key not in src:
        sys.exit(f"ERROR: '{key}' not found in {path} (version mismatch?)")
    src = src.replace(key, key + "\n        ggml-cpu/ggml_student_sgemm.cpp", 1)
    open(path, "w").write(src)
    print("  patched CMakeLists.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: inject.py <llama.cpp-root>")
    root = sys.argv[1]
    patch_cpu_c(root)
    patch_cmake(root)
    print("injection complete")
