// ggml_student_sgemm.cpp  adapter that exposes the student's matmul_optimized to ggml.
//
// Copied into ggml/src/ggml-cpu/ by run_demo.sh and compiled into libggml-cpu. The injected
// call in ggml_compute_forward_mul_mat (see llama/inject.py) calls student_mul_mat_f32 for the
// single-threaded F32 x F32 case.
//
// ggml's mul_mat semantics (from ggml.h): with src0 = A [ne01 rows x ne00 cols] and
// src1 = B [ne11 rows x ne00 cols], the destination is dst[i1*ldc + i0] = dot(A_row_i0, B_row_i1)
// for i0 in [0,ne01), i1 in [0,ne11).  The student's matmul computes
//     C[i*ldc + j] = dot(Amat_row_i, Bmat_row_j).
// Mapping (dot is symmetric in its operands): call matmul_optimized with Amat = src1, Bmat = src0,
// M = ne11, N = ne01, K = ne00  then C[i1*ldc + i0] = dot(src1_i1, src0_i0), exactly ggml's dst.

#include <cstdint>
#include <cstdio>

#include "matmul.h"
#include "matmul_naive.cpp"      // provides matmul_naive (some kernels use it for tails)
#include "matmul_optimized.cpp"  // the student's kernel (defines matmul_optimized)

extern "C" void student_mul_mat_f32(int64_t, int64_t, int64_t, const float*, int64_t,
                                    const float*, int64_t, float*, int64_t);

extern "C" void student_mul_mat_f32(int64_t ne01, int64_t ne11, int64_t ne00,
                                    const float* src0, int64_t lda,   // A = src0, stride lda
                                    const float* src1, int64_t ldb,   // B = src1, stride ldb
                                    float* dst, int64_t ldc) {
    static bool announced = false;
    if (!announced) {
        announced = true;
        std::fprintf(stderr, "[student] matmul_optimized is now serving ggml F32 mul_mat\n");
    }
    matmul_optimized(src1, src0, dst,
                     static_cast<int>(ne11), static_cast<int>(ne01), static_cast<int>(ne00),
                     static_cast<int>(ldb), static_cast<int>(lda), static_cast<int>(ldc));
}
