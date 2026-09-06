# task2/bench  analysis-only build (NOT graded, NOT part of the submission's src/).
#
# Same pinned flags as the top-level Makefile so instruction/cycle counts stay
# comparable to what the graded ./bin/matmul would produce. The variant files
# (simd_widths / prefetch_tuned / opt_variants) hold bench-only entry points
# that keep src/matmul_*.cpp untouched (README section 5).

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include -Wall

COMMON_SRC := ../src/matmul_naive.cpp ../src/matmul_simd.cpp ../src/matmul_prefetch.cpp \
              ../src/matmul_optimized.cpp
VARIANT_SRC := simd_widths.cpp prefetch_tuned.cpp opt_variants.cpp

.PHONY: all clean
all: bench_stage bench_prefetch

bench_stage: $(COMMON_SRC) $(VARIANT_SRC) bench_stage.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

bench_prefetch: $(COMMON_SRC) $(VARIANT_SRC) bench_prefetch.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

# Optional: separate AVX-512 build of the SIMD-width variant, ONLY for the
# 512-bit point of the "speedup vs SIMD width" plot. Not built by `all` and
# not usable on CPUs without AVX-512 (this i7-12650H has none).
bench_stage_avx512: $(COMMON_SRC) prefetch_tuned.cpp opt_variants.cpp bench_stage.cpp simd_widths.cpp
	$(CXX) -std=c++17 -O2 -fno-tree-vectorize -mavx512f -mfma -I../include -Wall $^ -o $@

clean:
	rm -f bench_stage bench_prefetch bench_stage_avx512
