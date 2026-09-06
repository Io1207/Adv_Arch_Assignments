# task2/bench  analysis-only build (NOT graded, NOT part of the submission's src/).
#
# Same pinned flags as the top-level Makefile so instruction/cycle counts stay
# comparable to what the graded ./bin/matmul would produce.

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include -Wall

COMMON_SRC := ../src/matmul_naive.cpp ../src/matmul_simd.cpp ../src/matmul_prefetch.cpp \
              ../src/matmul_optimized.cpp

.PHONY: all clean
all: bench_stage bench_prefetch

bench_stage: $(COMMON_SRC) bench_stage.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

bench_prefetch: $(COMMON_SRC) bench_prefetch.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f bench_stage bench_prefetch
