# task1/bench  analysis-only build (NOT graded, NOT part of the submission's src/).
#
# Same pinned flags as the top-level Makefile so instruction/cycle counts stay
# comparable to what the graded ./bin/conv would produce.

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include -Wall

COMMON_SRC := ../src/conv_naive.cpp ../src/conv_reorder.cpp ../src/conv_unroll.cpp \
              ../src/conv_tile.cpp ../src/conv_simd.cpp ../src/conv_optimized.cpp

.PHONY: all clean
all: bench_stage

bench_stage: $(COMMON_SRC) bench_stage.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f bench_stage
