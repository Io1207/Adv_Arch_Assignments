#!/usr/bin/env python3
"""make_plots.py -- generates every figure task1_report.pdf embeds, from the
CSV logs in task1/bench/logs/. Not part of the graded submission; run this
before building the report whenever the logs change.

All figures are grouped bar charts: one bar group per image size, one bar
per variant/stage within a group, faceted (as separate subplots) by kernel
size where more than one K is being compared.

Usage: ./make_plots.py   (writes PNGs into report/figs/)
"""
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "bench" / "logs"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

KS = [3, 5, 7, 9]
SIZES = [128, 256, 512, 1024, 2048, 4096]

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})


def load(name):
    with open(LOGS / name, newline="") as f:
        return list(csv.DictReader(f))


def series(rows, stage, K, extra=None):
    """Return {size: value} of speedup_vs_naive for one stage at one K,
    optionally filtered by an extra {column: value} predicate."""
    out = {}
    for r in rows:
        if r["stage"] != stage or int(r["K"]) != K:
            continue
        if extra and any(r.get(k) != v for k, v in extra.items()):
            continue
        out[int(r["H"])] = float(r["speedup_vs_naive"])
    return out


def metric_series(rows, stage, K, col, extra=None, normalize_by_reps=False):
    out = {}
    for r in rows:
        if r["stage"] != stage or int(r["K"]) != K:
            continue
        if extra and any(r.get(k) != v for k, v in extra.items()):
            continue
        v = r.get(col)
        if v is None or v == "":
            continue
        v = float(v)
        if normalize_by_reps:
            v = v / float(r["perf_reps"])
        out[int(r["H"])] = v
    return out


def bars(ax, series_list, logy=False):
    """series_list: [(label, color, {size: value}), ...]. Draws one grouped
    bar chart with a bar per series in each of len(SIZES) groups."""
    n = len(series_list)
    x = np.arange(len(SIZES))
    width = 0.8 / n
    for i, (label, color, data) in enumerate(series_list):
        vals = [data.get(s, 0.0) for s in SIZES]
        offset = (i - (n - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, color=color, label=label,
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SIZES], rotation=45)
    if logy:
        ax.set_yscale("log", base=2)


def facet_by_k(build_series_for_k, title, ylabel, outfile, logy=False,
               figsize=(11, 3), hline=1.0):
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=False)
    for ax, K in zip(axes, KS):
        series_list = build_series_for_k(K)
        bars(ax, series_list, logy=logy)
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--", linewidth=1, zorder=0)
        ax.set_title(f"K={K}")
        ax.set_xlabel("image size N (N x N)")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGS / outfile)
    plt.close(fig)
    print(f"wrote {outfile}")


# ---------------------------------------------------------------------------
# 1. The mandated figure: speedup of each FINAL graded stage vs naive,
#    faceted by K, varying image size.
# ---------------------------------------------------------------------------
reorder_rows = load("task1a_reorder.csv")
unroll_rows = load("task1a_unroll.csv")
tile_rows = load("task1b_tile.csv")
simd_rows = load("task1c_simd.csv")
opt_rows = load("task1_optimized.csv")

FINAL = [
    ("reorder", reorder_rows, "reorder_oy_ky_kx_ox", None, "tab:blue"),
    ("unroll", unroll_rows, "unroll_ox8", None, "tab:orange"),
    ("tile", tile_rows, "tile_square", {"tile_size": "16"}, "tab:green"),
    ("simd", simd_rows, "simd256", None, "tab:red"),
    ("optimized", opt_rows, "acc8", None, "tab:purple"),
]


def summary_for_k(K):
    return [(label, color, series(rows, stage, K, extra))
            for label, rows, stage, extra, color in FINAL]


facet_by_k(summary_for_k, "Speedup vs conv_naive -- final graded stages",
           "speedup (x)", "summary_speedup.png", logy=True, figsize=(12, 3.2))

# ---------------------------------------------------------------------------
# 2. Stage 1: reorder variants
# ---------------------------------------------------------------------------
def reorder_for_k(K):
    return [
        ("reorder_ky_kx_oy_ox", "tab:red", series(reorder_rows, "reorder_ky_kx_oy_ox", K)),
        ("reorder_oy_ky_kx_ox", "tab:blue", series(reorder_rows, "reorder_oy_ky_kx_ox", K)),
    ]


facet_by_k(reorder_for_k, "Stage 1: loop reordering -- ky,kx,oy,ox vs oy,ky,kx,ox",
           "speedup (x)", "reorder_variants.png")


def reorder_mpki_for_k(K):
    return [
        ("reorder_ky_kx_oy_ox", "tab:red",
         metric_series(reorder_rows, "reorder_ky_kx_oy_ox", K, "l1d_mpki")),
        ("reorder_oy_ky_kx_ox", "tab:blue",
         metric_series(reorder_rows, "reorder_oy_ky_kx_ox", K, "l1d_mpki")),
    ]


facet_by_k(reorder_mpki_for_k, "Stage 1: L1-D MPKI -- ky,kx,oy,ox vs oy,ky,kx,ox",
           "L1-D MPKI", "reorder_mpki.png", logy=True, hline=None)

# ---------------------------------------------------------------------------
# 3. Stage 2: unroll variants
# ---------------------------------------------------------------------------
def unroll_for_k(K):
    return [
        ("unroll_ox8", "tab:blue", series(unroll_rows, "unroll_ox8", K)),
        ("unroll_kx4", "tab:red", series(unroll_rows, "unroll_kx4", K)),
    ]


facet_by_k(unroll_for_k, "Stage 2: loop unrolling -- ox x8 (8 accumulators) vs kx x4 (1 accumulator)",
           "speedup (x)", "unroll_variants.png")

# ---------------------------------------------------------------------------
# 4. Stage 3: tiling -- shape comparison, and the auto-formula failure
# ---------------------------------------------------------------------------
def tile_shape_for_k(K):
    return [
        ("row, ts=16", "tab:blue", series(tile_rows, "tile_row", K, {"tile_size": "16"})),
        ("square, ts=16", "tab:red", series(tile_rows, "tile_square", K, {"tile_size": "16"})),
    ]


facet_by_k(tile_shape_for_k, "Stage 3: row vs square tiling, fixed tile size = 16",
           "speedup (x)", "tile_shape.png")


def auto_tile_size(W, K):
    L1D = 48 * 1024
    budget = L1D - 4 * K * K
    return max(1, budget // (2 * 4 * W))


def tile_auto_for_k(K):
    fixed = series(tile_rows, "tile_row", K, {"tile_size": "16"})
    auto = {}
    for N in SIZES:
        ts = str(auto_tile_size(N, K))
        d = series(tile_rows, "tile_row", K, {"tile_size": ts})
        if N in d:
            auto[N] = d[N]
    return [
        ("fixed ts=16", "tab:blue", fixed),
        ("L1D-capacity formula", "tab:red", auto),
    ]


facet_by_k(tile_auto_for_k, "Stage 3: fixed tile size vs the L1D-capacity formula (row tiling)",
           "speedup (x)", "tile_auto_vs_fixed.png")

# ---------------------------------------------------------------------------
# 5. Stage 4: SIMD widths
# ---------------------------------------------------------------------------
def simd_for_k(K):
    return [
        ("simd128", "tab:red", series(simd_rows, "simd128", K)),
        ("simd256", "tab:blue", series(simd_rows, "simd256", K)),
    ]


facet_by_k(simd_for_k, "Stage 4: SIMD -- SSE (__m128, 4-wide) vs AVX2 (__m256, 8-wide)",
           "speedup (x)", "simd_variants.png")


def simd_instr_for_k(K):
    def scaled(stage):
        d = metric_series(simd_rows, stage, K, "instructions", normalize_by_reps=True)
        return {k: v / 1e6 for k, v in d.items()}
    return [
        ("simd128", "tab:red", scaled("simd128")),
        ("simd256", "tab:blue", scaled("simd256")),
    ]


facet_by_k(simd_instr_for_k, "Stage 4: instructions per call (millions) -- SSE vs AVX2",
           "instructions / call (M)", "simd_instructions.png", logy=True, hline=None)

# ---------------------------------------------------------------------------
# 6. Stage 5: accumulator-count sweep
# ---------------------------------------------------------------------------
ACC_COLORS = {"acc1": "tab:red", "acc2": "tab:orange", "acc4": "tab:green",
              "acc8": "tab:blue", "acc16": "tab:purple"}


def optimized_for_k(K):
    return [(stage, ACC_COLORS[stage], series(opt_rows, stage, K))
            for stage in ["acc1", "acc2", "acc4", "acc8", "acc16"]]


facet_by_k(optimized_for_k, "Stage 5: accumulator count N (chunk width N x 8) vs naive",
           "speedup (x)", "optimized_variants.png", logy=True, figsize=(12, 3.2))

# ---------------------------------------------------------------------------
# Mean-speedup-by-accumulator-count bar chart (single summary number per N)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.2))
means = {}
for stage in ["acc1", "acc2", "acc4", "acc8", "acc16"]:
    vals = [float(r["speedup_vs_naive"]) for r in opt_rows if r["stage"] == stage]
    means[stage] = sum(vals) / len(vals)
labels = list(means.keys())
ax.bar(labels, [means[l] for l in labels],
       color=[ACC_COLORS[l] for l in labels], edgecolor="black", linewidth=0.3)
for i, l in enumerate(labels):
    ax.text(i, means[l] + 0.2, f"{means[l]:.2f}x", ha="center", fontsize=8)
ax.set_ylabel("mean speedup vs naive (24 configs)")
ax.set_title("Stage 5: mean speedup by accumulator count N")
fig.tight_layout()
fig.savefig(FIGS / "optimized_mean_bar.png")
plt.close(fig)
print("wrote optimized_mean_bar.png")

print("\nall figures written to", FIGS)
