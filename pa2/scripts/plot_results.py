#!/usr/bin/env python3
"""Plot ChampSim results as grouped bar charts (one group per trace, one bar per config).

Reads <dir>/<config>/<trace>.txt (the layout run_task2a.sh writes) and produces:
  <dir>/plots/ipc.png, speedup.png, l2c_load_mpki.png,
  l2c_instr_mpki.png, l2c_data_mpki.png
  <dir>/plots/summary.csv and summary.md   (the tables for the report)

Usage:
  python3 plot_results.py                                  # output/task2a, baseline rrip
  python3 plot_results.py --dir output/task2a --baseline lru
  python3 plot_results.py --configs rrip irrip_hit0 irrip_hit1 \
      --labels RRIP "I-RRIP (data hit->0)" "I-RRIP (data hit->1)"

Needs matplotlib:  pip3 install matplotlib
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# metric key -> (regex on the ChampSim output, plot title, y-axis label)
METRICS = {
    "ipc":            (r"^CPU 0 cumulative IPC:\s+([\d.]+)",          "IPC",                    "IPC"),
    "l1d_load_mpki":  (r"^L1D LOAD\s+ACCESS:.*MPKI:\s+([\d.]+)",      "L1D Load MPKI",          "MPKI"),
    "l2c_load_mpki": (r"^L2C LOAD\s+ACCESS:.*MPKI:\s+([\d.]+)",      "L2C Load MPKI",          "MPKI"),
    "l2c_instr_mpki": (r"^L2C INSTRUCTION LOAD MPKI:\s+([\d.]+)",     "L2C Instruction Load MPKI", "MPKI"),
    "l2c_data_mpki":  (r"^L2C DATA LOAD MPKI:\s+([\d.]+)",            "L2C Data Load MPKI",     "MPKI"),
}

# fixed categorical order: a config keeps its colour no matter which subset is plotted
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# table-only extras, present only for some policies (dynamic I-RRIP prints these)
EXTRAS = {
    "irrip_pct":  r"follower updates in I-RRIP mode: \d+ \(([\d.]+)%\)",
    "ratio_mean": r"instr ratio per epoch: min [\d.]+\s+mean ([\d.]+)",
    # L1D prefetcher: accuracy is ChampSim's (useful / issued to lower level); "nan" without a prefetcher -> None
    "pf_accuracy":   r"^L1D USEFUL LOAD PREFETCHES:.*ACCURACY:\s+([\d.]+)",
    "l1d_load_miss": r"^L1D LOAD\s+ACCESS:\s+\d+\s+HIT:\s+\d+\s+MISS:\s+(\d+)",
}


def parse(path):
    text = path.read_text(errors="replace")
    vals = {}
    for key, pattern in [(k, v[0]) for k, v in METRICS.items()] + list(EXTRAS.items()):
        m = re.search(pattern, text, re.MULTILINE)
        vals[key] = float(m.group(1)) if m else None
    return vals


def geomean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def style_axes(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=12, pad=10)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.set_xlabel("Traces", color=INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors=INK_MUTED)


def bar_chart(groups, configs, labels, colors, series, title, ylabel, out, base=0.0, ref_line=None):
    """series[config] = list of values aligned with groups; bars grow from `base`.
    ref_line = (y, label) draws a dashed reference line, e.g. the speedup baseline."""
    n = len(configs)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(8.5, 1.6 * len(groups) * max(1, n / 2.5)), 4.2))
    if ref_line is not None:
        ax.axhline(ref_line[0], color=INK_MUTED, linestyle="--", linewidth=1, zorder=3, label=ref_line[1])
    for i, cfg in enumerate(configs):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        heights = [(v - base) if v is not None else 0 for v in series[cfg]]
        ax.bar(xs, heights, width, bottom=base, label=labels[i],
               color=colors[i], edgecolor="white", linewidth=1.5, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    style_axes(ax, title, ylabel)
    if base != 0.0:     # speedup: zoom around 1.0 with some headroom
        vals = [v for cfg in configs for v in series[cfg] if v is not None] + [base]
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.25, 0.005)
        ax.set_ylim(lo - pad, hi + pad)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="output/task2a", help="results directory (one sub-folder per config)")
    ap.add_argument("--baseline", default="rrip", help="config that speedup is measured against")
    ap.add_argument("--configs", nargs="+", help="configs to plot, in order (default: all, baseline first)")
    ap.add_argument("--labels", nargs="+", help="legend names, one per --configs entry")
    ap.add_argument("--out", help="where to write plots and tables (default: <dir>/plots)")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        sys.exit(f"No such directory: {root}")

    configs = args.configs or sorted(
        (d.name for d in root.iterdir() if d.is_dir() and d.name != "plots" and any(d.glob("*.txt"))),
        key=lambda c: (c != args.baseline, c))
    labels = args.labels or configs
    if len(labels) != len(configs):
        sys.exit("--labels needs exactly one entry per config")
    if args.baseline not in configs:
        sys.exit(f"Baseline '{args.baseline}' not found among {configs}")

    traces = sorted({p.stem for c in configs for p in (root / c).glob("*.txt")})
    data = {c: {t: parse(root / c / f"{t}.txt") if (root / c / f"{t}.txt").exists() else {}
                for t in traces} for c in configs}
    for c in configs:
        for t in traces:
            if not data[c][t] or data[c][t]["ipc"] is None:
                print(f"warning: no IPC for {c}/{t} (run missing or unfinished)")

    # speedup = IPC(config) / IPC(baseline), per trace, plus geomean
    for c in configs:
        for t in traces:
            ipc, base = data[c][t].get("ipc"), data[args.baseline][t].get("ipc")
            data[c][t]["speedup"] = ipc / base if ipc and base else None

    # coverage = (baseline L1D load misses - config's) / baseline's, only for configs with a prefetcher
    for c in configs:
        for t in traces:
            miss, base = data[c][t].get("l1d_load_miss"), data[args.baseline][t].get("l1d_load_miss")
            has_pf = data[c][t].get("pf_accuracy") is not None
            data[c][t]["coverage"] = 100.0 * (base - miss) / base if has_pf and base else None
    pf_configs = [i for i, c in enumerate(configs)
                  if any(data[c][t].get("pf_accuracy") is not None for t in traces)]

    out =Path(args.out) if args.out else root / "plots"
    out.mkdir(parents=True, exist_ok=True)

    # --- plots ---
    colors = [COLORS[i % len(COLORS)] for i in range(len(configs))]
    base_label = labels[configs.index(args.baseline)]

    # speedup: baseline is the dashed line at 1.0, the other configs are bars
    others = [i for i, c in enumerate(configs) if c != args.baseline]
    sp = {configs[i]: [data[configs[i]][t]["speedup"] for t in traces]
                      + [geomean([data[configs[i]][t]["speedup"] for t in traces])] for i in others}
    bar_chart(traces + ["GeoMean"], [configs[i] for i in others], [labels[i] for i in others],
              [colors[i] for i in others], sp, f"Speedup over {base_label}", "Speedup",
              out / "speedup.png", base=1.0, ref_line=(1.0, f"{base_label} (baseline)"))

    for key, (_, title, ylabel) in METRICS.items():
        series = {c: [data[c][t].get(key) for t in traces] for c in configs}
        bar_chart(traces, configs, labels, colors, series, title, ylabel, out / f"{key}.png")

    # prefetch accuracy / coverage: only the configs that have a prefetcher
    if pf_configs:
        for key, title in (("pf_accuracy", "L1D Prefetch Accuracy"), ("coverage", "L1D Prefetch Coverage")):
            series = {configs[i]: [data[configs[i]][t].get(key) for t in traces] for i in pf_configs}
            bar_chart(traces, [configs[i] for i in pf_configs], [labels[i] for i in pf_configs],
                      [colors[i] for i in pf_configs], series, title, "%", out / f"{key}.png")

    # --- tables ---
    cols = ["ipc", "speedup", "l1d_load_mpki", "l2c_load_mpki", "l2c_instr_mpki", "l2c_data_mpki"]
    heads = ["IPC", "Speedup", "L1D Load MPKI", "L2C Load MPKI", "L2C Instr MPKI", "L2C Data MPKI"]
    if pf_configs:
        cols += ["pf_accuracy", "coverage"]
        heads += ["PF Accuracy %", "Coverage %"]
    if any(data[c][t].get("irrip_pct") is not None for c in configs for t in traces):
        cols += ["ratio_mean", "irrip_pct"]
        heads += ["Mean Instr Ratio", "I-RRIP Mode %"]
    fmt = lambda v: "-" if v is None else f"{v:.4f}"
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Trace", "Config"] + heads)
        for t in traces:
            for c, l in zip(configs, labels):
                w.writerow([t, l] + [fmt(data[c][t].get(k)) for k in cols])
    with open(out / "summary.md", "w") as f:
        f.write("| Trace | Config | " + " | ".join(heads) + " |\n")
        f.write("|---|---|" + "---|" * len(heads) + "\n")
        for t in traces:
            for c, l in zip(configs, labels):
                f.write(f"| {t} | {l} | " + " | ".join(fmt(data[c][t].get(k)) for k in cols) + " |\n")

    print((out / "summary.md").read_text())
    print(f"Plots and tables written to {out}/")


if __name__ == "__main__":
    main()
