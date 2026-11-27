#!/usr/bin/env python3
"""Plots speedup against thread count from the benchmark CSV, one PNG per graph.

    plot_bench.py results/benchmarks.csv -o results

Reads the CSV written by bench/run_bench.sh and draws one line per parallel
variant plus a dashed ideal line (speedup = threads). pagerank_serial is the
baseline the speedups are relative to, so it is not drawn. Variants, graphs
and thread counts all come from the file.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

# Categorical slots 1-5 in fixed order (dataviz reference palette, light mode);
# markers differ per variant so the lines also read without colour.
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#191919"
MUTED = "#666666"
GRID = "#e4e4e4"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", default="results/benchmarks.csv")
    ap.add_argument("-o", "--output-dir", default="results")
    args = ap.parse_args()

    # graph -> (vertices, edges); graph -> variant -> [(threads, speedup)]
    sizes = {}
    series = defaultdict(lambda: defaultdict(list))
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["variant"] == "pagerank_serial":
                continue
            g = row["graph"]
            sizes[g] = (int(row["vertices"]), int(row["edges"]))
            series[g][row["variant"]].append((int(row["threads"]), float(row["speedup"])))

    os.makedirs(args.output_dir, exist_ok=True)
    for graph, variants in series.items():
        threads = sorted({t for pts in variants.values() for t, _ in pts})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(threads, threads, "--", color=MUTED, linewidth=1.5, label="ideal")
        for i, (variant, pts) in enumerate(variants.items()):
            pts.sort()
            ax.plot(*zip(*pts), color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)],
                    markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                    linewidth=2, label=variant.removeprefix("pagerank_"))

        ax.set_xscale("log", base=2)
        ax.set_xticks(threads, [str(t) for t in threads])
        ax.minorticks_off()
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup over serial")
        ax.set_ylim(bottom=0)
        n, m = sizes[graph]
        ax.set_title(f"{graph} ({n:,} vertices, {m:,} edges)", color=INK)
        ax.grid(color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        out = os.path.join(args.output_dir, f"speedup-{graph}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
