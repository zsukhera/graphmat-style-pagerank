#!/usr/bin/env python3
"""Reports the sliced-ELLPACK padding: stored slots / edges, for several sigma.

    sell_padding.py data/soc-LiveJournal1.txt [--sigma 16384 ...]

Mirrors build_sell() in src/sell.h: rows are vertices, row length is the
in-degree (repeated u -> v edges counted once, self-loops kept, n = max id + 1,
as in common.h); rows are sorted by descending length inside windows of sigma
ids, cut into chunks of SELL_C = 4, and each chunk stores 4 x its longest row.
"""

import argparse

import numpy as np

SELL_C = 4


def slots(indeg, sigma):
    n = len(indeg)
    total = 0
    for w in range(0, n, sigma):
        d = np.sort(indeg[w:w + sigma])[::-1]
        # Chunks are cut over the whole perm array, and sigma is a multiple of
        # SELL_C, so every chunk lies inside one window. Pad the last one.
        pad = (-len(d)) % SELL_C
        if pad:
            d = np.concatenate([d, np.zeros(pad, dtype=d.dtype)])
        total += int(d.reshape(-1, SELL_C).max(axis=1).sum()) * SELL_C
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("graph", help="SNAP-style edge list")
    ap.add_argument("--sigma", type=int, nargs="*",
                    default=[SELL_C, 1024, 4096, 16384, 65536, 262144, 0],
                    help="window sizes (0 = n, a full sort); multiples of 4")
    args = ap.parse_args()

    edges = np.loadtxt(args.graph, dtype=np.int64, comments="#", ndmin=2)
    src, dst = edges[:, 0], edges[:, 1]
    n = int(edges.max()) + 1
    keys = np.unique(dst * n + src)
    indeg = np.bincount(keys // n, minlength=n).astype(np.int64)
    m = int(indeg.sum())
    print(f"n = {n}  edges = {m} (deduplicated)")
    for sigma in args.sigma:
        s = n if sigma == 0 else sigma
        label = "n (full sort)" if sigma == 0 else str(sigma)
        if s == SELL_C:
            label += " (no sort)"
        st = slots(indeg, s)
        print(f"sigma = {label:>16}  slots = {st:>12}  slots/edges = {st / m:.3f}")


if __name__ == "__main__":
    main()
