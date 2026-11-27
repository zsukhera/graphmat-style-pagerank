#!/usr/bin/env python3
"""Reports how unevenly OpenMP schedule(static) splits SPMV work across threads.

    edge_balance.py data/rmat-20.txt

Reads the graph exactly as common.h does (ids used as-is, n = max id + 1,
repeated u -> v edges counted once, self-loops kept). The SPMV work for
vertex v is its in-degree. For T threads, GCC's schedule(static) gives each
thread one block of consecutive vertices: the first n % T threads get
n // T + 1 vertices, the rest n // T. For each T this prints the edges in the
heaviest block divided by the mean edges per block (1.000 = perfectly even).
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("graph", help="SNAP-style edge list")
    args = ap.parse_args()

    # loadtxt's C parser is ~10x faster than np.fromstring on this file size.
    edges = np.loadtxt(args.graph, dtype=np.int64, comments="#", ndmin=2)
    src, dst = edges[:, 0], edges[:, 1]
    n = int(edges.max()) + 1

    # One int64 key per edge; n < 2^31 so dst * n + src cannot overflow.
    keys = np.unique(dst * n + src)
    in_deg = np.bincount(keys // n, minlength=n)
    m = len(keys)

    print(f"n = {n}  edges = {m} (deduplicated)  max in-degree = {in_deg.max()}")
    prefix = np.concatenate(([0], np.cumsum(in_deg)))
    for T in (2, 4, 8, 16):
        q, r = divmod(n, T)
        starts = np.arange(T + 1) * q + np.minimum(np.arange(T + 1), r)
        block_edges = np.diff(prefix[starts])
        print(f"T = {T:2d}  max/mean = {block_edges.max() / (m / T):.3f}")


if __name__ == "__main__":
    main()
