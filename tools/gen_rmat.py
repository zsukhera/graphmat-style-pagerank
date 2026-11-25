#!/usr/bin/env python3
"""Generates a Graph500-style RMAT graph as a SNAP edge list.

    gen_rmat.py --scale 20 --edge-factor 16 -o data/rmat-20.txt

RMAT (Chakrabarti et al., 2004) places each edge by recursively choosing
one quadrant of the adjacency matrix with probabilities a, b, c, d. The
Graph500 values (0.57, 0.19, 0.19, 0.05) give a skewed, power-law degree
distribution like real social graphs, which is what exposes load imbalance
in the static OpenMP variant.

As in Graph500, vertex ids are randomly permuted afterwards: without it the
high-degree vertices would all have small ids, which exaggerates static
scheduling's imbalance far beyond what real graphs show.

Output has 2^scale vertex ids and edge_factor * 2^scale edges, including the
duplicates and self-loops RMAT produces (common.h removes duplicates, as
NetworkX's DiGraph does). Deterministic for a given --seed.
"""

import argparse
import sys

import numpy as np


def rmat_edges(scale, m, a, b, c, rng):
    """Returns (src, dst) arrays of m RMAT edges on 2^scale vertices."""
    src = np.zeros(m, dtype=np.int64)
    dst = np.zeros(m, dtype=np.int64)
    ab = a + b
    c_norm = c / (1.0 - ab)  # P(lower-left | lower half)
    a_norm = a / ab          # P(upper-left | upper half)
    for bit in range(scale):
        # One bit of src and dst per level: pick the half, then the side.
        src_bit = rng.random(m) > ab
        dst_bit = rng.random(m) > np.where(src_bit, c_norm, a_norm)
        src |= src_bit.astype(np.int64) << bit
        dst |= dst_bit.astype(np.int64) << bit
    return src, dst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", type=int, required=True, help="2^scale vertices")
    ap.add_argument("--edge-factor", type=int, default=16, help="edges per vertex (default 16)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-a", type=float, default=0.57)
    ap.add_argument("-b", type=float, default=0.19)
    ap.add_argument("-c", type=float, default=0.19)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    if not 1 <= args.scale <= 30:
        sys.exit("error: --scale must be in 1..30 (ids must fit int32)")
    n = 1 << args.scale
    m = args.edge_factor * n
    rng = np.random.default_rng(args.seed)

    src, dst = rmat_edges(args.scale, m, args.a, args.b, args.c, rng)
    perm = rng.permutation(n)
    src, dst = perm[src], perm[dst]

    # Write in chunks: one formatted block per chunk is much faster than
    # np.savetxt, and avoids building one giant string.
    with open(args.output, "w") as f:
        f.write(f"# RMAT scale={args.scale} edge_factor={args.edge_factor} "
                f"a={args.a} b={args.b} c={args.c} seed={args.seed}\n")
        f.write(f"# Nodes: {n} Edges: {m} (before removing duplicates)\n")
        chunk = 1 << 20
        for i in range(0, m, chunk):
            s, d = src[i:i + chunk], dst[i:i + chunk]
            f.write("\n".join(map("{0[0]}\t{0[1]}".format, zip(s.tolist(), d.tolist()))))
            f.write("\n")
    print(f"wrote {args.output}: {n} vertices, {m} edges", file=sys.stderr)


if __name__ == "__main__":
    main()
