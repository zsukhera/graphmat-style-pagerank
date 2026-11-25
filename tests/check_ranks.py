#!/usr/bin/env python3
"""Checks a PageRank ranks file against a reference.

Two modes:

    check_ranks.py --graph G.txt  --ranks R.txt   compare R to NetworkX's
                                                   pagerank of graph G
    check_ranks.py --ref REF.txt  --ranks R.txt   compare R to another ranks
                                                   file (e.g. the serial output)

A ranks file has one "vertex rank" line per vertex, as written by
`bin/<variant> <graph> -o R.txt`.

The pass criterion is an L1 distance below --tol (default 1e-6, from the
README). Parallel variants sum in a different order than the serial one,
so results are compared within a tolerance, never bit-for-bit.

Exit status: 0 = pass, 1 = fail, 2 = bad input.
"""

import argparse
import math
import sys


def read_ranks(path):
    """Reads a ranks file into a list indexed by vertex id."""
    ranks = {}
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                sys.exit(f"error: {path}:{line_no}: expected 'vertex rank'")
            v, r = int(parts[0]), float(parts[1])
            if v in ranks:
                sys.exit(f"error: {path}:{line_no}: vertex {v} listed twice")
            ranks[v] = r

    # Vertices must be exactly 0..n-1, as the C++ code numbers them.
    n = len(ranks)
    if set(ranks) != set(range(n)):
        sys.exit(f"error: {path}: vertex ids are not exactly 0..{n - 1}")
    return [ranks[v] for v in range(n)]


def networkx_ranks(graph_path):
    """Computes the reference ranks for a SNAP edge list with NetworkX."""
    import networkx as nx

    # DiGraph keeps one copy of each u -> v and keeps self-loops, which is
    # exactly what common.h's CSR builder does.
    g = nx.read_edgelist(graph_path, create_using=nx.DiGraph,
                         nodetype=int, comments="#")

    # The C++ loader sets n = max id + 1, so ids that never appear in an
    # edge are still vertices (isolated and dangling). Add them here too.
    n = max(g.nodes) + 1 if g.number_of_nodes() else 0
    g.add_nodes_from(range(n))

    # Converge far tighter than the tolerance so NetworkX's own error
    # doesn't count against the variant being checked.
    pr = nx.pagerank(g, alpha=0.85, tol=1e-13, max_iter=10000)
    return [pr[v] for v in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--graph", help="graph file: compare against NetworkX")
    src.add_argument("--ref", help="ranks file: compare against it")
    ap.add_argument("--ranks", required=True, help="ranks file to check")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max allowed L1 distance (default 1e-6)")
    args = ap.parse_args()

    try:
        got = read_ranks(args.ranks)
        want = networkx_ranks(args.graph) if args.graph else read_ranks(args.ref)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if len(got) != len(want):
        print(f"FAIL  vertex count {len(got)}, expected {len(want)}")
        return 1

    # Any NaN or inf means the run broke (e.g. 0/0 on a dangling vertex).
    if not all(math.isfinite(r) for r in got):
        print("FAIL  ranks contain NaN or inf")
        return 1

    diffs = [abs(a - b) for a, b in zip(got, want)]
    l1 = math.fsum(diffs)
    worst = max(range(len(diffs)), key=diffs.__getitem__) if diffs else 0
    total = math.fsum(got)

    ok = l1 < args.tol
    print(f"{'PASS' if ok else 'FAIL'}  L1 = {l1:.3e} (tol {args.tol:.0e})"
          f"  max |diff| = {max(diffs, default=0):.3e} at vertex {worst}"
          f"  rank sum = {total:.12f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
