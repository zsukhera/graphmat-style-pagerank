// pagerank_serial.cpp — serial PageRank baseline (Stage 0).
//
// Every parallel variant is checked against this file's output and timed
// against its runtime (speedup = serial time / parallel time), so it is kept
// deliberately simple: one thread, no SIMD, no tricks.
//
// PageRank is written the way GraphMat expresses a vertex program: as a
// generalized sparse matrix-vector multiply (SPMV) over the transposed
// adjacency matrix Aᵀ. One iteration is:
//
//     x_new[v] = base + d * Σ_{u -> v} x[u] / outdeg(u)
//     base     = (1 - d) / n  +  d * (rank held by dangling vertices) / n
//
// with the GraphMat steps mapped as:
//     SEND_MESSAGE     contrib[u] = x[u] / outdeg(u)
//     PROCESS_MESSAGE  read contrib[u] for each in-neighbor u of v
//     REDUCE           sum those contributions
//     APPLY            x_new[v] = base + d * sum
//
// Usage:  bin/pagerank_serial <graph-file> [-o ranks.txt]

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <omp.h>  // only for omp_get_wtime(); this variant runs on one thread

#include "common.h"

using namespace std;

// Result of one PageRank run.
struct PRResult {
    vector<double> rank;  // final PageRank of each vertex; sums to 1.0
    int iterations = 0;   // iterations performed, including the converging one
    double seconds = 0;   // time spent in the iteration loop only (no loading)
};

// Runs PageRank on g until the L1 change between iterations drops below eps,
// or max_iters is reached.
//
//   d          damping factor: probability of following a link instead of
//              jumping to a random vertex (0.85 matches NetworkX's default)
//   eps        convergence threshold on the L1 distance between iterations;
//              kept far below the 1e-6 tolerance of the NetworkX check
//   max_iters  safety cap in case the ranks never converge
//
// Ranks are doubles: with tens of thousands of vertices each rank is ~1e-5,
// and float cannot converge to a tolerance this tight.
PRResult pageRank_serial(const CSRGraph& g, double d = 0.85,
                         double eps = 1e-10, int max_iters = 100)
{
    const vid_t n = g.n;
    PRResult res;

    // Start from the uniform distribution. Note 1.0, not 1: 1 / n would be
    // integer division and set every rank to 0.
    vector<double> x(n, 1.0 / n);   // ranks from the previous iteration
    vector<double> x_new(n, 0.0);   // ranks being computed this iteration
    vector<double> contrib(n, 0.0); // what each vertex sends along each out-edge

    const double t_start = omp_get_wtime();

    for (int it = 0; it < max_iters; ++it) {
        // (a) Dangling vertices have no out-edges, so their rank would leak
        //     out of the system. Collect it here and share it equally among
        //     all vertices below (NetworkX does the same).
        double dangling_sum = 0.0;
        for (vid_t u = 0; u < n; ++u)
            if (g.out_deg[u] == 0) dangling_sum += x[u];

        // (b) SEND_MESSAGE: each vertex splits its rank evenly over its
        //     out-edges. Dividing here costs one division per vertex instead
        //     of one per edge inside the inner loop below.
        for (vid_t u = 0; u < n; ++u)
            contrib[u] = (g.out_deg[u] > 0) ? x[u] / g.out_deg[u] : 0.0;

        // (c) Rank every vertex receives regardless of its in-edges: the
        //     random-jump share plus its share of the dangling rank.
        const double base = (1.0 - d) / n + d * dangling_sum / n;

        // (d) The SPMV. Each vertex pulls from its in-neighbors (row v of
        //     Aᵀ) and writes only x_new[v], so iterations of the outer loop
        //     are independent; Stages 1-2 parallelize it. The inner loop is
        //     one indexed read and one add; Stage 3 vectorizes it, so keep
        //     it that simple.
        for (vid_t v = 0; v < n; ++v) {
            double sum = 0.0;
            // PROCESS_MESSAGE + REDUCE: add up what every in-neighbor sent.
            for (eid_t k = g.row_ptr[v]; k < g.row_ptr[v + 1]; ++k)
                sum += contrib[g.col_idx[k]];
            // APPLY: combine the received rank with the base share.
            x_new[v] = base + d * sum;
        }

        // (e) Convergence: L1 distance between this iteration and the last.
        double diff = 0.0;
        for (vid_t v = 0; v < n; ++v) diff += fabs(x_new[v] - x[v]);

        // Swap instead of copying: x now holds the newest ranks.
        swap(x, x_new);
        res.iterations = it + 1;
        if (diff < eps) break;
    }

    res.seconds = omp_get_wtime() - t_start;
    res.rank = move(x);
    return res;
}

// Prints the k highest-ranked vertices, highest first.
static void print_top(const vector<double>& rank, size_t k)
{
    // List of all vertex ids, 0..n-1.
    vector<vid_t> ids(rank.size());
    iota(ids.begin(), ids.end(), 0);

    // Only the first k need to be in order, so a partial sort is enough.
    k = min(k, ids.size());
    partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                 [&](vid_t a, vid_t b) { return rank[a] > rank[b]; });

    cout << "top " << k << " vertices:\n";
    cout << "  vertex  rank\n";
    for (size_t i = 0; i < k; ++i)
        cout << "  " << setw(6) << ids[i] << "  " << setprecision(6)
             << rank[ids[i]] << "\n";
}

// Writes one "vertex rank" line per vertex, for the NetworkX check.
// 17 significant digits round-trip a double exactly, so printing never
// limits the accuracy of the comparison.
static void write_ranks(const string& path, const vector<double>& rank)
{
    ofstream out(path);
    if (!out) {
        cerr << "error: cannot open output file: " << path << "\n";
        exit(1);
    }
    out << setprecision(17);
    for (size_t v = 0; v < rank.size(); ++v) out << v << " " << rank[v] << "\n";
}

int main(int argc, char** argv)
{
    // Parse arguments: <graph-file> [-o ranks.txt]
    if (argc < 2) {
        cerr << "usage: " << argv[0] << " <graph-file> [-o ranks.txt]\n";
        return 1;
    }
    const string graph_path = argv[1];
    string out_path;
    if (argc == 4 && strcmp(argv[2], "-o") == 0) {
        out_path = argv[3];
    } else if (argc != 2) {
        cerr << "usage: " << argv[0] << " <graph-file> [-o ranks.txt]\n";
        return 1;
    }

    // Load the graph (not timed: loading isn't part of the algorithm).
    CSRGraph g;
    try {
        g = load_graph(graph_path);
    } catch (const exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    if (g.n == 0) {
        cerr << "error: graph has no vertices: " << graph_path << "\n";
        return 1;
    }

    // Run PageRank.
    PRResult r = pageRank_serial(g);

    // Report. The rank sum should be 1.0 (up to rounding): the cheapest
    // check that no rank was lost, e.g. from forgetting dangling vertices.
    const double rank_sum = accumulate(r.rank.begin(), r.rank.end(), 0.0);
    cout << "graph:       " << graph_path << "\n";
    cout << "vertices:    " << g.n << "\n";
    cout << "edges:       " << g.m << "\n";
    cout << "dangling:    " << count_dangling(g) << "\n";
    cout << "iterations:  " << r.iterations << "\n";
    cout << "time (s):    " << fixed << setprecision(6) << r.seconds << "\n";
    cout << "rank sum:    " << setprecision(12) << rank_sum << "\n";
    cout.unsetf(ios::floatfield);
    print_top(r.rank, 10);

    // Save every rank to a file if -o was given.
    if (!out_path.empty()) {
        write_ranks(out_path, r.rank);
        cout << "ranks written to " << out_path << "\n";
    }
    return 0;
}
