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
// Usage:  bin/pagerank_serial <graph-file> [-o ranks.txt] [options]
//         (options are listed in driver.h)

#include <cmath>
#include <vector>

#include <omp.h>  // only for omp_get_wtime(); this variant runs on one thread

#include "driver.h"

using namespace std;

// Runs PageRank on g until the L1 change between iterations drops below
// p.eps, or p.max_iters is reached (see PRParams in driver.h).
//
// Ranks are doubles: with tens of thousands of vertices each rank is ~1e-5,
// and float cannot converge to a tolerance this tight.
PRResult pageRank_serial(const CSRGraph& g, const PRParams& p)
{
    const vid_t n = g.n;
    const double d = p.d;
    PRResult res;

    // Start from the uniform distribution. Note 1.0, not 1: 1 / n would be
    // integer division and set every rank to 0.
    vector<double> x(n, 1.0 / n);   // ranks from the previous iteration
    vector<double> x_new(n, 0.0);   // ranks being computed this iteration
    vector<double> contrib(n, 0.0); // what each vertex sends along each out-edge

    // Raw pointers, exactly as in the parallel variants: with vector
    // indexing the compiler must reload g.row_ptr / g.col_idx data pointers
    // around stores, which made serial ~5% slower than the 1-thread static
    // variant and inflated every speedup. Re-read xo / xn after each swap.
    const eid_t* row_ptr = g.row_ptr.data();
    const vid_t* col_idx = g.col_idx.data();
    const vid_t* out_deg = g.out_deg.data();
    double* c = contrib.data();

    const double t_start = omp_get_wtime();

    for (int it = 0; it < p.max_iters; ++it) {
        const double* xo = x.data();
        double* xn = x_new.data();

        // (a) Dangling vertices have no out-edges, so their rank would leak
        //     out of the system. Collect it here and share it equally among
        //     all vertices below (NetworkX does the same).
        double dangling_sum = 0.0;
        for (vid_t u = 0; u < n; ++u)
            if (out_deg[u] == 0) dangling_sum += xo[u];

        // (b) SEND_MESSAGE: each vertex splits its rank evenly over its
        //     out-edges. Dividing here costs one division per vertex instead
        //     of one per edge inside the inner loop below.
        for (vid_t u = 0; u < n; ++u)
            c[u] = (out_deg[u] > 0) ? xo[u] / out_deg[u] : 0.0;

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
            for (eid_t k = row_ptr[v]; k < row_ptr[v + 1]; ++k)
                sum += c[col_idx[k]];
            // APPLY: combine the received rank with the base share.
            xn[v] = base + d * sum;
        }

        // (e) Convergence: L1 distance between this iteration and the last.
        double diff = 0.0;
        for (vid_t v = 0; v < n; ++v) diff += fabs(xn[v] - xo[v]);

        // Swap instead of copying: x now holds the newest ranks.
        swap(x, x_new);
        res.iterations = it + 1;
        if (diff < p.eps) {
            res.converged = true;
            break;
        }
    }

    res.seconds = omp_get_wtime() - t_start;
    res.rank = move(x);
    return res;
}

int main(int argc, char** argv)
{
    return run_variant(argc, argv, "serial", pageRank_serial,
                       /*parallel=*/false);
}
