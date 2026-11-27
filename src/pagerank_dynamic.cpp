// pagerank_dynamic.cpp — OpenMP PageRank with dynamic over-partitioning (Stage 2).
//
// Identical to pagerank_static.cpp except for the SPMV loop (d), which uses
// `schedule(dynamic, chunk)` with chunk = ceil(n / (K * T)): the rows are
// split into P = K * T equal-vertex partitions (K = -p, T = threads), many
// more than threads, and handed out at run time. This is GraphMat's load
// balancing: on power-law graphs a few partitions hold most of the in-edges,
// and a thread that draws a heavy one no longer holds the others up, because
// they keep taking the remaining partitions. Loops (a), (b) and (e) do the
// same work for every vertex, so they stay `schedule(static)`.
//
// Races: none, for the same reason as static. A partition is a contiguous
// range of rows handed to exactly one thread, and loop (d) writes only xn[v]
// for its own row v. Reading the shared contrib array is read-only here.
// Which thread computes a row changes nothing about its value (each row's sum
// is in a fixed order), so only the reductions in (a) and (e) can differ from
// serial in the last bits.
//
// Usage:  bin/pagerank_dynamic <graph-file> [-t threads] [-p partitions] [-o ranks.txt] [options]
//         (options are listed in driver.h)

#include <cmath>
#include <vector>

#include <omp.h>

#include "driver.h"

using namespace std;

PRResult pageRank_dynamic(const CSRGraph& g, const PRParams& p)
{
    const vid_t n = g.n;
    const double d = p.d;
    PRResult res;

    vector<double> x(n, 1.0 / n);   // ranks from the previous iteration
    vector<double> x_new(n, 0.0);   // ranks being computed this iteration
    vector<double> contrib(n, 0.0); // what each vertex sends along each out-edge

    // Raw pointers keep the loop bodies free of vector bounds/aliasing
    // overhead and are what OpenMP shares between threads. They must be
    // re-read after each swap(x, x_new).
    const eid_t* row_ptr = g.row_ptr.data();
    const vid_t* col_idx = g.col_idx.data();
    const vid_t* out_deg = g.out_deg.data();
    double* c = contrib.data();

    // At least 1: K * T can exceed n on tiny graphs. 64-bit so K * T cannot overflow.
    const int64_t parts = static_cast<int64_t>(p.partitions) * omp_get_max_threads();
    const int chunk = static_cast<int>(max<int64_t>(1, (n + parts - 1) / parts));

    const double t_start = omp_get_wtime();

    for (int it = 0; it < p.max_iters; ++it) {
        const double* xo = x.data();
        double* xn = x_new.data();

        // (a) Rank held by dangling vertices, shared equally below.
        double dangling_sum = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:dangling_sum)
        for (vid_t u = 0; u < n; ++u)
            if (out_deg[u] == 0) dangling_sum += xo[u];

        // (b) SEND_MESSAGE: each vertex splits its rank over its out-edges.
        #pragma omp parallel for schedule(static)
        for (vid_t u = 0; u < n; ++u)
            c[u] = (out_deg[u] > 0) ? xo[u] / out_deg[u] : 0.0;

        // (c) Random-jump share plus dangling share.
        const double base = (1.0 - d) / n + d * dangling_sum / n;

        // (d) The SPMV, over-partitioned and handed out dynamically.
        #pragma omp parallel for schedule(dynamic, chunk)
        for (vid_t v = 0; v < n; ++v) {
            double sum = 0.0;
            for (eid_t k = row_ptr[v]; k < row_ptr[v + 1]; ++k)
                sum += c[col_idx[k]];
            xn[v] = base + d * sum;
        }

        // (e) Convergence: L1 distance between this iteration and the last.
        double diff = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:diff)
        for (vid_t v = 0; v < n; ++v) diff += fabs(xn[v] - xo[v]);

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
    return run_variant(argc, argv, "dynamic", pageRank_dynamic);
}
