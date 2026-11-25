// pagerank_static.cpp — OpenMP PageRank with static scheduling (Stage 1).
//
// Identical to pagerank_serial.cpp except that every per-vertex loop is
// split across threads with `schedule(static)`: thread t gets one
// contiguous block of n / T vertices. Nothing else changes, so the speedup
// over serial measures naive parallelism alone.
//
// Static blocks contain equal numbers of *vertices*, not equal numbers of
// *edges*. On power-law graphs a few vertices own most of the in-edges, so
// the SPMV loop (d) leaves some threads with far more work than others.
// That load imbalance is what Stage 2 (dynamic over-partitioning) targets,
// so it is measured here, not fixed.
//
// Races: none. Loops (b) and (d) write only their own index (the pull /
// Aᵀ layout is what makes (d) race-free). Loops (a) and (e) are sums, done
// with OpenMP reductions. A reduction adds per-thread partial sums in a
// different order than the serial loop, so ranks differ from serial in the
// last bits; the harness compares within a tolerance.
//
// Usage:  bin/pagerank_static <graph-file> [-t threads] [-o ranks.txt] [options]
//         (options are listed in driver.h)

#include <cmath>
#include <vector>

#include <omp.h>

#include "driver.h"

using namespace std;

PRResult pageRank_static(const CSRGraph& g, const PRParams& p)
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

        // (d) The SPMV, split into equal vertex blocks per thread.
        #pragma omp parallel for schedule(static)
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
    return run_variant(argc, argv, "static", pageRank_static);
}
