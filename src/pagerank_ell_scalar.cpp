// pagerank_ell_scalar.cpp — PageRank on the sliced-ELLPACK layout, scalar kernel.
//
// Identical to pagerank_static.cpp except for the SPMV loop (d), which runs
// over chunks of SELL_C = 4 rows (see sell.h) instead of over single rows:
// four accumulators, and step j adds the j-th in-neighbor of each of the four
// rows. The j-outer / lane-inner order is the memory access order of the AVX2
// kernel on this layout, so this variant is the control that separates the
// effect of the layout (padding, permuted rows, different access pattern)
// from the effect of SIMD. The layout is built before the timer starts.
//
// The kernel must stay scalar. At -O3 -march=native GCC may turn the
// four-lane step into gathers by itself, which would silently turn this
// control into the SIMD variant. Check after any change to this loop or to
// the compiler: `objdump -d bin/pagerank_ell_scalar` must show no vgather*
// and no vaddpd on ymm/zmm registers.
//
// Padding and the extra element: padding entries index contrib[n], which is
// kept at 0.0, so the kernel needs no masks. The rank arrays also get one
// scratch element at index n, where the empty lanes of a partial last chunk
// write; the result is cut back to n entries. Loops (a), (b) and (e) run over
// 0..n-1 only, so the extra elements never enter the sums or the result.
//
// Races: none. Each row slot of perm is written by exactly one thread (a
// chunk belongs to one thread), and every real vertex appears in exactly one
// slot. The scratch slot n is written only by the last chunk, so by one
// thread. Each row's sum is in CSR order, as in serial; only the reductions
// in (a) and (e) can differ from serial in the last bits.
//
// Usage:  bin/pagerank_ell_scalar <graph-file> [-t threads] [-o ranks.txt] [options]
//         (options are listed in driver.h)

#include <cmath>
#include <vector>

#include <omp.h>

#include "driver.h"
#include "sell.h"

using namespace std;

PRResult pageRank_ell_scalar(const CSRGraph& g, const PRParams& p)
{
    const vid_t n = g.n;
    const double d = p.d;
    PRResult res;

    const SELLGraph s = build_sell(g, SELL_SIGMA);

    vector<double> x(n + 1, 1.0 / n);   // ranks from the previous iteration
    vector<double> x_new(n + 1, 0.0);   // ranks being computed this iteration
    vector<double> contrib(n + 1, 0.0); // what each vertex sends along each out-edge
    x[n] = 0.0;                         // scratch slot, never read as a rank

    // Raw pointers keep the loop bodies free of vector bounds/aliasing
    // overhead and are what OpenMP shares between threads. They must be
    // re-read after each swap(x, x_new).
    const vid_t* out_deg = g.out_deg.data();
    const eid_t* chunk_ptr = s.chunk_ptr.data();
    const vid_t* col_idx = s.col_idx.data();
    const vid_t* perm = s.perm.data();
    const vid_t num_chunks = s.num_chunks;
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

        // (d) The SPMV, split into equal chunk blocks per thread.
        #pragma omp parallel for schedule(static)
        for (vid_t k = 0; k < num_chunks; ++k) {
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            const vid_t* idx = col_idx + chunk_ptr[k];
            const vid_t width = s.width(k);
            for (vid_t j = 0; j < width; ++j) {
                sum0 += c[idx[SELL_C * j + 0]];
                sum1 += c[idx[SELL_C * j + 1]];
                sum2 += c[idx[SELL_C * j + 2]];
                sum3 += c[idx[SELL_C * j + 3]];
            }
            const vid_t* slot = perm + static_cast<size_t>(k) * SELL_C;
            xn[slot[0]] = base + d * sum0;
            xn[slot[1]] = base + d * sum1;
            xn[slot[2]] = base + d * sum2;
            xn[slot[3]] = base + d * sum3;
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
    x.resize(n);
    res.rank = move(x);
    return res;
}

int main(int argc, char** argv)
{
    return run_variant(argc, argv, "ell_scalar", pageRank_ell_scalar);
}
