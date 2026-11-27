// pagerank_combined.cpp — dynamic over-partitioning plus the AVX2 kernel (Stage 4).
//
// Identical to pagerank_simd.cpp except for the schedule of the SPMV loop (d):
// `schedule(dynamic, chunk)` over the SELL chunks, with
// chunk = ceil(num_chunks / (K * T)) (K = -p, T = threads), the same rule as
// pagerank_dynamic.cpp uses for rows. It combines the two optimisations, so
// comparing it with simd isolates dynamic scheduling on the SIMD kernel, and
// comparing it with dynamic isolates the SIMD kernel under the same schedule.
// Loops (a), (b) and (e) stay `schedule(static)`, as in dynamic.
//
// The layout, the AVX2 kernel (gather, no scatter, why intrinsics rather than
// `#pragma omp simd`) and the padding and scratch element are described in
// pagerank_simd.cpp and sell.h.
//
// Races: none. A partition is a contiguous range of chunks handed to exactly
// one thread, so each row slot of perm is written by one thread, and every
// real vertex appears in exactly one slot. The scratch slot n is written only
// by the last chunk, so by one thread. contrib is read-only in loop (d). Which
// thread computes a chunk does not change its value (each row's sum is in CSR
// order), so only the reductions in (a) and (e) can differ from serial in the
// last bits.
//
// Usage:  bin/pagerank_combined <graph-file> [-t threads] [-p partitions] [-o ranks.txt] [options]
//         (options are listed in driver.h)

#if !defined(__AVX2__)
#error "pagerank_combined.cpp needs AVX2: build with -mavx2 or -march=native on an AVX2 CPU"
#endif

#include <immintrin.h>

#include <cmath>
#include <vector>

#include <omp.h>

#include "driver.h"
#include "sell.h"

using namespace std;

static_assert(SELL_C == 4, "the kernel in loop (d) hard-codes four lanes (__m256d)");

PRResult pageRank_combined(const CSRGraph& g, const PRParams& p)
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

    // At least 1: K * T can exceed num_chunks on tiny graphs. 64-bit so K * T cannot overflow.
    const int64_t parts = static_cast<int64_t>(p.partitions) * omp_get_max_threads();
    const int chunk = static_cast<int>(max<int64_t>(1, (num_chunks + parts - 1) / parts));

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
        for (vid_t k = 0; k < num_chunks; ++k) {
            __m256d acc = _mm256_setzero_pd();
            const vid_t* idx = col_idx + chunk_ptr[k];
            const vid_t width = s.width(k);
            for (vid_t j = 0; j < width; ++j) {
                const __m128i lanes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(idx + SELL_C * j));
                acc = _mm256_add_pd(acc, _mm256_i32gather_pd(c, lanes, 8));
            }
            double out[SELL_C];
            _mm256_storeu_pd(out, _mm256_add_pd(_mm256_set1_pd(base),
                                               _mm256_mul_pd(_mm256_set1_pd(d), acc)));
            const vid_t* slot = perm + static_cast<size_t>(k) * SELL_C;
            for (int lane = 0; lane < SELL_C; ++lane) xn[slot[lane]] = out[lane];
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
    return run_variant(argc, argv, "combined", pageRank_combined);
}
