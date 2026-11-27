// pagerank_simd.cpp — PageRank on the sliced-ELLPACK layout, AVX2 kernel.
//
// Identical to pagerank_ell_scalar.cpp (same layout, same `schedule(static)`
// over chunks) except for the SPMV loop (d): one __m256d accumulator per
// chunk, one lane per row. Step j loads the four adjacent 32-bit indices,
// gathers the four contributions with vgatherdpd and adds them. Each lane
// adds its row's entries in the same order as the scalar variant, so the two
// agree to the last bits, apart from fused multiply-add in `base + d * sum`.
// Comparing this variant with ell_scalar measures SIMD alone.
//
// AVX2 has no scatter instruction, so the four results are stored to a small
// buffer and written to xn[perm[slot]] with four scalar stores: the rows of a
// chunk are not adjacent vertices.
//
// Intrinsics, not `#pragma omp simd`: the proposal was to try the pragma
// first. On GCC 16.2 (-O3 -march=native) every formulation tried, an
// accumulator array with and without simdlen(4) and a lane-outer loop, left
// the gather scalar (no vgatherdpd in the binary). Intrinsics make the kernel
// the same on every compiler, so the measurement does not depend on whether
// the team's GCC 13 would have vectorized the pragma.
//
// Padding and the extra element: padding entries index contrib[n], which is
// kept at 0.0, so the kernel needs no masks, and every gather index is at
// most n, inside contrib's n + 1 elements. The rank arrays also get one
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
// Usage:  bin/pagerank_simd <graph-file> [-t threads] [-o ranks.txt] [options]
//         (options are listed in driver.h)

#if !defined(__AVX2__)
#error "pagerank_simd.cpp needs AVX2: build with -mavx2 or -march=native on an AVX2 CPU"
#endif

#include <immintrin.h>

#include <cmath>
#include <vector>

#include <omp.h>

#include "driver.h"
#include "sell.h"

using namespace std;

static_assert(SELL_C == 4, "the kernel in loop (d) hard-codes four lanes (__m256d)");

PRResult pageRank_simd(const CSRGraph& g, const PRParams& p)
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
    return run_variant(argc, argv, "simd", pageRank_simd);
}
