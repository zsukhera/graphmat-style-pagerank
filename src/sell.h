// sell.h — sliced ELLPACK (SELL-C-sigma) layout of the CSR, for SIMD-friendly SPMV.
//
// Header-only, like common.h. Include it after common.h.
//
// In CSR the SPMV inner loop `sum += c[col_idx[k]]` is a floating-point
// reduction along one row, which the compiler may not reorder, and most rows
// are short, so it does not vectorize. Here each SIMD lane handles a
// *different* row: step j adds the j-th in-neighbor of SELL_C rows at once.
// That needs rows of equal length, so rows are padded. Padding every row to
// the longest row of the matrix is impossible on power-law graphs, so rows
// are sorted by length inside windows of sigma rows, cut into chunks of
// SELL_C rows, and each chunk is padded only to its own longest row
// (Kreutzer et al., SIAM J. Sci. Comput. 2014).
#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

#include "common.h"

constexpr int SELL_C = 4;  // rows per chunk = doubles in one AVX2 register

// sigma = 16384: on rmat-20 it stores 17,280,772 slots, 1.07x the edge count,
// versus 3.60x with no sorting. A full sort (sigma = n) would pad slightly
// less (1.01x) but would put every hub row in the first chunks, so an equal
// split of chunks between threads would give thread 0 all the heavy rows and
// the comparison with the CSR static variant would no longer be like for like.
constexpr vid_t SELL_SIGMA = 16384;

// Sliced ELLPACK of the same matrix as a CSRGraph (row v = in-neighbors of v).
struct SELLGraph {
    vid_t n = 0;                  // number of vertices (rows)
    vid_t num_chunks = 0;         // ceil(n / SELL_C)
    std::vector<eid_t> chunk_ptr; // size num_chunks + 1, start of each chunk in col_idx
    std::vector<vid_t> col_idx;   // entry (j, lane) of chunk c is at
                                  // chunk_ptr[c] + SELL_C * j + lane; padding = n
    std::vector<vid_t> perm;      // size SELL_C * num_chunks: vertex whose rank each
                                  // row slot computes; n for the empty lanes of a
                                  // last partial chunk

    // Chunk width = number of j steps = longest row in the chunk.
    vid_t width(vid_t c) const {
        return static_cast<vid_t>((chunk_ptr[c + 1] - chunk_ptr[c]) / SELL_C);
    }
};

// Row entries keep CSR order (ascending id), so each row's sum is added in the
// same order as in the serial code. Padding entries hold the index n: the
// caller keeps contrib[n] = 0.0, so the kernel needs no masks or bounds checks.
// sigma must be a multiple of SELL_C.
inline SELLGraph build_sell(const CSRGraph& g, vid_t sigma) {
    SELLGraph s;
    s.n = g.n;
    s.num_chunks = (g.n + SELL_C - 1) / SELL_C;
    s.perm.assign(static_cast<size_t>(s.num_chunks) * SELL_C, g.n);

    // Order the rows: descending in-degree within each window of sigma ids.
    for (vid_t w = 0; w < g.n; w += sigma) {
        const vid_t end = std::min(g.n, w + sigma);
        vid_t* first = s.perm.data() + w;
        std::iota(first, first + (end - w), w);
        std::stable_sort(first, first + (end - w), [&](vid_t a, vid_t b) {
            return g.in_degree(a) > g.in_degree(b);
        });
    }

    s.chunk_ptr.assign(static_cast<size_t>(s.num_chunks) + 1, 0);
    for (vid_t c = 0; c < s.num_chunks; ++c) {
        eid_t width = 0;
        for (int lane = 0; lane < SELL_C; ++lane) {
            const vid_t v = s.perm[static_cast<size_t>(c) * SELL_C + lane];
            if (v < g.n) width = std::max(width, g.in_degree(v));
        }
        s.chunk_ptr[c + 1] = s.chunk_ptr[c] + width * SELL_C;
    }

    s.col_idx.assign(static_cast<size_t>(s.chunk_ptr[s.num_chunks]), g.n);
    for (vid_t c = 0; c < s.num_chunks; ++c) {
        for (int lane = 0; lane < SELL_C; ++lane) {
            const vid_t v = s.perm[static_cast<size_t>(c) * SELL_C + lane];
            if (v >= g.n) continue;
            const eid_t deg = g.in_degree(v);
            for (eid_t j = 0; j < deg; ++j)
                s.col_idx[s.chunk_ptr[c] + SELL_C * j + lane] = g.col_idx[g.row_ptr[v] + j];
        }
    }
    return s;
}
