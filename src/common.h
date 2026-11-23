// common.h — graph loading and CSR construction shared by every PageRank variant.
//
// Header-only: include it from any .cpp file; all functions are `inline`.
//
// The graph is stored as the CSR of the *transposed* adjacency matrix (Aᵀ),
// i.e. for each vertex v we store the vertices u with an edge u -> v.
// PageRank then "pulls" from in-neighbors, so each vertex writes only its
// own entry and the outer loop parallelizes without races.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using vid_t = int32_t;  // vertex id (fits soc-LiveJournal1's ~4.8M vertices)
using eid_t = int64_t;  // edge offset (64-bit so row_ptr never overflows)

// Raw directed edges in file order.
struct EdgeList {
    vid_t n = 0;              // number of vertices = max id + 1
    std::vector<vid_t> src;   // src[i] -> dst[i]
    std::vector<vid_t> dst;
};

// CSR of Aᵀ: in-neighbors of v are col_idx[row_ptr[v] .. row_ptr[v+1]).
struct CSRGraph {
    vid_t n = 0;                  // number of vertices
    eid_t m = 0;                  // number of directed edges
    std::vector<eid_t> row_ptr;   // size n + 1
    std::vector<vid_t> col_idx;   // size m, in-neighbor ids
    std::vector<vid_t> out_deg;   // size n, out-degree of each vertex (0 = dangling)
    eid_t duplicates_removed = 0; // repeated u -> v edges dropped while building

    eid_t in_degree(vid_t v) const { return row_ptr[v + 1] - row_ptr[v]; }
};

// Reads a SNAP-style edge list: one "u v" pair per line (tabs or spaces),
// lines starting with '#' are comments, blank lines are skipped, and
// Windows "\r\n" line endings are accepted. Vertex ids must be non-negative;
// they are used directly as indices (no remapping), so n = max id + 1.
inline EdgeList load_edge_list(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open graph file: " + path);

    EdgeList el;
    std::string line;
    long long line_no = 0;
    long long max_id = -1;

    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        const char* p = line.c_str();
        char* end = nullptr;
        long long u = std::strtoll(p, &end, 10);
        if (end == p) throw std::runtime_error(path + ":" + std::to_string(line_no) + ": bad source id");
        p = end;
        long long v = std::strtoll(p, &end, 10);
        if (end == p) throw std::runtime_error(path + ":" + std::to_string(line_no) + ": bad target id");
        if (u < 0 || v < 0 || u > INT32_MAX - 1 || v > INT32_MAX - 1)
            throw std::runtime_error(path + ":" + std::to_string(line_no) + ": vertex id out of range");

        el.src.push_back(static_cast<vid_t>(u));
        el.dst.push_back(static_cast<vid_t>(v));
        if (u > max_id) max_id = u;
        if (v > max_id) max_id = v;
    }

    el.n = static_cast<vid_t>(max_id + 1);
    return el;
}

// Builds the CSR of Aᵀ with a counting sort over destinations, then sorts
// each row's in-neighbors ascending and removes duplicate edges.
//
// Duplicates are removed so the graph matches NetworkX's DiGraph (which keeps
// one copy of each u -> v); otherwise a repeated edge would count twice in
// both the rank contribution and the out-degree. RMAT generators produce many
// duplicates. Self-loops (u -> u) are kept, as NetworkX keeps them too.
inline CSRGraph build_csr_transposed(const EdgeList& el) {
    CSRGraph g;
    g.n = el.n;
    const eid_t m_raw = static_cast<eid_t>(el.src.size());
    g.row_ptr.assign(static_cast<size_t>(g.n) + 1, 0);
    g.col_idx.resize(static_cast<size_t>(m_raw));

    // Count in-degrees (shifted by one for the prefix sum).
    for (eid_t i = 0; i < m_raw; ++i) ++g.row_ptr[el.dst[i] + 1];
    // Exclusive prefix sum: row_ptr[v] = start of v's in-neighbor list.
    for (vid_t v = 0; v < g.n; ++v) g.row_ptr[v + 1] += g.row_ptr[v];

    // Scatter each edge into its destination row.
    std::vector<eid_t> next(g.row_ptr.begin(), g.row_ptr.end() - 1);
    for (eid_t i = 0; i < m_raw; ++i) g.col_idx[next[el.dst[i]]++] = el.src[i];

    // Sort each row, drop repeats, and compact col_idx in place.
    eid_t write = 0;
    eid_t row_start = g.row_ptr[0];
    for (vid_t v = 0; v < g.n; ++v) {
        const eid_t row_end = g.row_ptr[v + 1];
        auto first = g.col_idx.begin() + row_start;
        auto last = g.col_idx.begin() + row_end;
        std::sort(first, last);
        auto uniq_end = std::unique(first, last);
        g.row_ptr[v] = write;
        for (auto it = first; it != uniq_end; ++it) g.col_idx[write++] = *it;
        row_start = row_end;
    }
    g.row_ptr[g.n] = write;
    g.col_idx.resize(static_cast<size_t>(write));
    g.col_idx.shrink_to_fit();
    g.m = write;
    g.duplicates_removed = m_raw - write;

    // Out-degrees from the deduplicated edges.
    g.out_deg.assign(static_cast<size_t>(g.n), 0);
    for (vid_t u : g.col_idx) ++g.out_deg[u];

    return g;
}

// Convenience: load a file straight into the transposed CSR.
inline CSRGraph load_graph(const std::string& path) {
    return build_csr_transposed(load_edge_list(path));
}

// Number of vertices with no out-edges (their rank must be redistributed).
inline vid_t count_dangling(const CSRGraph& g) {
    vid_t c = 0;
    for (vid_t v = 0; v < g.n; ++v) c += (g.out_deg[v] == 0);
    return c;
}
