// driver.h — command-line driver shared by every PageRank variant.
//
// Each src/<variant>.cpp defines only its PageRank function and calls
// run_variant() from main(). That keeps the CLI, timing and output format
// identical across variants, so the test harness can compare their ranks
// and the benchmarks can compare their times.
//
// Usage:  bin/<variant> <graph-file> [options]
//   -o FILE         write one "vertex rank" line per vertex to FILE
//   -t N            number of OpenMP threads (default: OMP_NUM_THREADS / all cores)
//   -r N            run PageRank N times and report min / median time (default 1)
//   -i N            maximum iterations (default 1000)
//   -e EPS          L1 convergence threshold (default 1e-10); 0 = always run
//                   exactly -i iterations (fixed-work benchmarking)
//   -d D            damping factor (default 0.85)
//   -p K            partitions per thread, K >= 1 (default 32); used by the
//                   dynamic and combined variants, the others ignore it
#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <omp.h>

#include "common.h"

// Algorithm parameters, passed to every variant.
struct PRParams {
    double d = 0.85;       // damping factor (NetworkX's default alpha)
    double eps = 1e-10;    // stop when the L1 change between iterations < eps;
                           // far below the 1e-6 tolerance of the NetworkX check
    int max_iters = 1000;  // safety cap; PageRank's error shrinks by about d
                           // per iteration, so 1e-10 needs ~150 iterations
    int partitions = 32;   // partitions per thread (dynamic, combined); chosen
                           // by a K = 1..64 sweep at 16 threads on an i7-14700:
                           // flat from K = 16 on soc-LiveJournal1, best or
                           // near-best at 32 on email-Enron
};

// Result of one PageRank run.
struct PRResult {
    std::vector<double> rank;  // final PageRank of each vertex; sums to 1.0
    int iterations = 0;        // iterations performed, including the converging one
    bool converged = false;    // false = stopped by max_iters, not by eps
    double seconds = 0;        // time spent in the iteration loop only (no loading)
};

using PageRankFn = PRResult (*)(const CSRGraph&, const PRParams&);

namespace driver_detail {

[[noreturn]] inline void usage(const char* prog) {
    std::cerr << "usage: " << prog << " <graph-file> [-o ranks.txt] [-t threads]"
                 " [-r repeats] [-i max-iters] [-e eps] [-d damping]"
                 " [-p partitions]\n";
    std::exit(1);
}

// Parses a whole-string number; anything else is a usage error.
inline long parse_long(const char* prog, const char* s, long lo) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < lo) usage(prog);
    return v;
}

inline double parse_double(const char* prog, const char* s) {
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || *end != '\0') usage(prog);
    return v;
}

// Prints the k highest-ranked vertices, highest first.
inline void print_top(const std::vector<double>& rank, size_t k) {
    std::vector<vid_t> ids(rank.size());
    std::iota(ids.begin(), ids.end(), 0);

    // Only the first k need to be in order, so a partial sort is enough.
    k = std::min(k, ids.size());
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                      [&](vid_t a, vid_t b) { return rank[a] > rank[b]; });

    std::cout << "top " << k << " vertices:\n";
    std::cout << "  vertex  rank\n";
    for (size_t i = 0; i < k; ++i)
        std::cout << "  " << std::setw(6) << ids[i] << "  " << std::setprecision(6)
                  << rank[ids[i]] << "\n";
}

// Writes one "vertex rank" line per vertex, for the NetworkX check.
// 17 significant digits round-trip a double exactly, so printing never
// limits the accuracy of the comparison.
inline void write_ranks(const std::string& path, const std::vector<double>& rank) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "error: cannot open output file: " << path << "\n";
        std::exit(1);
    }
    out << std::setprecision(17);
    for (size_t v = 0; v < rank.size(); ++v) out << v << " " << rank[v] << "\n";
}

}  // namespace driver_detail

// Parses arguments, loads the graph, runs `fn` (repeats times), and reports.
// `parallel` = false marks a single-threaded variant: -t is accepted (so
// scripts can pass it to every variant) but threads is reported as 1.
inline int run_variant(int argc, char** argv, const char* name, PageRankFn fn,
                       bool parallel = true) {
    using namespace driver_detail;
    const char* prog = argv[0];

    std::string graph_path, out_path;
    PRParams params;
    int repeats = 1;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage(prog);
            return argv[++i];
        };
        if      (!std::strcmp(a, "-o")) out_path = next();
        else if (!std::strcmp(a, "-t")) omp_set_num_threads(static_cast<int>(parse_long(prog, next(), 1)));
        else if (!std::strcmp(a, "-r")) repeats = static_cast<int>(parse_long(prog, next(), 1));
        else if (!std::strcmp(a, "-i")) params.max_iters = static_cast<int>(parse_long(prog, next(), 1));
        else if (!std::strcmp(a, "-e")) params.eps = parse_double(prog, next());
        else if (!std::strcmp(a, "-d")) params.d = parse_double(prog, next());
        else if (!std::strcmp(a, "-p")) params.partitions = static_cast<int>(parse_long(prog, next(), 1));
        else if (a[0] == '-' && a[1] != '\0') usage(prog);
        else if (graph_path.empty()) graph_path = a;
        else usage(prog);
    }
    if (graph_path.empty()) usage(prog);
    if (!(params.d >= 0.0 && params.d < 1.0) || !(params.eps >= 0.0)) {
        std::cerr << "error: need 0 <= damping < 1 and eps >= 0\n";
        return 1;
    }

    // Load the graph (not timed: loading isn't part of the algorithm).
    CSRGraph g;
    try {
        g = load_graph(graph_path);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    if (g.n == 0) {
        std::cerr << "error: graph has no vertices: " << graph_path << "\n";
        return 1;
    }

    // Run PageRank. With -r, every run must give the same iteration count;
    // the last run's ranks are reported.
    // Start the OpenMP thread pool now: the first parallel region creates the
    // threads (milliseconds), which would otherwise be timed as part of the
    // first PageRank iteration.
    #pragma omp parallel
    { volatile int warm = omp_get_thread_num(); (void)warm; }

    std::vector<double> times;
    PRResult r;
    for (int k = 0; k < repeats; ++k) {
        r = fn(g, params);
        times.push_back(r.seconds);
    }
    std::sort(times.begin(), times.end());
    const double t_min = times.front();
    const double t_med = times[times.size() / 2];

    // Report. The rank sum should be 1.0 (up to rounding): the cheapest
    // check that no rank was lost, e.g. from forgetting dangling vertices.
    const double rank_sum = std::accumulate(r.rank.begin(), r.rank.end(), 0.0);
    std::cout << "variant:     " << name << "\n";
    std::cout << "graph:       " << graph_path << "\n";
    std::cout << "vertices:    " << g.n << "\n";
    std::cout << "edges:       " << g.m << "\n";
    std::cout << "dangling:    " << count_dangling(g) << "\n";
    std::cout << "threads:     " << (parallel ? omp_get_max_threads() : 1) << "\n";
    std::cout << "iterations:  " << r.iterations
              << (r.converged ? ""
                  : params.eps > 0.0 ? " (NOT converged: hit max iterations)"
                                     : " (fixed count, -e 0)") << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "time (s):    " << t_med << "\n";  // median of the repeats
    if (repeats > 1) {
        std::cout << "time min:    " << t_min << "  (" << repeats << " runs)\n";
        std::cout << "time max:    " << times.back() << "\n";
    }
    std::cout << "time/iter:   " << t_med / r.iterations << "\n";
    std::cout << "iter min/max:" << " " << t_min / r.iterations << " "
              << times.back() / r.iterations << "\n";
    std::cout << "rank sum:    " << std::setprecision(12) << rank_sum << "\n";
    std::cout.unsetf(std::ios::floatfield);
    print_top(r.rank, 10);

    if (!r.converged && params.eps > 0.0)
        std::cerr << "warning: not converged after " << r.iterations
                  << " iterations (raise -i or -e)\n";

    // Save every rank to a file if -o was given.
    if (!out_path.empty()) {
        write_ranks(out_path, r.rank);
        std::cout << "ranks written to " << out_path << "\n";
    }
    return 0;
}
