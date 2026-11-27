# Implementation plan: what is left

Covers everything between the current repo (Stages 0 and 1 done) and the Milestone 3 report. Proposal: `SecG_B_7/GraphMat_Proposal_Paper7.pdf`.

## Where we are

| Proposal item | State |
|---|---|
| Serial baseline, CSR/CSC SPMV (4.1) | Done: `src/pagerank_serial.cpp` |
| Correctness vs. serial and NetworkX (5.1) | Done: `make test` |
| Stage 1, OpenMP static (4.2) | Done: `src/pagerank_static.cpp`. Benchmarked only on a 4-core / 8-thread laptop under WSL2 |
| Stage 2, OpenMP dynamic, over-partitioned | Not started |
| Stage 3, SIMD | Not started |
| Stage 4, combined | Not started |
| Thread sweep 1–16 (5.2, slide 7) | Not done: highest count run so far is 8 |
| Two real-world graphs (5.2) | One (`email-Enron`). The `soc-LiveJournal1` download failed |
| `perf` profiling (5.3) | Not started; WSL2 does not expose the counters |
| Comparison to the paper's Fig. 5a, report (5.4) | Not started |

## Decisions

**Storage for Stages 0–2: CSR of Aᵀ.** This is what `common.h` already builds. The same three arrays are the CSC of A, so it satisfies the proposal's "CSR (or CSC)" either way. We keep it because each row is one vertex pulling from its in-neighbors, so every thread writes only its own `x_new[v]` and the SPMV loop has no races.

**Storage for Stages 3–4: sliced ELLPACK (SELL-4-σ), built from the CSR.** ELLPACK is what makes SIMD work here: with rows padded to a common width, the four AVX2 lanes each sum a *different row*, so there is no per-row reduction to break up and short rows vectorize as well as long ones. Plain ELLPACK, which pads every row to the longest row in the matrix, cannot be used on these graphs. Measured on `rmat-20` (1,048,576 vertices, 16,087,413 edges, max in-degree 39,547, median in-degree 1):

| Layout | Stored index slots | vs. edges |
|---|---|---|
| Plain ELLPACK | 41,468,035,072 (154.5 GiB of `int32`) | 2577.7× |
| SELL-4, no sorting (σ = 1) | 57,867,336 | 3.60× |
| SELL-4, σ = 1,024 | 22,261,456 | 1.38× |
| **SELL-4, σ = 16,384** | **17,280,772** | **1.07×** |
| SELL-4, σ = n (full sort) | 16,193,344 | 1.01× |

SELL-C-σ (Kreutzer et al., 2014) sorts rows by length inside windows of σ rows, cuts them into chunks of C rows, and pads each chunk only to its own longest row. We use C = 4 (four `double` lanes in AVX2) and σ = 16,384. A full sort pads slightly less, but it would put every hub row at the front, so a static split would hand all of them to thread 0 and the SIMD-static variant would no longer be comparable to the CSR-static one.

**Threads: every variant is run at 1, 2, 4, 8 and 16 threads.** This needs a machine with at least 16 physical cores. On the current laptop (4 cores, 8 hardware threads) a 16-thread run is oversubscribed and says nothing about scaling. The same machine must be native Linux, because `perf` needs the hardware counters.

## Variants after this plan

Scheduling and kernel are varied independently, so each effect can be read off as a ratio at the same thread count.

| Binary | Layout | Kernel | Schedule | Stage |
|---|---|---|---|---|
| `pagerank_serial` | CSR | scalar | none (1 thread) | 0, done |
| `pagerank_static` | CSR | scalar | static | 1, done |
| `pagerank_dynamic` | CSR | scalar | dynamic, P = K·T | 2 |
| `pagerank_ell_scalar` | SELL-4-σ | scalar | static | 3 (control) |
| `pagerank_simd` | SELL-4-σ | AVX2 gather | static | 3 |
| `pagerank_combined` | SELL-4-σ | AVX2 gather | dynamic, P = K·T | 4 |

- Load-balancing gain = `static` ÷ `dynamic`
- Cost or gain of the layout change alone = `static` ÷ `ell_scalar`
- SIMD gain alone = `ell_scalar` ÷ `simd`
- End to end = `serial` ÷ `combined`

Without `pagerank_ell_scalar`, a difference between `static` and `simd` could come from the row sorting and padding instead of from vectorization.

## Steps

Each step ends with `make test` passing before the next one starts.

### 1. Benchmark machine and thread sweep

- Get access to a native Linux machine with ≥ 16 physical cores and AVX2. Record `lscpu` output in the results file.
- `bench/run_bench.sh`: change the default `THREADS` from "powers of two up to `nproc`" to `1 2 4 8 16`, and write a warning into the report header when a thread count exceeds `nproc`.
- Pinning: the script's `OMP_PLACES=threads` comment describes the laptop's sibling numbering. On the new machine check `lscpu -e`; with 16 threads on ≥ 16 cores, `OMP_PLACES=cores OMP_PROC_BIND=spread` puts one thread on each core.
- If the machine has two sockets, the rank vectors are allocated by one thread and land on one NUMA node. Either keep all 16 threads on one socket with `numactl`, or note the effect in the report.
- Re-run Stages 0 and 1 there. These replace the laptop numbers in `results/`.

Verify: `results/benchmarks.txt` shows 16 as the highest thread count and no oversubscription warning.

### 2. Data: second real-world graph and imbalance figures

- Retry `soc-LiveJournal1`. The Makefile rule restarts from zero on every attempt; add `-C -` to its `curl` so a stalled download resumes. If SNAP stays unusable, use another directed SNAP graph with its original vertex ids.
- Finish the edges-per-thread analysis that Stage 1 left open: `tools/edge_balance.py <graph>` prints, for T = 2, 4, 8, 16, the edges in the heaviest equal-vertex block divided by the mean block. That ratio is the most a load balancer can recover over static.
- Already measured for `rmat-20`: 1.008 (T=2), 1.028 (T=4), 1.058 (T=8), 1.070 (T=16). Its ids are randomly permuted, so static blocks are already balanced to within 7% and Stage 2 cannot gain more than that on this graph. The load-balancing question can only be answered on a graph with clustered ids, which is why the real graph is required and not optional.

Verify: the script's edge total for `rmat-20` equals the 16,087,413 the binaries report.

### 3. Stage 2: `src/pagerank_dynamic.cpp`

- Copy of `pagerank_static.cpp` with one change in the SPMV loop (d): `schedule(dynamic, chunk)` with `chunk = ceil(n / (K·T))`. That is equal-vertex over-partitioning into P = K·T blocks, handed out at run time. Loops (a), (b) and (e) do the same work for every vertex and stay static.
- `driver.h`: add `-p K` (partitions per thread) to `PRParams`. The static variants ignore it.
- Sweep K ∈ {1, 2, 4, 8, 16, 32, 64} at 16 threads on the skewed graph, then fix one K for all final runs and state it in the report.

Verify: `make test`; then `static` ÷ `dynamic` at each thread count, reported next to the imbalance ratio from step 2.

### 4. SELL layout: `src/sell.h`

- `build_sell(const CSRGraph&, sigma)` returns: `cols` (`int32`, stored column-major inside each chunk so one SIMD step reads four adjacent indices), `chunk_ptr` (start of each chunk in `cols`), and `perm` (row slot → vertex id).
- Row order inside a window: descending in-degree, stable. Neighbor order inside a row is kept from the CSR, so each row's sum adds in the same order as serial.
- Padding slots hold index `n`, and `contrib` gets one extra element that is always `0.0`. The kernel then needs no masks.
- If `n` is not a multiple of 4, the missing lanes of the last chunk are empty rows whose `perm` entry is `n`; the rank vectors get the same extra element and are cut back to `n` before returning.
- Built inside the variant's function before its timer starts, so build time is never in the reported time.

Verify: the builder prints its slot count; for `rmat-20` at σ = 16,384 it must be 17,280,772. `tests/tiny.txt` has 6 vertices and covers the not-a-multiple-of-4 case.

### 5. Stage 3: `src/pagerank_ell_scalar.cpp` and `src/pagerank_simd.cpp`

- Both loop over chunks with `schedule(static)` and write results through `perm`.
- `ell_scalar`: plain loops. GCC can vectorize the across-rows loop with gathers on its own at `-O3 -march=native`, so vectorization is switched off for this kernel and confirmed off with `-fopt-info-vec`.
- `simd`: first `#pragma omp simd` on the lane loop, as the proposal says, checked with `-fopt-info-vec`. If the compiler does not emit gathers, use intrinsics: `_mm_loadu_si128` loads four indices, `_mm256_i32gather_pd(c, idx, 8)` fetches the four contributions, `_mm256_add_pd` accumulates. The four results are stored to a small buffer and written to `x_new[perm[..]]` one by one (AVX2 has no scatter).

Verify: `make test`, plus `make debug` (ASan catches a gather that reads past `contrib`).

### 6. Stage 4: `src/pagerank_combined.cpp`

- The `simd` kernel with `schedule(dynamic, chunk)` over SELL chunks, using the K fixed in step 3.

Verify: `make test`.

### 7. Test harness

- `tests/run_tests.sh` runs each variant at the default thread count only. Run every variant at `-t 1`, `-t 3` and `-t 16`: the odd count catches block-boundary mistakes, and 16 is the count the report uses.

Verify: `make test` output lists all three thread counts for all six binaries, 0 failed.

### 8. Final benchmarks

- Six binaries × threads {1, 2, 4, 8, 16} × graphs {`email-Enron`, the second real graph, `rmat-20`}, 20 fixed iterations, median of 10 runs, on the step 1 machine, idle.
- Commit `results/benchmarks.txt` and `.csv` from that run.

### 9. `perf` profiling: `bench/run_perf.sh`

- `perf stat` counts the whole process, and graph loading is far longer than the iterations. Run each configuration twice, at 10 and at 110 fixed iterations, and subtract: the difference is exactly 100 iterations with loading and the SELL build cancelled out.
- Counters: instructions, cycles (IPC = instructions ÷ cycles), cache misses, stalled cycles. Event names for stalls differ between Intel and AMD; pick them from `perf list` on the machine.
- All six binaries at 1 and 16 threads on the largest graph. Output: `results/perf.csv`.

### 10. Plots and report

- `tools/plot_bench.py` reads `results/benchmarks.csv` and draws speedup against threads (1–16) per graph, one line per variant plus the ideal line. This adds `matplotlib` to `requirements.txt`; nothing already installed draws figures.
- GraphMat's Fig. 5a trend goes next to ours as a qualitative reference only.
- Report: the four ratios from the variants table at each thread count, the `perf` table, the fixed K and σ with the sweeps that chose them, final dataset list, and the individual contribution breakdown the proposal promises.
- Tick the README roadmap.

## Risks

- **SIMD may gain little or nothing.** The kernel is a gather over an 8 MB `contrib` array on `rmat-20`, so it is bound by memory latency, not arithmetic. Gathers are also slowed by the "Downfall" microcode fix on many Intel CPUs; check `/sys/devices/system/cpu/vulnerabilities/gather_data_sampling` on the benchmark machine and record it. A flat result still answers the SIMD research question, and the `perf` numbers explain it.
- **`email-Enron` will scale poorly at 16 threads.** One iteration is about 1 ms split over four parallel regions, so thread start-up cost is a large share. Report it as the small-graph case.
- **Memory bandwidth can cap scaling on the large graphs** before 16 threads, for every variant. The stalled-cycle counter in step 9 is what shows this.

## Reference added by this plan

M. Kreutzer, G. Hager, G. Wellein, H. Fehske, A. R. Bishop, "A Unified Sparse Matrix Data Format for Efficient General Sparse Matrix-Vector Multiplication on Modern Processors with Wide SIMD Units," SIAM J. Sci. Comput. 36(5), 2014. Format description only; no code is reused.
