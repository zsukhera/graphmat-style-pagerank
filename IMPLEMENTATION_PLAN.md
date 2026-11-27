# Implementation plan: what is left

All the code the proposal calls for is on `main`. What is left is checking it on Linux, running it on a 16-core machine, and writing the report. Proposal: `SecG_B_7/GraphMat_Proposal_Paper7.pdf`.

## Where we are

| Proposal item | State |
|---|---|
| Serial baseline, CSR SPMV (4.1) | Done: `src/pagerank_serial.cpp` |
| Stage 1, OpenMP static | Done: `src/pagerank_static.cpp` |
| Stage 2, OpenMP dynamic, over-partitioned | Done: `src/pagerank_dynamic.cpp`, option `-p K` |
| Stage 3, SIMD | Done: `src/sell.h` (sliced ELLPACK), `src/pagerank_ell_scalar.cpp` (scalar control), `src/pagerank_simd.cpp` (AVX2) |
| Stage 4, combined | Done: `src/pagerank_combined.cpp` |
| Correctness vs. serial and NetworkX (5.1) | Done: `make test`, every variant at 1, 3 and 16 threads |
| Thread sweep 1–16 (5.2) | Script done: `bench/run_bench.sh` defaults to 1, 2, 4, 8, 16. **Not run**: needs a 16-core machine |
| Two real-world graphs (5.2) | One (`email-Enron`). `make data-large` now resumes a broken download; `soc-LiveJournal1` still not fetched |
| `perf` profiling (5.3) | Script done: `bench/run_perf.sh`. **Never run against real `perf`** |
| Comparison to the paper, report (5.4) | Plot script done: `tools/plot_bench.py`. Report not started |

The decisions from the first version of this plan stand: CSR is the one sparse format, the SIMD variants use a sliced-ELLPACK copy built from it (SELL-4-σ, σ = 16,384), and every variant is benchmarked at 1, 2, 4, 8 and 16 threads.

## What has not been verified

Everything above was built and tested on one Windows laptop with MinGW g++ 16.2, because no Linux machine was available. The tests pass there (108 of 108). These things are still unknown:

- Whether it builds without warnings and passes on Linux with GCC 13, the team's compiler.
- Whether the ELLPACK variants are memory-safe under AddressSanitizer. `make debug` does not work with MinGW, and these variants index one element past the vertex count by design.
- Whether GCC 13 leaves the `ell_scalar` kernel scalar. GCC 16.2 does. If GCC 13 vectorizes it, the control is no longer a control.
- Whether `bench/run_bench.sh` works with the real `bc` and `lscpu`. It was run with stand-ins.
- Whether `bench/run_perf.sh` parses real `perf stat -x,` output. It was run against an imitation written from the man page.
- Every timing. The laptop numbers in `results/` and in the pull requests are from 4 cores and are not for the report.

## The six variants and what they measure

| | scalar, CSR | scalar, ELLPACK | AVX2, ELLPACK |
|---|---|---|---|
| 1 thread | `serial` | | |
| static schedule | `static` | `ell_scalar` | `simd` |
| dynamic schedule | `dynamic` | | `combined` |

At the same thread count:

- Load-balancing gain = `static` ÷ `dynamic`
- Effect of the layout change alone = `static` ÷ `ell_scalar`
- SIMD gain alone = `ell_scalar` ÷ `simd`
- End to end = `serial` ÷ `combined`

## Steps

### 1. Linux check (any Linux machine; WSL2 is enough except for `perf`)

```bash
./setup.sh                      # toolchain, venv, email-Enron, build
make 2>&1 | grep -i warning     # expect nothing
make test                       # expect "0 failed"
make debug
for v in bin/debug/pagerank_*; do
  for g in tests/tiny.txt tests/edge_cases.txt tests/rmat-10.txt data/email-Enron.txt; do
    "$v" "$g" -t 3 > /dev/null || echo "FAILED: $v $g"
  done
done
objdump -d bin/pagerank_ell_scalar | grep -c vgather   # expect 0
objdump -d bin/pagerank_simd       | grep -c vgather   # expect at least 1
ITERS=2 REPEATS=1 OUT_DIR=/tmp/benchcheck bench/run_bench.sh data/email-Enron.txt
```

- If `ell_scalar` contains gathers under GCC 13, switch vectorization off for its SPMV loop (its header comment says how the requirement was checked) and send that as a pull request.
- Use `email-Enron` for the benchmark-script check, not the tiny test graphs: on those an iteration takes under a microsecond, the time prints as `0.000000`, and the script's speedup calculation divides by zero.
- Fix whatever this turns up before step 4. Nothing later is worth running on unchecked code.

### 2. Benchmark machine

- Native Linux, at least 16 physical cores, AVX2. Record `lscpu` in the results file (the script already writes the CPU line).
- Pinning: check `lscpu -e`. With 16 threads on at least 16 cores, run with `OMP_PLACES=cores OMP_PROC_BIND=spread`; the script's default (`threads`) was tuned on a laptop.
- Two sockets: the rank vectors are allocated by one thread and land on one NUMA node. Keep all 16 threads on one socket with `numactl`, or state the effect in the report.
- Read `/sys/devices/system/cpu/vulnerabilities/gather_data_sampling` and record it. If the "Downfall" fix is active, gathers are slower and the SIMD result must be read with that in mind.
- Repeat step 1 on this machine, including one real `bench/run_perf.sh` run.

Verify: `results/benchmarks.txt` has 16 as the highest thread count and no oversubscription warning.

### 3. Data

- `make data-large` for `soc-LiveJournal1`. It resumes if the connection drops; re-run it until it finishes. If SNAP stays unusable, pick another directed SNAP graph with its original vertex ids.
- `make data-rmat` for `rmat-20`.
- Run `python tools/edge_balance.py <graph>` on each and keep the output for the report. `rmat-20` is already known: 1.008, 1.028, 1.058 and 1.070 at 2, 4, 8 and 16 threads. Its ids are randomly permuted, so dynamic scheduling can recover at most 7% there.
- If the real graph's ratio at 16 threads is also close to 1, the answer to the load-balancing question is "there is little to recover on these graphs". An RMAT graph without the id permutation would then be a useful stress case; `tools/gen_rmat.py` always permutes today, so that needs a small option.
- ELLPACK padding: σ = 16,384 was chosen on `rmat-20` (1.07× the edge count). Nothing in the repo prints the padding for another graph yet. Add a few lines of NumPy next to `tools/edge_balance.py` that report stored slots ÷ edges, and run it on the real graph before trusting σ there.

### 4. Choose K (partitions per thread)

`-p` defaults to 8, which is a placeholder.

```bash
for K in 1 2 4 8 16 32 64; do
  for v in dynamic combined; do
    printf '%s K=%s ' $v $K
    bin/pagerank_$v data/<skewed graph>.txt -t 16 -p $K -i 20 -e 0 -r 10 | grep '^time/iter'
  done
done
```

Pick one K, set it as the default in `src/driver.h` (the comment there marks it provisional), and send that as a pull request. `bench/run_bench.sh` does not pass `-p`, so the default is what the final run uses.

### 5. Final benchmarks

- `make bench` on the idle benchmark machine: five parallel variants × threads 1, 2, 4, 8, 16 × three graphs, 20 fixed iterations, median of 10 runs.
- Commit the new `results/benchmarks.txt` and `.csv`. They replace the laptop numbers, including the notes at the bottom of the `.txt`, which describe the laptop run.

### 6. `perf` profiling

- Choose the stall event for the CPU from `perf list` and pass it through `EVENTS`.
- `bench/run_perf.sh` on the largest graph, threads 1 and 16. Commit `results/perf.csv`.
- If counters jump between repeats, raise `ITERS_HI`.

### 7. Plots

- `python tools/plot_bench.py` writes `results/speedup-<graph>.png`. Commit them with the final CSV.

### 8. Report (Milestone 3) and README

- The four ratios above at each thread count, per graph.
- The `perf` table, and whether "fewer instructions, fewer stalls" holds here.
- K and σ, each with the sweep that chose it; the imbalance ratios from step 3 next to the measured static-versus-dynamic gap.
- The SIMD finding as measured. `#pragma omp simd` was tried first and GCC 16.2 did not vectorize the gather in any of three forms, so the kernel uses AVX2 intrinsics. On the laptop, SIMD was about 9–14% faster than the scalar control and most of the gain over CSR came from the layout change; expect to report something of that shape, and report it as it comes out.
- GraphMat's Fig. 5a trend next to ours, as a qualitative reference only.
- Final dataset list, and the individual contribution breakdown the proposal promises.
- README: tick the roadmap, replace the "tentative" dataset table, add the results, and add the `ell_scalar` control to the variants table. The `-p K` row says "dynamic variant only"; `combined` uses it too. If the course requires naming one sparse format, the Stage 0 row should say CSR only.
- Fill in the `[GROUP NAME]` and `[SECTION]` placeholders in the proposal and slides.

## Risks

- **SIMD may gain little.** The kernel is a gather over an 8 MB array on `rmat-20`, so it is bound by memory latency. A small or flat result still answers the research question, and the `perf` numbers explain it.
- **`email-Enron` will scale poorly at 16 threads.** One iteration is about 1 ms split over four parallel regions. Report it as the small-graph case.
- **Memory bandwidth can cap scaling on the large graphs** before 16 threads, for every variant.
- **The test suite is slow.** Every variant runs at three thread counts and each check starts Python, so with `email-Enron` and `soc-LiveJournal1` present a full `make test` takes a long time. `THREADS="3" tests/run_tests.sh pagerank_simd` narrows it to one thread count and one variant.

## Reference for the ELLPACK layout

M. Kreutzer, G. Hager, G. Wellein, H. Fehske, A. R. Bishop, "A Unified Sparse Matrix Data Format for Efficient General Sparse Matrix-Vector Multiplication on Modern Processors with Wide SIMD Units," SIAM J. Sci. Comput. 36(5), 2014. Format description only; no code is reused.
