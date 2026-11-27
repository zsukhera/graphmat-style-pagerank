# Implementation plan: what is left

All the code and all the experiments are done. What is left is merging the fixes the experiments needed, and writing the report (Milestone 3) and the README. Proposal: `SecG_B_7/GraphMat_Proposal_Paper7.pdf`.

Every number below comes from `results/linux_check_results.txt`, which is the summary to write the report from. Raw outputs are next to it in `results/`.

## Where we are

| Proposal item | State |
|---|---|
| Serial baseline, CSR SPMV (4.1) | Done: `src/pagerank_serial.cpp` |
| Stage 1, OpenMP static | Done: `src/pagerank_static.cpp` |
| Stage 2, OpenMP dynamic, over-partitioned | Done: `src/pagerank_dynamic.cpp`, `-p K`, default now K = 32 |
| Stage 3, SIMD | Done: `src/sell.h`, `src/pagerank_ell_scalar.cpp` (scalar control), `src/pagerank_simd.cpp` (AVX2) |
| Stage 4, combined | Done: `src/pagerank_combined.cpp` |
| Correctness vs. serial and NetworkX (5.1) | Done on Linux: 147 of 147 including `soc-LiveJournal1` |
| Thread sweep 1–16 (5.2) | Done: `results/benchmarks.txt` / `.csv`, idle 20-core machine |
| Two real-world graphs (5.2) | Done: `email-Enron`, `soc-LiveJournal1`, plus synthetic `rmat-20` |
| `perf` profiling (5.3) | Done: `results/perf.csv` (`soc-LiveJournal1`, 1 and 16 threads) |
| Comparison to the paper, report (5.4) | Plots done: `results/speedup-*.png`. **Report not started** |

## What was run, and where

- **Benchmark machine**: the lab machine `gpu-lab-Precision-3680`, an Intel Core i7-14700, Ubuntu 26.04, GCC 15.2. Ask the group for the login; it is not in the repo. The project copy there is `~/graphmat-style-pagerank`, with all three graphs in `data/`.
- **GCC 13 check**: WSL2 with GCC 13.3. It builds with no warnings, passes all tests, and `ell_scalar` contains no gathers. GCC 13 is not packaged for Ubuntu 26.04, so the timed runs use GCC 15.2 (also no gathers in `ell_scalar`). Say so in the report.
- **How each step was run**: `results/logs/scripts/`. `wait_idle.sh` holds a run until the machine has been idle for 3 minutes; another user's ML job shares the machine. `final.sh` runs `make bench` and then `bench/run_perf.sh`, lowering `kernel.perf_event_paranoid` to 1 just for `perf` and restoring it to 4 (needs sudo). Their output is in `results/logs/lab_*.log`.

## Changes the experiments needed (to merge)

All four are in the working tree. Send them as pull requests, as the earlier plan asked:

1. **`Makefile`**: `DEBUGFLAGS` gains `-march=native`. Without it `make debug` stops at the `#error "... needs AVX2"` in `simd` and `combined`. With it, all six variants run clean under ASan and UBSan.
2. **`src/driver.h`**: default `-p` changes from 8 to 32, from the K sweep (`results/logs/lab_ksweep.log`). The comment now says `combined` uses it too.
3. **`bench/run_perf.sh`**: `perf stat --no-scale`. On a hybrid CPU each event is opened on both the P-core and the E-core PMU. perf scaled the E-core counts up from milliseconds of running time, which added billions of phantom events. The first run was wrong because of this and was discarded. With the fix, instructions per iteration match between 1 and 16 threads.
4. **`tools/sell_padding.py`** (new): ELLPACK stored slots ÷ edges for several σ, which step 3 of the old plan asked for. It reproduces the `rmat-20` numbers in `sell.h`.

Also in the working tree: new `results/` files, `results/logs/`, and `CLAUDE.md` in `.gitignore`. Never commit `CLAUDE.md`; it holds the machine password.

## Decisions taken, with their evidence

- **Pinning**: `OMP_PLACES=cores OMP_PROC_BIND=close`. Threads 1–8 get one P-core each; threads 9–16 go to E-cores 16–23. The scripts' default (`threads`/`spread`) was **not** used for the final run. To reproduce a number, set these two variables.
- **K = 32**: `soc-LiveJournal1` is flat from K = 16 (16, 32 and 64 within 3%). On `email-Enron`, 32 is best for `dynamic` and second for `combined`.
- **σ = 16,384 stays**: padding is 1.007 on Enron, 1.074 on `rmat-20`, and 1.005 on LiveJournal.
- **Graphs**: `email-Enron` (small), `soc-LiveJournal1` (large real graph), `rmat-20` (synthetic, balanced). A non-permuted RMAT stress case is no longer needed, because both real graphs are already very skewed (static imbalance 6.5 and 4.1 at 16 threads).

## Steps left

### 1. Merge

- One PR per change above, or one PR "Linux run: fixes and results" if the group prefers. Include the new `results/` files. They replace the laptop numbers, and `results/benchmarks.txt` no longer has the laptop notes at the bottom.

### 2. Report (Milestone 3)

Everything needed is in `results/linux_check_results.txt`:

- **Setup**: machine, compiler, flags, pinning, 20 iterations, median of 10 runs. State the hybrid CPU plainly: at 16 threads, half the threads run on E-cores.
- **The four ratios** per graph and thread count (table in the results file): load balancing = `static`/`dynamic`, layout = `static`/`ell_scalar`, SIMD = `ell_scalar`/`simd`, end to end = `serial`/`combined`.
- **Load balancing is the largest effect**: up to 2.46× on LiveJournal and 1.54× on Enron. Put it next to the `edge_balance.py` ratios.
- **The E-core caveat, with the evidence**: `rmat-20` is balanced (1.07), and `static` ≈ `dynamic` up to 8 threads. At 16 threads `static` falls to 5.87× while `dynamic` reaches 8.95×. Part of every 16-thread static-versus-dynamic gap is core heterogeneity, not graph skew. If the group wants a clean split, re-run the 16-thread `static` and `dynamic` points on the 8 P-cores with hyperthreading (`OMP_PLACES=threads`, CPUs 0–15) and report both.
- **SIMD**: −1% to +10% over the scalar control, typically 5–8%. The layout change gives more at low thread counts (2.3× on Enron at 1 thread). This matches the laptop finding the old plan predicted.
- **The `perf` table**: "fewer instructions" holds (ELLPACK −32%, AVX2 a further −26%, −50% in total). "Fewer stalls" does **not**: stall cycles and cache misses are flat across variants, and 75–80% of stalls are L3-miss stalls. The kernel is memory-latency bound, so halving the instructions buys about 6–10%. `cycle_activity.*` events exist only on P-cores, so at 16 threads they cover the 8 P-core threads only.
- **K and σ**, each with its sweep table.
- **Small-graph case**: `combined` on Enron is slower at 16 threads than at 8 (0.14 vs 0.06 ms per iteration). An iteration is about 0.1 ms, so fork/join overhead and E-core stragglers dominate.
- GraphMat's Fig. 5a trend next to ours, as a qualitative reference only.
- Final dataset list (three graphs above), and the individual contribution breakdown the proposal promises.

### 3. README

- Tick the roadmap, replace the "tentative" dataset table, and add the results with one or more plots.
- Variants table: add the `ell_scalar` control. The `-p K` row should say "dynamic and combined, default 32". If the course requires naming one sparse format, the Stage 0 row should say CSR only.
- Note the pinning used for the published numbers.

### 4. Proposal and slides

- Fill in the `[GROUP NAME]` and `[SECTION]` placeholders.

## If anything needs re-running

- Re-run on the lab machine, never on a laptop or WSL. First check that the machine is idle (`uptime`, `top`); another user's ML job shares it. `results/logs/scripts/wait_idle.sh` does this check.
- `OMP_PLACES=cores OMP_PROC_BIND=close make bench`. That is about 10 minutes for all three graphs.
- `perf` needs `sudo sysctl -w kernel.perf_event_paranoid=1` first, and `=4` after. Use the stall events `cycle_activity.stalls_total,cycle_activity.stalls_l3_miss` (see `results/logs/scripts/final.sh`).
- `make test` with LiveJournal present takes about 30 minutes. `THREADS="3" tests/run_tests.sh pagerank_simd` narrows it.

## Reference for the ELLPACK layout

M. Kreutzer, G. Hager, G. Wellein, H. Fehske, A. R. Bishop, "A Unified Sparse Matrix Data Format for Efficient General Sparse Matrix-Vector Multiplication on Modern Processors with Wide SIMD Units," SIAM J. Sci. Comput. 36(5), 2014. Format description only; no code is reused.
