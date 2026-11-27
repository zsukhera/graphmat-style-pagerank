#!/usr/bin/env bash
# run_bench.sh — times every PageRank variant across thread counts and graphs.
#
# Every run does a fixed number of iterations (-i ITERS -e 0), so all
# variants do identical work, and the metric is the median time per
# iteration over REPEATS in-process runs (graph loading is never timed).
# Speedup = serial time/iter / variant time/iter, on the same graph.
#
# Writes:
#   results/benchmarks.txt   human-readable report (machine, settings, tables)
#   results/benchmarks.csv   one row per run, for plots
#
# Usage:  bench/run_bench.sh [graph ...]
#   default graphs: data/email-Enron.txt, data/rmat-*.txt and
#   data/soc-LiveJournal1.txt, whichever are present
# Environment overrides:
#   THREADS="1 2 4 8"   thread counts      (default: 1 2 4 8 16, whatever nproc is;
#                       counts above nproc are run but flagged as oversubscribed)
#   ITERS=20            iterations per run
#   REPEATS=10          runs per configuration (median reported)
#   VARIANTS="pagerank_static ..."  (default: everything in bin/ except serial)
#   OUT_DIR=results

set -euo pipefail
cd "$(dirname "$0")/.."

ITERS=${ITERS:-20}
REPEATS=${REPEATS:-10}
OUT_DIR=${OUT_DIR:-results}
NPROC=$(nproc)

THREADS=${THREADS:-1 2 4 8 16}

# More threads than hardware threads time-slice, so those numbers say
# nothing about scaling.
OVERSUB=""
for t in $THREADS; do
    [ "$t" -gt "$NPROC" ] && OVERSUB+="$t "
done
WARNING=""
[ -z "$OVERSUB" ] || WARNING="WARNING: thread counts ${OVERSUB% } exceed the $NPROC hardware threads; those runs are oversubscribed and do not show scaling"
[ -z "$WARNING" ] || echo "$WARNING" >&2

# Pin one thread per physical core first. On a CPU with hyperthreading,
# "threads" places with "spread" binding use one sibling per core up to the
# core count (if the siblings are numbered 2k and 2k+1, as on the laptop this
# was tuned on) and both siblings beyond it; check `lscpu -e` for the
# numbering on another machine. If the highest thread count does not exceed
# the number of physical cores, OMP_PLACES=cores OMP_PROC_BIND=spread (set
# through the environment overrides) gives one thread per core. With more
# threads than cores, OMP_PLACES=cores is a trap: it binds two spinning
# threads to one place, ~10x slower.
# The default OMP_WAIT_POLICY (spin briefly, then sleep) is kept: it was
# fastest; "passive" was ~4x slower on email-Enron.
export OMP_PLACES=${OMP_PLACES:-threads}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}

SERIAL=bin/pagerank_serial
[ -x "$SERIAL" ] || { echo "error: $SERIAL not built; run make" >&2; exit 2; }

if [ -z "${VARIANTS:-}" ]; then
    VARIANTS=""
    for v in bin/*; do
        [ -f "$v" ] && [ -x "$v" ] && [ "$v" != "$SERIAL" ] && VARIANTS+="$(basename "$v") "
    done
fi

if [ $# -gt 0 ]; then
    GRAPHS=("$@")
else
    GRAPHS=()
    [ -f data/email-Enron.txt ]      && GRAPHS+=(data/email-Enron.txt)
    for r in data/rmat-*.txt; do [ -f "$r" ] && GRAPHS+=("$r"); done
    [ -f data/soc-LiveJournal1.txt ] && GRAPHS+=(data/soc-LiveJournal1.txt)
fi
[ ${#GRAPHS[@]} -gt 0 ] || { echo "error: no graphs; run make data" >&2; exit 2; }

mkdir -p "$OUT_DIR"
TXT="$OUT_DIR/benchmarks.txt"
CSV="$OUT_DIR/benchmarks.csv"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Runs one configuration; prints "median_per_iter min_per_iter max_per_iter".
run_one() {  # run_one <binary> <graph> <threads>
    "$1" "$2" -t "$3" -i "$ITERS" -e 0 -r "$REPEATS" > "$TMP/out" 2>"$TMP/err" || {
        echo "error: $1 $2 -t $3 failed: $(head -1 "$TMP/err")" >&2; exit 1; }
    awk '/^time\/iter:/ {med=$2} /^iter min\/max:/ {lo=$3; hi=$4}
         END {if (lo == "") {lo = med; hi = med}; print med, lo, hi}' "$TMP/out"
}

field() { awk -v k="$1" 'index($0, k) == 1 {print $2; exit}' "$TMP/out"; }

echo "variant,graph,vertices,edges,threads,iters,repeats,median_s_per_iter,min_s_per_iter,max_s_per_iter,speedup" > "$CSV"

{
    echo "PageRank benchmarks"
    echo "==================="
    echo
    echo "date:        $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "commit:      $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (+ uncommitted changes)')"
    echo "cpu:         $(lscpu | sed -n 's/^Model name: *//p')"
    echo "cores:       $(lscpu | sed -n 's/^Core(s) per socket: *//p') physical x $(lscpu | sed -n 's/^Socket(s): *//p') socket(s), $NPROC hardware threads"
    [ -z "$WARNING" ] || echo "$WARNING"
    echo "kernel:      $(uname -r)"
    echo "compiler:    $(${CXX:-g++} --version | head -1)"
    echo "flags:       -std=c++17 -O3 -march=native -fopenmp"
    echo "OpenMP env:  OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-default}"
    echo "method:      $ITERS fixed iterations per run (-e 0), median of $REPEATS runs;"
    echo "             time = PageRank iteration loop only (loading not timed)"
    echo "             speedup = serial median / variant median"
    echo "             min-max = fastest and slowest of the runs (shows noise)"
} > "$TXT"

for graph in "${GRAPHS[@]}"; do
    gname=$(basename "$graph" .txt)
    echo "benchmarking $gname ..." >&2

    read -r serial_t s_lo s_hi < <(run_one "$SERIAL" "$graph" 1)
    n=$(field "vertices:"); m=$(field "edges:")
    echo "pagerank_serial,$gname,$n,$m,1,$ITERS,$REPEATS,$serial_t,$s_lo,$s_hi,1.000" >> "$CSV"

    {
        echo
        echo "$gname  ($n vertices, $m edges)"
        echo "$(printf '%.0s-' $(seq 1 80))"
        printf '%-10s %7s %14s %19s %8s %11s\n' variant threads "median ms/it" "min-max ms/it" speedup efficiency
        printf '%-10s %7s %14.3f %19s %8s %11s\n' serial 1 "$(echo "$serial_t * 1000" | bc -l)" \
            "$(printf '%.3f-%.3f' "$(echo "$s_lo * 1000" | bc -l)" "$(echo "$s_hi * 1000" | bc -l)")" 1.00 -
    } >> "$TXT"

    for vname in $VARIANTS; do
        for t in $THREADS; do
            echo "  $vname t=$t" >&2
            read -r vt v_lo v_hi < <(run_one "bin/$vname" "$graph" "$t")
            sp=$(echo "$serial_t / $vt" | bc -l)
            eff=$(echo "100 * $sp / $t" | bc -l)
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.3f\n' "$vname" "$gname" "$n" "$m" "$t" \
                "$ITERS" "$REPEATS" "$vt" "$v_lo" "$v_hi" "$sp" >> "$CSV"
            printf '%-10s %7s %14.3f %19s %8.2f %10.0f%%\n' "${vname#pagerank_}" "$t" \
                "$(echo "$vt * 1000" | bc -l)" \
                "$(printf '%.3f-%.3f' "$(echo "$v_lo * 1000" | bc -l)" "$(echo "$v_hi * 1000" | bc -l)")" \
                "$sp" "$eff" >> "$TXT"
        done
    done
done

echo >&2
cat "$TXT"
echo
echo "written: $TXT, $CSV" >&2
