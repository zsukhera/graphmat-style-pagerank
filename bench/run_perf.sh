#!/usr/bin/env bash
# run_perf.sh — hardware counters per PageRank iteration, for every variant.
#
# `perf stat` counts the whole process, and here loading the graph (and, for
# the SIMD variants, building the extra data structure) costs far more than
# the iterations. So each configuration runs twice with a fixed iteration
# count (-e 0 makes the binary do exactly -i iterations): ITERS_LO and
# ITERS_HI. Everything but the iteration loop is identical in the two runs,
# so for each counter
#     (count_hi - count_lo) / (ITERS_HI - ITERS_LO)
# is the count per iteration with loading cancelled out. If the difference
# is not positive the noise exceeds the signal and the counter is NA.
#
# Writes:
#   results/perf.csv   one row per variant x threads (also printed as a table)
#
# Usage:  bench/run_perf.sh [graph]
#   default graph: the largest of data/soc-LiveJournal1.txt, data/rmat-*.txt
#   and data/email-Enron.txt that is present (profile the big one)
# Environment overrides:
#   THREADS="1 16"      thread counts       (serial always runs at 1 only)
#   ITERS_LO=10 ITERS_HI=110   the two iteration counts
#   EVENTS="instructions,cycles,..."  perf events. Stall event names differ
#       between CPUs (on recent Intel use cycle_activity.stalls_total
#       instead of stalled-cycles-*); check `perf list`. Unsupported events
#       are recorded as NA. ipc needs both "instructions" and "cycles".
#   VARIANTS="pagerank_static ..."  (default: everything in bin/)
#   OUT_DIR=results
#   OMP_PLACES, OMP_PROC_BIND   as in run_bench.sh

set -euo pipefail
cd "$(dirname "$0")/.."

ITERS_LO=${ITERS_LO:-10}
ITERS_HI=${ITERS_HI:-110}
THREADS=${THREADS:-1 16}
EVENTS=${EVENTS:-instructions,cycles,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend}
OUT_DIR=${OUT_DIR:-results}

[ "$ITERS_HI" -gt "$ITERS_LO" ] || { echo "error: ITERS_HI must exceed ITERS_LO" >&2; exit 2; }

# Same pinning as run_bench.sh; see the comment there.
export OMP_PLACES=${OMP_PLACES:-threads}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}

if ! command -v perf >/dev/null 2>&1 || ! perf stat -e instructions true >/dev/null 2>&1; then
    echo "error: perf hardware counters unavailable (WSL2 and VMs usually lack them);" >&2
    echo "       run this on a native Linux machine" >&2
    exit 2
fi

if [ -z "${VARIANTS:-}" ]; then
    VARIANTS=""
    for v in bin/*; do
        if [ -f "$v" ] && [ -x "$v" ]; then VARIANTS+="$(basename "$v") "; fi
    done
fi
[ -n "$VARIANTS" ] || { echo "error: no binaries in bin/; run make" >&2; exit 2; }

if [ $# -gt 0 ]; then
    GRAPH=$1
else
    GRAPH=""
    size=-1
    for g in data/soc-LiveJournal1.txt data/rmat-*.txt data/email-Enron.txt; do
        if [ -f "$g" ] && [ "$(wc -c < "$g")" -gt "$size" ]; then
            GRAPH=$g
            size=$(wc -c < "$g")
        fi
    done
fi
[ -n "$GRAPH" ] || { echo "error: no graphs; run make data" >&2; exit 2; }
gname=$(basename "$GRAPH" .txt)

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/perf.csv"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

IFS=, read -ra EVLIST <<< "$EVENTS"

# perf stat -x, lines are "value,unit,event,run-time,pct-running[,...]"
# (perf-stat(1), CSV FORMAT); lines starting with # are comments.
# Runs one measurement; leaves the counts in $TMP/counts.<iters>.
measure() {  # measure <binary> <threads> <iters>
    perf stat -x, -o "$TMP/perf.out" -e "$EVENTS" -- \
        "$1" "$GRAPH" -t "$2" -i "$3" -e 0 > "$TMP/prog.out" 2> "$TMP/prog.err" || {
        echo "error: $1 $GRAPH -t $2 -i $3 failed: $(tail -1 "$TMP/prog.err")" >&2; exit 1; }
    mv "$TMP/perf.out" "$TMP/counts.$3"
}

# count <iters> <event>: the value, or empty if perf did not report it.
# perf does not always print the requested name: user-only counting adds a
# modifier ("instructions:u"), and hybrid CPUs print one line per PMU
# ("cpu_core/instructions/", "cpu_atom/instructions/"), which are summed. A
# non-numeric line counts only if no other matching line is numeric.
count() {
    awk -F, -v e="$2" '
        function norm(s) {
            sub(/^[A-Za-z0-9_]+\//, "", s); sub(/\/.*$/, "", s); sub(/:[A-Za-z]+$/, "", s)
            return s
        }
        BEGIN {e = norm(e)}
        $1 !~ /^#/ && norm($3) == e {
            if ($1 ~ /^[0-9]+$/) {sum += $1; num = 1} else if (bad == "") bad = $1
        }
        END {if (num) printf "%.0f\n", sum; else print bad}' "$TMP/counts.$1"
}

field() { awk -v k="$1" 'index($0, k) == 1 {print $2; exit}' "$TMP/prog.out"; }

{
    printf 'variant,graph,vertices,edges,threads,iters_lo,iters_hi'
    printf ',%s' "${EVLIST[@]}"
    echo ',ipc'
} > "$CSV"

for variant in $VARIANTS; do
    bin=bin/$variant
    threads=$THREADS
    case "$variant" in *serial*) threads=1 ;; esac

    for t in $threads; do
        echo "profiling $variant on $gname, $t thread(s) ..." >&2
        measure "$bin" "$t" "$ITERS_LO"
        measure "$bin" "$t" "$ITERS_HI"
        n=$(field "vertices:"); m=$(field "edges:")

        row="$variant,$gname,$n,$m,$t,$ITERS_LO,$ITERS_HI"
        instr=NA; cyc=NA
        for ev in "${EVLIST[@]}"; do
            lo=$(count "$ITERS_LO" "$ev"); hi=$(count "$ITERS_HI" "$ev")
            per=$(awk -v lo="$lo" -v hi="$hi" -v dn=$((ITERS_HI - ITERS_LO)) 'BEGIN {
                if (lo !~ /^[0-9]+$/ || hi !~ /^[0-9]+$/) {print "NA"; exit}
                d = hi - lo
                if (d <= 0) {print "NEG"; exit}
                printf "%.0f\n", d / dn }')
            if [ "$per" = NEG ]; then
                echo "warning: $variant -t $t: $ev did not grow with iterations (lo=$lo hi=$hi); recorded NA" >&2
                per=NA
            fi
            row+=",$per"
            if [ "$ev" = instructions ]; then instr=$per; fi
            if [ "$ev" = cycles ]; then cyc=$per; fi
        done
        ipc=$(awk -v i="$instr" -v c="$cyc" 'BEGIN {
            if (i == "NA" || c == "NA") print "NA"; else printf "%.3f\n", i / c }')
        echo "$row,$ipc" >> "$CSV"
    done
done

awk -F, '{for (i = 1; i <= NF; i++) {r[NR, i] = $i; if (length($i) > w[i]) w[i] = length($i)}; nf = NF}
     END {for (j = 1; j <= NR; j++) {for (i = 1; i <= nf; i++) printf "%-*s  ", w[i], r[j, i]; print ""}}' "$CSV"
echo "wrote $CSV" >&2
