#!/usr/bin/env bash
# Steps 5 and 6 of IMPLEMENTATION_PLAN.md on the benchmark machine.
# Needs SUDO_PW in the environment for the perf_event_paranoid toggle.
cd ~/graphmat-style-pagerank
export OMP_PLACES=cores OMP_PROC_BIND=close

echo "=== rebuild"
make 2>&1 | grep -iE "warning|error" || echo "(clean build)"
echo "=== load before bench: $(cat /proc/loadavg)"

echo "=== step 5: make bench"
make bench > ~/gm_bench_stdout.log 2> ~/gm_bench_stderr.log
echo "bench rc=$?"
echo "=== load after bench: $(cat /proc/loadavg)"

echo "=== step 6: perf"
old=$(cat /proc/sys/kernel/perf_event_paranoid)
echo "$SUDO_PW" | sudo -S sysctl -w kernel.perf_event_paranoid=1
EVENTS="instructions,cycles,cache-references,cache-misses,cycle_activity.stalls_total,cycle_activity.stalls_l3_miss" \
    bench/run_perf.sh data/soc-LiveJournal1.txt > ~/gm_perf_stdout.log 2> ~/gm_perf_stderr.log
echo "perf rc=$?"
echo "$SUDO_PW" | sudo -S sysctl -w kernel.perf_event_paranoid="$old"
echo "perf_event_paranoid restored to $(cat /proc/sys/kernel/perf_event_paranoid)"
echo "=== load at end: $(cat /proc/loadavg)"
echo FINAL_DONE
