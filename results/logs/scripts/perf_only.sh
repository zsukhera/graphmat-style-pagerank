#!/usr/bin/env bash
cd ~/graphmat-style-pagerank
export OMP_PLACES=cores OMP_PROC_BIND=close
echo "load before: $(cat /proc/loadavg)"
echo "$SUDO_PW" | sudo -S sysctl -w kernel.perf_event_paranoid=1
EVENTS="instructions,cycles,cache-references,cache-misses,cycle_activity.stalls_total,cycle_activity.stalls_l3_miss" \
    bench/run_perf.sh data/soc-LiveJournal1.txt > ~/gm_perf_stdout.log 2> ~/gm_perf_stderr.log
echo "perf rc=$?"
echo "$SUDO_PW" | sudo -S sysctl -w kernel.perf_event_paranoid=4
echo "paranoid now $(cat /proc/sys/kernel/perf_event_paranoid)"
echo "load after: $(cat /proc/loadavg)"
echo PERF_DONE
