#!/usr/bin/env bash
# Blocks until the machine has been idle (1-min load < 1.5, no process above
# 20% CPU) for three checks in a row, 60 s apart.
ok=0
while [ "$ok" -lt 3 ]; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    busy=$(ps -eo pcpu= | awk '$1 > 20' | wc -l)
    if awk -v l="$load" 'BEGIN {exit !(l < 1.5)}' && [ "$busy" -eq 0 ]; then
        ok=$((ok + 1))
    else
        ok=0
    fi
    echo "$(date '+%H:%M:%S') load=$load busy_procs=$busy idle_checks=$ok"
    [ "$ok" -lt 3 ] && sleep 60
done
echo IDLE
