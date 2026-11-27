#!/usr/bin/env bash
# Step 4 of IMPLEMENTATION_PLAN.md: partitions per thread (K) at 16 threads.
cd ~/graphmat-style-pagerank
export OMP_PLACES=cores OMP_PROC_BIND=close
echo "K sweep: 16 threads, -i 20 -e 0 -r 10, OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND"
echo "load at start: $(cat /proc/loadavg)"
for g in data/soc-LiveJournal1.txt data/email-Enron.txt; do
  echo "== $(basename "$g" .txt)"
  for K in 1 2 4 8 16 32 64; do
    for v in dynamic combined; do
      printf '%-9s K=%-3s ' $v $K
      bin/pagerank_$v "$g" -t 16 -p $K -i 20 -e 0 -r 10 | grep '^time/iter'
    done
  done
done
echo "load at end: $(cat /proc/loadavg)"
echo KSWEEP_DONE
