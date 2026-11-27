#!/usr/bin/env bash
# Step 1 of IMPLEMENTATION_PLAN.md, on the benchmark machine.
cd ~/graphmat-style-pagerank
echo "=== compiler"; g++ --version | head -1
echo "=== make clean && make (warnings)"
make clean >/dev/null
make 2>&1 | grep -iE "warning|error" || echo "(no warnings or errors)"
echo "=== objdump vgather counts"
echo "ell_scalar: $(objdump -d bin/pagerank_ell_scalar | grep -c vgather)"
echo "simd:       $(objdump -d bin/pagerank_simd | grep -c vgather)"
echo "combined:   $(objdump -d bin/pagerank_combined | grep -c vgather)"
echo "=== make debug"
make debug 2>&1 | grep -iE "warning|error" || echo "(no warnings or errors)"
echo "=== ASan/UBSan runs, -t 3"
for v in bin/debug/pagerank_*; do
  for g in tests/tiny.txt tests/edge_cases.txt tests/rmat-10.txt data/email-Enron.txt; do
    "$v" "$g" -t 3 > /dev/null 2> /tmp/asan.$$ || { echo "FAILED: $v $g"; head -8 /tmp/asan.$$; }
  done
done
rm -f /tmp/asan.$$
echo "(sanitizer runs finished)"
echo "=== make test"
make test 2>&1 | grep -E "^FAIL|passed,"
echo "=== bench script check"
ITERS=2 REPEATS=1 OUT_DIR=/tmp/benchcheck.$$ bench/run_bench.sh data/email-Enron.txt 2>&1 | tail -40
rm -rf /tmp/benchcheck.$$
echo STEP1_DONE
