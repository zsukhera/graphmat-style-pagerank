#!/usr/bin/env bash
# run_tests.sh — correctness harness for every PageRank variant in bin/.
#
# For each test graph, every variant is run and its ranks are checked:
#   1. against NetworkX's pagerank (graphs small enough for NetworkX), and
#   2. against bin/pagerank_serial (every variant except serial itself),
# each with the README's criterion: L1 distance below 1e-6.
#
# Graphs: tests/*.txt, plus data/email-Enron.txt and
# data/soc-LiveJournal1.txt when they have been downloaded. LiveJournal is
# too big for NetworkX, so it only gets the check against serial.
#
# Usage:  tests/run_tests.sh [variant ...]    (default: everything in bin/)
# Usually run via `make test`, which builds first.

set -uo pipefail

cd "$(dirname "$0")/.."

# Use the project's venv if it exists (setup.sh creates it).
PYTHON=python3
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python
if ! "$PYTHON" -c "import networkx" 2>/dev/null; then
    echo "error: networkx not found; run ./setup.sh or: pip install -r requirements.txt" >&2
    exit 2
fi

SERIAL=bin/pagerank_serial
if [ ! -x "$SERIAL" ]; then
    echo "error: $SERIAL not built; run make" >&2
    exit 2
fi

# Variants to test: the ones named on the command line, or all of bin/.
if [ $# -gt 0 ]; then
    VARIANTS=()
    for v in "$@"; do VARIANTS+=("bin/$(basename "$v")"); done
else
    VARIANTS=()
    for v in bin/*; do [ -f "$v" ] && [ -x "$v" ] && VARIANTS+=("$v"); done
fi

# Graphs to test, with whether NetworkX should check them.
GRAPHS=()
NX_CHECK=()
for gr in tests/*.txt; do GRAPHS+=("$gr"); NX_CHECK+=(1); done
[ -f data/email-Enron.txt ]      && { GRAPHS+=(data/email-Enron.txt);      NX_CHECK+=(1); }
[ -f data/soc-LiveJournal1.txt ] && { GRAPHS+=(data/soc-LiveJournal1.txt); NX_CHECK+=(0); }

# Ranks files go in a temporary directory, removed on exit.
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

pass=0
fail=0

# Prints one result line and updates the counters.
report() {  # report <status> <variant> <graph> <check> <details>
    printf '%-4s  %-22s %-28s %-8s %s\n' "$1" "$2" "$3" "$4" "$5"
    if [ "$1" = PASS ]; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

for i in "${!GRAPHS[@]}"; do
    graph=${GRAPHS[$i]}
    gname=$(basename "$graph")

    # Serial output is the reference for the other variants on this graph.
    ref="$OUT/serial-$gname"
    "$SERIAL" "$graph" -o "$ref" >/dev/null 2>"$OUT/err"
    rc=$?
    if [ $rc -ne 0 ]; then
        report FAIL pagerank_serial "$gname" run "exit $rc: $(head -1 "$OUT/err")"
        continue
    fi

    for variant in "${VARIANTS[@]}"; do
        vname=$(basename "$variant")
        ranks="$OUT/$vname-$gname"

        # Run the variant; a crash or non-zero exit is a failure.
        if [ "$variant" = "$SERIAL" ]; then
            cp "$ref" "$ranks"
        else
            "$variant" "$graph" -o "$ranks" >/dev/null 2>"$OUT/err"
            rc=$?
            if [ $rc -ne 0 ]; then
                report FAIL "$vname" "$gname" run "exit $rc: $(head -1 "$OUT/err")"
                continue
            fi
        fi

        # Check 1: against NetworkX.
        if [ "${NX_CHECK[$i]}" = 1 ]; then
            if res=$("$PYTHON" tests/check_ranks.py --graph "$graph" --ranks "$ranks" 2>&1)
            then status=PASS; else status=FAIL; fi
            report "$status" "$vname" "$gname" networkx "${res#* }"
        fi

        # Check 2: against serial (pointless for serial itself).
        if [ "$variant" != "$SERIAL" ]; then
            if res=$("$PYTHON" tests/check_ranks.py --ref "$ref" --ranks "$ranks" 2>&1)
            then status=PASS; else status=FAIL; fi
            report "$status" "$vname" "$gname" serial "${res#* }"
        fi
    done
done

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
