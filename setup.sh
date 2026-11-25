#!/usr/bin/env bash
# First-time setup for graphmat-style-pagerank.
#
#   ./setup.sh           check the toolchain, create the Python venv,
#                        download email-Enron, and build
#   ./setup.sh --large   also download soc-LiveJournal1 (~250 MB compressed)
#
# Safe to re-run: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"

LARGE=0
for arg in "$@"; do
    case "$arg" in
        --large) LARGE=1 ;;
        -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
    esac
done

ok()   { printf '  \033[32m[ok]\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m[fail]\033[0m %s\n' "$1"; FAILED=1; }
FAILED=0

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ---------------------------------------------------------------- toolchain
echo "[1/4] Checking toolchain"

for tool in g++ make curl gunzip python3; do
    if command -v "$tool" >/dev/null 2>&1; then ok "$tool"; else fail "$tool not found"; fi
done

if command -v g++ >/dev/null 2>&1; then
    cat > "$TMP/omp.cpp" <<'EOF'
#include <omp.h>
int main() { int n = 0;
#pragma omp parallel reduction(+:n)
    n += 1;
    return n > 0 ? 0 : 1; }
EOF
    if g++ -std=c++17 -fopenmp "$TMP/omp.cpp" -o "$TMP/omp" 2>/dev/null && "$TMP/omp"; then
        ok "g++ builds C++17 + OpenMP ($(g++ -dumpfullversion))"
    else
        fail "g++ cannot build C++17 with -fopenmp"
    fi
fi

if grep -qw avx2 /proc/cpuinfo 2>/dev/null; then
    ok "CPU supports AVX2 ($(nproc) cores)"
else
    warn "CPU lacks AVX2: the AVX2-intrinsics variant (Stage 3) will not run here"
fi

if command -v perf >/dev/null 2>&1 && perf stat -e instructions true >/dev/null 2>&1; then
    ok "perf hardware counters available"
else
    warn "perf hardware counters unavailable (common on WSL2 and VMs); timing works,"
    warn "but run the profiling stage on a native Linux machine"
fi

if [ "$FAILED" -ne 0 ]; then
    echo
    echo "Missing required tools. On Ubuntu/Debian:"
    echo "  sudo apt install build-essential curl python3 python3-venv"
    exit 1
fi

# ------------------------------------------------------------------- python
echo "[2/4] Python environment (.venv)"

# A failed earlier attempt can leave a .venv without pip; rebuild it.
if [ -d .venv ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    warn "existing .venv is broken, recreating"
    rm -rf .venv
fi
if [ ! -d .venv ]; then
    if ! python3 -m venv .venv; then
        rm -rf .venv
        printf '  \033[31m[fail]\033[0m %s\n' "could not create venv. On Ubuntu/Debian: sudo apt install python3-venv"
        exit 1
    fi
    ok "created .venv"
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
ok "networkx $(.venv/bin/python -c 'import networkx; print(networkx.__version__)') installed"

# ----------------------------------------------------------------- datasets
echo "[3/4] Datasets"
make --no-print-directory data
ok "data/email-Enron.txt"
if [ "$LARGE" -eq 1 ]; then
    make --no-print-directory data-large
    ok "data/soc-LiveJournal1.txt"
fi

# -------------------------------------------------------------------- build
echo "[4/4] Build"
make --no-print-directory
ok "build finished"

echo
echo "Setup complete. Activate the Python environment with:"
echo "  source .venv/bin/activate"
