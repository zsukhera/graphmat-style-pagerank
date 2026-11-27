# Every src/<name>.cpp builds into its own executable bin/<name>,
# so each PageRank variant (serial, static, dynamic, simd, ...) is one file.
#
#   make              optimized build of every variant into bin/
#   make debug        sanitizer build (ASan + UBSan) into bin/debug/
#   make test         check every variant against NetworkX and serial
#   make data         download email-Enron from SNAP into data/
#   make data-large   download soc-LiveJournal1 (~250 MB compressed)
#   make data-rmat    generate a 1M-vertex / 16M-edge RMAT graph locally
#   make bench        time every variant; writes results/benchmarks.txt
#   make clean        remove bin/

CXX        ?= g++
CXXFLAGS   ?= -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra
DEBUGFLAGS := -std=c++17 -O1 -g -fopenmp -Wall -Wextra \
              -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS    ?=

SRC_DIR  := src
BIN_DIR  ?= bin
DATA_DIR := data
SNAP_URL := https://snap.stanford.edu/data

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
HEADERS := $(wildcard $(SRC_DIR)/*.h)
TARGETS := $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%,$(SOURCES))

.PHONY: all debug test bench data data-large data-rmat clean help

all: $(TARGETS)
ifeq ($(strip $(SOURCES)),)
	@echo "No variants in $(SRC_DIR)/ yet (add a src/<name>.cpp to build bin/<name>)."
endif

# Any header change rebuilds every variant (they all include common.h).
$(BIN_DIR)/%: $(SRC_DIR)/%.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $@

debug:
	@$(MAKE) --no-print-directory all CXXFLAGS="$(DEBUGFLAGS)" BIN_DIR=bin/debug

# Builds first, then runs the correctness harness on every variant.
test: all
	tests/run_tests.sh

bench: all
	bench/run_bench.sh

data: $(DATA_DIR)/email-Enron.txt

data-large: $(DATA_DIR)/soc-LiveJournal1.txt

data-rmat: $(DATA_DIR)/rmat-20.txt

# Generated, not downloaded (needs the venv for NumPy). Must come before the
# SNAP download pattern rule so make picks it for rmat-* files.
$(DATA_DIR)/rmat-%.txt:
	@mkdir -p $(DATA_DIR)
	.venv/bin/python tools/gen_rmat.py --scale $* -o $@.part
	mv $@.part $@

# Downloads to a .part file first so an interrupted download never
# leaves a truncated graph that looks complete. Re-running resumes the
# .gz.part (curl -C -). Older curl exits non-zero when
# the part file is already complete (HTTP 416), so a failed curl is accepted
# if the part file is a whole gzip. A corrupt part file is never repaired by
# resuming: delete it by hand.
$(DATA_DIR)/%.txt:
	@mkdir -p $(DATA_DIR)
	curl -fL -C - --retry 2 -o $@.gz.part $(SNAP_URL)/$*.txt.gz || gunzip -t $@.gz.part
	gunzip -c $@.gz.part > $@.part || { rm -f $@.part; echo "corrupt $@.gz.part: delete it and re-run make" >&2; false; }
	mv $@.part $@
	rm -f $@.gz.part

clean:
	rm -rf $(BIN_DIR)

help:
	@sed -n '1,12p' Makefile | sed 's/^# \{0,1\}//'
