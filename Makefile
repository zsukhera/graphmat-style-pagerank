# Every src/<name>.cpp builds into its own executable bin/<name>,
# so each PageRank variant (serial, static, dynamic, simd, ...) is one file.
#
#   make              optimized build of every variant into bin/
#   make debug        sanitizer build (ASan + UBSan) into bin/debug/
#   make data         download email-Enron from SNAP into data/
#   make data-large   download soc-LiveJournal1 (~250 MB compressed)
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

.PHONY: all debug data data-large clean help

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

data: $(DATA_DIR)/email-Enron.txt

data-large: $(DATA_DIR)/soc-LiveJournal1.txt

# Downloads to a .part file first so an interrupted download never
# leaves a truncated graph that looks complete.
$(DATA_DIR)/%.txt:
	@mkdir -p $(DATA_DIR)
	curl -fL --retry 2 -o $@.gz.part $(SNAP_URL)/$*.txt.gz
	gunzip -c $@.gz.part > $@.part
	mv $@.part $@
	rm -f $@.gz.part

clean:
	rm -rf $(BIN_DIR)

help:
	@sed -n '1,9p' Makefile | sed 's/^# \{0,1\}//'
