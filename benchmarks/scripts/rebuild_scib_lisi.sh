#!/usr/bin/env bash
# Rebuild scib's bundled LISI helper binary against the host glibc.
#
# scib (Theis-lab) ships a precompiled `knn_graph.o` that requires
# glibc 2.38+ / GLIBCXX_3.4.32+. On Ubuntu 22.04 (glibc 2.35) the
# shipped binary fails to load with:
#
#     /knn_graph.o: /usr/lib/x86_64-linux-gnu/libc.so.6:
#     version `GLIBC_2.38' not found
#
# The source (`knn_graph.cpp`) ships alongside the binary, and the
# rebuild is a one-liner. This script compiles in-place so the
# `scib.metrics.lisi.*` codepath works on any host with g++ and
# C++11 support.
#
# Idempotent: safe to re-run; just overwrites the .o.
#
# Usage:
#     bash benchmarks/scripts/rebuild_scib_lisi.sh
set -euo pipefail

KNN_DIR=$(python -c 'import scib, pathlib; print(pathlib.Path(scib.__file__).parent / "knn_graph")')
echo "Rebuilding LISI binary in: $KNN_DIR"

if [[ ! -f "$KNN_DIR/knn_graph.cpp" ]]; then
    echo "ERROR: $KNN_DIR/knn_graph.cpp not found — is scib installed?" >&2
    exit 1
fi

g++ -O3 -std=c++11 "$KNN_DIR/knn_graph.cpp" -o "$KNN_DIR/knn_graph.o"
chmod +x "$KNN_DIR/knn_graph.o"

# Smoke-test: the binary should print its usage when invoked with no args.
if ! "$KNN_DIR/knn_graph.o" 2>&1 | head -1 | grep -q 'usage'; then
    echo "WARN: rebuilt binary did not print expected usage line; check it manually" >&2
fi

echo "OK — $KNN_DIR/knn_graph.o rebuilt and runnable."
