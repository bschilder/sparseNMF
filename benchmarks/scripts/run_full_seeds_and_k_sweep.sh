#!/usr/bin/env bash
# Comprehensive benchmark run on a GPU pod.
#
# Two phases:
# 1. Multi-seed × full-data: 5 seeds × 5 methods × 2 datasets,
#    no subsampling (Luecken 2022 reference scale).
# 2. Multi-k sweep for sparseNMF only: k ∈ {10, 20, 30, 50, 100}
#    on both datasets at seed 0. Shows where sparseNMF's sweet
#    spot lives.
#
# Outputs land under /workspace/runs/ in a layout the orchestrator
# already understands (results.csv per run, figures auto-generated).
#
# Usage on the pod:
#     bash benchmarks/scripts/run_full_seeds_and_k_sweep.sh
set -euo pipefail

REPO=${REPO:-/workspace/sparseNMF}
RUNS=${RUNS:-/workspace/runs}
mkdir -p "$RUNS"
cd "$REPO"

SEEDS=(0 1 2)
KS=(10 20 30 50 100)
METHODS="PCA NMF sparseNMF Harmony scVI"
# All scIB-canonical RNA integration datasets except mouse_brain (978k
# cells, multi-hour scVI training per seed) and immune_hum_mou (97k
# cells, cross-species, much slower to converge). Lung + sim1 + sim2
# round out the standard benchmark set.
DATASETS="pancreas immune lung sim1 sim2"

echo "=== Phase 1: multi-seed full-data run ==="
for seed in "${SEEDS[@]}"; do
    out="$RUNS/full-seeds/seed-$seed"
    if [[ -s "$out/results.csv" ]]; then
        echo "skip seed=$seed (results.csv exists)"
        continue
    fi
    echo ">>> seed=$seed"
    python -m benchmarks.run_benchmark \
        --out-dir "$out" \
        --full \
        --seed "$seed" \
        --k 30 \
        --datasets $DATASETS \
        --methods $METHODS \
        --metrics-impl scib_yosef \
        2>&1 | tee "$out.log" | tail -3
done

echo ""
echo "=== Phase 2: multi-k sweep for sparseNMF ==="
for k in "${KS[@]}"; do
    out="$RUNS/k-sweep/k-$k"
    if [[ -s "$out/results.csv" ]]; then
        echo "skip k=$k (results.csv exists)"
        continue
    fi
    echo ">>> k=$k"
    python -m benchmarks.run_benchmark \
        --out-dir "$out" \
        --full \
        --seed 0 \
        --k "$k" \
        --datasets $DATASETS \
        --methods sparseNMF \
        --metrics-impl scib_yosef \
        2>&1 | tee "$out.log" | tail -3
done

echo ""
echo "=== Phase 3: scib_original head-to-head on seed=0 embeddings ==="
# Reuse the seed-0 embeddings; rerun metrics with the canonical impl.
bash "$REPO/benchmarks/scripts/rebuild_scib_lisi.sh" || true
out0="$RUNS/full-seeds/seed-0"
if [[ -d "$out0" ]]; then
    for ds in $DATASETS; do
        echo ">>> scib_original on $ds"
        python -m benchmarks.metrics.scib_original \
            --dataset "$ds" \
            --out-dir "$out0" \
            --methods $METHODS \
            --seed 0 \
            2>&1 | tee -a "$out0.log" | tail -2
    done
fi

echo ""
echo "=== Done. Run tree: ==="
find "$RUNS" -name 'results.csv' -printf '%p %s\n' | sort
