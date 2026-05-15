"""Driver for ``sparse_nmf.sweep_hyperparameters``.

Runs the user-facing sweep on at least two scIB datasets, writes
the result CSV and figures into ``docs/_static/`` so the docs page
can reference them, and prints a one-line per-config summary for
quick eyeballing.

CLI::

    # Default: pancreas + sim1, 5 k values × 2 normalize × 2 loss modes
    #          = 20 configs/dataset = 40 configs total, full data.
    python -m benchmarks.scripts.run_hyperparam_sweep

    # Lighter: subsample for faster iteration.
    python -m benchmarks.scripts.run_hyperparam_sweep --cells-per-cohort 200

    # Include batch-aware variant in the sweep.
    python -m benchmarks.scripts.run_hyperparam_sweep --include-batch-aware
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import pandas as pd

from benchmarks.io import counts, load_scib_dataset
from sparse_nmf import sweep_hyperparameters


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DOCS_STATIC = REPO / "docs" / "_static" / "hyperparam_sweep"


# Sweep grid. Kept compact so the full run fits in <30 min on one GPU.
DEFAULT_K_VALUES = [10, 20, 30, 50, 100]
DEFAULT_NORM_VALUES = [True, False]
DEFAULT_LOSS_MODES = [
    # (label, kwargs)
    ("mse_all_entries", {"nonzero_mse_weight": 0.0, "mse_weight": 1.0}),
    # nonzero-only loss switches sparseNMF to the gradient-descent
    # path which is much slower; omit by default and let the user
    # opt in via --include-nonzero.
]
DEFAULT_BATCH_AW_ALIGN_VALUES = [0.5, 2.0, 8.0]


def _configs_standard(k_values, norm_values, loss_modes, seed=0):
    """Cross-product of k × normalize × loss-mode kwargs."""
    out = []
    for k, norm, (_lossname, lkwargs) in itertools.product(k_values, norm_values, loss_modes):
        cfg = {"n_components": k, "normalize_inputs": norm,
               "max_iter": 500, "patience": 10,
               "random_state": seed}
        cfg.update(lkwargs)
        out.append(cfg)
    return out


def _configs_batch_aware(k_values, align_values, seed=0):
    """k × alignment_weight for the batch-aware variant."""
    out = []
    for k, av in itertools.product(k_values, align_values):
        out.append({
            "n_components": k,
            "alignment_weight": av,
            "sparsity_weight": 0.01,
            "max_iter": 500,
            "patience": 10,
            "random_state": seed,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["pancreas", "sim1"],
                        help="scIB dataset names to sweep on (default: pancreas sim1).")
    parser.add_argument("--cells-per-cohort", type=int, default=None,
                        help="Subsample size per (batch, label) cohort. None = full data.")
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--include-batch-aware", action="store_true",
                        help="Also sweep the batch-aware variant (k × alignment_weight).")
    parser.add_argument("--include-nonzero", action="store_true",
                        help="Also sweep nonzero_mse_weight=1.0 (gradient-descent path; slower).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DOCS_STATIC,
                        help=f"Where to write CSV + figures (default: {DOCS_STATIC}).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build sweep grids.
    loss_modes = list(DEFAULT_LOSS_MODES)
    if args.include_nonzero:
        loss_modes.append(
            ("mse_nonzero_only", {"nonzero_mse_weight": 1.0, "mse_weight": 0.0}),
        )

    standard_configs = _configs_standard(
        args.k_values, DEFAULT_NORM_VALUES, loss_modes, seed=args.seed,
    )
    batch_aw_configs = (
        _configs_batch_aware(args.k_values, DEFAULT_BATCH_AW_ALIGN_VALUES, seed=args.seed)
        if args.include_batch_aware else []
    )

    print(f"Sweeping {len(args.datasets)} datasets × "
          f"{len(standard_configs)} standard configs"
          + (f" + {len(batch_aw_configs)} batch-aware configs" if batch_aw_configs else "")
          + f" = {len(args.datasets) * (len(standard_configs) + len(batch_aw_configs))} runs")

    all_dfs: list[pd.DataFrame] = []
    t_start = time.perf_counter()
    for ds in args.datasets:
        adata, batch_key, label_key, counts_layer = load_scib_dataset(
            ds, cells_per_cohort=args.cells_per_cohort,
            seed=args.seed, n_hvg=args.n_hvg,
        )
        print(f"\n=== {ds} === (shape={adata.shape})")
        X = counts(adata, counts_layer)
        labels = adata.obs[label_key].astype(str).values
        batch = adata.obs[batch_key].astype(str).values

        if standard_configs:
            print(f"  -- standard sparseNMF ({len(standard_configs)} configs) --")
            res = sweep_hyperparameters(
                X, standard_configs, labels=labels, batch=batch,
                mode="standard", dataset_name=ds, verbose=True,
            )
            all_dfs.append(res.df)

        if batch_aw_configs:
            print(f"  -- batch-aware sparseNMF ({len(batch_aw_configs)} configs) --")
            res_bw = sweep_hyperparameters(
                X, batch_aw_configs, labels=labels, batch=batch,
                mode="batch_aware", dataset_name=ds, verbose=True,
            )
            all_dfs.append(res_bw.df)

    df = pd.concat(all_dfs, ignore_index=True)
    csv_path = args.out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}  ({len(df)} rows)")

    # Re-wrap for the .plot() helper so figures get rendered with the
    # same multi-panel layout the docs page expects.
    from sparse_nmf._hyper_sweep import SweepResult
    paths = SweepResult(df).plot(args.out_dir)
    for label, p in paths.items():
        print(f"Wrote {p}  ({label})")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal: {elapsed/60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
