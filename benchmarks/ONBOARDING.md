# benchmarks/ — onboarding

A subprocess-isolated scIB-style integration benchmark. The
architecture exists because PyTorch (scVI, sparseNMF) and JAX
(scib-metrics) can't safely share a CUDA context in one Python
process — JAX captures the device and PyTorch ends up with
`cudaErrorDevicesUnavailable`. Each method gets its own process.

## Layout

```
benchmarks/
  io.py                    # dataset loaders, fingerprinting,
                           # embedding I/O, MethodTiming dataclass,
                           # common method-subprocess scaffolding
  run_benchmark.py         # orchestrator CLI
  viz.py                   # 4 plotting functions
  methods/                 # one module per method (callable as -m)
    pca.py
    nmf.py
    sparse_nmf.py
    sparse_nmf_nonzero.py  # opt-in: --methods sparseNMF+nonzero
    harmony.py
    scvi.py
  metrics/                 # one module per scoring impl
    scib_yosef.py          # JAX rewrite (default)
    scib_original.py       # canonical Theis-lab impl
  scripts/
    rebuild_scib_lisi.sh   # rebuild scib's LISI .o against host glibc
    run_full_seeds_and_k_sweep.sh
  runs/<run-id>/<dataset>/<method>/
    X_emb.npz              # (n_cells, k) embedding + fingerprint
    timing.json            # fit/infer/peak_rss/gpu_peak + metadata
    metrics_yosef.json     # one of these per metrics impl
    metrics_original.json
    error.txt              # iff the method's subprocess failed
```

## CLI

```bash
# Default: 5 methods × 2 datasets × subsample-50 × scib_yosef metrics.
python -m benchmarks.run_benchmark

# Full data, alternate metrics impl, into a named run dir:
python -m benchmarks.run_benchmark \
    --out-dir benchmarks/runs/2026-05-15 \
    --full \
    --metrics-impl scib_original

# Just one method on one dataset:
python -m benchmarks.run_benchmark \
    --methods sparseNMF --datasets immune --full
```

## How a run works

1. **Embed phase**: for each (dataset, method) pair, the orchestrator
   spawns `python -m benchmarks.methods.<method>` as a subprocess.
   That subprocess loads the dataset, runs preprocessing per the
   method's input convention, fits, writes `X_emb.npz` + `timing.json`
   to its run directory. On failure it writes `error.txt`.
2. **Metrics phase**: for each dataset, the orchestrator spawns
   `python -m benchmarks.metrics.<impl>` once with the method list.
   That subprocess loads all the dataset's embeddings, runs the
   scIB metric suite once per embedding, writes
   `metrics_{yosef,original}.json` per method.
3. **Aggregate**: the orchestrator reads back the per-method
   artifacts, builds the long-form results DataFrame, writes
   `results.csv`/`results.md`/`results.json` to the run root, and
   renders the four figures via `viz.plot_all`.

## Adding a method

1. Create `benchmarks/methods/<your_method>.py` with an `embed()`
   function matching this signature:

   ```python
   def embed(adata, batch_key, label_key, counts_layer, k, seed):
       # ... preprocess + fit ...
       return W, MethodTiming(fit_seconds, infer_seconds, ...)
   ```

2. Add a `__main__` block:

   ```python
   if __name__ == "__main__":
       import argparse
       from benchmarks.io import add_common_method_args, run_method_subprocess
       parser = argparse.ArgumentParser()
       add_common_method_args(parser)
       args = parser.parse_args()
       raise SystemExit(run_method_subprocess(args, embed))
   ```

3. Register in `benchmarks/run_benchmark.py`:

   ```python
   METHOD_MODULES["YourMethod"] = "benchmarks.methods.your_method"
   ```

4. Update `DEFAULT_METHODS` if it should run by default.

The `embed()` signature has more args than most methods need
(`label_key` is unused by every current method; `counts_layer`
only by sparseNMF and scVI). That's intentional so all method
modules share a uniform calling convention.

## Adding a metrics impl

Same pattern as methods. The metrics module's `__main__` should
accept `--dataset`, `--out-dir`, `--methods`, and the standard
preprocessing args (`--cells-per-cohort`, `--n-hvg`, etc.) so
fingerprints match the embed subprocesses. Write
`metrics_<impl>.json` files alongside each method's embedding.

Then add the module path to `METRICS_MODULES` in
`run_benchmark.py` and `_load_method_row` so the orchestrator
knows where to find the scores.

## Data integrity

Every embedding ships with a 16-char SHA256 fingerprint of the
cell-order + dataset shape (`io.adata_fingerprint`). The metrics
subprocess recomputes this from its freshly-loaded adata and
refuses to score when it doesn't match — catches the case where
two subprocesses are given different preprocessing args (different
HVG seed, different subsample size). If you see "fingerprint
mismatch" in metrics output, the orchestrator's `--cells-per-cohort`
or `--n-hvg` were inconsistent between phases.

## Gotchas

- **scib's LISI binary** needs a one-shot rebuild on Ubuntu 22.04
  / macOS. See `scripts/rebuild_scib_lisi.sh`. The rebuild is
  idempotent — re-run safely.
- **`track_memory`** assumes one-shot use per process (it diffs
  `ru_maxrss` high-water marks). Don't call it twice in the same
  process and trust the second result.
- **`scaled_X`** does per-batch z-scoring. Single-batch datasets
  fall through to a global path — equivalent in that degenerate
  case.
- **scib_original composite ≠ scib_yosef composite.** They use
  different per-category weight schemes. Per-metric values
  (NMI, ARI, etc.) match within numerical noise; the bio/batch/
  Total aggregates differ by ~hundredths. Pick one and stick to
  it for cross-method comparisons.
