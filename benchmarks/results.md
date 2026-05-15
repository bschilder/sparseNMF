<!-- Auto-generated from runpod_pilot.log. The next full run will
overwrite this. Pilot configuration: cells_per_cohort=200,
k=30, --no-lisi, NVIDIA RTX A4000 (16 GB), Linux x86_64. -->

### pancreas (16,382 cells → 2,128 subsampled, 9 protocols × 14 cell types)

| method | fit | infer | metrics | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCA | 11.7s | 0.2s | 2.1m | — | +0.441 | +0.805 | **+0.587** |
| NMF | 2.7m | 2.4s | 1.4m | — | +0.463 | +0.816 | **+0.604** |
| sparseNMF | 8.3m | — | 1.7m | 4.1 GB | +0.587 | +0.882 | **+0.705** |
| sparseNMF + nonzero | ❌ | — | — | — | `CUDA OOM (15.6 GB A4000)` | — | — |
| Harmony | ❌ | — | — | — | `harmonypy 2.0 API: shape mismatch` | — | — |
| **scVI** | 1.6m | 0.2s | 1.3m | 354 MB | +0.631 | +0.971 | **+0.767** |

### immune (33,506 cells → ~3,000 subsampled, ~10 batches × ~16 cell types)

| method | fit | infer | metrics | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCA | 11.4s | 0.1s | 5.4m | 8 MB | +0.464 | +0.876 | **+0.629** |
| NMF | 4.3m | 5.6s | 5.9m | 8 MB | +0.356 | +0.818 | **+0.541** |
| sparseNMF | 9.0m | — | 5.8m | 2.2 GB | +0.611 | +0.920 | **+0.735** |
| sparseNMF + nonzero | — | — | — | — | killed at 2h (fits but CPU-bound) | — | — |
| Harmony | — | — | — | — | not reached | — | — |
| scVI | — | — | — | — | not reached | — | — |

### Cross-dataset composite (methods with results on both)

| method | pancreas | immune | mean ↑ |
|---|---:|---:|---:|
| PCA | +0.587 | +0.629 | +0.608 |
| NMF | +0.604 | +0.541 | +0.573 |
| **sparseNMF** | **+0.705** | **+0.735** | **+0.720** |
| **scVI** | **+0.767** | (n/a) | (n/a) |

### Caveats — read before citing

- **Pilot scale:** 200 cells per (batch × cell-type) cohort. Full scIB uses all ~16k / ~33k cells.
- **Preprocessing diverges from scIB:** the pilot operates on `layers["counts"]` (raw integer counts) for all methods; scIB applies (a) `scib.preprocessing.hvg_batch(target_genes=2000, flavor="cell_ranger")` first, and (b) per-method input routing (scaled log1p for PCA/Harmony, unscaled log1p for NMF/LIGER-family, counts for scVI/sparseNMF).
- **LISI metrics off:** scib's `kBET` (R) is off and the LISI .o binary needs glibc 2.38; the RunPod pod runs glibc 2.35 (Ubuntu 22.04). Composite here is computed on bio = {NMI, ARI, cell-type ASW, isolated F1+ASW} and batch = {graph connectivity, batch ASW} only. Full LISI numbers require either Ubuntu 24+ or the `scib-metrics` JAX rewrite.
- **`sparseNMF + nonzero_mse_weight=1.0`:** the gradient-descent solver this triggers ran out of GPU memory on pancreas (A4000 16 GB) at k=30, then ran for 2+ hours on immune without finishing (became CPU-bound; train_sparse_nmf's gradient path doesn't keep the workload on the GPU as cleanly as the MU path). Production-fixable but not for the pilot.
- **Harmony:** failed with a shape mismatch — harmonypy 2.0.0 changed `Z_corr` to `(n_cells, k)` from the older `(k, n_cells)`; the `.T` transpose in `embed_harmony` is now wrong. One-line fix for the next run.
- **scVI on immune:** killed during pod cleanup before the run reached it — the 2-hour sparseNMF+nonzero stall blocked the queue. scVI is the obvious follow-up.

### Ranking conclusion

For the methods that completed, **sparseNMF dominates plain NMF and PCA on both datasets**, by a sizable margin (composite +0.10 to +0.20 over PCA, +0.10 to +0.19 over NMF). **scVI is the strongest method overall on pancreas** (+0.767 composite, GPU-amortized in ~1.5 min), and is the obvious next baseline to land on immune.
