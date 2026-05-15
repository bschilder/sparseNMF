### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCA | 1.1s | 13ms | 1.2m | — | — | +0.668 | +0.436 | **+0.575** |
| NMF | 31.2s | 1.2s | 16.7s | — | — | +0.679 | +0.427 | **+0.578** |
| sparseNMF | 2.0m | — | 16.1s | — | 374 MB | +0.694 | +0.518 | **+0.623** |
| Harmony | 2.9s | — | 16.9s | — | — | +0.749 | +0.578 | **+0.680** |
| scVI | 1.6m | 98ms | 15.9s | — | 52 MB | +0.754 | +0.619 | **+0.700** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCA | 0.8s | 9ms | 1.2m | — | — | +0.753 | +0.470 | **+0.640** |
| NMF | 26.9s | 0.7s | 13.2s | — | — | +0.709 | +0.446 | **+0.604** |
| sparseNMF | 1.3m | — | 11.8s | — | 445 MB | +0.765 | +0.575 | **+0.689** |
| Harmony | 2.9s | — | 11.1s | — | — | +0.860 | +0.689 | **+0.792** |
| scVI | 1.1m | 70ms | 12.0s | — | 52 MB | +0.843 | +0.565 | **+0.731** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| Harmony | **+0.736** | 2.9s | — |
| scVI | **+0.716** | 1.3m | 52 MB |
| sparseNMF | **+0.656** | 1.7m | 409 MB |
| PCA | **+0.608** | 0.9s | — |
| NMF | **+0.591** | 29.0s | — |
