### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCA | 2.2s | 53ms | 1.3m | — | — | +0.441 | +0.805 | **+0.587** |
| NMF | 1.3m | 1.2s | 54.3s | — | — | +0.455 | +0.832 | **+0.606** |
| sparseNMF | 4.8m | — | 59.4s | 576 MB | 4.1 GB | +0.577 | +0.889 | **+0.702** |
| Harmony | 2.8s | — | 52.0s | 2 MB | 8 MB | +0.399 | +0.713 | **+0.525** |
| scVI | 43.2s | 90ms | 58.9s | — | 354 MB | +0.607 | +0.962 | **+0.749** |

### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCA | 2.3s | 48ms | 3.9m | 456 MB | 8 MB | +0.464 | +0.876 | **+0.629** |
| NMF | 2.0m | 2.6s | 4.3m | — | 8 MB | +0.359 | +0.821 | **+0.543** |
| sparseNMF | 6.4m | — | 4.1m | — | 2.1 GB | +0.626 | +0.908 | **+0.739** |
| Harmony | 2.2s | — | 4.6m | — | 8 MB | +0.461 | +0.886 | **+0.631** |
| scVI | 56.2s | 101ms | 3.5m | — | 234 MB | +0.708 | +0.991 | **+0.821** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| scVI | **+0.785** | 49.7s | 294 MB |
| sparseNMF | **+0.720** | 5.6m | 3.1 GB |
| PCA | **+0.608** | 2.3s | 8 MB |
| Harmony | **+0.578** | 2.5s | 8 MB |
| NMF | **+0.575** | 1.7m | 8 MB |
