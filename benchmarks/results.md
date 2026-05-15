### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Harmony | 2.0s | — | 17.4s | — | — | +0.749 | +0.578 | **+0.681** |
| NMF | 19.1s | 0.7s | 17.1s | — | — | +0.675 | +0.427 | **+0.576** |
| PCA | 1.0s | 11ms | 40.9s | — | — | +0.668 | +0.436 | **+0.575** |
| scVI | 1.2m | 85ms | 15.2s | — | 52 MB | +0.756 | +0.610 | **+0.698** |
| sparseNMF | 3.4m | — | 38.8s | — | 372 MB | +0.698 | +0.531 | **+0.632** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Harmony | 2.0s | — | 10.0s | — | — | +0.860 | +0.689 | **+0.792** |
| NMF | 16.5s | 427ms | 9.8s | — | — | +0.708 | +0.446 | **+0.604** |
| PCA | 1.9s | 11ms | 44.0s | — | — | +0.753 | +0.470 | **+0.640** |
| scVI | 52.6s | 57ms | 11.0s | — | 52 MB | +0.844 | +0.557 | **+0.729** |
| sparseNMF | 51.3s | — | 1.0m | — | 445 MB | +0.794 | +0.588 | **+0.712** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| Harmony | **+0.736** | 2.0s | — |
| scVI | **+0.713** | 1.0m | 52 MB |
| sparseNMF | **+0.672** | 2.1m | 409 MB |
| PCA | **+0.608** | 1.4s | — |
| NMF | **+0.590** | 17.8s | — |
