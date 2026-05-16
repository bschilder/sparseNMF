### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 12.1s | — | 6.1m | — | 593 MB | +0.662 | +0.576 | **+0.628** |

### lung

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 13.2s | — | 8.0m | — | 445 MB | +0.553 | +0.531 | **+0.544** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 7.1s | — | 7.0m | — | 282 MB | +0.723 | +0.566 | **+0.660** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 5.7s | — | 4.1m | — | 214 MB | +0.726 | +0.546 | **+0.654** |

### sim2

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 7.0s | — | 6.5m | — | 337 MB | +0.629 | +0.415 | **+0.544** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF_supervised | **+0.606** | 9.0s | 374 MB |
