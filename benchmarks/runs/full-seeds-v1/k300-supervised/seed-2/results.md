### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 12.4s | — | 6.4m | — | 593 MB | +0.652 | +0.564 | **+0.617** |

### lung

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 13.3s | — | 8.2m | — | 445 MB | +0.570 | +0.514 | **+0.547** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 7.1s | — | 7.4m | — | 282 MB | +0.751 | +0.589 | **+0.686** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 5.9s | — | 4.1m | — | 214 MB | +0.747 | +0.549 | **+0.667** |

### sim2

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 7.0s | — | 7.0m | — | 335 MB | +0.630 | +0.413 | **+0.543** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF_supervised | **+0.612** | 9.1s | 374 MB |
