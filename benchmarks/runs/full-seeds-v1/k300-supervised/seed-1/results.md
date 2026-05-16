### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 12.2s | — | 6.2m | — | 593 MB | +0.659 | +0.569 | **+0.623** |

### lung

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 13.4s | — | 8.5m | — | 445 MB | +0.549 | +0.513 | **+0.535** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 7.0s | — | 7.1m | — | 282 MB | +0.718 | +0.585 | **+0.664** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 5.6s | — | 4.3m | — | 214 MB | +0.739 | +0.545 | **+0.662** |

### sim2

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF_supervised | 7.1s | — | 6.8m | — | 335 MB | +0.630 | +0.420 | **+0.546** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF_supervised | **+0.606** | 9.0s | 374 MB |
