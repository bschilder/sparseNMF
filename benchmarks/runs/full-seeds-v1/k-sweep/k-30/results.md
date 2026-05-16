### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 3.2m | — | 1.6m | — | 455 MB | +0.691 | +0.516 | **+0.621** |

### lung

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 3.0m | — | 1.5m | — | 449 MB | +0.678 | +0.469 | **+0.594** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 1.7m | — | 1.1m | — | 469 MB | +0.773 | +0.510 | **+0.668** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 4.6m | — | 1.3m | — | 1.2 GB | +0.660 | +0.238 | **+0.491** |

### sim2

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 8.4m | — | 1.6m | — | 1.5 GB | +0.661 | +0.240 | **+0.493** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF | **+0.573** | 4.2m | 831 MB |
