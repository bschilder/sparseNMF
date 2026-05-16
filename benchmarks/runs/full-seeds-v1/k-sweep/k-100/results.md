### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 3.3m | — | 4.9m | — | 827 MB | +0.679 | +0.546 | **+0.626** |

### lung

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 3.0m | — | 4.8m | — | 819 MB | +0.659 | +0.482 | **+0.588** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 1.7m | — | 1.9m | — | 947 MB | +0.797 | +0.581 | **+0.711** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 4.7m | — | 1.8m | — | 2.7 GB | +0.642 | +0.545 | **+0.603** |

### sim2

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 8.6m | — | 2.7m | — | 3.1 GB | +0.637 | +0.325 | **+0.512** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF | **+0.608** | 4.2m | 1.7 GB |
