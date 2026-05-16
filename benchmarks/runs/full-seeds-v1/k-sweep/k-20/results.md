### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 3.3m | — | 1.6m | — | 403 MB | +0.686 | +0.515 | **+0.618** |

### lung

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 3.0m | — | 1.5m | — | 442 MB | +0.693 | +0.467 | **+0.603** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 1.7m | — | 1.1m | — | 433 MB | +0.714 | +0.496 | **+0.627** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 4.7m | — | 1.3m | — | 1.1 GB | +0.670 | +0.192 | **+0.479** |

### sim2

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 8.4m | — | 1.6m | — | 1.5 GB | +0.671 | +0.211 | **+0.487** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF | **+0.563** | 4.2m | 784 MB |
