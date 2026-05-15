### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 4.7s | — | 1.7m | — | 519 MB | +0.637 | +0.490 | **+0.578** |

### lung

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 5.7s | — | 1.7m | — | 355 MB | +0.634 | +0.500 | **+0.580** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 3.2s | — | 1.3m | — | 226 MB | +0.774 | +0.531 | **+0.677** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 2.5s | — | 1.2m | — | 172 MB | +0.661 | +0.281 | **+0.509** |

### sim2

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 2.8s | — | 1.3m | — | 276 MB | +0.665 | +0.248 | **+0.498** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF+batch | **+0.568** | 3.8s | 310 MB |
