### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 4.7s | — | 1.7m | — | 519 MB | +0.660 | +0.515 | **+0.602** |

### lung

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 5.7s | — | 1.7m | — | 355 MB | +0.629 | +0.487 | **+0.572** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 3.2s | — | 1.2m | — | 226 MB | +0.755 | +0.508 | **+0.656** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 2.5s | — | 1.2m | — | 172 MB | +0.662 | +0.289 | **+0.513** |

### sim2

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 2.8s | — | 1.3m | — | 276 MB | +0.663 | +0.240 | **+0.494** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF+batch | **+0.567** | 3.8s | 310 MB |
