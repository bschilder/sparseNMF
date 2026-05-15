### immune

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 4.9s | — | 1.7m | — | 519 MB | +0.657 | +0.494 | **+0.592** |

### lung

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 6.4s | — | 1.6m | — | 355 MB | +0.635 | +0.471 | **+0.570** |

### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 3.4s | — | 1.4m | — | 226 MB | +0.766 | +0.541 | **+0.676** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 2.5s | — | 1.2m | — | 172 MB | +0.662 | +0.347 | **+0.536** |

### sim2

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF+batch | 5.6s | — | 1.3m | — | 276 MB | +0.664 | +0.243 | **+0.496** |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF+batch | **+0.574** | 4.6s | 310 MB |
