### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 2.9m | — | 13.5m | — | 12.9 GB | +0.737 | +0.540 | **+0.658** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | — | — | — | — | — | `OutOfMemoryError: CUDA out of memory. Tr` | — | — |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF | **+0.658** | 2.9m | 12.9 GB |
