### pancreas

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | 2.7m | — | 7.3m | — | 6.6 GB | +0.745 | +0.559 | **+0.671** |

### sim1

| method | fit | infer | metrics | RSS Δ | GPU peak | bio ↑ | batch ↑ | composite ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparseNMF | — | — | — | — | — | `OutOfMemoryError: CUDA out of memory. Tr` | — | — |

### Cross-dataset composite average

| method | mean composite ↑ | mean fit | mean GPU peak |
|---|---:|---:|---:|
| sparseNMF | **+0.671** | 2.7m | 6.6 GB |
