| Variant | Seed | Final val loss | Avg of last 3 evals | Perplexity | vs matching control |
|---|---|---|---|---|---|
| Control (RMSNorm + RoPE + SwiGLU) | 0 | 1.609 | 1.619 | 5.05 | n/a |
| Control (RMSNorm + RoPE + SwiGLU) | 1 | 1.613 | 1.617 | 5.04 | n/a |
| LayerNorm instead of RMSNorm | 0 | 1.615 | 1.619 | 5.05 | +0.000 |
| LayerNorm instead of RMSNorm | 1 | 1.612 | 1.616 | 5.03 | -0.001 |
| Learned positions instead of RoPE | 0 | 1.626 | 1.629 | 5.10 | +0.011 |
| Learned positions instead of RoPE | 1 | 1.624 | 1.625 | 5.08 | +0.008 |
| GELU instead of SwiGLU | 0 | 1.620 | 1.624 | 5.07 | +0.005 |
| GELU instead of SwiGLU | 1 | 1.620 | 1.624 | 5.07 | +0.007 |
