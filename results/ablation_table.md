| Variant | Final val loss | Avg of last 3 evals | Perplexity | vs control |
|---|---|---|---|---|
| Control (RMSNorm+RoPE+SwiGLU) | 1.609 | 1.608 | 4.99 | +0.000 |
| Control, seed 1 (noise check) | 1.613 | 1.617 | 5.04 | +0.009 |
| LayerNorm instead of RMSNorm | 1.615 | 1.619 | 5.05 | +0.011 |
| Learned positions instead of RoPE | 1.626 | 1.629 | 5.10 | +0.021 |
| GELU instead of SwiGLU | 1.620 | 1.624 | 5.07 | +0.016 |
