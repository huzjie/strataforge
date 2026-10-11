# Benchmarks

Run `python -m strataforge bench`. Every number is deterministic.

| Benchmark | Measures | Trains with skill? |
|---|---|---|
| generation | decode tokens/sec | yes |
| prefill | prompt-processing tokens/sec + n-gram hit rate | yes |
| speculation | speedup vs no-spec baseline | yes |
| tier_hitrate | GPU hit rate / resident experts | yes |
| memory | quantized size + budget fit | no |

## Reference numbers (Strata, RTX 5070 12GB / 64GB RAM)

- Q2_0: 94 tok/s generation, 2,650 tok/s prefill
- IQ3_S: 53 tok/s generation, 1,620 tok/s prefill
- Speculative decoding: 1.6-1.8x end-to-end
- ~4,000 / 24,576 experts resident on GPU
