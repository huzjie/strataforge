# FAQ

**Why zero dependencies?**
So the reference core runs anywhere (managed Python 3.13 with no PyYAML /
numpy) and the economics are fully auditable. The heavy backends (openai /
vllm / transformers) are optional extras.

**Does it really run a 125B model?**
The reference build uses small dimensions so it runs on CPU in seconds. The
*real* economics -- 125B params, 24,576 experts, 10 active, 3-tier memory,
quantization, speculative speedup -- are modeled exactly by the memory budget,
tier scheduler and benchmarks, which is what you need to reason about and
prototype the placement policy before wiring real ggml kernels.

**Why is the mock trainable?**
To give a clean, reproducible signal: as `skill` rises, acceptance, tier
hit-rate, n-gram reuse and throughput all climb monotonically. That makes the
framework's "better scheduling -> faster" claim verifiable end-to-end.
