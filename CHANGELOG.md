# Changelog

## [1.0.0] - 2026-10-11

### Added
- Sparse MoE layer with deterministic Top-K hash routing + load-balance aux loss
- Tiered expert scheduler (GPU hot tier / RAM full set / SSD n-gram table)
- LRU eviction policy for the GPU hot-expert tier
- n-gram lookup table for prompt-processing reuse
- Speculative decoding (MTP draft + verifier) with monotone speedup math
- Five backends: mock (trainable) / cpu / openai / vllm / transformers
- OpenAI + Anthropic compatible stdlib HTTP server
- Benchmark suite: generation / prefill / speculation / tier hit-rate / memory
- CLI (doctor / train / bench / serve / quantize / run)
- Docker / docker-compose / Kubernetes / Helm / GitHub Actions CI
- Prompt library + recipes + architecture docs
