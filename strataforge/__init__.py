"""strataforge: heterogeneous tiered inference engine for consumer hardware.

A zero-dependency, fully runnable reference implementation of the engineering
ideas behind Strata -- the open-source inference engine that runs a ~125B
parameter sparse MoE (Qwen3.8-Flash-Next, 24,576 experts, ~10 active per token)
on an ordinary gaming PC with a 12GB GPU, by tiering experts across GPU / RAM /
SSD and adding speculative decoding.

The whole computer is treated as one unified multi-tier memory hierarchy:

    GPU  (12-24GB)  -> attention + routing + the ~4,000 hottest experts
    RAM  (32-64GB)  -> all 24,576 experts; CPU computes the ones missing on GPU
    SSD  (29GB)     -> a large n-gram lookup table, read a few rows per token

Three core techniques, distilled into reusable, runnable code:

1. Tiered expert scheduling -- a hot/cold LRU placement policy keeps the most
   frequently routed experts resident on the GPU while the CPU computes the
   cold tail in system RAM, so no tier sits idle.
2. Speculative decoding (MTP) -- a tiny draft head guesses a couple of next
   tokens; the main model verifies them in one forward pass, for a ~1.6x
   end-to-end speedup with byte-identical output.
3. Quantization tiers -- Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S / Coder levels model
   ggml-style low-bit weights, driving the memory-footprint math.

It ships with a deterministic trainable mock backend, five backends in total,
an OpenAI/Anthropic-compatible stdlib HTTP server, a throughput benchmark
suite, and CLI / Docker / K8s / Helm / CI scaffolding.

Highlights
----------
- Zero-dependency pure-Python core (MoE + tiering + speculation + quantize)
- Deterministic trainable mock backend (skill 0.5 -> 1.0, acceptance + hit-rate
  + throughput all monotonically climb)
- Speculative-decoding speedup benchmark (1.52x -> 1.68x)
- Tier hit-rate benchmark (GPU hit-rate 0.75 -> 1.00)
- OpenAI + Anthropic compatible stdlib HTTP server
- CLI / Docker / K8s / Helm / CI + prompts + recipes + docs
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
