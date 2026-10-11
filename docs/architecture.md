# Architecture

strataforge treats the whole computer as a **unified multi-tier memory hierarchy**
and routes work to whichever tier is cheapest for the current token.

## Component graph

```
                    +---------------------------+
   prompt tokens -> |  n-gram lookup (SSD)      |  prompt-processing reuse
                    +---------------------------+
                                |
                                v
   +-----------------+   +----------------------+   +------------------+
   |  Router (Top-K) |-->|  Tiered scheduler    |-->|  GPU hot tier    |
   |  24,576 experts |   |  (freq -> LRU promo) |   |  ~4,000 experts  |
   +-----------------+   +----------------------+   +------------------+
                                |  (miss)
                                v
   +-----------------+   +----------------------+   +------------------+
   |  SwiGLU experts |<--|  CPU computes tail   |   |  RAM full set    |
   +-----------------+   +----------------------+   |  24,576 experts  |
                                |
                                v
   +-----------------+   +----------------------+
   |  MTP draft head |-->|  Main-model verifier |  (speculative decode)
   +-----------------+   +----------------------+
```

## Modules

| Module | Responsibility |
|---|---|
| `core/tensor.py` | zero-dep linear algebra (matmul/softmax/layernorm) |
| `core/quantize.py` | bit-budget per quant level (Q2_0..Coder) |
| `core/memory.py` | three-tier byte budget + fit check |
| `experts/moe.py` | sparse MoE + deterministic Top-K routing |
| `experts/tier.py` | hot/cold placement + LRU eviction |
| `experts/ngram.py` | SSD n-gram table for prompt reuse |
| `draft/mtp.py` + `draft/spec.py` | speculative decoding |
| `engine/strata_engine.py` | wires everything into a token loop |
| `model/backends.py` | mock/cpu/openai/vllm/transformers |
| `serving/http.py` | OpenAI/Anthropic stdlib server |
