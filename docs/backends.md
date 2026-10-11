# Backends

Five backends share one interface (`generate(prompt, **kw)`):

| Backend | Description |
|---|---|
| `mock` | deterministic, **trainable** (skill 0.5 -> 1.0) -- reference target |
| `cpu` | same math, non-trainable, local reference |
| `openai` | any OpenAI-compatible endpoint (stdlib urllib) |
| `vllm` | vLLM OpenAI-compatible completions endpoint |
| `transformers` | HuggingFace pipeline (lazy import) |

Select via `backend.name` in config or `--backend mock` on the CLI.
