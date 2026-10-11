# Quantization Tiers

Low-bit weights are what make a 125B model physically fit consumer RAM. We
model five ggml-style levels:

| Level | bits/weight | 125B on-disk | Use case |
|---|---|---|---|
| q2_0 | 2.0 | ~31 GB | fastest, most aggressive |
| iq2_xs | 2.3 | ~36 GB | balanced default |
| iq3_xxs | 3.0 | ~47 GB | better quality |
| iq3_s | 3.4 | ~53 GB | best 3-bit, long context |
| coder | 3.2 | ~50 GB | code-tuned mix |

## What the scheduler does with it

The memory budget decides placement: `active_params` (attention + routing +
hot experts) must fit GPU, `total_params` must fit RAM, and the n-gram table
sizes to SSD. `core/memory.tier_layout()` computes the exact split and returns
a `fits` boolean.

See `python -m strataforge quantize` for the table computed live from config.
