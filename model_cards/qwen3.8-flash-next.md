# Qwen3.8-Flash-Next (125B MoE)

- **Total params:** 125B (24,576 experts)
- **Active per token:** ~10 experts (~6.2B)
- **Quantization:** GSQ-RCO (ISTA-DASLab)
- **Target hardware:** RTX 20/30/40/50, AMD RX 9070 XT, 12GB+ VRAM, 32-64GB RAM
- **Measured (RTX 5070 12GB):** Q2_0 94 tok/s gen, 2,650 tok/s prefill
- **Speculative:** MTP draft, 1.6-1.8x end-to-end

This is the reference model strataforge models. Note: strataforge does not
redistribute weights; use the upstream GGUF from the Strata project.
