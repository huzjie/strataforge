# Glossary

- **MoE** — Mixture-of-Experts: many experts, few active per token.
- **Top-K routing** — send each token to the K highest-scoring experts.
- **Tiered scheduling** — place data across GPU/RAM/SSD by access heat.
- **Speculative decoding** — draft + verify to emit >1 token per pass.
- **MTP** — multi-token prediction draft head.
- **LRU** — least-recently-used eviction.
- **bpw** — bits per weight (quantization measure).
