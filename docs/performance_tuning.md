# Performance Tuning

- **Hot-tier size**: bigger GPU tier -> higher hit-rate, but watch VRAM budget.
- **Draft length**: longer draft helps *only* if acceptance stays high; monitor
  `speedup` not `draft_len`.
- **Quant level**: lower bits -> smaller memory + faster, at some quality cost.
  `coder` is the sweet spot for coding workloads.
- **n-gram n**: `n=3` balances table size vs reuse; larger `n` = fewer hits.
