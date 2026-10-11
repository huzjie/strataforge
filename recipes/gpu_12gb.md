# Recipe: 12GB GPU budget

```yaml
tier: { gpu_capacity_experts: 4000, gpu_bytes: 12884901888, ram_bytes: 68719476736, ssd_bytes: 31138512896 }
quant: { level: iq2_xs }
spec: { draft_len: 2 }
```

Targets ~79 tok/s generation (RTX 5070 class). Hot tier 4,000 experts.
