# Recipe: 24GB GPU budget

```yaml
tier: { gpu_capacity_experts: 8000, gpu_bytes: 25769803776, ram_bytes: 68719476736, ssd_bytes: 31138512896 }
quant: { level: iq3_s }
spec: { draft_len: 3 }
```

Larger hot tier + 3-bit + 3-token draft for higher quality and ~1.7x spec.
