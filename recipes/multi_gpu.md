# Recipe: Multi-GPU split

```yaml
tier: { gpu_capacity_experts: 16000 }
quant: { level: iq2_xs }
```

Two 12GB cards: double the hot tier, split the layer range.
