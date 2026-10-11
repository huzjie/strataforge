# Recipe: Multi-GPU layer split

```yaml
tier: { gpu_capacity_experts: 16000 }
quant: { level: iq2_xs }
```

Two 12GB cards: double the hot tier and split the layer range across GPUs.
