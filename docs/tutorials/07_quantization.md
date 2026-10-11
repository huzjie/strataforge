# Tutorial 7: Quantization

```bash
python -m strataforge quantize
```

See the five levels and their on-disk size for 125B params. Then flip
`config.quant.level` and watch the memory benchmark change.
