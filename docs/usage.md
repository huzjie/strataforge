# Usage

```bash
pip install -e .            # or: pip install .

python -m strataforge doctor     # self-check all backends
python -m strataforge train      # train the mock skill 0.5 -> 1.0
python -m strataforge bench      # run the benchmark suite
python -m strataforge quantize   # quantization tier table
python -m strataforge run --prompt "hello" --max-tokens 64
python -m strataforge serve --host 0.0.0.0 --port 8080
```

## Configuration

Edit `configs/default.yaml` (or pass `--config path.yaml`). Key knobs:

```yaml
model:
  n_experts: 24576
  top_k: 10
tier:
  gpu_capacity_experts: 4000
spec:
  draft_len: 2
quant:
  level: iq2_xs
backend:
  name: mock
```

## Docker

```bash
docker build -t strataforge .
docker run -p 8080:8080 strataforge
```
