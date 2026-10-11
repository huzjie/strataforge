# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| decode slow | low tier hit-rate | raise `gpu_capacity_experts` or `skill` |
| low speedup | low draft acceptance | raise `acceptance_base` / `acceptance_gain` |
| OOM | quant level too high | drop to `q2_0` |
| first token slow | n-gram cold start | pre-warm the table (tutorial 6) |
