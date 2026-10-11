# Tutorial 5: Serve + integrate

```bash
python -m strataforge serve --port 8080
# in another shell:
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"hi"}]}'
```
