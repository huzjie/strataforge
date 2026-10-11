# HTTP API

The server is OpenAI- and Anthropic-compatible, so it drops into Cursor,
Claude Code, and any OpenAI SDK without changes.

## Endpoints

- `GET /health` -> `{"status": "ok"}`
- `GET /v1/models` -> model list
- `POST /v1/chat/completions` -> OpenAI chat completion
- `POST /v1/messages` -> Anthropic messages

## OpenAI example

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

## Claude Code / Cursor

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
```

## Run it

```bash
python -m strataforge serve --host 0.0.0.0 --port 8080
```
