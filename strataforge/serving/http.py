"""OpenAI/Anthropic-compatible stdlib HTTP server.

Zero dependencies: uses http.server. Exposes:

    GET  /health              -> {"status": "ok"}
    GET  /v1/models           -> OpenAI model list
    POST /v1/chat/completions -> OpenAI chat completion
    POST /v1/messages         -> Anthropic messages (for Claude Code / Cursor)
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..model.backends import get_backend


class _Handler(BaseHTTPRequestHandler):
    backend = None

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            return self._send(200, {"status": "ok", "engine": "strataforge"})
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": "qwen3.8-flash-next", "object": "model",
                 "owned_by": "strataforge"}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception as e:
            return self._send(400, {"error": "bad json: %s" % e})

        if self.path == "/v1/chat/completions":
            msgs = body.get("messages", [])
            prompt = "".join(m.get("content", "") for m in msgs) or "hello"
            max_tokens = body.get("max_tokens", 64)
            res = self.backend.generate(prompt, max_tokens=max_tokens)
            return self._send(200, {
                "id": "chatcmpl-strataforge",
                "object": "chat.completion",
                "model": body.get("model", "qwen3.8-flash-next"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": res["text"]},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(str(prompt)), "completion_tokens": len(res["tokens"])},
            })

        if self.path == "/v1/messages":
            prompt = body.get("prompt") or body.get("messages")
            if isinstance(prompt, list):
                prompt = "".join(m.get("content", "") for m in prompt)
            max_tokens = body.get("max_tokens", 64)
            res = self.backend.generate(prompt or "hello", max_tokens=max_tokens)
            return self._send(200, {
                "id": "msg_strataforge",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": res["text"]}],
                "model": body.get("model", "qwen3.8-flash-next"),
                "stop_reason": "end_turn",
            })

        return self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


def create_app(cfg, backend=None):
    class Bound(_Handler):
        pass
    Bound.backend = backend or get_backend(cfg.backend.name, cfg)
    return Bound


def serve(cfg, host="127.0.0.1", port=8080):
    handler = create_app(cfg)
    httpd = HTTPServer((host, port), handler)
    print("strataforge serving on http://%s:%d" % (host, port))
    print("  OpenAI    POST /v1/chat/completions")
    print("  Anthropic POST /v1/messages")
    print("  health    GET  /health")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
