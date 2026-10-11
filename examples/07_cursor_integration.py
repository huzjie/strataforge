"""strataforge example: Cursor integration

Run from the project root:
    python examples/07_cursor_integration.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


"""Point Cursor / Claude Code at the local OpenAI-compatible server.

Start the server first:
    python examples/06_serve_http.py
Then in Cursor / Claude Code set:
    ANTHROPIC_BASE_URL=http://127.0.0.1:8080
    (or OpenAI base URL http://127.0.0.1:8080/v1 for OpenAI-compatible tools)
"""
import json
import urllib.request

body = json.dumps({"messages": [{"role": "user", "content": "What is tiered inference?"}],
                   "model": "qwen3.8-flash-next", "max_tokens": 32}).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print(r.read().decode("utf-8"))
except Exception as e:
    print("server not running -- start it with: python examples/06_serve_http.py")
    print("error:", e)
