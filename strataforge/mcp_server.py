"""Standalone MCP server entrypoint (minimal, stdlib JSON-RPC over stdio).

Run `python -m strataforge.mcp_server` to expose a tiny MCP server with a
single `strataforge.status` tool. Real deployments can point it at the local
HTTP server via STRATA_HOST / STRATA_PORT.
"""
import json
import os
import sys


def _tool_status():
    return {
        "jsonrpc": "2.0",
        "result": {
            "content": [{"type": "text", "text": json.dumps({
                "engine": "strataforge",
                "host": os.environ.get("STRATA_HOST", "127.0.0.1"),
                "port": int(os.environ.get("STRATA_PORT", "8080")),
            })}],
        },
    }


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("method") == "tools/list":
            resp = {"jsonrpc": "2.0", "id": msg.get("id"), "result": {"tools": [
                {"name": "strataforge.status", "description": "Report the local Strata server status",
                 "inputSchema": {"type": "object", "properties": {}}}]}}
        elif msg.get("method") == "tools/call":
            resp = _tool_status()
            resp["id"] = msg.get("id")
        else:
            resp = {"jsonrpc": "2.0", "id": msg.get("id"), "result": {}}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
