"""MCP server config helper for Strata.

Strata ships an MCP server so AI assistants can install and manage the engine
itself. This module emits the JSON config an MCP client (Claude Desktop,
Cursor, etc.) expects to discover the local server.
"""


class StrataMCPConfig:
    def __init__(self, host="127.0.0.1", port=8080):
        self.host = host
        self.port = port

    def server_config(self, name="strataforge"):
        return {
            "mcpServers": {
                name: {
                    "command": "python",
                    "args": ["-m", "strataforge.mcp_server"],
                    "env": {"STRATA_HOST": self.host, "STRATA_PORT": str(self.port)},
                }
            }
        }

    def to_json(self):
        import json
        return json.dumps(self.server_config(), indent=2)
