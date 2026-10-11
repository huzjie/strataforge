"""Optional integrations (LangChain, MCP)."""
from .langchain import StrataChatModel
from .mcp import StrataMCPConfig

__all__ = ["StrataChatModel", "StrataMCPConfig"]
