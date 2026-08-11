"""Real MCP (Model Context Protocol) server for QSH — full deployment only.

A FastMCP server exposes the 5 per-industry tools, bound to the project's quantum-safe
(PQC+QKD+ML-DSA+AEAD) transport via :mod:`qsh.pqc.mcp_bridge`. The public demo keeps the
client-side ``clientMcp.ts`` mirror instead, so the demo VPS takes zero per-user MCP load.
"""

from .server import TOOL_DOCS, TOOL_FUNCS, build_server, dispatch_tool, list_tools

__all__ = ["TOOL_FUNCS", "TOOL_DOCS", "build_server", "dispatch_tool", "list_tools"]
