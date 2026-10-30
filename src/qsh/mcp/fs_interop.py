"""Interop with a real EXTERNAL MCP server under the quantum-safe layer (full deployment).

Connects to the OFFICIAL ``@modelcontextprotocol/server-filesystem`` (Node, via ``npx``) using the MCP
Python SDK's stdio client, invokes a real tool (e.g. ``list_directory``), and then carries the genuine
result through the project's quantum-safe channel (PQC + ML-DSA + AEAD, optionally mixed with a
caller-supplied key; the API passes a fixed 32-byte constant, not a QKD key) via
:func:`qsh.pqc.mcp_bridge.secure_carry`. This proves interop with a real third-party MCP server while
keeping the transport quantum-safe — no trusted third party, no plaintext on the wire.

Requires Node/``npx`` (present in the full Docker image). Callers should guard with :func:`node_available`.
"""

from __future__ import annotations

import asyncio
import shutil


def node_available() -> bool:
    """True if ``npx`` is on PATH (the filesystem-MCP server is a Node package)."""
    return shutil.which("npx") is not None


async def _filesystem_mcp_call(directory: str, tool: str, arguments: dict, timeout: float) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", directory],
    )

    async def _run() -> dict:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                tool_names = [t.name for t in listed.tools]
                call = await session.call_tool(tool, arguments)
                text = "\n".join(
                    getattr(c, "text", "") for c in (call.content or []) if getattr(c, "type", "") == "text"
                )
                return {"server": "@modelcontextprotocol/server-filesystem", "tools": tool_names,
                        "tool": tool, "result_text": text[:2000], "is_error": bool(getattr(call, "isError", False))}

    return await asyncio.wait_for(_run(), timeout=timeout)


def secure_filesystem_interop(directory: str = ".", tool: str = "list_directory",
                              arguments: dict | None = None, *, qkd_key: bytes | None = None,
                              timeout: float = 45.0) -> dict:
    """Call the official filesystem MCP server, then carry the real result over the quantum-safe channel.

    Returns the external server's tool list + result, plus the quantum-safe envelope report
    (authenticated / tamper_rejected / sealed_bytes) showing the result rode the PQC + ML-DSA layer
    (with ``qkd_key`` mixed in when one is given).
    """
    if not node_available():
        raise RuntimeError("npx (Node) not available — the filesystem-MCP interop needs Node")

    from ..pqc.mcp_bridge import secure_carry

    arguments = arguments if arguments is not None else {"path": directory}
    external = asyncio.run(_filesystem_mcp_call(directory, tool, arguments, timeout))
    envelope = secure_carry(external, qkd_key=qkd_key, callee_name="filesystem-mcp")
    return {
        "external": external,
        "quantum_safe": {k: envelope[k] for k in
                         ("kem", "qkd_mixed", "authenticated", "signature_alg", "tamper_rejected", "sealed_bytes")},
        "verified_result": envelope["payload"],     # the external result, after open+verify
    }
