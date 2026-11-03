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

#: The official filesystem server, pinned (Round 85). `npx -y` without a version ran whatever npm's
#: `latest` was at call time. 2.0.0 and 2.1.0 (2026-10-19, 2026-10-28) declare `engines: node >= 24`,
#: which the API image's Debian Node 20 did not meet, so Round 85 pinned 1.0.0. Round 86 gave the image
#: Node 24.22.2 LTS from the official image (`docker/Dockerfile.api`) and moved to 2.1.0, npm's newest,
#: which depends on `@modelcontextprotocol/server` ^2.3.1. Its tools keep their names; `read_file` is
#: deprecated in favour of `read_text_file`, and allowed directories are still given on the command line.
#: It is also a security fix, which Round 85's pin was not: four advisories of 2026-10-25 hit every
#: release from 1.0.0 below 2.0.1 (GHSA-4gxg-m5jx-9j57, a case-variant path creating files outside the
#: allowed directories; GHSA-7rgq-gccq-qphx, path traversal through a trailing quote or whitespace;
#: GHSA-53mr-w429-c597 and GHSA-hhwq-wxfx-cj7g). They are listed in the servers repository's own
#: advisories and not in GitHub's global database, so a dependency audit did not see them.
FILESYSTEM_SERVER = "@modelcontextprotocol/server-filesystem@2.1.0"


def _call_outcome(call) -> tuple[str, bool]:
    """The text a tool result carries, and whether the server flagged it an error.

    The flag is `is_error` in the MCP Python SDK 2.x and was `isError` in 1.x (Round 86): reading only
    `isError` on 2.x reported every error the official server returned as a success.
    """
    text = "\n".join(getattr(c, "text", "") for c in (call.content or []) if getattr(c, "type", "") == "text")
    flag = getattr(call, "is_error", None)
    if flag is None:
        flag = getattr(call, "isError", False)
    return text, bool(flag)


def node_available() -> bool:
    """True if ``npx`` is on PATH (the filesystem-MCP server is a Node package)."""
    return shutil.which("npx") is not None


async def _filesystem_mcp_call(directory: str, tool: str, arguments: dict, timeout: float) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command="npx", args=["-y", FILESYSTEM_SERVER, directory])

    async def _run() -> dict:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                tool_names = [t.name for t in listed.tools]
                call = await session.call_tool(tool, arguments)
                text, is_error = _call_outcome(call)
                # The revision the handshake agreed: a low-level `initialize()` performs the pre-2026
                # handshake, so this says which era the interop exercised (Round 86).
                return {"server": FILESYSTEM_SERVER, "tools": tool_names, "tool": tool,
                        "result_text": text[:2000], "is_error": is_error,
                        "protocol_version": session.protocol_version}

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
