"""Real MCP server (full deployment): the 5 per-industry tools, dispatch over the quantum-safe channel,
and interop with the official filesystem MCP server under the PQC+QKD+ML-DSA+AEAD layer."""

import pytest

pytest.importorskip("mcp")

from qsh.mcp.server import build_server, dispatch_tool, list_tools
from qsh.pqc.mcp_bridge import Agent, secure_agent_call

TOOLS = {"score_transaction", "query_cohort", "pool_allele_counts",
         "inspect_qkd_telemetry", "relay_mission_telemetry"}


def test_mcp_server_lists_five_tools():
    tools = list_tools()
    assert {t["name"] for t in tools} == TOOLS
    assert all(t["description"] for t in tools)        # every tool documented


def test_fastmcp_server_builds():
    server = build_server()                            # a real FastMCP server object
    assert server.name == "qsh-quantum-safe"


def test_dispatch_pool_allele_counts_is_real_qsms():
    r = dispatch_tool("pool_allele_counts", {"counts": [10, 20, 30]})
    assert r["pooled_allele_count"] == 60              # genuine secure summation
    assert r["n_cohorts"] == 3


def test_inspect_qkd_telemetry_detects_eavesdropper():
    assert dispatch_tool("inspect_qkd_telemetry", {"qber": 0.25})["eavesdropper_detected"] is True
    assert dispatch_tool("inspect_qkd_telemetry", {"qber": 0.02})["eavesdropper_detected"] is False


def test_score_transaction_flags_account_draining_transfer():
    r = dispatch_tool("score_transaction", {"amount": 9000, "old_balance": 9000, "new_balance": 0})
    assert r["flag"] is True


def test_secure_agent_call_runs_real_tool_over_quantum_safe_channel():
    a, b = Agent.create("analyst"), Agent.create("genomics-silo")
    try:
        r = secure_agent_call(a, b, "tools/call",
                              {"name": "pool_allele_counts", "arguments": {"counts": [1, 2, 3, 4]}})
    finally:
        a.free()
        b.free()
    assert r["authenticated"] is True
    assert r["tamper_rejected"] is True
    assert r["result_verified"] is True                # callee's ML-DSA signature on the result verified
    assert r["tool"] == "pool_allele_counts"
    assert r["result"]["pooled_allele_count"] == 10    # the REAL result, end-to-end over PQC+QKD


@pytest.mark.slow
def test_filesystem_mcp_interop_over_quantum_safe_channel(tmp_path):
    """Interop with the OFFICIAL @modelcontextprotocol/server-filesystem under the quantum-safe seal."""
    from qsh.mcp.fs_interop import node_available, secure_filesystem_interop

    if not node_available():
        pytest.skip("Node/npx not available — filesystem-MCP interop needs Node")
    (tmp_path / "hello.txt").write_text("hi")
    r = secure_filesystem_interop(str(tmp_path), tool="list_directory",
                                  arguments={"path": str(tmp_path)}, timeout=90)
    assert "read_file" in r["external"]["tools"]       # the official server's real tool catalogue
    assert "hello.txt" in r["external"]["result_text"]
    assert r["quantum_safe"]["authenticated"] is True
    assert r["quantum_safe"]["tamper_rejected"] is True
