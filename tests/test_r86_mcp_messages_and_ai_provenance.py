"""The channel's messages are MCP 2026-07-28, and the panel's download carries a record that reads back.

WHY THIS EXISTS (Round 86, found by the fact-check sweep of 2026-11-03, the owner's two choices)

The channel signed `{"jsonrpc", "method", "params"}` with no `id` and no `_meta`, and replied with a
request-shaped `"tools/result"`. REFERENCES.md said it "depends on none of" revision 2026-07-28, which
requires an id on every request, `_meta` fields on every request and `resultType` on every result, and
requires a custom transport to keep the JSON-RPC format and the per-request metadata model.

The Agentic AI panel's JSON download carried generated text while the gap analysis said only a future
"download as PDF" button would change the Article 50(2) answer, and the file's own note said free-form
text "cannot carry metadata". The Code of Practice's Sub-measure 1.1.1 applies to content "exported in a
data format that supports attaching metadata", and its very-short-text exception is written into
Sub-measure 1.1.2 only.

WHAT IS ASSERTED

- A request and every kind of response validate against the MCP SDK's own 2026-07-28 types.
- The server side refuses what the revision says it must refuse, with the codes it names.
- The channel still verifies both directions, and a tool error is not a protocol error.
- `scripts/verify_ai_provenance.py` reads a file the production browser module signed
  (`tests/fixtures/ai_provenance_sample.json`, also read by `web/scripts/check_mcp_envelope.mjs`) and
  refuses each edit.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "tests" / "fixtures" / "ai_provenance_sample.json"

oqs = pytest.importorskip("oqs")
from qsh.pqc.mcp_bridge import (  # noqa: E402
    MCP_PROTOCOL_VERSION,
    Agent,
    mcp_request,
    mcp_tool_result,
    message_body,
    secure_agent_call,
    secure_carry,
    serve_tool_call,
)

PROTOCOL = "io.modelcontextprotocol/protocolVersion"
CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"


def _tool_call(name="query_cohort", arguments=None):
    return mcp_request("analyst", "tools/call", {"name": name, "arguments": arguments or {"phenotype": "x"}})


def test_a_request_is_what_the_sdk_calls_a_tool_call() -> None:
    types = pytest.importorskip("mcp.types")
    assert types.LATEST_PROTOCOL_VERSION == MCP_PROTOCOL_VERSION, "the SDK has moved to a newer revision"
    message = _tool_call()
    types.JSONRPCRequest.model_validate(message)
    parsed = types.CallToolRequest.model_validate({k: message[k] for k in ("method", "params")})
    assert parsed.params.name == "query_cohort"
    assert isinstance(message["id"], int) and not isinstance(message["id"], bool)
    meta = message["params"]["_meta"]
    assert meta[PROTOCOL] == MCP_PROTOCOL_VERSION and meta[CAPABILITIES] == {}
    types.Implementation.model_validate(meta["io.modelcontextprotocol/clientInfo"])


def test_every_response_is_a_response_of_the_revision() -> None:
    types = pytest.importorskip("mcp.types")
    ok = serve_tool_call(_tool_call(), "silo")
    types.JSONRPCResponse.model_validate(ok)
    assert ok["id"] == 1 and "method" not in ok
    assert ok["result"]["resultType"] == "complete" and ok["result"]["isError"] is False
    result = types.CallToolResult.model_validate(ok["result"])
    assert result.is_error is False
    # The serialized copy the spec recommends beside structured content.
    assert json.loads(ok["result"]["content"][0]["text"]) == ok["result"]["structuredContent"]

    tool_error = serve_tool_call(_tool_call("score_transaction", {"nosuch": 1}), "silo")
    assert types.CallToolResult.model_validate(tool_error["result"]).is_error is True
    assert "nosuch" in tool_error["result"]["content"][0]["text"]


def test_the_server_refuses_what_the_revision_says_to_refuse() -> None:
    no_meta = _tool_call()
    del no_meta["params"]["_meta"]
    refused = serve_tool_call(no_meta, "silo")
    assert refused["error"]["code"] == -32602 and PROTOCOL in refused["error"]["message"]

    for key in (PROTOCOL, CAPABILITIES):
        missing = _tool_call()
        del missing["params"]["_meta"][key]
        assert serve_tool_call(missing, "silo")["error"]["code"] == -32602, key

    old = _tool_call()
    old["params"]["_meta"][PROTOCOL] = "1900-01-01"
    refused = serve_tool_call(old, "silo")
    assert refused["error"]["code"] == -32022
    assert refused["error"]["data"] == {"supported": [MCP_PROTOCOL_VERSION], "requested": "1900-01-01"}

    assert serve_tool_call(_tool_call("no_such_tool"), "silo")["error"]["code"] == -32602
    assert serve_tool_call(mcp_request("a", "tools/list", {}), "silo")["error"]["code"] == -32601

    for bad_id in (None, True, 1.5, []):
        message = _tool_call()
        message["id"] = bad_id
        refused = serve_tool_call(message, "silo")
        assert refused["error"]["code"] == -32600 and "id" not in refused, bad_id
    del message["id"]
    assert serve_tool_call(message, "silo")["error"]["code"] == -32600
    unhashable = _tool_call()
    unhashable["params"]["name"] = ["query_cohort"]
    assert serve_tool_call(unhashable, "silo")["error"]["code"] == -32602


def test_the_channel_verifies_both_directions_and_reports_the_revision() -> None:
    a, b = Agent.create("analyst"), Agent.create("silo")
    try:
        r = secure_agent_call(a, b, "tools/call", {"name": "query_cohort", "arguments": {"phenotype": "x"}})
        failed = secure_agent_call(a, b, "tools/call", {"name": "pool_allele_counts", "arguments": {"counts": [-1]}})
        unknown = secure_agent_call(a, b, "tools/call", {"name": "no_such_tool"})
    finally:
        a.free()
        b.free()
    assert r["protocol_version"] == MCP_PROTOCOL_VERSION and r["request_id"] == 1
    assert r["kem"] == "ML-KEM-768+X25519", "the browser reports both halves; so does this"
    assert r["authenticated"] and r["result_verified"] and r["tamper_rejected"]
    assert r["tool"] == "query_cohort" and r["result"]["phenotype"] == "x"
    # A tool that ran and failed is a signed result; a tool that does not exist is a signed protocol error.
    assert "allele count" in failed["result"]["error"] and failed["result_verified"] and failed["tool"]
    assert unknown["result"]["code"] == -32602 and unknown["result_verified"] and unknown["tool"] is None


def test_the_carry_is_one_exchange_and_its_result_says_if_the_external_server_failed() -> None:
    payload = {"server": "official", "result_text": "[FILE] a.txt", "is_error": False}
    r = secure_carry(payload, tool="list_directory", arguments={"path": "."}, qkd_key=b"q" * 32)
    assert r["authenticated"] and r["tamper_rejected"] and r["payload"] == payload
    assert r["protocol_version"] == MCP_PROTOCOL_VERSION and r["sealed_bytes"] > 3309, "holds an ML-DSA-65 signature"
    assert mcp_tool_result(1, "s", {"is_error": True}, is_error=True)["result"]["isError"] is True


def test_the_signed_text_has_no_spaces_and_keeps_utf8() -> None:
    assert message_body({"b": "Ωmega", "a": 1}) == '{"a":1,"b":"Ωmega"}'


def test_the_stand_in_cohort_does_not_change_with_the_process() -> None:
    """It used builtin `hash`, salted per process, so "deterministic" held only until a restart."""
    from qsh.mcp.server import query_cohort

    assert query_cohort("rare-disease-X")["cohort_count"] == 287
    assert not query_cohort.__doc__.startswith("Medical: a privacy-preserving"), "no cohort data exists"


# --------------------------------------------------------------------------- #
#  The AI-generated record
# --------------------------------------------------------------------------- #
def _verifier():
    spec = importlib.util.spec_from_file_location("verify_ai_provenance", REPO / "scripts" / "verify_ai_provenance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample() -> dict:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_python_reads_a_record_the_browser_module_signed() -> None:
    r = _verifier().check(_sample())
    assert r["signature_valid"] and r["text_matches"] and r["copy_matches"] is True, r
    assert r["ai_generated"] is True and r["problem"] is None
    record = r["record"]
    assert record["spec"] == "qsh/ai-provenance/v1" and record["content_field"] == "run.text"
    for field in ("ai_system", "provider", "generator", "generated_at", "signed_at", "content_sha256"):
        assert record[field], field


def test_each_edit_is_refused() -> None:
    check = _verifier().check

    text = _sample()
    text["run"]["text"] += " edited"
    r = check(text)
    assert r["signature_valid"] and not r["text_matches"] and r["problem"]

    copy_differs = _sample()
    copy_differs["mcp"]["echoedPayload"]["answer"] = "other"
    assert check(copy_differs)["copy_matches"] is False

    body = _sample()
    body["ai_provenance"]["body"] = body["ai_provenance"]["body"].replace('"ai_generated":true', '"ai_generated":false')
    assert not check(body)["signature_valid"]

    sig = _sample()
    sig["ai_provenance"]["sig"] = sig["ai_provenance"]["sig"][:-2] + ("00" if not sig["ai_provenance"]["sig"].endswith("00") else "01")
    assert not check(sig)["signature_valid"]

    for field, value in (("sig", "zz"), ("public_key", "00" * 32), ("public_key", "")):
        bad = _sample()
        bad["ai_provenance"][field] = value
        assert not check(bad)["signature_valid"], (field, value)

    other = _sample()
    other["ai_provenance"]["alg"] = "none"
    assert "names none" in check(other)["problem"]


def test_junk_is_a_finding_and_not_a_crash() -> None:
    check = _verifier().check
    for doc in (None, 7, "text", [], {}, {"ai_provenance": 3}, {"ai_provenance": {"body": 1}}):
        r = check(doc)
        assert not r["signature_valid"] and r["problem"], doc
    no_text = _sample()
    del no_text["run"]
    r = check(no_text)
    assert r["signature_valid"] and not r["text_matches"]
    assert copy.deepcopy(_sample()) == _sample()


def test_the_panel_signs_what_it_downloads_and_the_gap_analysis_says_so() -> None:
    panel = (REPO / "web" / "src" / "components" / "AgenticAIPanel.tsx").read_text(encoding="utf-8")
    for needle in ("signProvenance", "checkProvenance", "ai_provenance:", "ai_provenance_note"):
        assert needle in panel, needle
    assert "cannot carry metadata" not in panel, "the file's own note said a file cannot carry metadata"
    gap = (REPO / "docs" / "ai-act-art50-gap-analysis.md").read_text(encoding="utf-8")
    for needle in ("qsh/ai-provenance/v1", "Sub-measure 1.1.1", "scripts/verify_ai_provenance.py", "not who signed"):
        assert needle in gap, needle
