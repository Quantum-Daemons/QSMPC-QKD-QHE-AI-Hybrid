"""The full deployment's filesystem-MCP endpoint reads one fixed folder, and nothing else reaches it.

WHY THIS EXISTS (Round 86, found by the fact-check sweep of 2026-11-03, the owner's choice of all four)

`/api/mcp/filesystem` passed the caller's `directory`, `tool` and `arguments` straight to the official
filesystem MCP server, unauthenticated. The server has write tools (`write_file`, `edit_file`,
`move_file`), the API allowed CORS from every origin, and the compose file published its port on
every interface: any client on the network, or any web page a user of the full deployment opened,
could read and write anywhere the API process can. The threat model said the server was "sandboxed to
a use-case data dir". The public demo serves no `/api` and was never exposed.

WHAT IS ASSERTED

- The folder is fixed server side (`MCP_FS_ROOT`); a request naming one is refused (422), and the
  interop is handed the fixed folder whatever the request carries.
- Only read-only tools are accepted, at the endpoint (400) and in the interop itself (ValueError),
  before any server is started.
- No CORS header goes to a foreign origin by default.
- The compose file binds the API's host port to loopback.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def api(monkeypatch):
    import api.main as main
    from qsh.mcp import fs_interop

    calls: list[tuple] = []

    def fake_interop(directory, tool, arguments, *, qkd_key=None, **_):
        calls.append((directory, tool, arguments))
        return {"external": {"tool": tool}, "quantum_safe": {}, "verified_result": {}}

    monkeypatch.setattr(fs_interop, "node_available", lambda: True)
    monkeypatch.setattr(fs_interop, "secure_filesystem_interop", fake_interop)
    return TestClient(main.app), main, calls


def test_a_write_tool_is_refused_before_any_server_starts(api) -> None:
    client, _, calls = api
    for tool in ("write_file", "edit_file", "move_file", "create_directory", "read_file"):
        r = client.post("/api/mcp/filesystem", json={"tool": tool, "arguments": {"path": "x"}})
        assert r.status_code == 400, (tool, r.status_code, r.text)
    assert calls == [], "a refused tool still reached the interop"


def test_the_folder_is_fixed_server_side(api) -> None:
    client, main, calls = api
    r = client.post("/api/mcp/filesystem", json={"directory": "/", "tool": "list_directory"})
    assert r.status_code == 422, "a request naming its own folder was accepted"
    r = client.post("/api/mcp/filesystem", json={"tool": "list_directory"})
    assert r.status_code == 200, r.text
    assert calls == [(str(main.MCP_FS_ROOT), "list_directory", None)]
    assert main.MCP_FS_ROOT == (REPO / "data").resolve() and main.MCP_FS_ROOT.is_dir()


def test_the_interop_itself_refuses_a_write_tool() -> None:
    from qsh.mcp import fs_interop

    assert {"write_file", "edit_file", "move_file", "create_directory"}.isdisjoint(fs_interop.READ_ONLY_TOOLS)
    with pytest.raises(ValueError, match="not a read-only tool"):
        fs_interop.secure_filesystem_interop(".", "write_file", {"path": "x", "content": "y"})


def test_no_cors_header_goes_to_a_foreign_origin(api) -> None:
    client, main, _ = api
    assert main.CORS_ORIGINS == [], "CORS origins are configured by default"
    r = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}, dict(r.headers)
    pre = client.options("/api/mcp/filesystem", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in {k.lower() for k in pre.headers}, dict(pre.headers)


def test_the_compose_file_binds_the_api_to_loopback() -> None:
    compose = (REPO / "docker" / "docker-compose.yml").read_text()
    api_block = compose.split("\n  api:\n", 1)[1].split("\n  web:", 1)[0]
    ports = re.findall(r'ports:\s*\[([^\]]*)\]', api_block)
    assert ports and all(p.strip().strip('"').startswith("127.0.0.1:") for p in ports[0].split(",")), ports
