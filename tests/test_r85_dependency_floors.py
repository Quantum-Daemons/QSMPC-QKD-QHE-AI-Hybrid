"""The dependency floors the 2026-11-01 sweep raised stay raised, and the MCP server stays pinned.

WHY THIS EXISTS (Round 85)

A research sweep against post-August-2026 advisories found the full deployment's floors admitting
releases with published fixes after them: websockets below 17.4 (GHSA-6m5v-9gmr-5665, memory
exhaustion through continuation frames, reachable on `/api/ws`), pydantic below 2.14.2, the MCP
Python SDK below 2.2.0, python-multipart below 0.0.31, cryptography below 50.0.0, onnx below 1.23.0
and onnxruntime below 1.28.3. And `qsh.mcp.fs_interop` started the official filesystem server with
`npx -y` and no version, so it ran whatever npm's `latest` was: since 2.0.0 that declares Node 24 or
newer, and the API image's Debian Node is 20. The owner chose to apply every finding in this round.

A floor is one line in a requirements file, and nothing else notices when a later edit lowers it.

WHAT IS ASSERTED

- Each floor, in each place it is declared (`requirements-*.txt` and the matching `pyproject.toml`
  extra).
- The filesystem server is started at an exact version.
- OpenFHE stays below 1.6, and the comment that says never 1.6.0 when the cap lifts is still there.
- The four noble packages are on one version set (post-quantum 0.7.1 and the 2.4.0 it depends on), and
  the lockfile resolves each to a single copy.

On the base commit (6bb8657) every test here fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: (package, minimum version, the files that must declare at least it)
FLOORS = [
    ("websockets", "17.4", ["requirements-api.txt", "pyproject.toml"]),
    ("pydantic", "2.14.2", ["requirements-api.txt", "pyproject.toml"]),
    ("python-multipart", "0.0.31", ["requirements-api.txt", "pyproject.toml"]),
    ("mcp", "2.2.0", ["requirements-ai.txt"]),
    ("onnx", "1.23.0", ["requirements-ai.txt", "pyproject.toml"]),
    ("onnxruntime", "1.28.3", ["requirements-ai.txt", "pyproject.toml"]),
    ("cryptography", "50.0.2", ["requirements-core.txt", "pyproject.toml"]),
]


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4])


def _declared(text: str, package: str) -> list[str]:
    """Every `package>=X` in a requirements file or a pyproject list, as X."""
    return re.findall(rf'(?<![\w.-]){re.escape(package)}(?:\[[^\]]*\])?>=([0-9][0-9.]*)', text)


@pytest.mark.parametrize("package,minimum,files", FLOORS, ids=[f[0] for f in FLOORS])
def test_the_floor_is_at_least_the_fixed_release(package: str, minimum: str, files: list[str]) -> None:
    for name in files:
        found = _declared((REPO / name).read_text(encoding="utf-8"), package)
        assert found, f"{name} does not declare a floor for {package}"
        for v in found:
            assert _version(v) >= _version(minimum), f"{name}: {package}>={v}, below {minimum}"


def test_the_filesystem_server_is_pinned_to_an_exact_version() -> None:
    from qsh.mcp import fs_interop

    assert re.fullmatch(r"@modelcontextprotocol/server-filesystem@\d+\.\d+\.\d+", fs_interop.FILESYSTEM_SERVER)
    src = (REPO / "src" / "qsh" / "mcp" / "fs_interop.py").read_text(encoding="utf-8")
    assert '"@modelcontextprotocol/server-filesystem"' not in src.split("FILESYSTEM_SERVER =")[1]


def test_openfhe_stays_below_1_6_and_says_never_1_6_0() -> None:
    text = (REPO / "requirements-fhe.txt").read_text(encoding="utf-8")
    assert "openfhe>=1.5.1,<1.6" in text
    assert "NEVER 1.6.0" in text and "#1364" in text


def test_the_noble_packages_move_together() -> None:
    pkg = json.loads((REPO / "web" / "package.json").read_text())["dependencies"]
    assert pkg["@noble/post-quantum"] == "^0.7.1"
    assert pkg["@noble/curves"] == "^2.4.0" and pkg["@noble/hashes"] == "^2.4.0"
    assert pkg["@noble/ciphers"] == "~2.4.0"
    lock = json.loads((REPO / "web" / "package-lock.json").read_text())["packages"]
    for name in ("@noble/post-quantum", "@noble/curves", "@noble/hashes", "@noble/ciphers"):
        copies = sorted({v["version"] for k, v in lock.items() if k.endswith(f"node_modules/{name}")})
        assert len(copies) == 1, f"{name} resolves to {copies}: more than one copy is bundled"
    assert lock["node_modules/@noble/post-quantum"]["version"] == "0.7.1"
