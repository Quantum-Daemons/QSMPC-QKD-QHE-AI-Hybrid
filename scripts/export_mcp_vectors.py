#!/usr/bin/env python3
"""Emit the MCP request text the channel signs, from the production functions.

`src/qsh/pqc/mcp_bridge.py` and `web/src/lib/clientMcp.ts` build the same `tools/call` request, and the
browser's comment said for two years that it signed "the exact same bytes" while Python wrote spaces
after every separator. So the two languages get a shared artifact: this calls the Python functions and
commits their text, `web/scripts/check_mcp_envelope.mjs` bundles the browser module and reproduces it,
and CI fails if this file is stale. Same mechanism as `export_combiner_vectors.py`.

The version is fixed here so a release does not change the file.

Run after editing the channel's messages:

    python scripts/export_mcp_vectors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qsh.pqc.mcp_bridge import MCP_PROTOCOL_VERSION, mcp_request, message_body  # noqa: E402

OUT = ROOT / "web" / "scripts" / "vectors" / "mcp_envelope_vectors.json"
VERSION = "0.0.0-vector"

CASES = [
    {"client": "analyst-agent", "method": "tools/call",
     "params": {"name": "query_cohort", "arguments": {"phenotype": "rare-disease-X"}}},
    # The panel's shape: a tool name and the model's text, which is not ASCII and must not be escaped.
    {"client": "browser-agent", "method": "tools/call",
     "params": {"name": "score_transaction",
                "arguments": {"prompt": "Why? \"quoted\"\nnext line", "answer": "Ωmega é ✓", "usecase": None,
                              "nested": {"b": 2, "a": [1, True, False]}}}},
]


def main() -> int:
    vectors = [{**case, "body": message_body(mcp_request(case["client"], case["method"], case["params"],
                                                         version=VERSION))}
               for case in CASES]
    OUT.write_text(json.dumps({
        "generated_by": "scripts/export_mcp_vectors.py",
        "protocol_version": MCP_PROTOCOL_VERSION, "version": VERSION, "vectors": vectors,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(vectors)} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
