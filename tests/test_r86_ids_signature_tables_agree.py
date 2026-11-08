"""The browser's IDS signature table and the Python one are one table (Round 86, the audit).

`web/src/lib/clientIDS.ts` carries a hand copy of the eight attack signatures in `src/qsh/qml_ids/attacks.py`, so the telecom page's
synthetic QKD telemetry and the full deployment's are the same only while nobody edits one of them. All eight rows matched on the day
this was written. This holds them together.
"""

from __future__ import annotations

import re
from pathlib import Path

from qsh.qml_ids import attacks

REPO = Path(__file__).resolve().parents[1]
TS = REPO / "web" / "src" / "lib" / "clientIDS.ts"


def _ts_signatures() -> dict[str, tuple[float, ...]]:
    block = re.search(r"const SIG: Record<string, number\[\]> = \{(.*?)\n\};", TS.read_text(), re.S)
    assert block, "clientIDS.ts no longer declares SIG this way; re-derive this test"
    return {m[1]: tuple(float(x) for x in m[2].split(","))
            for m in re.finditer(r"^\s*(\w+):\s*\[([^\]]+)\]", block[1], re.M)}


def test_the_two_signature_tables_are_the_same_table() -> None:
    ts = _ts_signatures()
    assert len(ts) == 8, ts
    assert ts == dict(attacks._SIGNATURES)


def test_the_classes_and_the_feature_count_agree() -> None:
    src = TS.read_text()
    listed = re.search(r"const ATTACKS = \[(.*?)\];", src, re.S)
    assert listed, "clientIDS.ts no longer declares ATTACKS this way; re-derive this test"
    assert re.findall(r'"(\w+)"', listed[1]) == attacks.ATTACKS
    assert int(re.search(r"const N_FEAT = (\d+)", src)[1]) == len(attacks.FEATURES)
