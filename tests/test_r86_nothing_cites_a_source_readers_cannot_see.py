"""Public text states its requirements as this project's own, and cites what a reader can open.

WHY THIS EXISTS (Round 86, found by reading every plan against the tree, the owner's choice)

About 30 tracked lines cited a confidential source document as "the report", and some published its
content: a target for a detector's F1, a PoC criterion for accuracy degradation, a quoted sentence with
no URL. A reader can check none of that, and "the report" read as this repository's own technical report,
which has also left it. They now state the requirement as this project's own and cite primary sources.

WHAT IS ASSERTED

The phrases those lines used do not come back in any tracked text. The register and the log are records,
and they use "the report" for the technical report in dated entries, so they are left out.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORDS = {"docs/deliverable-verification.md", "docs/verification-log.md"}
THIS = "tests/test_r86_nothing_cites_a_source_readers_cannot_see.py"
#: Each one attributes a requirement to a document no reader can open.
_CITATION = re.compile(
    r"(?i)\bsource report\b"
    r"|\bthe report frames\b"
    r"|\bthe report flags\b"
    r"|\bthe report asks\b"
    r"|\bthe report carefully\b"
    r"|\bthe report, section\b"
    r"|\bReport Phase-\d"
    r"|\breport WS\d"
    r"|\bthe report's (?:WS\d|QHE|QSDC|QSMPC|XAI|MCP|Phase-\d|benchmark|binary|selling|\"|PoC|gap)"
)
_ARTEFACTS = (".json", ".jpg", ".png", ".pdf", ".wasm", ".onnx", ".bin", ".tsv", ".csv", ".lock", ".svg")


def _tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\n")


def _hits(text: str) -> list[str]:
    return [m.group(0) for m in _CITATION.finditer(text)]


def test_no_tracked_text_cites_the_private_source() -> None:
    files = [f for f in _tracked() if f and f not in RECORDS and f != THIS and not f.endswith(_ARTEFACTS)]
    assert len(files) > 300, f"only {len(files)} files; the listing is looking at nothing"
    found = []
    for f in files:
        path = REPO / f
        if path.is_file():
            found += [f"{f}: {h}" for h in _hits(path.read_text(errors="replace"))]
    assert not found, f"public text attributes a requirement to a document no reader can open: {found[:8]}"


def test_the_check_can_fail() -> None:
    for seeded in ("the report's WS1 target", "The report frames MCP as a layer", "the source report's PoC",
                   'the report\'s "PQC-first hybrid"', "(Report Phase-4 of WS2.)", "# report WS1 target"):
        assert _hits(seeded), seeded
    for fine in ("the technical report's crypto table", "reads the report and never fetched it", "Nature reports"):
        assert not _hits(fine), fine
