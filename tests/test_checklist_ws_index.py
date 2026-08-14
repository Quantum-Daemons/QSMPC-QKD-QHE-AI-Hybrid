"""A count written beside the thing it counts must equal it.

WHY THIS EXISTS

Three hand-maintained counts have drifted in two rounds, and each was written by someone fixing the
previous one:

  - `CI1` asserted the local gate's last step is `9/9` while it printed `12/12` - its second drift,
    under a paragraph explaining the first and saying the row and the sentence "must be edited
    together";
  - the prefix legend listed 48 families against 53 in use, under its own instruction to regenerate
    rather than hand-edit;
  - the workstream index said WS1 had 20 rows beside a spec expanding to 29, and WS4 27 beside 35,
    under a sentence Round 39 wrote asserting *"the count above is now derived from the row IDs
    beside it, so the two cannot disagree without being visibly inconsistent."* Only WS2's had been
    derived. The same section also said "all 346 rows" against 380, and "Three rows" two lines under
    the 6 that replaced it.

Three instances is this project's threshold for writing the invariant instead of the next fix. So
this parses the WS table, expands each row-ID spec, and requires the written count to equal what the
spec expands to AND every named row to exist.

WHY IT CHECKS EXISTENCE TOO

A spec is only a count if the rows are real. `WS4` names `QR1`-`QR5`; those IDs were renamed once
already (the duplicate-ID guard fired on `QR1`-`QR5` and they became `QS1`-`QS4`), and a rename that
misses the index leaves a count that is arithmetically right about rows that are gone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "browser-verification-checklist.md"

#: `| **WS1** AI-QKD | ...claim... | **29** - `QK1`-`QK7`, `QN1`-`QN6`, ... | ...verdict... |`
_WS_ROW = re.compile(r"^\|\s*\*\*(WS\d)\*\*[^|]*\|[^|]*\|\s*\*\*(\d+)\*\*\s*-\s*([^|]*)\|", re.M)
#: Any row ID in the checklist's own form.
_ROW_ID = re.compile(r"^\| ([A-Z]{1,3}[0-9]{1,3}[a-z]?) \|", re.M)
#: A backticked ID inside a spec cell.
_SPEC_ID = re.compile(r"`([A-Z]{1,3}\d{1,3}[a-z]?)`")


@pytest.fixture(scope="module")
def doc() -> str:
    if not DOC.exists():
        pytest.skip("browser-verification-checklist.md is not present in this checkout")
    return DOC.read_text()


def _expand(cell: str) -> set[str]:
    """`QK1`-`QK7`, `SC4`, `CS4`/`CS7` -> the set of IDs it names.

    Ranges are written as two backticked IDs joined by a hyphen. Anything else backticked is a
    single ID. Both forms appear in the table today and both must be understood, because a spec the
    parser silently drops shrinks the expected count to match the written one - passing for exactly
    the wrong reason.
    """
    out: set[str] = set()
    for m in re.finditer(r"`([A-Z]{1,3})(\d{1,3})`\s*-\s*`([A-Z]{1,3})(\d{1,3})`", cell):
        p1, a, p2, b = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        assert p1 == p2, f"range across two prefixes in the WS index: {m.group(0)}"
        out |= {f"{p1}{i}" for i in range(a, b + 1)}
    consumed = re.sub(r"`([A-Z]{1,3})(\d{1,3})`\s*-\s*`([A-Z]{1,3})(\d{1,3})`", " ", cell)
    out |= set(_SPEC_ID.findall(consumed))
    return out


def test_the_ws_table_parses(doc: str) -> None:
    """A parse that finds no rows would pass every assertion below."""
    rows = _WS_ROW.findall(doc)
    assert len(rows) >= 4, f"only {len(rows)} workstream rows parsed; the index format changed"


def test_each_ws_count_equals_what_its_spec_expands_to(doc: str) -> None:
    wrong = []
    for ws, written, cell in _WS_ROW.findall(doc):
        expanded = _expand(cell)
        if len(expanded) != int(written):
            wrong.append(f"{ws}: written {written}, spec expands to {len(expanded)}")
    assert not wrong, (
        "workstream counts disagree with the row IDs beside them:\n  " + "\n  ".join(wrong)
        + "\n\nThis is the third count in two rounds to drift this way. Do not just correct the "
        "number - the previous two were corrected and drifted again. Derive it, or delete it."
    )


def test_every_row_a_ws_spec_names_actually_exists(doc: str) -> None:
    real = set(_ROW_ID.findall(doc))
    assert len(real) >= 300, f"row-ID parse collapsed ({len(real)} found); the table format changed"
    missing = []
    for ws, _written, cell in _WS_ROW.findall(doc):
        gone = sorted(_expand(cell) - real)
        if gone:
            missing.append(f"{ws} names rows that do not exist: {gone}")
    assert not missing, (
        "\n  ".join(missing)
        + "\n\nA count is only a count if its rows are real. `QR1`-`QR5` were renamed to `QS1`-`QS4` "
        "once already after the duplicate-ID guard fired; a rename that misses this index leaves a "
        "count that is arithmetically right about rows that are gone."
    )
