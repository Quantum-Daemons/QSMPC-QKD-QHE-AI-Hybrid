"""A merged pull request has a line in the register (Round 86, the audit).

Round 48 merged without writing its record, and `tests/test_the_register_records_every_round_that_shipped.py` was written so that could not
recur. It reads the round number from merged branch names, so once every branch was `round86-*` it saw nothing: six pull requests (#7 to #12) carried a new
student model, a Web Worker and four Hub republishes and touched neither the register nor the log. This reads the pull requests themselves.

A shallow clone has no merges, so there the check skips loudly rather than passes.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REGISTER = REPO / "docs" / "deliverable-verification.md"

#: Pull-request numbers restarted at 1 with the second history rewrite (2026-11-06), so only merges after it are comparable.
SINCE = "2026-11-07T00:00:00+09:00"
_MERGE = re.compile(r"^Merge pull request #(\d+) from ", re.M)


def _merged() -> list[int]:
    out = subprocess.run(["git", "log", "--merges", "--first-parent", "main", f"--since={SINCE}", "--format=%s"],
                         cwd=REPO, capture_output=True, text=True, check=False).stdout
    return sorted({int(n) for n in _MERGE.findall(out)})


def test_every_merged_pull_request_has_a_line_in_the_register() -> None:
    merged = _merged()
    if not merged:
        pytest.skip("no merge commits are visible (shallow clone or export): unchecked, not satisfied")
    text = REGISTER.read_text()
    missing = [n for n in merged if not re.search(rf"^\| #{n} \|", text, re.M)]
    assert not missing, (
        f"merged pull requests with no line in the register's table: {missing}. Add one row each to the section "
        "'what merged' before this branch merges.")
