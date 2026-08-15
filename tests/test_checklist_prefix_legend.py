"""The checklist's prefix legend must list every prefix the checklist uses.

WHY THIS EXISTS

The legend carries its own instruction - *"Regenerate this list rather than editing it by hand"* -
followed by the exact command to do it. By Round 39 it listed 48 families against 53 in use. The
five missing ones (`MS`, `PF`, `PI`, `PS`, `QS`) were precisely those added by Rounds 37-38, i.e.
every family added since the instruction was written.

A correct instruction that nothing enforces is the failure this round kept finding: the register
deadline that ran three rounds late, `DP7` right and ignored for five, `CI1`'s step count drifting
twice. The pattern is always the same - the rule is right, greppable, and has no executor. So this
runs the command the document names and compares.

WHY IT ASSERTS COVERAGE AND NOT EQUALITY

A legend entry for a family whose last row was deleted is stale but harmless - a reader looks up a
prefix that no longer appears and learns nothing wrong. A row whose prefix is NOT in the legend is
the harmful direction: the reader hits an unexplained ID. Requiring exact equality would also make
deleting the final row of a family fail a test in a way that invites deleting the legend entry
without thinking about whether the row should have gone. So: every prefix in use must be listed;
extras are reported but do not fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "browser-verification-checklist.md"

#: The row-ID pattern the document itself specifies, `[a-z]?` included. That optional suffix was
#: missing once and `RS2b` went uncounted, which also blinded the duplicate-ID guard to the one
#: collision a suffix makes likely.
_ROW_ID = re.compile(r"^\| ([A-Z]{1,3})[0-9]{1,3}[a-z]? \|", re.M)
#: Legend entries look like: `MS` model source and metric scope,
#: The description may start uppercase - `G` GPU claims, `HF` Hugging Face, `QN` QRNG - so this must
#: not require a lowercase letter. The first draft did, and reported ten false missing families
#: including every acronym-led one. A parser that over-reports is as useless as one that under-
#: reports: both get the test switched off.
_LEGEND = re.compile(r"`([A-Z]{1,3})`\s+[A-Za-z]")


def _legend_block(text: str) -> str:
    """The prose between the regenerate instruction and the 'Two collisions' paragraph."""
    start = text.find("The prefix says which section a row belongs to")
    end = text.find("Two collisions have been fixed", start)
    assert start != -1 and end != -1, "the legend block's landmarks moved; this test cannot locate it"
    return text[start:end]


@pytest.fixture(scope="module")
def doc() -> str:
    if not DOC.exists():
        pytest.skip(
            "LOCAL-ONLY GATE, SKIPPED. docs/browser-verification-checklist.md is gitignored by "
            "design (it must not be published), so this check is a no-op in CI and in every "
            "clone - it enforces nothing there and its green tick means only that the file was "
            "absent. `scripts/ci_local.sh` gate 15 fails when the file is missing, so the one "
            "machine that CAN run this cannot silently stop running it."
        )
    return DOC.read_text()


def test_the_parse_finds_rows_and_legend_entries(doc: str) -> None:
    """Both sides must be non-trivial, or the coverage assertion below passes vacuously."""
    assert len(set(_ROW_ID.findall(doc))) >= 40, "row-ID parse collapsed; the table format changed"
    assert len(set(_LEGEND.findall(_legend_block(doc)))) >= 40, "legend parse collapsed"


def test_every_prefix_in_use_is_explained(doc: str) -> None:
    in_use = set(_ROW_ID.findall(doc))
    listed = set(_LEGEND.findall(_legend_block(doc)))
    missing = sorted(in_use - listed)
    assert not missing, (
        f"{len(missing)} prefix families are used by rows but absent from the legend: {missing}.\n"
        "Regenerate it with the command the document itself gives:\n"
        "    grep -oE '^\\| [A-Z]{1,3}[0-9]{1,3}[a-z]? \\|' docs/browser-verification-checklist.md \\\n"
        "      | sed 's/[0-9].*//' | sort -u\n"
        "The legend already says to regenerate rather than hand-edit. It drifted five families "
        "behind anyway, which is why this test exists rather than a third reminder."
    )
