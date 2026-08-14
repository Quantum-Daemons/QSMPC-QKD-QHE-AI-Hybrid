"""Prose must not restate a count the file or the tree already computes.

WHY THIS EXISTS

Four hand-written counts have now drifted, and each was written by the person fixing the previous
one:

  1. `CI1` said the local gate's last step is `9/9` while it printed `12/12`.
  2. The prefix legend listed 48 families against 53 in use, under its own instruction to regenerate.
  3. The workstream table said WS1 had 20 rows beside a spec expanding to 29.
  4. The paragraph explaining (1) said *"the count is now 9"* while the script ran **fourteen** steps
     - eleven lines below the row Round 40 had just rewritten to stop naming a number, under a
     heading (*"Why CI1 names the number"*) that the same round had made false.

Rounds 39 and 40 each closed one of these with a test, and each test was written against the exact
artefact in front of it: `test_ci_local_step_numbering.py` reads the `| CI1 |` table row,
`test_checklist_ws_index.py` reads the `| **WS1** |` table rows. Both are correct and both are blind
to the sentence underneath, which is where the next instance landed. That is this round's subject -
a check that names a boundary and verifies one side of it.

WHAT THIS CHECKS, AND WHAT IT DELIBERATELY DOES NOT

It looks for sentences that assert a CURRENT count of something the repository can count, and fails
them. It does not try to validate the number: a count in prose is wrong eventually whatever its value
today, so the rule is that prose does not carry one.

Quoted history is explicitly allowed, and the file is full of it for good reason - the paragraph
above quotes *"the count is now 9"* to record how the drift happened. A sentence inside quotation
marks, or one that says a number "said"/"read"/"used to", is describing the past and is not an
assertion about now. Without that exemption this test would force the file to delete its own
evidence, which is how a gate makes a document worse.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "browser-verification-checklist.md"

#: Nouns naming THIS FILE's own structures, which a command computes. Bare "rows" is deliberately
#: absent: it means a checklist row here and a dataset row three sections over, and a first version
#: of this pattern failed on "across all 5,078,345 rows" (the AMLworld corpus) and "the held-out set
#: here is 80 rows" (a split size). Neither restates anything the file computes.
#:
#: That over-reach is the same shape as the Round 40 dated-blocker heuristic, which matched "iterated
#: until clean" and produced three false positives out of three. The lesson recorded then applies
#: here: a gate that cries wolf gets switched off, so it fires only on the file's own vocabulary.
_COUNTABLE = (r"(?:checks?|gates?|steps?|families|prefixes|workstreams?|"
              r"(?:numbered|checklist|verification|ID) rows?|row IDs)")

#: "the count is now **9**", "all 346 rows", "at six", "Three rows", "runs all seven blocking gates".
#: DIGITS AND WORDS. The first version matched `\d` only and caught "the count is now 9" while
#: missing "the smallest at six" - one of the two instances it was written for, and the two are the
#: same defect written two ways. A gate that covers one notation and not the other is this round's
#: subject happening to the test written for this round's subject.
_NUM = r"(?:\d[\d,]*|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen)"

_PATTERNS = [
    re.compile(rf"\bcount is (?:now )?\*?\*?({_NUM})\*?\*?\b", re.I),
    re.compile(rf"\ball ({_NUM}) {_COUNTABLE}\b", re.I),
    re.compile(rf"\b(?:is|are|has|have|now) \*?\*?({_NUM})\*?\*? {_COUNTABLE}\b", re.I),
    # "the smallest at six", "stands at 383" - a bare total attached to a superlative or "at".
    re.compile(rf"\b(?:smallest|largest|thinnest|deepest|total(?:s)?|stands) at \*?\*?({_NUM})\*?\*?\b", re.I),
]

#: Words that mark a sentence as describing the past rather than asserting the present.
_HISTORICAL = re.compile(
    r"\b(said|read|used to|had|was|were|until|previously|drift|stale|wrong|corrected)\b", re.I)


def _sentences(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.startswith("|"):          # table rows are the computed side, not prose
            continue
        out.append((i, line))
    return out


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


def test_the_scan_sees_prose_at_all(doc: str) -> None:
    """A parse that finds no prose lines would pass the assertion below vacuously."""
    lines = _sentences(doc)
    assert len(lines) > 500, f"only {len(lines)} non-table lines; the file format changed"


def test_no_sentence_asserts_a_current_count(doc: str) -> None:
    offenders = []
    for lineno, line in _sentences(doc):
        if _HISTORICAL.search(line):
            continue
        for pat in _PATTERNS:
            m = pat.search(line)
            if not m:
                continue
            # The exemption is per MATCH, not per line. A line-level `'"' in line` skip exempted
            # `the smallest at six` because the same sentence quoted a metric name twenty characters
            # away - so the gate written to catch two instances caught one, for the same
            # one-side-of-a-boundary reason it exists to prevent. Only a figure INSIDE quotes is
            # history.
            before = line[: m.start()]
            if before.count('"') % 2 == 1:
                continue
            offenders.append(f"{lineno}: {' '.join(line.split())[:130]}")
            break
    assert not offenders, (
        "prose restates a count the file or the tree computes:\n  " + "\n  ".join(offenders)
        + "\n\nFour of these have drifted, each written by someone fixing the previous one. Do not "
        "correct the number - remove it, and let the table or the test carry it. Quoted history is "
        "allowed and is how this file records the drifts."
    )
