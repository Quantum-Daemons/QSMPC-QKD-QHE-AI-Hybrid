"""`scripts/ci_local.sh` must number its own steps consistently.

WHY THIS EXISTS

Checklist row `CI1` asserts that the local gate script "exits 0, and its last step is numbered
`N/N`. The count is part of the check" - a good row, whose whole purpose is to fail when a gate is
added without the row being updated. It has now failed to do that TWICE:

  - it read `8` in one place and `9/9` eleven lines above, and the prose fixing that drift says in
    so many words that the row and the sentence "must be edited together";
  - by Round 39 the script printed `12/12` while both still said 9.

The row is correct and unenforceable, which is the same shape as the register deadline that ran
three rounds late and as `DP7`, right and ignored for five rounds. Fixing the number a third time
buys one round.

So the invariant replaces the constant. `CI1` no longer names a number - it asserts that the last
step's numerator equals its denominator equals the number of steps - and this test enforces the
property the number was standing in for. Adding a gate now requires only that the script stay
self-consistent, which is checkable, instead of requiring that someone remember a document.

WHY THE DENOMINATORS ARE CHECKED TOO

Before Round 39 the script's own denominators disagreed: steps 1-7 said `/10`, steps 8-9 said `/10`,
steps 10-11 said `/11`. A reader watching for "the last step is N/N" would have seen `11/11` and
called it consistent while three earlier steps promised a different total. A count is only evidence
if every step agrees on it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "ci_local.sh"

#: `step "7/12 no model artifacts committed"` - numerator, denominator, label.
_STEP = re.compile(r'^step\s+"(\d+)/(\d+)\s+([^"]*)"', re.M)


def _steps() -> list[tuple[int, int, str]]:
    return [(int(a), int(b), c) for a, b, c in _STEP.findall(SCRIPT.read_text())]


def test_the_script_still_has_numbered_steps() -> None:
    """A parse that finds nothing would pass every assertion below for the wrong reason."""
    steps = _steps()
    assert len(steps) >= 8, (
        f"only {len(steps)} numbered steps parsed from {SCRIPT.name}; the step() convention changed "
        "and this test is now measuring nothing"
    )


def test_every_step_agrees_on_the_total() -> None:
    """All denominators identical, and equal to the number of steps."""
    steps = _steps()
    denominators = {b for _, b, _ in steps}
    assert denominators == {len(steps)}, (
        f"the script has {len(steps)} steps but its denominators are {sorted(denominators)}. "
        "Every step must say /N for the same N, or the running count means nothing - a reader who "
        "checks only the last line sees a consistent-looking total while earlier steps promise "
        "a different one."
    )


def test_the_numerators_run_1_to_n_without_gaps() -> None:
    """Renumbering after an insert is where this drifts; a duplicate or skip is the symptom."""
    numerators = [a for a, _, _ in _steps()]
    assert numerators == list(range(1, len(numerators) + 1)), (
        f"step numerators are {numerators}; expected 1..{len(numerators)} with no gaps or repeats. "
        "A gate was inserted without renumbering the ones after it."
    )


def test_the_checklist_row_does_not_hardcode_the_count() -> None:
    """`CI1` must assert the invariant, not a number - it has drifted twice naming one.

    The checklist is gitignored and not published, so this reads it only if present; on a clone
    without it the test skips rather than failing for an absent file.
    """
    import pytest

    doc = REPO / "docs" / "browser-verification-checklist.md"
    if not doc.exists():
        pytest.skip(
            "LOCAL-ONLY GATE, SKIPPED. docs/browser-verification-checklist.md is gitignored by "
            "design (it must not be published), so this check is a no-op in CI and in every "
            "clone - it enforces nothing there and its green tick means only that the file was "
            "absent. `scripts/ci_local.sh` gate 15 fails when the file is missing, so the one "
            "machine that CAN run this cannot silently stop running it."
        )
    row = [ln for ln in doc.read_text().splitlines() if ln.startswith("| CI1 ")]
    assert row, "row CI1 has gone; the local-gate check is no longer in the checklist"
    stale = re.search(r"numbered `?(\d+)/(\d+)", row[0])
    assert not stale, (
        f"CI1 hardcodes the step count again ({stale.group(0)!r}). It has drifted twice this way. "
        "State the invariant - the last step's numerator equals its denominator equals the number "
        "of steps - and let this test carry the number."
    )
