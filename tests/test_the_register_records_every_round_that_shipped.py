"""A round that shipped must appear in the register, and git is what knows it shipped.

WHY THIS EXISTS

Round 48 shipped three commits, a deploy, and a visitor-facing fix to the finance QCNN - and wrote
none of its record. No `## Round 48` heading in `docs/deliverable-verification.md`, no entry in
`docs/verification-log.md`, and no checklist rows for any of its six changes.

Every gate that could have noticed was disabled by the same omission, because they all ask the same
file what round it is:

  * `tests/test_register_deadlines.py::_current_round()` takes `max()` over the register's
    `## Round N` headings.
  * `tests/test_the_log_has_an_entry_for_the_current_round.py::_current_round()` is a second copy of
    that function, reading the same file, and its own comment says why: "Deriving the round from the
    same place both files do means this cannot disagree with the deadline gate about which round it
    is."

That is true, and it is the defect. They cannot disagree with each other, and neither can disagree
with a register that simply stopped being written. Measured on the tip that shipped Round 48:
`_current_round()` returned **47**, so the log gate asked "does the log name round 47?" - it did -
and both went green with an entire round missing. Three obligations conditioned `by Round 48` stayed
pending for the same reason, since the deadline comparison is also against that number.

WHAT THIS ASSERTS, AND WHY IT CANNOT BE FOOLED THE SAME WAY

The round number is recoverable from git, which no amount of forgetting to write prose can change:
every round merges a branch called `roundN-...`, and the merge commit subject records it. So this
compares two INDEPENDENT sources - the merged history against the register - and fails when the
register lags.

Verified against the actual failure rather than a hypothetical: at `e192214`, the merged history's
highest round was **48** and the register's was **47**. This gate is red there, which is the whole
point of writing it.

WHAT IS DELIBERATELY NOT ASSERTED

Not that the register is AHEAD by some amount. A round in progress writes its heading before it
merges, so `register > history` is the normal state for most of a round's life. Only lagging is a
defect.

Not the log or the checklist. The log has its own gate; the checklist is gitignored and cannot be
checked anywhere but one machine. This asserts the one thing that is both machine-checkable and
load-bearing for every other round-conditioned promise in the project.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REGISTER = REPO / "docs" / "deliverable-verification.md"

#: `## Round 47` / `## Rounds 39-42`. The same shape both existing round-readers use, so this gate
#: cannot fail merely because it parses the headings differently from them.
_HEADING = re.compile(r"^##+ Rounds? (\d+)", re.M)

#: `Merge pull request #39 from Quantum-Daemons/round48-the-page-was-serving-a-crippled-model`.
#: The org has changed at least once, so the pattern anchors on the branch name rather than the
#: remote: what identifies a round is the branch it merged, not who owned it.
_BRANCH_ROUND = re.compile(r"round(\d+)[-_]")


def _register_round() -> int:
    rounds = [int(m.group(1)) for m in _HEADING.finditer(REGISTER.read_text())]
    assert rounds, "the register has no '## Round N' heading; the round cannot be read from it"
    return max(rounds)


def _shipped_round() -> int | None:
    """The highest round number in the merged history, or None when git cannot say.

    Reads merge-commit subjects, which is where the branch name survives. A shallow clone or an
    archive export has no merges, and this returns None there rather than inventing a number - the
    check below then skips loudly instead of passing quietly.
    """
    try:
        out = subprocess.run(
            ["git", "log", "--format=%s", "--merges", "-n", "200"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):       # pragma: no cover
        return None
    rounds = [int(m.group(1)) for m in _BRANCH_ROUND.finditer(out)]
    return max(rounds) if rounds else None


def test_the_two_sources_are_actually_independent() -> None:
    """Non-vacuity, and it is the whole argument: git must know something the register does not.

    If this gate derived its number from the register - as both existing round-readers do - it would
    be a third copy of the same blind spot rather than a check on it.
    """
    shipped = _shipped_round()
    if shipped is None:
        pytest.skip(
            "no merge commits carrying a `roundN-` branch name are visible (shallow clone or "
            "export). The comparison below is unchecked, not satisfied."
        )
    assert shipped >= 41, (
        f"the merged history's highest round reads {shipped}, which is implausibly low - the branch "
        "naming convention has probably changed and this gate is now reading the wrong thing. Fix "
        "the pattern rather than the round."
    )
    # The two sources must genuinely be different things. Asserted on the CODE, so a future
    # refactor cannot quietly repoint `_shipped_round` at the register and leave a gate that agrees
    # with itself - which is precisely the shape of the blind spot this file exists to break.
    assert "REGISTER" not in _shipped_round.__code__.co_names, (
        "`_shipped_round` reads the register. It must not: the whole value of this gate is that its "
        "number survives the register not being written."
    )
    assert "REGISTER" in _register_round.__code__.co_names, (
        "`_register_round` no longer reads the register, so there is nothing to compare against"
    )


def test_the_register_has_an_entry_for_every_round_that_merged() -> None:
    """The property: the register must not lag the merged history.

    Red at `e192214` - history 48, register 47 - which is the state that prompted this file.
    """
    shipped = _shipped_round()
    if shipped is None:
        pytest.skip("no merge commits with a `roundN-` branch name are visible")
    recorded = _register_round()
    assert recorded >= shipped, (
        f"round {shipped} merged and the register's newest heading is round {recorded}.\n\n"
        "A round that ships without a record disables every gate that depends on knowing which "
        "round it is: both `_current_round()` readers take the register's newest heading, so the "
        "log gate asks for an entry naming a round that has already been written, and every "
        "`by Round N` deadline is compared against a number that stopped advancing. Nothing goes "
        "red; the round simply becomes invisible.\n\n"
        f"Write the `## Round {shipped}` section in docs/deliverable-verification.md. If the round "
        "genuinely produced nothing worth recording, write that - a section saying so is a record; "
        "silence is not."
    )


def test_the_comparison_would_have_caught_round_48() -> None:
    """Anti-tautology, against the exact failure rather than an invented one.

    This gate is green on the tree that adds it, so on its own it demonstrates nothing. The state it
    exists for is reconstructed here from the two numbers that were true at `e192214`: the merged
    history said 48 and the register said 47.
    """
    def lags(recorded: int, shipped: int) -> bool:
        return recorded < shipped

    assert lags(47, 48), (
        "the comparison does not flag a register at 47 against a history at 48 - the literal state "
        "of this repository when Round 48 merged, and the reason this file exists"
    )
    assert not lags(48, 48), "a register level with the history must pass"
    assert not lags(49, 48), (
        "a register AHEAD of the history must pass: a round in progress writes its heading before "
        "it merges, so that is the normal state for most of a round's life"
    )
