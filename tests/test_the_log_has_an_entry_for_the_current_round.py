"""A round that raises the register's heading must leave a record of what it saw on the page.

WHY THIS EXISTS

`docs/verification-log.md` is where deployed-site measurements live. It has lost whole runs of rounds
four separate times:

  - Rounds 25-27: never reconstructable, gone.
  - Rounds 29-31: reconstructed in Round 32 - by a round that then recorded nothing of its own.
  - Rounds 39-42: became a pointer table in Round 43, which the file itself argued was the honest
    form, since "a reconstruction presented as a record is worse than an absent record, because the
    absence is visible and the imitation is not".
  - Round 44: no section at all - in the round IMMEDIATELY after the one that wrote that pointer, and
    while its own register entry documented having verified on the deployed site.

The file diagnosed its own root cause at `:364`: **the round that writes the checks is not the round
that runs them, and nothing forces the second half.** Four recurrences later, "nothing forces it" was
still literally true. Every other obligation in this project that survived repetition was eventually
given a gate; this one had prose.

WHAT IT ASSERTS, AND WHAT IT DELIBERATELY DOES NOT

It asserts only that the round is NAMED here. It does not check that the content is good, that
numbers were measured, or that a deploy happened - a gate cannot tell a real measurement from a
plausible sentence, and pretending otherwise would license exactly the reconstruction the file warns
against.

A POINTER COUNTS. Rounds 39-42 and 44 are recorded as pointers to where their measurements actually
live, and that is a legitimate entry: it is honest about the absence and tells a reader where to go.
The failure mode being closed is silence, not brevity.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTER = REPO / "docs" / "deliverable-verification.md"
LOG = REPO / "docs" / "verification-log.md"

#: The register's own round headings, which are what `_current_round` in test_register_deadlines.py
#: reads. Deriving the round from the same place both files do means this cannot disagree with the
#: deadline gate about which round it is.
_HEADING = re.compile(r"^##+ Rounds? (\d+)", re.M)


def _current_round() -> int:
    rounds = [int(m.group(1)) for m in _HEADING.finditer(REGISTER.read_text())]
    assert rounds, "the register has no '## Round N' headings; the round cannot be determined"
    return max(rounds)


def test_the_log_names_the_current_round() -> None:
    """The round on the register's newest heading appears somewhere in the log."""
    now = _current_round()
    text = LOG.read_text()
    # "## Round 45", "## Rounds 39-42", "## Round 44, added in Round 45" - any heading that names it.
    named = [m.group(0) for m in re.finditer(rf"^##+ .*\b{now}\b.*$", text, re.M)]
    assert named, (
        f"the register is at Round {now} and docs/verification-log.md has no heading naming it.\n\n"
        "This file has lost four separate runs of rounds - 25-27 are gone for good - and its own "
        "diagnosis is that 'the round that writes the checks is not the round that runs them, and "
        "nothing forces the second half'. This is the thing that forces it.\n\n"
        "A POINTER IS A VALID ENTRY. If the measurements live in the register instead, say so and "
        "link them, the way the Rounds 39-42 and Round 44 entries do. What is not acceptable is "
        "silence, because an absent record and an unverified round look identical afterwards."
    )


def test_the_log_still_carries_real_entries() -> None:
    """Non-vacuity: a file trimmed to one heading would satisfy the check above.

    The count is deliberately loose. It exists so that deleting history to make the gate pass is
    caught, not to police how many rounds are written up.
    """
    headings = re.findall(r"^##+ Rounds? \d+", LOG.read_text(), re.M)
    assert len(headings) >= 8, (
        f"only {len(headings)} round sections remain in docs/verification-log.md. The point of this "
        "file is the accumulated record; if entries are being removed, the gate above is being "
        "satisfied by emptying the thing it protects."
    )
