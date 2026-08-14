"""A deadline the register sets on itself must be executable, not remembered.

WHY THIS EXISTS

`docs/deliverable-verification.md` conditioned four metadata-only reference entries with an explicit
deadline: promoted by quoting a number with its section, or **removed by Round 35**. The register's
own reasoning for adding it was exactly right - *"an honest label with no deadline is a permanent
resident"*.

It was Round 39 before anyone noticed the deadline had passed. All four entries were still in
`references/REFERENCES.md`, still labelled "abstracts not individually opened", three rounds after
the date the register set. The rule was correct, greppable, and had no executor - which is the same
shape as checklist row `DP7`, correct and ignored for five consecutive rounds, and as
`test_agreement_disclosure_parity.py`'s subject, a disclosure shipped four times with the scope
wrong.

So the fix is not "remove the entries". It is that a deadline in the register fails a gate once it
expires. This parses the register for its own deadline sentences and checks the round we are in
against them.

WHERE THE CURRENT ROUND COMES FROM

The register's own section headings - "## Round 38: ..." - are the only place the project records
which round it is, and they are written as part of doing the round. Reading the maximum is therefore
self-maintaining: nothing extra to update, and a round that forgets to write its heading gets a
LOWER number, which fails safe by not firing deadlines early.

WHY THE EXEMPTION IS A STAMP AND NOT A KEYWORD

The first draft of this test exempted any paragraph containing "removed", "promoted", "executed" or
"closed". It passed on a register whose deadline had been blown by three rounds - because a sentence
that SETS a deadline says *"an entry is promoted only when ... is removed"*. The exemption matched
the deadline's own wording, so every deadline of this shape self-exempted and the gate could never
fire. That is this project's tautology class, the fourth instance: the FHE tie margin twice, the
`2*maxAbsError` bound, and now a gate written to catch unexecuted rules that could not catch one.

So the exemption is an explicit stamp, `DEADLINE EXECUTED Round <N>`, which nobody writes while
conditioning an item and which names the round that discharged it. The bare word "executed" would
have reintroduced the same bug the moment someone wrote "to be executed by Round 42".
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTER = REPO / "docs" / "deliverable-verification.md"

#: "any entry not promoted by **Round 35** is removed", and any future sentence of that shape.
_DEADLINE = re.compile(r"by \*\*Round (\d+)\*\*", re.I)
#: "## Round 38: the round ..." / "## Rounds 34-36, ..." - the register's own round markers.
_HEADING = re.compile(r"^##+ Rounds? (\d+)", re.M)
#: The ONLY thing that discharges a deadline. Deliberately unlike anything a conditioning sentence
#: says - see "WHY THE EXEMPTION IS A STAMP AND NOT A KEYWORD" above.
_EXECUTED = re.compile(r"DEADLINE EXECUTED Round (\d+)")
#: Start of the next register item ("**12. ...**") or section, whichever comes first. The unit of
#: discharge is the ITEM, not the paragraph: the stamp records what was done and why it ran late,
#: which is several sentences, and a paragraph-scoped search would push the author to cram it onto
#: the end of the conditioning sentence or, worse, to drop the reasoning to satisfy the parser.
_NEXT_ITEM = re.compile(r"^(?:\*\*\d+\.|#{2,} )", re.M)


def _item_containing(text: str, pos: int) -> str:
    """The register item the character at `pos` belongs to."""
    nxt = _NEXT_ITEM.search(text, pos)
    return text[pos: nxt.start() if nxt else len(text)]


#: The register's OTHER deadline shape. The round-based gate below was written first and would have
#: sailed straight past this one: the ZeroGPU Space item is blocked on "a Hugging Face account-age
#: unlock around 2026-09-01", a date the register set on itself and nothing watches. Covering one
#: shape and not the other is the "gate blind to its own scope" defect - the same one that left 122
#: of 356 tracked files unscanned by the Japanese-character check while its comment claimed the scope
#: was "exactly tracked". A deadline is a deadline whichever notation it wears.
#: A dated blocker must SAY it is one. The first attempt scanned for any date near words like
#: "blocker", "unlock" or "until", and produced three false positives out of three - "iterated until
#: clean", "the wording this file carried until Round 23", "Until now this paragraph said". The file
#: is dense with access dates because its credibility rests on them, so a heuristic over prose either
#: fires on provenance or is tuned until it fires on nothing. A gate that cries wolf gets ignored,
#: which is the DP7 failure this whole exercise exists to fix, so the marker is explicit.
_BLOCKED_UNTIL = re.compile(r"BLOCKED UNTIL (20\d\d)-(\d\d)-(\d\d)")

#: KNOWN LIMIT, stated rather than papered over: this catches dated blockers that use the marker, and
#: nothing else. Someone can still write "blocked until September" in prose and no gate will see it -
#: exactly as the round-based check only sees the `by **Round N**` phrasing. That is a real hole, and
#: naming it is the point; the alternative on offer was a heuristic with a 100% false-positive rate,
#: and claiming a scope the code does not have is this project's most-repeated defect.


def _current_round() -> int:
    rounds = [int(m) for m in _HEADING.findall(REGISTER.read_text())]
    assert rounds, "no '## Round N' heading in the register; the round can no longer be read from it"
    return max(rounds)


def test_the_register_records_which_round_it_is() -> None:
    """A parse that finds nothing would pass the deadline test below for the wrong reason."""
    n = _current_round()
    assert n >= 38, f"latest round heading reads {n}; the register has stopped being written"


def test_no_deadline_in_the_register_has_expired() -> None:
    """Every 'by Round N' the register sets on itself must still be in the future, or be gone.

    Failing here means one of two things, and both are actions rather than annoyances: the deadline
    passed and the thing it governs must now be promoted or removed, or the deadline was met and the
    sentence stating it should say so instead of reading as still-pending.
    """
    text = REGISTER.read_text()
    now = _current_round()
    expired = []
    for m in _DEADLINE.finditer(text):
        due = int(m.group(1))
        if due >= now:
            continue
        # Discharged deadlines stay in the file as history, marked with the stamp. Anything else is
        # still asserting a pending obligation, whatever else the item happens to say.
        item = _item_containing(text, m.start())
        if _EXECUTED.search(item):
            continue
        expired.append(f"Round {due} (we are in {now}): {' '.join(item.split())[:160]}")
    assert not expired, (
        "the register set these deadlines on itself and they have passed with nothing done:\n  "
        + "\n  ".join(expired)
        + "\n\nThe register's own words: 'an honest label with no deadline is a permanent resident'. "
        "A deadline with no executor is the same thing wearing a date. Promote the item by quoting "
        "the number and the section it came from, or remove it - then write 'DEADLINE EXECUTED "
        "Round <N>' in that paragraph. Do not move the date; that is how this one reached Round 39."
    )


def test_the_gate_fires_on_an_expired_deadline() -> None:
    """Prove the check can fail, with the seed entering where the defect entered.

    The first draft of this file passed on a register that HAD a three-round-overdue deadline,
    because its exemption matched the deadline's own wording. A green tick was worth nothing. So
    the seed here is a real conditioning paragraph in the register's own house style - the kind
    someone would actually write - and the assertion is that it is caught.
    """
    seeded = (
        "\n\n**99. A conditioned item.** This entry is promoted only when a number from it is "
        "quoted with its section, and any entry not promoted by **Round 2** is removed.\n\n"
    )
    text = REGISTER.read_text() + seeded
    now = max(int(x) for x in _HEADING.findall(text))
    caught = [
        int(m.group(1)) for m in _DEADLINE.finditer(text)
        if int(m.group(1)) < now and not _EXECUTED.search(_item_containing(text, m.start()))
    ]
    assert 2 in caught, (
        "the seeded overdue deadline was NOT caught - the exemption is matching the conditioning "
        "sentence again, which is exactly the bug this test was rewritten to remove"
    )


def test_no_dated_blocker_in_the_register_has_come_due() -> None:
    """A date the register set on itself is a deadline too, and needs the same stamp.

    This one fails on a wall-clock date with no code change, which is deliberate: the ZeroGPU Space
    item is the longest-carried entry in the file, and the register's own words are that it "must not
    appear in a fifth deferred list unqualified". A gate that only fires when someone edits the file
    would never fire on an item whose defining property is that nobody is editing it.

    When this goes red, do the thing the item says - wire it or delete it - and stamp the paragraph.
    Do NOT move the date forward; that is precisely how the round-based deadline reached Round 39.
    """
    text = REGISTER.read_text()
    today = _dt.date.today()
    due = []
    for m in _BLOCKED_UNTIL.finditer(text):
        when = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        item = _item_containing(text, m.start())
        if when > today or _EXECUTED.search(item):
            continue
        due.append(f"{when} (today {today}): {' '.join(item.split())[:160]}")
    assert not due, (
        "these dated blockers in the register have come due and nothing was recorded:\n  "
        + "\n  ".join(due)
        + "\n\nDo what the item says and write 'DEADLINE EXECUTED Round <N>' in it. Moving the date "
        "is not discharging it."
    )


def test_the_dated_gate_can_fire_and_ignores_provenance_dates() -> None:
    """Prove both directions: a due marker is caught, an access date is not.

    The seeded blocker uses the register's own house wording, and the seeded provenance line is
    copied from the shape the file actually uses hundreds of times. If the marker were ever loosened
    back into a prose heuristic, the second assertion is what would go red - and it is the one that
    matters, because a gate firing on provenance would be "fixed" by deleting provenance.
    """
    past = _dt.date.today() - _dt.timedelta(days=30)
    future = _dt.date.today() + _dt.timedelta(days=30)

    def _due(extra: str) -> list[str]:
        text = REGISTER.read_text() + extra
        return [
            "-".join(m.groups()) for m in _BLOCKED_UNTIL.finditer(text)
            if _dt.date(*map(int, m.groups())) <= _dt.date.today()
            and not _EXECUTED.search(_item_containing(text, m.start()))
        ]

    assert str(past) in _due(f"\n\n**96. Item.** BLOCKED UNTIL {past}, an account-age unlock.\n\n"), (
        "a dated blocker that has come due was NOT caught - the gate has collapsed to catching "
        "nothing, which passes for the same reason an empty check passes"
    )
    assert str(past) not in _due(f"\n\n**97. Item.** Fetched and read {past}. Nothing pending.\n\n"), (
        "an access date was mistaken for a deadline; the file is full of them and the only way to "
        "quieten that is to delete provenance, which would make the register worse to keep a test green"
    )
    assert str(future) not in _due(f"\n\n**98. Item.** BLOCKED UNTIL {future}, not yet due.\n\n"), (
        "a blocker that has NOT come due was reported; the gate would be red from the day it is written"
    )
    assert str(past) not in _due(
        f"\n\n**99. Item.** BLOCKED UNTIL {past}. DEADLINE EXECUTED Round 39, the Space was wired.\n\n"
    ), "the stamp did not discharge a dated blocker, so a discharged item stays red forever"


def test_the_stamp_discharges_only_the_item_it_is_written_in() -> None:
    """A stamp must not silence a deadline in a DIFFERENT item.

    Widening the search from paragraph to item is what made the real stamp count. Widened one step
    further - to the whole file - a single 'DEADLINE EXECUTED' anywhere would exempt every deadline
    forever, which is the self-exempting bug again in a new costume. This pins the boundary.
    """
    text = REGISTER.read_text() + (
        "\n\n**98. A discharged item.** DEADLINE EXECUTED Round 39. Nothing pending here.\n\n"
        "**99. An item that is still pending.** Any entry not promoted by **Round 3** is removed.\n\n"
    )
    now = max(int(x) for x in _HEADING.findall(text))
    caught = [
        int(m.group(1)) for m in _DEADLINE.finditer(text)
        if int(m.group(1)) < now and not _EXECUTED.search(_item_containing(text, m.start()))
    ]
    assert 3 in caught, (
        "a stamp in item 98 silenced the overdue deadline in item 99 - the discharge boundary has "
        "widened past the item, so one stamp now exempts unrelated rules"
    )
