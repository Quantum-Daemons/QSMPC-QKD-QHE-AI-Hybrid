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

ROUND 42: THIS GATE NAMED A BOUNDARY AND CHECKED ONE SIDE OF IT

`references/REFERENCES.md` carried *"Next round: find the Cooperation Group publication itself before
writing any of it down as a date"* from **Round 33 to Round 42** - nine rounds. Nothing was broken.
The obligation was simply outside this file's scope on **both** axes at once:

- the **file set** was one file, `docs/deliverable-verification.md`, and the promise was in another;
- the **phrasing** was `by **Round N**` and `BLOCKED UNTIL <date>`, and the promise was a third.

That is this round's subject arriving inside the test written to stop obligations from being
forgotten: a gate that verifies one side of a boundary it names. The docstring above even *stated*
the phrasing hole - "someone can still write 'blocked until September' in prose and no gate will see
it" - and stating a hole is not covering it.

Both axes are now widened. The file set is every **tracked** markdown file under `docs/` and
`references/`, so it is what a clone contains rather than what one machine has. The phrasings are
three.

WHY `Next round:` IS ALWAYS OVERDUE

`by **Round 35**` carries its own due date; `Next round:` does not. It is relative to a round the
sentence never names, and the only file with round headings is the register, so there is no general
way to compute when it falls due. The rule is therefore that this phrasing is due **on sight**:
whoever reads it is, by construction, in a round at or after the one it meant.

That makes the phrasing unusable without discharging it, which is the intended outcome and not a
side effect. The register's own words are that *"an honest label with no deadline is a permanent
resident"*; a deadline nobody can evaluate is the same thing with a date-shaped hole in it. Write
`by **Round N**` and the gate will wait for you.
"""

from __future__ import annotations

import datetime as _dt
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTER = REPO / "docs" / "deliverable-verification.md"

#: Gitignored, so it is not in a clone and a promise in it binds one machine. Same exclusion, and the
#: same reason, as `test_model_artifact_inventory.py`'s `UNPUBLISHED`.
UNPUBLISHED = {"docs/browser-verification-checklist.md"}


def _watched() -> list[Path]:
    """Tracked markdown under `docs/` and `references/`.

    TRACKED, not globbed. A filesystem walk would pull in the gitignored checklist, and a promise
    only one machine can see is not the thing this gate is about. `git ls-files` is the same
    definition `test_model_artifact_inventory.py` uses, and it failed there first - in CI, where the
    checklist does not exist - which is why it is the definition here rather than a glob.
    """
    out = subprocess.run(
        ["git", "ls-files", "docs", "references"], cwd=REPO, capture_output=True, text=True, check=True
    )
    files = [f for f in out.stdout.split() if f.endswith(".md") and f not in UNPUBLISHED]
    assert files, "git ls-files returned no markdown under docs/ or references/ - the scope is empty"
    return [REPO / f for f in files]

#: TOLERANT OF ONE LINE WRAP between its words, and it governs ALL THREE markers below.
#:
#: These files are hard-wrapped at 100 columns, so any multi-word marker can be split by a reflow.
#: Round 42 learned this on the stamp: two of its four `DEADLINE EXECUTED Round <N>` stamps landed
#: near the margin, the wrap separated `DEADLINE` from `EXECUTED`, and the gate reported discharged
#: items as pending. The fix was applied to `_EXECUTED` and to nothing else.
#:
#: ROUND 44: the identical hole was still open in `_DEADLINE`, twelve lines away in the same file.
#: `docs/deliverable-verification.md:1992` carries "**Resolve or condition by\n**Round 45**.**" - the
#: wrap fell between `by` and `**Round`, so the register's authoritative statement of that obligation,
#: the one that says HOW to discharge it, was invisible to the gate that exists to surface it. The
#: lesson was learned on one regex and not carried to its sibling: a boundary named on one side.
#:
#: At most ONE newline per gap, never a blank line - a marker must not be assembled out of one
#: paragraph's last word and the next paragraph's first.
_WRAP = r"[ \t]*\n?[ \t]*"

#: "any entry not promoted by **Round 35** is removed", and any future sentence of that shape.
#:
#: ROUND 45: the opening `**` is now OPTIONAL, because an obligation can be bolded as a whole clause
#: rather than round-by-round. `references/qcnn-width-ablation.md:127` writes
#:
#:     the only one not re-run at five seeds. **Re-run it by Round 45** or restate it as one draw.
#:
#: so the literal text at the marker is `by Round 45**`: the emphasis opens before `Re-run`, and no
#: `**` sits in front of `Round` for the old pattern to anchor on. That file returned ZERO matches
#: while carrying an obligation due this round. Round 44 taught this regex to tolerate a LINE WRAP
#: between its words and never asked what else could sit between them - the same lesson one costume
#: over, which is the shape this entire round is about.
#:
#: The closing `**` stays REQUIRED, and that is load-bearing rather than incidental. Dropping it too
#: was measured against the corpus and sweeps in four narrative sentences that are not obligations:
#: "had been *restored* by Round 20 on the stated grounds", "filed by Round 33 as a defect against
#: itself", "SUPERSEDED IN PART BY ROUND 30", and one at `references/REFERENCES.md:1044`. Prose says
#: "by Round N"; a deadline bolds the round. That is the discriminator, so it is kept.
#:
#: `_EXECUTED` and `_BLOCKED_UNTIL` below need no equivalent change and are deliberately untouched:
#: neither pattern contains an asterisk, so no emphasis boundary can hide either. Stated here rather
#: than left silent, because "fixed one regex of a sibling set" is the very defect being repaired and
#: the reader is owed the check rather than the assurance.
#: ROUND 46: the closing `**` no longer has to be ADJACENT to the digits either.
#:
#: Round 45 made the opening `**` optional and asserted that the closing one was the discriminator
#: against narrative prose. Both halves were right and the second was still one-sided: it required
#: the `**` immediately after the number, so a clause bolded as a whole
#:
#:     **Re-run it by Round 46 or restate it as one draw**
#:
#: leaves `by Round 46 or restate...` at the marker with no adjacent `**` at all - zero matches, the
#: same hole one word further along. Round 45 fixed the opening and left the closing, which is the
#: sibling-pair defect that file's own comment says it exists to stop.
#:
#: `_inside_bold` below replaces adjacency with the property adjacency was standing in for: is this
#: marker inside an emphasis span? Counting `**` from the start of the line and asking whether the
#: count is ODD answers that exactly, and it still excludes the four narrative sentences ("restored
#: by Round 20", "SUPERSEDED IN PART BY ROUND 30") because those carry no emphasis at all.
#: CASE-SENSITIVE, and that is what makes the widening safe. Dropping adjacency alone swept in
#: `**SUPERSEDED IN PART BY ROUND 30, marked in Round 33**` - bolded narrative, not an
#: obligation. The register writes deadlines in sentence case ("by **Round N**") and writes
#: supersession markers in caps, so case is the discriminator adjacency was standing in for.
#: Measured over the whole corpus: 9 matches, all real, zero narrative.
_DEADLINE_LOOSE = re.compile(r"by" + _WRAP + r"(?:\*\*)?Round" + _WRAP + r"(\d+)(\*\*)?")


def _inside_bold(text: str, pos: int) -> bool:
    """Is `pos` inside a `**...**` span? True when an odd number of `**` precede it on its line."""
    line_start = text.rfind("\n", 0, pos) + 1
    return text.count("**", line_start, pos) % 2 == 1


class _Deadline:
    """`_DEADLINE`'s interface, with adjacency replaced by emphasis-membership."""

    pattern = _DEADLINE_LOOSE.pattern

    @staticmethod
    def finditer(text):
        for m in _DEADLINE_LOOSE.finditer(text):
            if m.group(2) or _inside_bold(text, m.start()):
                yield m

    @staticmethod
    def search(text):
        return next(_Deadline.finditer(text), None)


_DEADLINE = _Deadline
#: "## Round 38: the round ..." / "## Rounds 34-36, ..." - the register's own round markers.
_HEADING = re.compile(r"^##+ Rounds? (\d+)", re.M)
#: The ONLY thing that discharges a deadline. Deliberately unlike anything a conditioning sentence
#: says - see "WHY THE EXEMPTION IS A STAMP AND NOT A KEYWORD" above.
#:
_EXECUTED = re.compile(_WRAP.join(["DEADLINE", "EXECUTED", "Round", r"(\d+)"]))
#: Start of an item: the register's numbered form ("**12. ...**"), a top-level bold bullet
#: ("- **The EU NIS ...**", which is how `references/REFERENCES.md` and half the register write an
#: entry), or a section heading. The unit of discharge is the ITEM, not the paragraph: the stamp
#: records what was done and why it ran late, which is several sentences, and a paragraph-scoped
#: search would push the author to cram it onto the end of the conditioning sentence or, worse, to
#: drop the reasoning to satisfy the parser.
#: Anchored at column 0 on purpose - indented `  - **...**` sub-bullets elaborate an item rather than
#: starting one, and treating them as boundaries would cut an entry into pieces the stamp cannot span.
_ITEM_START = re.compile(r"^(?:\*\*\d+\.|- \*\*|#{2,} )", re.M)


def _item_containing(text: str, pos: int) -> str:
    """The item the character at `pos` belongs to - from its START, not from `pos`.

    The first version sliced forward from the match, so a stamp written ABOVE the deadline sentence
    did not count. That is not hypothetical house style: the corrected EU entry opens by saying it is
    resolved and quotes the promise it discharged several paragraphs later, which is the natural way
    to write it and the way a reader wants to read it. Searching from the item's start accepts both
    orders and still refuses to look past the item, which is the boundary the tests below pin.
    """
    starts = [m.start() for m in _ITEM_START.finditer(text) if m.start() <= pos]
    begin = starts[-1] if starts else 0
    nxt = _ITEM_START.search(text, pos)
    return text[begin: nxt.start() if nxt else len(text)]


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
#: Wrap-tolerant for the same reason as its two siblings. It has not been split yet; fixing only the
#: regex that has already failed is how this hole reached Round 44 in the first place.
_BLOCKED_UNTIL = re.compile(_WRAP.join(["BLOCKED", "UNTIL", r"(20\d\d)-(\d\d)-(\d\d)"]))

#: KNOWN LIMIT, stated rather than papered over: this catches dated blockers that use the marker, and
#: nothing else. Someone can still write "blocked until September" in prose and no gate will see it -
#: exactly as the round-based check only sees the `by **Round N**` phrasing. That is a real hole, and
#: naming it is the point; the alternative on offer was a heuristic with a 100% false-positive rate,
#: and claiming a scope the code does not have is this project's most-repeated defect.
#: Round 42 closed the half of that hole this project actually writes - see below - and the sentence
#: above is left standing because the other half is still open and a gate should not be read as
#: covering more than it does.

#: The third phrasing, and the one that survived nine rounds: a promise relative to a round it never
#: names. Due on sight - see "WHY `Next round:` IS ALWAYS OVERDUE" in the module docstring.
#:
#: CASE-SENSITIVE, and that is the whole of its precision. These files are dense with prose ABOUT
#: future rounds that promises nothing: "so the next round does not re-derive", "one caution for the
#: next round", "instructions for the next round:". Every one of those is lowercase because it is a
#: clause inside a sentence; an obligation is written as a sentence opener, `Next round:`, or as the
#: register's italic reopening marker. Measured on the tracked set before this was written: the
#: case-sensitive form matched 5 lines, all five genuine obligations or quotations of one, and the
#: case-insensitive form added 3 lines, all three false. A gate that fires on commentary gets an
#: exemption bolted on within two rounds, and the exemption is where the tautologies come from.
_NEXT_ROUND = re.compile(r"Next round:|Reopens next round")


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


def test_no_relative_promise_is_left_pending_anywhere_in_docs_or_references() -> None:
    """The widened gate: three phrasings, every tracked doc, not one file and one phrasing.

    This is the check that would have caught the EU lead in Round 34 instead of Round 42. It reads
    the same three markers the register uses and applies them to every file a clone contains under
    `docs/` and `references/`, because the promise that survived nine rounds was in `references/`.

    Discharging one means the same thing it has always meant: do the thing, then write
    `DEADLINE EXECUTED Round <N>` in the item. Quoting a discharged promise as history is fine and is
    how three of these read today - the quote sits inside an item that carries the stamp.
    """
    now = _current_round()
    today = _dt.date.today()
    pending: list[str] = []

    for path in _watched():
        text = path.read_text()
        rel = path.relative_to(REPO)
        for regex, due_here in (
            (_NEXT_ROUND, lambda m: True),
            (_DEADLINE, lambda m: int(m.group(1)) < now),
            (_BLOCKED_UNTIL, lambda m: _dt.date(*map(int, m.groups())) <= today),
        ):
            for m in regex.finditer(text):
                if not due_here(m):
                    continue
                item = _item_containing(text, m.start())
                if _EXECUTED.search(item):
                    continue
                line = text.count("\n", 0, m.start()) + 1
                pending.append(f"{rel}:{line}: {' '.join(item.split())[:150]}")

    assert not pending, (
        "these promises are due and nothing records them as done:\n  "
        + "\n  ".join(pending)
        + "\n\n'Next round:' is due on sight - it names no round, so no round can be waited for. Do "
        "the thing and stamp the item 'DEADLINE EXECUTED Round <N>', or rewrite the promise as "
        "'by **Round N**' and the gate will wait. Moving the promise is not discharging it: that is "
        "how one sentence in references/REFERENCES.md reached its ninth round."
    )


def test_the_widened_gate_fires_on_the_side_that_was_added() -> None:
    """Seed the NEW side, not the side the old gate already watched.

    A seed in `docs/deliverable-verification.md` would prove nothing here: that file and the
    `by **Round N**` phrasing were already covered, and a green tick from re-testing covered ground
    is how this gate stayed narrow for nine rounds. So the seed is BOTH new axes at once - a
    `Next round:` sentence in a `references/` file - and the assertion names the file it was found
    in, so a future narrowing that quietly drops `references/` from the scope goes red here rather
    than passing on the register alone.
    """
    watched = {str(p.relative_to(REPO)) for p in _watched()}
    assert "references/REFERENCES.md" in watched, (
        "references/REFERENCES.md has left the watched set - the file that carried the nine-round "
        "promise is exactly the one this gate was widened to reach"
    )

    real = (REPO / "references" / "REFERENCES.md").read_text()

    def _pending(extra: str) -> set[int]:
        text = real + extra
        return {
            m.start() for m in _NEXT_ROUND.finditer(text)
            if not _EXECUTED.search(_item_containing(text, m.start()))
        }

    # DIFFED against the unseeded file, not asserted as a non-empty list. The first version checked
    # `assert caught` and went green off a promise that was already in REFERENCES.md and not yet
    # discharged - the seed contributed nothing and the test would have passed with the seeding line
    # deleted. Appending leaves every existing offset where it was, so the difference is exactly the
    # seed.
    base = _pending("")
    seeded = (
        "\n\n- **A metadata-only entry.** Seen in secondary summaries only, so nothing may be quoted "
        "from it. Next round: find the publication itself before writing any of it down as a date.\n\n"
    )
    assert _pending(seeded) - base, (
        "the seeded promise in references/ was not caught - either the phrasing is no longer "
        "matched or the file dropped out of scope, which returns the gate to the shape that missed "
        "a promise for nine rounds"
    )

    stamped = seeded.rstrip() + " DEADLINE EXECUTED Round 42, the primary source was loaded.\n\n"
    assert not _pending(stamped) - base, (
        "the stamp did not discharge a promise in references/, so a genuinely finished obligation "
        "would stay red and the only way to green would be to delete the history"
    )


def test_the_stamp_survives_a_line_wrap_but_not_a_paragraph_break() -> None:
    """A four-word marker in a file hard-wrapped at 100 columns will be split. Both sides pinned.

    Found by writing the Round 42 stamps: two of the four landed near the margin, the wrap put
    `DEADLINE` and `EXECUTED` on separate lines, and the gate reported items as pending that read as
    discharged. Two of the stamps in the tracked files are wrapped today, so this tolerance is load
    bearing and not a convenience.

    The second half is the limit. `\\s+` would have let a stamp be assembled from one paragraph's
    last word and the next paragraph's first - a marker that appears where nobody wrote it, which is
    a worse failure than one that fails to appear where somebody did.
    """
    assert _EXECUTED.search("... its own source. DEADLINE\n  EXECUTED Round 42."), (
        "a stamp split by an ordinary line wrap is not recognised; reflowing a paragraph would "
        "silently re-open every item in it"
    )
    assert _EXECUTED.search("resolved.\nDEADLINE EXECUTED\nRound 42, the source was loaded."), (
        "a stamp wrapped before the round number is not recognised - the number is the last word, "
        "so it is the one most likely to be pushed onto the next line"
    )
    assert not _EXECUTED.search("... nothing was DEADLINE\n\nEXECUTED Round 42 says the next item."), (
        "a blank line was spanned, so a stamp can now be assembled from two different paragraphs - "
        "the marker would appear where nobody wrote it"
    )


def test_the_stamp_is_read_when_it_is_written_above_the_promise() -> None:
    """Order within an item must not decide whether a discharge counts.

    The first `_item_containing` sliced forward from the deadline match, so this passed only when the
    stamp came after. The corrected EU entry opens with its resolution and quotes the promise several
    paragraphs down - the readable order - and under the old slicing that entry would have been
    reported as pending while saying, in its first line, that it was not.
    """
    text = (
        "- **An entry.** DEADLINE EXECUTED Round 42, resolved at the primary source.\n\n"
        "  Its own history: this entry ended *\"Next round: find the publication\"* and nothing "
        "enforced it.\n\n"
        "- **A different entry.** Next round: do the other thing.\n"
    )
    pending = [
        text.count("\n", 0, m.start()) + 1 for m in _NEXT_ROUND.finditer(text)
        if not _EXECUTED.search(_item_containing(text, m.start()))
    ]
    # Line 3 is the quoted history inside the stamped entry; line 5 is the unstamped one.
    assert 5 in pending, "the unstamped entry was missed, so the gate has stopped seeing promises"
    assert 3 not in pending, (
        f"the quoted history under a stamped entry was reported as pending (lines {pending}); a "
        "stamp above its own promise is the order these entries are written in"
    )


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


def test_every_marker_survives_a_line_wrap_and_none_spans_a_paragraph() -> None:
    """All THREE markers, not the one that has already failed.

    Round 42 made `_EXECUTED` wrap-tolerant after two of its own stamps were split by a reflow, and
    stopped there. `_DEADLINE` kept a literal space for two more rounds, and
    `docs/deliverable-verification.md:1992` - "**Resolve or condition by\\n**Round 45**.**" - was
    invisible to the gate the whole time. That is the register's authoritative statement of the RI9
    obligation, the sentence that says how to discharge it, unseen by the check that exists to
    surface it.

    The same obligation is also written same-line 134 lines earlier, so the gate would still have
    fired eventually - from a different item, carrying less. "It would have been caught anyway" is
    how a hole survives; the sentence that explains the work is the one that has to be visible.

    Both directions are asserted for each marker, because tolerance without a bound is how a marker
    starts matching text nobody wrote.
    """
    cases = [
        (_DEADLINE, "resolve or condition by **Round 45**.", "condition by\n**Round 45**.",
         "condition by\n\n**Round 45**."),
        (_EXECUTED, "DEADLINE EXECUTED Round 43, done.", "DEADLINE\n  EXECUTED Round 43, done.",
         "DEADLINE\n\nEXECUTED Round 43, done."),
        (_BLOCKED_UNTIL, "BLOCKED UNTIL 2026-09-01, an unlock.", "BLOCKED\n  UNTIL 2026-09-01.",
         "BLOCKED\n\nUNTIL 2026-09-01."),
    ]
    for rx, same_line, wrapped, across_blank in cases:
        assert rx.search(same_line), f"{rx.pattern!r} no longer matches its own unwrapped form"
        assert rx.search(wrapped), (
            f"{rx.pattern!r} is not wrap-tolerant. These files are hard-wrapped at 100 columns, so a "
            "reflow would silently hide the marker - and reflowing prose to satisfy a parser is the "
            "'fix' this tolerance exists to make unnecessary."
        )
        assert not rx.search(across_blank), (
            f"{rx.pattern!r} spans a blank line, so a marker can be assembled from one paragraph's "
            "last word and the next paragraph's first - it would appear where nobody wrote it"
        )


def test_a_deadline_survives_being_bolded_as_a_whole_clause() -> None:
    """The other way a marker can be split: by emphasis, not by whitespace.

    Round 44 made `_DEADLINE` tolerant of a line wrap. The very next obligation written in this
    project used a different shape and was invisible anyway:

        **Re-run it by Round 45** or restate it as one draw.      references/qcnn-width-ablation.md:127

    The bold span opens before `Re-run`, so at the marker the literal text is `by Round 45**` with no
    `**` before `Round`. One file, one obligation, due this round, zero matches. Round 44 asked "can
    whitespace get between these words" and not "what else can".

    The bound is asserted in the same breath, because a marker that matches everything is worse than
    one that matches too little - it fires on history and teaches the reader to ignore it. Plain
    narrative prose says "by Round N" with no emphasis at all, and four such sentences exist in the
    watched corpus today; the closing `**` is what separates an obligation from a recollection.
    """
    bolded_clause = "**Re-run it by Round 45** or restate it as one draw."
    m = _DEADLINE.search(bolded_clause)
    assert m, (
        "`_DEADLINE` cannot see an obligation whose whole clause is bolded, which is how "
        "references/qcnn-width-ablation.md:127 is written. The opening `**` must be optional."
    )
    assert int(m.group(1)) == 45, f"matched but extracted round {m.group(1)!r} instead of 45"

    # Both older forms keep working - this is a widening, and a widening that drops a case is a swap.
    assert _DEADLINE.search("resolve or condition by **Round 45**."), "the fully-bolded form broke"
    assert _DEADLINE.search("condition by\n**Round 45**."), "the wrap tolerance was lost"

    # The bound. Each of these is real text from the watched corpus, and none is an obligation.
    # ROUND 46: the closing `**` no longer has to touch the digits either.
    m46 = _DEADLINE.search("**Re-run it by Round 46 or restate it as one draw**")
    assert m46 and int(m46.group(1)) == 46, (
        "`_DEADLINE` cannot see a clause whose bold span closes AFTER the round number. Round 45 "
        "made the opening `**` optional and left the closing one adjacent - the same hole one word "
        "further along, in the sibling this file's own comment says it exists to stop."
    )

    for narrative in (
        "**SUPERSEDED IN PART BY ROUND 30, marked in Round 33**",
        "had been *restored* by Round 20 on the stated grounds that it moves the bit count",
        "made MORE specific in Round 33 and then filed by Round 33 as a defect against itself",
        "**SUPERSEDED IN PART BY ROUND 30, marked in Round 33**",
        "renumbered with its section, or removed by Round 35. **Removed in Round 39.**",
    ):
        assert not _DEADLINE.search(narrative), (
            f"`_DEADLINE` now fires on narrative prose: {narrative!r}. A deadline bolds the round; "
            "recollection does not. Dropping the closing `**` requirement is what breaks this."
        )
