"""A component that can render nothing must own its own frame.

WHY THIS EXISTS

Round 41 put `RealDataScatter` on `/orchestration` inside an unconditional `<div className="card">`
and wrote, on the lines immediately above it, that the component *"renders null where no overlay
ships, which covers the generic `demo` entry"*. Checklist row `OR12`, written in the same commit,
promised *"It renders nothing for the generic `demo` entry"*.

The component does render nothing. The wrapper did not. So `/orchestration?usecase=demo` drew an
empty bordered box after every run - a frame around an absence, which reads to a visitor as a chart
that failed to load rather than a chart that was never applicable.

This is the round's own subject one level in: **a boundary was named and one side of it verified.**
The component's behaviour was checked; the wrapper's was not. And it was not carelessness about the
component - the comment proves it was read.

WHY THE FIX IS A PROP AND NOT A CONDITION AT THE CALL SITE

The caller cannot answer the question. `demo` has no `REAL_2D_PATH` entry, and even a mapped use case
can fetch a payload with no points - an answer that arrives inside a `useEffect`, after render. Any
call-site condition is therefore a second, necessarily weaker copy of a decision the component
already makes. `RealDataScatter` takes `card` and draws the frame after its own `return null`, so
there is one decision and the frame is downstream of it.

WHAT THIS GATE ASSERTS, AND WHAT IT DOES NOT

It asserts the general shape, not the one instance: no `<div className="card">` in `web/src` may
contain a lone component that can `return null`. It does not check components that render an empty
`<div>` of their own accord, or a card that also contains a heading - a heading with nothing under it
is a different defect and this gate would report a false one if it tried to guess.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"

#: A card whose entire content is a single self-closing component call. Deliberately narrow: this is
#: the shape where an empty component leaves a visibly empty box and nothing else explains it.
_LONE_IN_CARD = re.compile(r'<div className="card"[^>]*>(.*?)</div>', re.S)
_ONLY_ONE_TAG = re.compile(r"<(\w+)\b[^>]*/>", re.S)

_COMMENT = re.compile(r"\{/\*.*?\*/\}|/\*.*?\*/|//[^\n]*", re.S)


def _code_only(src: str) -> str:
    """Comments blanked, line numbering preserved.

    THE FIRST VERSION OF THIS FILE STRIPPED COMMENTS AFTER MATCHING, AND CAUGHT NOTHING. The comment
    that explains this very fix, in `Orchestration.tsx`, contains the words `<div className="card">`
    as a quotation. The card regex matched inside the comment, `.*?</div>` then ran to the first real
    closing tag, and the seeded defect twenty lines below went unreported - the gate was reading
    prose about code as code.

    That is this project's recurring one: the QCNN gate was defeated by a single comment and was
    fixed the same way. Prose describing a defect is the most likely place for a defect's own text to
    appear, so a source gate that does not blank comments is at its blindest exactly where someone
    took the trouble to write down what went wrong.

    Newlines are kept so reported line numbers still point at the real line.
    """
    return _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), src)


def _tsx() -> list[Path]:
    files = sorted(WEB.rglob("*.tsx"))
    assert files, "no .tsx under web/src - the scope is empty and this gate would pass on nothing"
    return files


def _nullable_components() -> set[str]:
    """Component names whose body contains a `return null`.

    Read from the source rather than listed here on purpose. A hand-maintained list is a second
    place to forget something, and forgetting is the defect under test.
    """
    found: set[str] = set()
    for path in _tsx():
        src = path.read_text()
        src = _code_only(src)
        starts = [(m.start(), m.group(1)) for m in re.finditer(r"^(?:export )?function (\w+)\(", src, re.M)]
        for i, (pos, name) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(src)
            if re.search(r"\breturn null\b", src[pos:end]):
                found.add(name)
    assert found, "no component with a `return null` was found; the parse has stopped working"
    return found


def _frames_around_nullables(extra: tuple[Path, str] | None = None) -> list[str]:
    nullable = _nullable_components()
    hits: list[str] = []
    for path in _tsx():
        src = _code_only(extra[1] if extra and extra[0] == path else path.read_text())
        for m in _LONE_IN_CARD.finditer(src):
            only = _ONLY_ONE_TAG.fullmatch(m.group(1).strip())
            if only and only.group(1) in nullable:
                line = src.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(REPO)}:{line}: card wraps a lone <{only.group(1)}/>")
    return hits


def test_the_component_owns_its_frame() -> None:
    """The specific fix, pinned where it lives.

    Kept separate from the general check below because this is the instance that shipped, and a
    general gate going green tells you nothing about whether the particular defect was fixed or the
    call site merely deleted.
    """
    # `_code_only` here too, and not only in the scanner. This component's own docstring quotes
    # `<div className="card">` while explaining the defect, and the ordering assertion below is an
    # index comparison - the one kind of check a stray mention silently inverts.
    chart = _code_only((WEB / "components" / "VerificationChart.tsx").read_text())
    assert re.search(r"function RealDataScatter\(\{[^}]*\bcard\b", chart), (
        "RealDataScatter no longer takes `card`, so the frame has gone back to a caller that cannot "
        "know whether there is anything to frame"
    )
    # Bounded by the next function declaration, not by a comment. The first version cut the body at
    # `// The genomics model` - a comment, which `_code_only` had just blanked - so the `if/else`
    # fell through and the slice ran to end of file. The ordering assertion would then have compared
    # this component's `return null` against SOME OTHER component's card, and passed either way. A
    # boundary marker has to survive the normalisation applied before it is used.
    body = chart[chart.index("function RealDataScatter"):]
    nxt = re.search(r"\n(?:export )?function \w+\(", body)
    body = body[: nxt.start()] if nxt else body
    assert "RealDataScatter" not in body[10:], "the body slice has run past its own function"
    null_at = body.index("return null")
    frame_at = body.index('className="card"')
    assert frame_at > null_at, (
        "the card frame is drawn before RealDataScatter's `return null`, so the empty box is back - "
        "the frame has to be downstream of the decision, not beside it"
    )

    page = (WEB / "pages" / "Orchestration.tsx").read_text()
    assert "<RealDataScatter usecase={usecase} card />" in page, (
        "the orchestration page is no longer asking the component for its frame"
    )


def test_no_card_wraps_a_component_that_can_render_nothing() -> None:
    """The class, not the instance."""
    hits = _frames_around_nullables()
    assert not hits, (
        "these frames are drawn unconditionally around components that can render nothing:\n  "
        + "\n  ".join(hits)
        + "\n\nAn empty bordered box reads as a chart that failed to load. The component knows "
        "whether it will render and the caller does not, so give the component a `card` prop and "
        "draw the frame after its own `return null` - as RealDataScatter does."
    )


def test_the_gate_fires_on_the_defect_that_shipped() -> None:
    """Seed the exact wrapper Round 41 wrote, at the exact call site, and require a red.

    Not a synthetic component and not a synthetic file: the seed is the deleted code, restored into
    the page it was deleted from. A gate for a defect that shipped should be provable against that
    defect rather than against a stand-in that might differ in the way that mattered.
    """
    page = WEB / "pages" / "Orchestration.tsx"
    src = page.read_text()
    assert "<RealDataScatter usecase={usecase} card />" in src, "call site moved; re-derive the seed"
    seeded = src.replace(
        "<RealDataScatter usecase={usecase} card />",
        '<div className="card" style={{ marginTop: 16 }}>\n'
        "          <RealDataScatter usecase={usecase} />\n"
        "        </div>",
    )
    hits = _frames_around_nullables(extra=(page, seeded))
    assert any("RealDataScatter" in h for h in hits), (
        "the restored Round 41 wrapper was NOT reported. Either `return null` is no longer detected "
        "in the component or the card shape is no longer matched; either way this gate is passing "
        "for the same reason an empty check passes."
    )
