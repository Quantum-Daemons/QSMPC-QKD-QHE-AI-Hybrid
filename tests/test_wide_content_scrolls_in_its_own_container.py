"""Content wider than the viewport must scroll inside its own container, not the page.

WHY THIS EXISTS

`.card` carries `overflow-x: auto`, and its comment in `styles.css` records the audit that put it
there: at 390px the document measured 1326px, i.e. 936px of horizontal page scroll, "from two diagram
SVGs (990px, 930px), five mini-tables (968px), the 12-column benchmark table (930px) and a <code>
block (916px)".

`TwoPartyDiagram`'s SVG sets `minWidth: 680` ON PURPOSE - so it scrolls rather than shrinking to
~4.5px step labels, which its own comment argues is worse because "off-screen can be scrolled to".
That comment then says the scrolling is "what `.card { overflow-x: auto }` already provides".

It was right about the rule and wrong about the element. `UseCasePage` renders the diagram inside a
hand-styled div that copies the card's background, radius and padding and NOT its `overflow-x`.
Measured on the deployed site at a 500px viewport: **204px of document overflow, one genuine cause**
once nav-internal scrollers are discounted.

WHAT IS ASSERTED

That every element which deliberately sets a `minWidth` wider than a phone viewport is rendered
inside a container that scrolls - either `.card` or an explicit `overflowX`. Asserted on the source,
because the alternative is a live browser at a real mobile viewport and this has to hold in CI.

WHY IT IS NOT A DOM MEASUREMENT

It should be. The honest position is recorded in the browser checklist: a source check cannot see a
container introduced by a wrapper three levels up, so this is the cheap guard and the DOM measurement
at a real mobile viewport is the real one. What made the DOM measurement finally possible was CDP
device emulation; a window resize does not move the viewport, which is why this went unmeasured for
several rounds.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"

#: Narrower than any phone this would be read on. An element pinned wider than this cannot fit and
#: must therefore be inside something that scrolls.
PHONE_PX = 420


#: `/* ... */` (including the `{/* ... */}` form JSX uses) and `//` to end of line. Blocks first.
_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_LINE = re.compile(r"//[^\n]*")


def _tsx():
    """Source with COMMENTS STRIPPED, and this gate needed it for the usual reason.

    The comment added to `UseCasePage` explaining why the wrapper must scroll QUOTES `minWidth: 680`.
    Scanning raw text therefore classified `UseCasePage` itself as a wide component and flagged
    `App.tsx` for rendering it - a gate satisfied by its own documentation, which is the third
    instance of that trap in this session after the registry-note gate and the QRNG call-site check.
    Prose about a property is exactly what a file gets when someone fixes that property.
    """
    for p in sorted(WEB.rglob("*.tsx")):
        yield p, _LINE.sub(" ", _BLOCK.sub(" ", p.read_text()))


def test_the_card_rule_still_carries_the_overflow():
    """The rule the diagram's comment relies on. If it moves, every element trusting it is exposed."""
    css = (WEB / "styles.css").read_text()
    block = re.search(r"\.card\s*\{(.*?)\}", css, re.S)
    assert block, "`.card` is gone from styles.css; re-derive this test"
    assert re.search(r"overflow-x:\s*auto", block.group(1)), (
        "`.card` no longer sets `overflow-x: auto`. Its own comment records the audit that added it - "
        "936px of horizontal page scroll at 390px - and at least one component sets a deliberate "
        "`minWidth` on the explicit understanding that the card scrolls.")


def test_every_deliberately_wide_component_is_rendered_inside_a_scroller():
    """The property, checked at the RENDER SITE - which is where it actually failed.

    The first version of this checked the file that SETS the `minWidth`, and flagged
    `TwoPartyDiagram.tsx` even after the defect was fixed, because the scroller lives in
    `UseCasePage.tsx` - a different file. A gate that fires on correct code gets switched off, and
    this round has already recorded one of those.

    So: find components that pin a width wider than a phone, then find every place they are RENDERED
    and require that call site to provide a scrolling container. That is the relationship that broke -
    a wrapper reproducing the card's background, radius and padding and dropping its `overflow-x`.
    """
    wide_components = {}
    for path, src in _tsx():
        for m in re.finditer(r"minWidth:\s*(\d{3,})", src):
            if int(m.group(1)) > PHONE_PX:
                wide_components.setdefault(path.stem, int(m.group(1)))
    assert wide_components, "no wide components found; the anti-tautology test covers this"

    offenders = []
    for name, width in sorted(wide_components.items()):
        rendered_anywhere = False
        for path, src in _tsx():
            if path.stem == name:
                continue
            for m in re.finditer(rf"<{re.escape(name)}\b", src):
                rendered_anywhere = True
                # The enclosing markup ABOVE the tag - far enough to reach the wrapping element.
                window = src[max(0, m.start() - 900): m.start()]
                if "overflowX" in window or re.search(r'className="[^"]*\bcard\b', window):
                    continue
                line = src[: m.start()].count("\n") + 1
                offenders.append(
                    f"{path.relative_to(REPO)}:{line} renders <{name}> (pins minWidth {width}) with "
                    "no `overflowX` and no `.card` on the enclosing element")
        if not rendered_anywhere:
            offenders.append(f"<{name}> pins minWidth {width} and is rendered nowhere - dead, or the "
                             "scan cannot see its call site; either way re-derive this test")

    assert not offenders, (
        "these render sites put deliberately-wide content in a container that does not scroll:\n  "
        + "\n  ".join(offenders)
        + "\n\nA deliberate `minWidth` is a fine choice - shrinking a diagram to illegibility is worse "
        "than making it scrollable, and `TwoPartyDiagram` argues exactly that. But the scrolling has "
        "to exist AT THE CALL SITE. Hand-styling a wrapper with the card's background, radius and "
        "padding reproduces the look and drops the one property that mattered, which is how the "
        "use-case diagram overflowed the deployed page by 204px at a 500px viewport."
    )


def test_the_scan_has_a_subject():
    """Anti-tautology: no pinned widths at all would make the gate above vacuous."""
    found = [(p.name, int(m.group(1)))
             for p, s in _tsx() for m in re.finditer(r"minWidth:\s*(\d{3,})", s)
             if int(m.group(1)) > PHONE_PX]
    assert found, (
        f"no element pins a minWidth above {PHONE_PX}px, so this gate checks nothing. If the wide "
        "diagrams were made responsive instead, that is a better outcome - re-derive this test "
        "rather than leaving it green on an empty set.")
