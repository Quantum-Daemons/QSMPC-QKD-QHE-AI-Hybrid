"""A chart's accessibility mode has to match what its chart type can actually do.

WHY THIS EXISTS

`accessibilityLayer` is not a decoration. On a recharts bar or line chart it installs a real keyboard
path: focus the surface, press ArrowRight, and the tooltip walks the categorical axis reporting
values. On a **ScatterChart it does not**, because recharts steps an `activeIndex` along a
categorical axis and a scatter with two numeric axes has none - measured on the deployed page in
Round 36, not inferred. What it still does is set `role="application"` and `tabindex="0"`, which
tells a screen reader to leave browse mode and hand every keystroke to a widget that handles none.

`KB6` in the browser checklist says an application role appears **only** where the keyboard path
passed. That row is about where the role appears; it cannot say on which element, and the element is
where the distinction lives.

WHAT WENT WRONG, AND WHY A PROSE ROW DID NOT CATCH IT

Round 37 added `EncryptedAgreementChart` with `accessibilityLayer` on a ScatterChart - twenty lines
above `RealDataScatter`, which uses `role="img"`, and directly below the comment recording why. It
was introduced in the same commit that FIXED a missing accessible name: the title was verified on the
deployed page and the row governing the element was not re-read. One accessibility defect repaired
while another was created, in one edit.

This asserts the invariant on the source, per chart type, so the next person copying a neighbouring
chart cannot copy the wrong neighbour.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COMPONENTS = REPO / "web" / "src" / "components"

#: Chart types recharts gives a working `activeIndex` keyboard walk. Bar and line charts have a
#: categorical axis; a scatter does not.
KEYBOARD_CAPABLE = {"BarChart", "LineChart", "AreaChart"}
#: Chart types with no categorical axis, so no activeIndex, so no keyboard path to advertise.
KEYBOARD_INCAPABLE = {"ScatterChart"}

#: RECHARTS primitives only, named explicitly.
#:
#: The first version matched `<(\w+Chart)\b`, which also matched this project's own wrapper
#: components - `AncestryStrataChart` is a React component whose name ends in "Chart" and whose
#: BODY renders a correctly-configured `<BarChart accessibilityLayer title=...>`. The test reported
#: it as a chart with no accessible name, which is a defect in the check's scope, not in the code.
#: Same shape as Round 37's linkage check: a wrong scope reports confidently in both directions.
RECHARTS = KEYBOARD_CAPABLE | KEYBOARD_INCAPABLE | {"ComposedChart", "PieChart", "RadarChart"}

_OPEN = re.compile(r"<(\w+Chart)\b([^>]*)>", re.S)


def _charts() -> list[tuple[str, str, str, int]]:
    """(file, chart type, its opening-tag attributes, line number) for every RECHARTS chart in the UI."""
    out: list[tuple[str, str, str, int]] = []
    for path in sorted(COMPONENTS.glob("*.tsx")):
        text = path.read_text()
        for m in _OPEN.finditer(text):
            if m.group(1) not in RECHARTS:
                continue                      # a local wrapper component, not a recharts primitive
            out.append((path.name, m.group(1), m.group(2), text[: m.start()].count("\n") + 1))
    return out


def test_there_are_charts_to_check() -> None:
    """A parse that finds nothing passes every assertion below for the wrong reason."""
    charts = _charts()
    assert len(charts) >= 8, f"only {len(charts)} charts found - did the regex or the directory move?"


@pytest.mark.parametrize("kind", sorted(KEYBOARD_INCAPABLE))
def test_charts_without_a_keyboard_path_do_not_claim_one(kind: str) -> None:
    offenders = [
        f"{f}:{ln}" for f, k, attrs, ln in _charts()
        if k == kind and "accessibilityLayer" in attrs
    ]
    assert not offenders, (
        f"{offenders} put `accessibilityLayer` on a {kind}. That sets role=\"application\" and "
        "tabindex=0 without giving the points a keyboard path - recharts walks an activeIndex along a "
        "categorical axis and this chart type has none. An inert application role is worse than no "
        "role: it takes a screen-reader user out of browse mode and hands them a widget that ignores "
        "every key. Use `role=\"img\"` with a title, as the scatters in VerificationChart.tsx do."
    )


@pytest.mark.parametrize("kind", sorted(KEYBOARD_INCAPABLE))
def test_charts_without_a_keyboard_path_declare_an_image_role(kind: str) -> None:
    missing = [
        f"{f}:{ln}" for f, k, attrs, ln in _charts()
        if k == kind and 'role="img"' not in attrs
    ]
    assert not missing, (
        f"{missing} are a {kind} with no explicit `role=\"img\"`. Without it the SVG has no role a "
        "screen reader can announce, and the figcaption beside it - which is the text alternative "
        "this project actually provides for a scatter - is not tied to anything."
    )


@pytest.mark.parametrize("kind", sorted(KEYBOARD_CAPABLE))
def test_charts_with_a_keyboard_path_keep_it(kind: str) -> None:
    """The other direction. Dropping `accessibilityLayer` from a bar chart removes a working path."""
    missing = [
        f"{f}:{ln}" for f, k, attrs, ln in _charts()
        if k == kind and "accessibilityLayer" not in attrs
    ]
    assert not missing, (
        f"{missing} are a {kind} without `accessibilityLayer`. On this chart type it installs a real "
        "keyboard walk over the categorical axis, so removing it takes away a path that worked."
    )


def test_every_chart_has_an_accessible_name() -> None:
    """Round 34 shipped six charts whose `<title>` rendered empty; Round 37 shipped a seventh."""
    nameless = [
        f"{f}:{ln} ({k})" for f, k, attrs, ln in _charts() if "title=" not in attrs
    ]
    assert not nameless, (
        f"{nameless} have no `title` prop. recharts does not forward a `<title>` CHILD, so writing "
        "one there produces a role with a null accessible name - which is how Round 37's chart "
        "shipped, and how Round 34's six did before it. It must be the prop."
    )
