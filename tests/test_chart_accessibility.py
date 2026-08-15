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


# --------------------------------------------------------------------------------------------- #
#  SC 1.4.1 Use of Colour: a category must not be distinguishable by colour alone
# --------------------------------------------------------------------------------------------- #

#: Round 39 gave a second channel to `RealDataScatter` (circle vs triangle), the histogram, the
#: forecast lines (solid vs dashed) and the confusion bars (solid vs hatched fill), and MISSED the
#: private-inference agreement scatter - the one chart where colour carries the most, because red
#: there means the encrypted path disagreed with the plaintext model.
#:
#: The sweep was done from memory rather than from the source. So this check enumerates the series
#: rather than listing the charts: any chart drawing MORE THAN ONE `<Scatter>` must give each one a
#: `shape`. One series has no category to confuse and is left alone.
#:
#: Scatter only, deliberately. Lines carry `strokeDasharray`, bars carry a pattern fill, and each
#: needs a differently-shaped check; claiming to cover all four here and testing one is the defect
#: this file already records in its own scope comment above.
_CHART_BLOCK = re.compile(r"<ScatterChart\b(?:[^>]|\n)*?>((?:.|\n)*?)</ScatterChart>", re.S)


def _scatter_tags(block: str) -> list[str]:
    """Attribute text of each `<Scatter ... />` in a chart body.

    A BRACE-AWARE SCAN, NOT A REGEX, AND THE REGEX VERSION SKIPPED THE ONE CHART THIS CHECK EXISTS
    FOR. It was `<Scatter\b((?:[^>]|\n)*?)/>`, and the agreement scatter's props read
    `data={plot.points.filter((p) => p.agree)}` - an arrow function, so a `>` inside the tag. `[^>]`
    stopped dead there, the tag never matched, and the whole chart fell out of scope.

    The guard did not save it. `assert charts` passed because `RealDataScatter`, whose props happen
    to contain no arrow, matched fine - so the check reported healthy while examining a chart that
    was already correct and silently ignoring the one that was not. A scope guard that counts is
    weaker than one that names: `test_the_agreement_scatter_is_in_scope` below names it.
    """
    out: list[str] = []
    i = 0
    while (i := block.find("<Scatter", i)) != -1:
        if block[i + len("<Scatter")].isalnum():      # <ScatterChart, not <Scatter
            i += 1
            continue
        j, depth = i + len("<Scatter"), 0
        while j < len(block):
            c = block[j]
            if c in "{(":
                depth += 1
            elif c in "})":
                depth -= 1
            elif c == ">" and depth == 0:
                break
            j += 1
        out.append(block[i:j])
        i = j + 1
    return out


def _flat_scatter_series() -> tuple[list[str], list[str]]:
    """(offending series, names of every multi-series scatter chart examined)."""
    flat: list[str] = []
    seen: list[str] = []
    for path in sorted(COMPONENTS.glob("*.tsx")):
        text = path.read_text()
        for block in _CHART_BLOCK.finditer(text):
            series = _scatter_tags(block.group(1))
            if len(series) < 2:
                continue                      # one series: no category, nothing to confuse
            line = text[: block.start()].count("\n") + 1
            names = [
                (re.search(r'name=\{?"?([^"}\n]+)', a).group(1).strip()
                 if re.search(r'name=\{?"?([^"}\n]+)', a) else "?")
                for a in series
            ]
            seen.append(f"{path.name}:{line} [{', '.join(names)}]")
            for attrs, name in zip(series, names):
                if "shape=" not in attrs:
                    flat.append(f"{path.name}:{line} series {name!r} has no shape")
    return flat, seen


def test_the_agreement_scatter_is_in_scope() -> None:
    """Name the chart, do not count the charts.

    The private-inference agreement scatter is the reason this section exists and was the chart the
    first parse could not see. Counting found two series elsewhere and reported healthy. This fails
    if the parse ever loses it again, whatever else it finds.
    """
    _, seen = _flat_scatter_series()
    assert any("agrees" in s for s in seen), (
        "the private-inference agreement scatter is not among the multi-series charts examined:\n  "
        + "\n  ".join(seen or ["(none)"])
        + "\n\nIt is the chart this check was written for - red there means the ENCRYPTED path "
        "disagreed with plaintext. If the parse cannot reach it, the check is green about charts "
        "that were never the problem."
    )


def test_a_scatter_category_is_never_colour_alone() -> None:
    """SC 1.4.1, Level A, asserted per series rather than per file."""
    flat, seen = _flat_scatter_series()
    assert seen, (
        "no multi-series scatter chart was found; this check would pass on an empty set, which is "
        "how the agreement scatter went three rounds without a second channel"
    )
    assert not flat, (
        "these scatter series are told apart by fill colour alone:\n  "
        + "\n  ".join(flat)
        + "\n\nSC 1.4.1 Level A. Add a `shape` - the pattern is in this same file, where "
        "`RealDataScatter` uses circle against triangle - and name the shape in the series name so "
        "the legend and tooltip say it too."
    )


def test_a_shaped_scatter_does_not_animate_its_symbols() -> None:
    """A `shape` that renders at zero size is worse than no `shape` at all.

    FOUND ON THE DEPLOYED PAGE, WITH EVERY GATE GREEN. Round 42 added `shape="circle"` and
    `shape="cross"` to the agreement scatter for SC 1.4.1 and shipped it. On the live site all 30
    symbols were in the DOM, correctly positioned, and rendered `d="M0,0"` - a legend naming two
    shapes above an empty plot area. Passing `shape` moves recharts off its default renderer onto
    `<Symbols>`, whose `size` is an animated prop, and with animation left on the symbols never left
    their `from` state.

    `RealDataScatter` in the same file has carried `shape` AND `isAnimationActive={false}` together
    since Round 34. Half the pattern was copied - the half the checklist row named.

    The gate above cannot catch this: it reads the source, finds the `shape` prop it asked for, and
    has no way to see a path of length zero. So this is the pairing asserted as its own invariant,
    which is what "write the invariant at the boundary" means when the boundary is between a prop
    being PRESENT and it having an EFFECT.
    """
    bad: list[str] = []
    for path in sorted(COMPONENTS.glob("*.tsx")):
        text = path.read_text()
        for block in _CHART_BLOCK.finditer(text):
            for attrs in _scatter_tags(block.group(1)):
                if "shape=" in attrs and "isAnimationActive={false}" not in attrs:
                    line = text[: block.start()].count("\n") + 1
                    name = re.search(r'name=\{?"?([^"}\n]+)', attrs)
                    bad.append(
                        f"{path.name}:{line} series "
                        f"{(name.group(1).strip() if name else '?')!r} has shape but animates"
                    )
    assert not bad, (
        "these scatter series pass `shape` while recharts animates the symbol size:\n  "
        + "\n  ".join(bad)
        + "\n\nThey will render d=\"M0,0\" - present, positioned, invisible. Add "
        "`isAnimationActive={false}`, as RealDataScatter does. A shape channel that draws nothing is "
        "not an accessibility fix, it is a blank chart with a more detailed legend."
    )
