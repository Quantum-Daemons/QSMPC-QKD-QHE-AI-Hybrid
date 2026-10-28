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

import json
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


# --------------------------------------------------------------------------------------------- #
#  SC 1.4.11 Non-text Contrast: every data mark reaches 3:1 against the card it is drawn on
# --------------------------------------------------------------------------------------------- #

#: Open since Round 34 and "still unmeasured as of 2026-08-15" in the checklist. Measured in Round 74
#: with the colours resolved the way the browser resolves them - through CSS variables, constants,
#: helper functions and the base fill of a hatch pattern - and one mark failed: the score
#: histogram's negative class, #4a5568 at 2.26:1. The latency chart's legend swatch failed as well,
#: at 1.23:1, which is the second check below.
STYLES = REPO / "web" / "src" / "styles.css"
MARKS = ("Bar", "Line", "Scatter", "Area", "Cell", "ReferenceLine")
SERIES = ("Bar", "Line", "Scatter", "Area")
NON_TEXT_MIN = 3.0

_HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")


def _tags(text: str, name: str) -> list[tuple[str, int]]:
    """(attribute text, line) of each `<name ...>` tag, scanned brace-aware like `_scatter_tags`."""
    out: list[tuple[str, int]] = []
    i = 0
    while (i := text.find(f"<{name}", i)) != -1:
        if text[i + len(name) + 1].isalnum():          # <BarChart, not <Bar
            i += 1
            continue
        j, depth = i + len(name) + 1, 0
        while j < len(text):
            c = text[j]
            if c in "{(":
                depth += 1
            elif c in "})":
                depth -= 1
            elif c == ">" and depth == 0:
                break
            j += 1
        out.append((text[i:j], text[:i].count("\n") + 1))
        i = j + 1
    return out


def _root_colours() -> dict[str, str]:
    root = re.search(r":root\s*\{([^}]*)\}", STYLES.read_text()).group(1)
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{3,6})\b", root))


def _contrast(a: str, b: str) -> float:
    def lum(h: str) -> float:
        h = h.lstrip("#")
        h = "".join(c * 2 for c in h) if len(h) == 3 else h
        lin = [(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
               for v in (int(h[k:k + 2], 16) / 255 for k in (0, 2, 4))]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _colours(expr: str, text: str, seen: frozenset = frozenset()) -> set[str]:
    """Every colour an attribute expression can evaluate to, resolved within its file."""
    out: set[str] = set()
    for m in re.finditer(r"url\(#([\w-]*)", expr):          # a pattern: its base fill
        for pid, body in re.findall(r"<pattern id=(\"[^\"]*\"|\{[^}]*\})(.*?)</pattern>", text, re.S):
            if pid.strip('"{}').startswith(m.group(1)) or pid.startswith("{"):
                rect = re.search(r"<rect\b[^>]*fill=(\"[^\"]*\"|\{[^}]*\})", body)
                if rect:
                    out |= _colours(rect.group(1), text, seen)
    plain = re.sub(r"url\([^)]*\)", "", expr)
    out |= {h.lower() for h in _HEX.findall(plain)}
    roots = _root_colours()
    out |= {roots[v].lower() for v in re.findall(r"var\((--[\w-]+)\)", plain) if v in roots}
    for ident in set(re.findall(r"\b([A-Za-z_]\w*)\b", plain)) - seen:
        d = re.search(rf"^const {ident}\b[^=]*=(.*?);\s*$", text, re.S | re.M)
        if d:
            out |= _colours(d.group(1), text, seen | {ident})
    return out


def _marks() -> list[tuple[str, str, str, set[str]]]:
    """(where, element, attribute expression, colours it resolves to) for every data mark."""
    out = []
    for path in sorted(COMPONENTS.glob("*.tsx")):
        text = path.read_text()
        for name in MARKS:
            for attrs, line in _tags(text, name):
                for m in re.finditer(r"\b(fill|stroke)=(\"[^\"]*\"|\{)", attrs):
                    if m.group(2) == "{":                         # a braced expression
                        j, depth = m.end(), 1
                        while depth and j < len(attrs):
                            depth += {"{": 1, "}": -1}.get(attrs[j], 0)
                            j += 1
                        expr = attrs[m.end():j - 1]
                    else:
                        expr = m.group(2)
                    out.append((f"{path.name}:{line}", name, expr, _colours(expr, text)))
    return out


def test_every_data_mark_resolves_to_a_colour() -> None:
    """A colour the scan cannot resolve is a mark it cannot check, so it fails rather than skips."""
    marks = _marks()
    unresolved = [f"{w} <{n} {e}>" for w, n, e, c in marks if not c and e.strip('"') != "none"]
    assert not unresolved, "chart mark colours the contrast check cannot resolve:\n  " + "\n  ".join(
        unresolved)
    found = set().union(*(c for *_, c in marks))
    # Named, not counted: the agreement scatter's disagreement red, the histogram's negative class
    # (the mark that failed), a CSS variable, a constant and a helper function's literal.
    assert {"#ff6b6b", "#717d92", "#36d399", "#f5a623", "#b072ff"} <= found, sorted(found)


def _is_separator(colours: set[str]) -> bool:
    """A stroke in the card's own colour draws a gap between segments, not a mark."""
    return colours == {_root_colours()["--panel"].lower()}


def test_every_data_mark_reaches_3_to_1_against_the_card() -> None:
    card = _root_colours()["--panel"]
    low = sorted({f"{w} <{n}> {c} at {_contrast(c, card):.2f}:1"
                  for w, n, _, cs in _marks() if not _is_separator(cs)
                  for c in cs if _contrast(c, card) < NON_TEXT_MIN})
    assert not low, (
        f"WCAG 2.2 SC 1.4.11 (AA): these marks fall below {NON_TEXT_MIN}:1 against the card "
        f"({card}):\n  " + "\n  ".join(low))


def test_stacked_segments_contrast_with_each_other_or_are_separated() -> None:
    """Stacked segments meet each other as well as the card. WCAG's pie-chart example: adjacent
    segments need 3:1 between them, or a border that contrasts with both. Round 74 lightened the
    histogram's negative class to pass against the card and, in the same edit, took it to 1.10:1
    against the hatch base it is stacked on (1.65:1 before). The card-coloured stroke separates them."""
    card = _root_colours()["--panel"].lower()
    bad = []
    for path in sorted(COMPONENTS.glob("*.tsx")):
        text = path.read_text()
        stacks: dict[str, list[tuple[str, int]]] = {}
        for attrs, line in _tags(text, "Bar"):
            m = re.search(r"stackId=\"([^\"]+)\"", attrs)
            if m:
                stacks.setdefault(m.group(1), []).append((attrs, line))
        for sid, bars in stacks.items():
            fills, separated = [], True
            for attrs, _ in bars:
                f = re.search(r"\bfill=(\"[^\"]*\"|\{[^}]*\})", attrs)
                st = re.search(r"\bstroke=(\"[^\"]*\"|\{[^}]*\})", attrs)
                fills.append(_colours(f.group(1), text) if f else set())
                separated &= bool(st) and _colours(st.group(1), text) == {card}
            pairs = [(a, b) for i, fa in enumerate(fills) for fb in fills[i + 1:] for a in fa for b in fb]
            if not separated and any(_contrast(a, b) < NON_TEXT_MIN for a, b in pairs):
                worst = min(pairs, key=lambda ab: _contrast(*ab))
                bad.append(f"{path.name}:{bars[0][1]} stack {sid!r}: {worst[0]} against {worst[1]} "
                           f"at {_contrast(*worst):.2f}:1 and no card-coloured separator")
    assert not bad, "stacked segments that meet each other below 3:1:\n  " + "\n  ".join(bad)


def test_a_legend_only_names_series_that_carry_their_own_colour() -> None:
    """A series coloured only through `<Cell>` children has no colour of its own, so its legend
    swatch renders in the SVG default, black. The latency chart's did, at 1.23:1 against the card,
    keying a single series and none of the per-bar colours. Removed in Round 74."""
    bad = []
    for path in sorted(COMPONENTS.glob("*.tsx")):
        text = path.read_text()
        for chart in re.finditer(r"<(\w+Chart)\b.*?</\1>", text, re.S):
            if chart.group(1) not in RECHARTS or "<Legend" not in chart.group(0):
                continue
            for name in SERIES:
                for attrs, line in _tags(chart.group(0), name):
                    own = "stroke=" if name == "Line" else "fill="
                    if own not in attrs:
                        bad.append(f"{path.name}:{text[:chart.start()].count(chr(10)) + line} <{name}>")
    assert not bad, ("a legend keys these series, which carry no colour of their own, so the swatch "
                     "draws in black:\n  " + "\n  ".join(bad))


# --------------------------------------------------------------------------------------------- #
#  SC 1.4.13 Content on Hover or Focus: a fixed readout instead of a floating tooltip
# --------------------------------------------------------------------------------------------- #

#: Measured on the deploy in Round 74, with real key presses: recharts 2.15.4 dismisses its tooltip on
#: Escape (TooltipBoundingBox listens on the document, recharts PR #2925), so the checklist's "zero
#: Escape handlers" had counted this repository and not the library; and its keyboard layer has no
#: blur handler, so Tab out of a chart left the tooltip open over the plot. Hoverable was not met: the
#: tooltip follows the pointer and can never itself be hovered. By the user's decision every chart
#: now writes into a fixed line under it (`ChartReadout`), which is not hover content at all.
#:
#: The readout depends on two recharts behaviours, so the release is pinned: the keyboard layer needs
#: a `<Tooltip>` child to exist, and a custom `content` receives the Tooltip's `formatter` and
#: `labelFormatter`. A different release must be measured again on the deploy, not assumed.
MEASURED_RECHARTS = "2.15.4"


def test_the_readout_was_measured_on_this_recharts() -> None:
    lock = json.loads((REPO / "web" / "package-lock.json").read_text())
    version = lock["packages"]["node_modules/recharts"]["version"]
    assert version == MEASURED_RECHARTS, (
        f"recharts is {version}; the chart readout was measured on {MEASURED_RECHARTS}. Re-measure on "
        "the deploy - arrow keys still move the readout, formatted values still reach it - then update "
        "this pin.")


def _tooltip_sites() -> list[tuple[str, int, str, str]]:
    """(file, line, the Tooltip tag, the text before it) for every recharts Tooltip in the UI."""
    out = []
    for path in sorted(COMPONENTS.glob("*.tsx")):
        text = path.read_text()
        if 'from "recharts"' not in text:
            continue                  # ChartReadout.tsx names <Tooltip> in its documentation only
        for attrs, line in _tags(text, "Tooltip"):
            out.append((path.name, line, attrs, text[: text.find(attrs)]))
    return out


def test_every_chart_tooltip_writes_into_a_fixed_readout() -> None:
    sites = _tooltip_sites()
    assert len(sites) >= 15, f"only {len(sites)} Tooltip sites found - did the scan or the charts move?"
    floating = [f"{f}:{ln}" for f, ln, tag, _ in sites if "content={<ReadoutTooltip" not in tag]
    assert not floating, (
        "these charts still draw a floating tooltip, which follows the pointer and can never itself be "
        "hovered (WCAG 2.2 SC 1.4.13); give them content={<ReadoutTooltip />}:\n  " + "\n  ".join(floating))


def test_every_readout_tooltip_has_a_readout_to_write_into() -> None:
    """Outside a `ChartReadout` the content renders nothing at all - a chart whose values are
    reachable nowhere, which is worse than the tooltip it replaced."""
    orphans = []
    for f, ln, tag, before in _tooltip_sites():
        if "ReadoutTooltip" not in tag:
            continue
        if before.rfind("<ChartReadout") <= before.rfind("</ChartReadout>"):
            orphans.append(f"{f}:{ln}")
    assert not orphans, "ReadoutTooltip outside any ChartReadout:\n  " + "\n  ".join(orphans)


def test_the_active_highlight_clears_when_the_chart_has_neither_hover_nor_focus() -> None:
    css = STYLES.read_text()
    assert re.search(
        r"\.recharts-wrapper:not\(:hover\):not\(:focus-within\)\s+\.recharts-tooltip-cursor\s*"
        r"\{\s*visibility:\s*hidden", css), (
        "the rule that clears a chart's highlight once the chart has neither hover nor focus is gone; "
        "recharts leaves it on the last category after Tab moves on")
