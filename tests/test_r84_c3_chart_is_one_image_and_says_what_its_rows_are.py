"""A chart is ONE image to assistive technology, and the agreement figure says what its rows are.

WHY THIS EXISTS (Round 84, checklist UX39, PI7 and PI9)

UX39 failed live: recharts 2.15 renders every scatter symbol as `<g role="img">` with no name and no
prop to change it, so finance's 8,192-point overlay was 8,192 unnamed images to a screen reader and
to Lighthouse's `svg-img-alt`. The fix is `MarksHidden`, which strips that role and sets
`aria-hidden` on every symbol (an observer, because a resize remounts them), while the chart's
`<svg role="img">` keeps its one `title` and points at its caption with `aria-describedby`.

PI7/PI9: the agreement figure's caption hardcoded which rows each page encrypted - and called
finance's and medical's fitting rows held out - and its y axis had no title. The caption now prints
what the plot says (`heldOut`, `fitNote`, `rowsAre`, the licence with its link) and both axes are
titled.

Source-level, like `test_chart_accessibility.py`: the effect runs only in a DOM, which this suite does
not have. Every assertion fails on the baseline.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPONENTS = REPO / "web" / "src" / "components"
CHART = COMPONENTS / "VerificationChart.tsx"


def _scatter_charts() -> list[tuple[str, str, int]]:
    """(file, text before the chart, line) for every recharts ScatterChart in the components."""
    out = []
    for path in sorted(COMPONENTS.glob("*.tsx")):
        text = path.read_text()
        for m in re.finditer(r"<ScatterChart\b", text):
            out.append((path.name, text[: m.start()], text[: m.start()].count("\n") + 1))
    return out


def test_there_are_scatter_charts_to_check() -> None:
    assert len(_scatter_charts()) >= 4, "the agreement, decision, overlay and calibration scatters"


def test_every_scatter_sits_inside_marks_hidden() -> None:
    """Open `<MarksHidden>` tags before the chart must outnumber the closes: the chart is inside one."""
    outside = [f"{f}:{ln}" for f, before, ln in _scatter_charts()
               if before.count("<MarksHidden>") <= before.count("</MarksHidden>")]
    assert not outside, (
        f"{outside} render a ScatterChart outside <MarksHidden>, so each of its symbols reaches the "
        "accessibility tree as an unnamed role=img (UX39)")


def test_every_scatter_names_its_text_alternative() -> None:
    text = CHART.read_text()
    tags = re.findall(r"<ScatterChart\b([^>]*)>", text, re.S)
    missing = [t[:60] for t in tags if "aria-describedby=" not in t]
    assert tags and not missing, (
        f"ScatterCharts without aria-describedby: {missing}. The caption or table is the text "
        "alternative and the image has to point at it.")


def test_marks_hidden_strips_the_role_and_follows_remounts() -> None:
    text = CHART.read_text()
    m = re.search(r"function MarksHidden\(.*?\n\}\n", text, re.S)
    assert m, "MarksHidden is gone"
    body = m.group(0)
    assert ".recharts-scatter-symbol" in body
    assert 'removeAttribute("role")' in body and 'setAttribute("aria-hidden", "true")' in body
    assert "MutationObserver" in body and "childList: true" in body and "subtree: true" in body


def _agreement_component() -> str:
    text = CHART.read_text()
    start = text.index("export function EncryptedAgreementChart")
    return text[start: text.index("\nfunction DecisionPlane", start)]


def test_the_agreement_caption_prints_what_the_plot_says_about_its_rows() -> None:
    body = _agreement_component()
    for read in ("plot.heldOut === true", "plot.heldOut === false", "plot.rowsAre", "plot.fitNote",
                 "plot.licence", "plot.ciphertexts", "plot.medianAbsError", "plot.smallestMargin"):
        assert read in body, f"the agreement figure no longer reads `{read}`"
    assert "None of these rows was" in body and "used to fit the plotted model" in body
    # The per-use-case sentence that called fitting rows held out is gone.
    assert "on finance and medical the bundled rows" not in body
    # "held-out" reaches the figure's name only through the plot's own flag.
    name = re.search(r"const name = (.*?);\n", body, re.S).group(1)
    assert "plot.heldOut" in name


def test_both_agreement_axes_are_titled() -> None:
    body = _agreement_component()
    y = re.search(r'<YAxis type="number" dataKey="enc"(.*?)/>', body, re.S)
    x = re.search(r'<XAxis type="number" dataKey="plain"(.*?)/>', body, re.S)
    assert x and "label=" in x.group(1)
    assert y and "label=" in y.group(1) and "(decrypted)" in y.group(1), (
        "the encrypted axis has no title; PI9 asks for e.g. 'encrypted score (decrypted)'")


def test_the_decisions_are_drawn_on_the_overlay_plane_with_a_table() -> None:
    text = CHART.read_text()
    start = text.index("function DecisionPlane")
    body = text[start: text.index("\n}\n", start)]
    assert "plane.x.label" in body and "plane.y.label" in body
    table = re.search(r"<table id=\{(\w+)\} className=\"facts-table small\"", body)
    assert table, "the counts table, its text alternative"
    # The image points at the table as well as the caption.
    assert re.search(r"aria-describedby=\{`\$\{\w+\} \$\{" + table.group(1) + r"\}`\}", body)
    assert "cannot\n          sit on the log axis" in body or "cannot sit on the log axis" in body, (
        "rows at 0 on a log axis are dropped silently")
