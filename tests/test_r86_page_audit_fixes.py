"""The defects the Round 86 audit of the deployed pages found, fixed at their source, and kept fixed.

WHY THIS EXISTS (Round 86, by the owner's choice to fix them all)

A headless audit of every route of the deployed main, at Lighthouse's desktop and mobile sizes and
after a default run on every use-case page, found:

- chart tick text in recharts' grey #666 inside a light OUTLINE, 2.96:1 on the chart panel, up to 84
  labels a page: `VerificationChart.tsx` set the ticks' `stroke`, not their `fill`;
- legend labels in their series colour: "true class 0 (solid)" at 4.09:1, a colour chosen to read as
  a mark (3:1) and read as 10 px text (4.5:1);
- the provenance badge's caret and separator dimmed to 3.0:1 and 4.0:1 by `opacity`;
- 15 key-value tables, and a one-row table of alternating labels and values, with no header cell;
- a layout shift of 0.19 to 0.32 while loading at desktop width: the crypto selects painted empty and
  filled in an effect after the first paint, and the loading placeholder's top margin collapsed into
  the body, which jumped when the app replaced it;
- the Value page citing NIST, EO 14412, CNSA 2.0 and the AI Act as text, with no link to any source.

The browser scan (`web/scripts/check_text_overlap.mjs`) now measures contrast, header cells and the
load's layout shift on every page after a run: 281 findings on the deployed base, none after the fix.
This file holds the causes at their source, including the tables a scan cannot reach because they
appear only after an agent run.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
SRC = WEB / "src"


def _read(rel: str) -> str:
    return (WEB / rel).read_text()


def test_every_table_in_the_app_has_a_header_cell() -> None:
    """Source-level, so a table that appears only after an interaction is held too."""
    bad, seen = [], 0
    for path in sorted(SRC.rglob("*.tsx")):
        text = path.read_text()
        for m in re.finditer(r"<table\b[^>]*>(.*?)</table>", text, flags=re.S):
            seen += 1
            if "<th" not in m.group(1):
                bad.append(f"{path.relative_to(WEB)}:{text.count(chr(10), 0, m.start()) + 1}")
    assert seen >= 25, f"only {seen} tables found in web/src; the parse has stopped working"
    assert not bad, f"tables with no header cell (SC 1.3.1): {bad}"


def test_the_key_column_of_a_key_value_table_is_a_row_header() -> None:
    """The fix's own shape: the first cell of each key-value row is `<th scope="row">`."""
    for rel, at_least in (("src/pages/Validations.tsx", 20), ("src/components/NetKeyCards.tsx", 15),
                          ("src/components/AgenticAIPanel.tsx", 10), ("src/components/SatNetworkPanel.tsx", 10),
                          ("src/components/RunMonitor.tsx", 1), ("src/components/BenchmarkPanel.tsx", 1)):
        n = _read(rel).count('<th scope="row"')
        assert n >= at_least, f"{rel} has {n} row headers, expected at least {at_least}"
    monitor = _read("src/components/RunMonitor.tsx")
    assert "<p className=\"muted\">no metrics yet</p>" in monitor, "the empty Metrics panel is a table again"


def test_a_row_header_keeps_the_look_of_the_cell_it_replaced() -> None:
    css = _read("src/styles.css")
    assert 'th:not([scope="row"]) { color: var(--muted); font-weight: 600; }' in css
    assert not re.search(r"^th \{ color: var\(--muted\)", css, flags=re.M), "a bare `th` rule mutes row headers again"
    assert '.facts-table td:first-child, .facts-table th[scope="row"]' in css


def test_chart_ticks_are_filled_not_outlined() -> None:
    chart = _read("src/components/VerificationChart.tsx")
    assert 'const AXIS = { fontSize: 10, fill: "#8a94a6" };' in chart
    for path in sorted(SRC.rglob("*.tsx")):
        text = path.read_text()
        for m in re.finditer(r"const [A-Z_]*(?:AXIS|TICK)[A-Z_]* = \{([^}]*)\}", text):
            assert "stroke" not in m.group(1), f"{path.name}: a tick style sets stroke: {m.group(0)}"
        assert not re.search(r"tick=\{\{[^}]*\bstroke:", text), f"{path.name}: an inline tick style sets stroke"


def test_legend_text_and_the_badge_read_at_body_contrast() -> None:
    css = _read("src/styles.css")
    assert ".recharts-legend-item-text { color: var(--text) !important; }" in css
    for cls in (".prov-detail", ".prov-caret"):
        rule = re.search(re.escape(cls) + r" \{([^}]*)\}", css)
        assert rule and "opacity" not in rule.group(1), f"{cls} is dimmed again: {rule and rule.group(0)}"


def test_the_crypto_selects_are_filled_on_the_first_render() -> None:
    hook = _read("src/lib/usePipelineRun.ts")
    assert "families: { kem: [], qkd: [], fhe: [], mpc: [], sig: [] }" not in hook, "the selects start empty again"
    for field in ("kem", "qkd", "fhe", "mpc", "sig"):
        assert f"useState(DEMO_CRYPTO.defaults.{field})" in hook, f"{field} starts empty again"


def test_the_loading_placeholder_has_no_collapsing_top_margin() -> None:
    html = (WEB / "index.html").read_text()
    m = re.search(r'<div id="root"><div style="([^"]+)"', html)
    assert m, "the inline loading placeholder moved"
    assert "padding:14vh" in m.group(1) and not re.search(r"margin:\s*\d+vh", m.group(1)), m.group(1)


#: The hosts a regulatory-clock source may live on: the issuing body's own site.
OFFICIAL = {"www.nist.gov", "csrc.nist.gov", "eur-lex.europa.eu", "www.whitehouse.gov", "www.federalregister.gov",
            "digital-strategy.ec.europa.eu", "media.defense.gov", "blog.cloudflare.com", "blog.google"}


def test_every_instrument_on_the_clock_links_its_primary_source() -> None:
    clock = _read("src/lib/regulatoryClock.ts")
    rows = re.findall(r"\{ lane: .*?firmness: \"[a-z ]+\", sources: \[([^\]]*)\] \},", clock, flags=re.S)
    total = len(re.findall(r"\{ lane: ", clock))
    assert total >= 15 and len(rows) == total, f"{len(rows)} of {total} clock rows carry sources"
    assert all(r.strip() for r in rows), "a clock row has an empty source list"
    urls = re.findall(r'url: "([^"]+)"', clock)
    assert len(urls) >= 10
    for u in urls:
        parsed = urlparse(u)
        assert parsed.scheme == "https" and parsed.netloc in OFFICIAL, f"not an official https source: {u}"
    view = _read("src/components/RegulatoryClock.tsx")
    assert "r.sources.map(" in view and 'rel="noopener noreferrer"' in view


def test_the_page_scan_measures_contrast_headers_and_the_load_shift() -> None:
    scan = _read("scripts/check_text_overlap.mjs")
    for needle in ("const A11Y", "[contrast]", "[table] no header cell", "layout-shift", "while loading, 0.1 or more",
                   "button:disabled"):
        assert needle in scan, f"the page scan no longer checks {needle!r}"


def test_the_page_scan_opens_every_disclosure_before_it_measures() -> None:
    """Round 86: content inside a closed <details> is not laid out, so the scan never measured it. The deployed walk found the finance page's
    transaction graphs, the QCNN circuit card, the genomics locus plot and the telecom confusion matrix failing there, all inside disclosures."""
    scan = (WEB / "scripts" / "check_text_overlap.mjs").read_text()
    opened = scan.index("document.querySelectorAll('details').forEach((d) => { d.open = true; })")
    assert opened < scan.index("// Through the page a screen at a time"), "the disclosures must be open before the page is scrolled and measured"
