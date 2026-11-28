"""Every card with its own download is either carried by Results (JSON) or named in its left_out, with why (Round 86).

The owner's choice of 2026-11-28 ("Results (JSON) every card"): the button promised "every card's rows" while the bundle
carried four kinds of card (the four hospitals, differential privacy, the multi-party PSI and the re-scorings), so the
published-figure comparison, the cards that show a committed record and the cards run on their own button were missing
without a word. `CARD_EXPORT_CLASSES` in `web/src/lib/resultsBundle.ts` now classifies every card by its export name:
the key the bundle carries it under, or the reason it is left out, which the bundle's `left_out` states on each page the
card is on. Held here from the source: every `<CardExport name=...>` in `web/src` is classified, a page-specific card's
pages are the page its name carries, every key the classes name is one `buildResultsBundle` writes, and the page passes
the published comparison's rows to the bundle from the same list its card draws.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "web" / "src"
BUNDLE = (SRC / "lib" / "resultsBundle.ts").read_text()
PAGES = ("finance_aml", "medical_fl", "genomics", "telecom_ids", "satellite_comms", "defi_lending")


def _exports() -> list[tuple[str, str | None, str]]:
    """(suffix, the page the name names or None for a `${usecase}-...` name, the file) for every CardExport."""
    out = []
    for f in sorted(SRC.rglob("*.tsx")):
        for m in re.finditer(r'<CardExport\s+name=(?:"([a-z_]+)-([a-z0-9-]+)"|\{`\$\{usecase\}-([a-z0-9-]+)`\})', f.read_text()):
            if m.group(3):
                out.append((m.group(3), None, f.name))
            else:
                assert m.group(1) in PAGES, (m.group(1), f.name)
                out.append((m.group(2), m.group(1), f.name))
    return out


def _classes() -> dict[str, dict]:
    block = BUNDLE[BUNDLE.index("export const CARD_EXPORT_CLASSES"):]
    block = block[:block.index("\n};")]
    out = {}
    for line in block.splitlines():
        m = re.match(r'\s*"([a-z0-9-]+)": \{ (card|leftOut): "([^"]+)"', line)
        if not m:
            continue
        p = re.search(r"pages: \[([^\]]*)\]", line)
        pages = [x.strip().strip('"') for x in p.group(1).split(",")] if p else None
        out[m.group(1)] = {"kind": m.group(2), "value": m.group(3), "pages": pages}
    return out


def test_every_card_with_a_download_is_classified() -> None:
    exports, classes = _exports(), _classes()
    assert len(exports) >= 13, exports
    missing = sorted({f"{s} ({f})" for s, _, f in exports if s not in classes})
    assert not missing, f"cards with their own download that Results (JSON) neither carries nor names: {missing}"
    unused = sorted(set(classes) - {s for s, _, _ in exports})
    assert not unused, f"classes no card uses: {unused}"


def test_a_page_specific_card_is_classified_on_its_page() -> None:
    classes = _classes()
    for suffix, page, f in _exports():
        c = classes[suffix]
        if page is None:
            assert c["pages"] is None, f"{suffix} ({f}) is on every page but classified for {c['pages']}"
        else:
            assert c["pages"] and page in c["pages"], f"{suffix} ({f}) is on {page}, classified for {c['pages']}"


def test_every_carried_card_is_one_the_bundle_writes() -> None:
    for suffix, c in _classes().items():
        if c["kind"] == "card":
            assert f"cards.{c['value']} = " in BUNDLE, f"{suffix}: the bundle writes no cards.{c['value']}"


def test_the_page_gives_the_bundle_the_rows_its_card_draws() -> None:
    page = (SRC / "pages" / "UseCasePage.tsx").read_text()
    assert "rows={publishedComparison}" in page
    assert "publishedComparison: publishedComparison as unknown as Record<string, unknown>[]" in page
    assert "every card's rows" not in page, "the button's title still promises every card's rows"
    assert "...cardsLeftOut(i.usecase)" in BUNDLE
