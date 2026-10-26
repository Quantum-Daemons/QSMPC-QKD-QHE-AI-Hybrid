"""Every page's heading outline starts at h1 and never skips a level.

WHY THIS EXISTS

WCAG 2.2 SC 1.3.1 (Info and Relationships, Level A): structure a reader sees must be available to
assistive technology, and a screen reader's heading list is how most of its users move through a long
page. The app's convention was h1 for the page and h3 for every card title, in the page files and in
the components they render, so `/orchestration`, `/usecases`, every `/usecase/<id>` and the not-found
page went from h1 straight to h3. The checklist declared it open on 2026-08-15 and it stayed open
because a visible fix would change the visual weight of every card.

Round 74 closed it without a visual change: hidden h2 section headings on the two composed pages,
and real h2 card titles where the cards ARE the page's sections, styled as the h3 they replaced.

WHAT IS ASSERTED, AND HOW

The outline of a rendered page is the page's own headings interleaved with those of every component
it renders, so a check of the page file alone would miss most of them. Each page is expanded instead:
every `<hN` and every `<Component` in source order, a component replaced by its own expansion,
resolved through the file's imports. JSX renders in source order, so this is the order a reader
meets them in.

The limit: both branches of a conditional are included. A conditional heading between two others
could hide a skip that only appears when it does not render. None does today; the live walk measures
the rendered DOM.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "web" / "src"

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"^\s*//.*$", re.M)
_DEF = re.compile(r"^(?:export\s+)?(default\s+)?function\s+([A-Z]\w*)\s*[(<]", re.M)
_IMPORT = re.compile(
    r"^import\s+(?:type\s+)?(?:(\w+)\s*,?\s*)?(?:\{([^}]*)\})?\s*from\s+[\"'](\.{1,2}/[^\"']+)[\"']",
    re.M)
_TOKEN = re.compile(r"<(h[1-6])\b|<([A-Z]\w*)\b")
DEFAULT = "<default>"


def _resolve(origin: Path, spec: str) -> Path | None:
    base = (origin.parent / spec).resolve()
    for cand in (base.with_suffix(".tsx"), base.with_suffix(".ts"), base / "index.tsx"):
        if cand.exists():
            return cand
    return None


@cache
def _parse(path: Path) -> tuple[dict[str, str], dict[str, tuple[Path, str]], str | None]:
    """(component name -> body, imported name -> (file, name), the default export's name)."""
    text = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", path.read_text()))
    defs = list(_DEF.finditer(text))
    bodies, default = {}, None
    for i, m in enumerate(defs):
        end = defs[i + 1].start() if i + 1 < len(defs) else len(text)
        bodies[m.group(2)] = text[m.start():end]
        if m.group(1):
            default = m.group(2)
    imports: dict[str, tuple[Path, str]] = {}
    for m in _IMPORT.finditer(text):
        target = _resolve(path, m.group(3))
        if target is None or target.suffix != ".tsx":
            continue
        if m.group(1):
            imports[m.group(1)] = (target, DEFAULT)
        for item in (m.group(2) or "").split(","):
            parts = item.split(" as ")
            if parts[0].strip():
                imports[parts[-1].strip()] = (target, parts[0].strip())
    return bodies, imports, default


def _outline(path: Path, name: str, stack: tuple = ()) -> list[tuple[int, str]]:
    """(level, where) for every heading a component renders, its children's expanded in place."""
    bodies, imports, default = _parse(path)
    if name == DEFAULT:
        name = default or ""
    if name not in bodies or (path, name) in stack:
        return []
    out: list[tuple[int, str]] = []
    where = f"{path.relative_to(SRC)}:{name}"
    for m in _TOKEN.finditer(bodies[name]):
        if m.group(1):
            out.append((int(m.group(1)[1]), where))
        elif m.group(2) in bodies:
            out += _outline(path, m.group(2), stack + ((path, name),))
        elif m.group(2) in imports:
            target, imported = imports[m.group(2)]
            out += _outline(target, imported, stack + ((path, name),))
    return out


#: Every page a visitor can land on: each route's page, and the not-found page App renders.
PAGES = sorted(p.stem for p in (SRC / "pages").glob("*.tsx"))


def _page_outline(stem: str) -> list[tuple[int, str]]:
    return _outline(SRC / "pages" / f"{stem}.tsx", DEFAULT)


def _skips(outline: list[tuple[int, str]]) -> list[str]:
    bad = [] if outline and outline[0][0] == 1 else [f"starts at h{outline[0][0]}, not h1"]
    for (a, wa), (b, wb) in zip(outline, outline[1:], strict=False):
        if b > a + 1:
            bad.append(f"h{a} ({wa}) -> h{b} ({wb})")
    return bad


def test_the_pages_are_all_found() -> None:
    assert len(PAGES) >= 6, f"only {PAGES} found - did web/src/pages move?"


def test_the_expansion_reaches_into_the_components() -> None:
    """A parse that sees only the page file would pass every page below for the wrong reason."""
    outline = _page_outline("UseCasePage")
    sources = {w.split(":")[0] for _, w in outline}
    assert len(outline) >= 15 and 4 in {lvl for lvl, _ in outline}, outline
    assert {"components/BenchmarkPanel.tsx", "components/SdnControlPlanePanel.tsx",
            "components/RunMonitor.tsx"} <= sources, sorted(sources)


@pytest.mark.parametrize("page", PAGES)
def test_a_page_outline_never_skips_a_level(page: str) -> None:
    outline = _page_outline(page)
    assert outline, f"{page}: no headings found"
    bad = _skips(outline)
    assert not bad, f"{page}: the heading outline skips a level:\n  " + "\n  ".join(bad)


def test_the_not_found_page_never_skips_a_level() -> None:
    outline = _outline(SRC / "App.tsx", "NotFoundPage")
    assert outline, "App.tsx's NotFoundPage renders no headings - did it move?"
    assert not _skips(outline), _skips(outline)


def test_hidden_section_headings_use_the_shared_class() -> None:
    """A hidden h2 must be hidden by `.sr-only`, which keeps it in the accessibility tree. `display:
    none` or `hidden` would remove it from the heading list as well as from the screen."""
    css = (SRC / "styles.css").read_text()
    assert re.search(r"\.sr-only\s*\{[^}]*clip:\s*rect\(0", css), "the .sr-only rule changed"
    for stem in ("UseCasePage", "Orchestration"):
        text = (SRC / "pages" / f"{stem}.tsx").read_text()
        hidden = re.findall(r"<h2 className=\"sr-only\">([^<]+)</h2>", text)
        assert len(hidden) >= 2, f"{stem}: expected its hidden section headings, found {hidden}"
        assert not re.search(r"<h[1-6][^>]*(?:hidden\b|display:\s*\"?none)", text), stem
