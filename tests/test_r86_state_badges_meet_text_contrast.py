"""Every state badge's text meets WCAG's 4.5:1 against what it sits on (Round 86, 2026-11-22).

WHY THIS EXISTS

`.badge-state.error` drew white on `--danger` (#f56565), 3.03:1 at 12 px. No page showed that state in a normal run until
the ResNet-18 card's published comparison turned from "reproduced" to "differs" on 2026-11-22, and CI's text-layout check
failed the medical page on it. The page check sees only the states a run reaches; this test reads the stylesheet, so a state
no page shows yet is held to the same ratio. The same reading found the two striped transitional states drawing their text
across `--accent` stripes at 2.34:1; on the owner's choice of 2026-11-22 the stripes became a ring and the text sits on a
solid fill inside it.

WHAT IS ASSERTED

- Each `.badge-state.<state>` rule with a solid background resolves, through `:root`, to a text colour of at least 4.5:1
  against it (12 px bold is not large text under WCAG 2.2, so 3:1 does not apply).
- The striped states keep their stripes to the border box and put their text on a solid padding-box fill of at least
  4.5:1, and the collapsed sidebar gives the ring's width back as the full-width badge does.
- Seeded: the error rule and the striped rule as they stood are refused.
"""

from __future__ import annotations

import re
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "web" / "src" / "styles.css"
STRIPED = ("pausing", "stopping")


def _linear(c: int) -> float:
    s = c / 255
    return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _root(css: str) -> dict[str, str]:
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{3,6})\s*;", css[css.index(":root"):css.index("}")]))


def _resolve(root: dict[str, str], value: str) -> str:
    m = re.fullmatch(r"var\(--([a-z0-9-]+)\)", value.strip())
    return root[m.group(1)] if m else value.strip()


def badge_problems(css: str) -> list[str]:
    root = _root(css)
    rules = re.findall(r"\.badge-state\.([a-z]+)\s*\{\s*background:\s*([^;]+);\s*color:\s*([^;]+);\s*\}", css)
    assert len(rules) >= 6, f"found {len(rules)} badge-state rules; the pattern is reading nothing"
    problems = []
    for state, background, colour in rules:
        ratio = contrast(_resolve(root, colour), _resolve(root, background))
        if ratio < 4.5:
            problems.append(f".badge-state.{state}: {_resolve(root, colour)} on {_resolve(root, background)} is {ratio:.2f}:1")
    return problems


def striped_problems(css: str) -> list[str]:
    root = _root(css)
    m = re.search(r"\.badge-state\.pausing,\s*\.badge-state\.stopping\s*\{([^}]*)\}", css)
    if not m:
        return ["no rule for the striped states"]
    body = m.group(1)
    fill = re.search(r"linear-gradient\(\s*(var\(--[a-z0-9-]+\)|#[0-9a-fA-F]{3,6})\s*,\s*\1\s*\)\s*padding-box", body)
    colour = re.search(r"(?<![-\w])color:\s*([^;]+);", body)
    problems = []
    if not fill:
        problems.append("the striped states have no solid padding-box fill under their text")
    if "border-box" not in body or not re.search(r"border:\s*2px solid transparent", body):
        problems.append("the stripes are not confined to a 2 px transparent border")
    if fill and colour:
        ratio = contrast(_resolve(root, colour.group(1)), _resolve(root, fill.group(1)))
        if ratio < 4.5:
            problems.append(f"striped states: text on their fill is {ratio:.2f}:1")
    collapsed = re.search(r"\.app\.nav-collapsed \.run-status-box \.badge-state\.pausing,\s*"
                          r"\.app\.nav-collapsed \.run-status-box \.badge-state\.stopping\s*\{\s*padding:\s*1px 4px;\s*\}", css)
    if not collapsed:
        problems.append("the collapsed sidebar does not give the ring's 2 px back (padding 1px 4px)")
    return problems


def test_every_solid_state_badge_meets_4_5_to_1() -> None:
    assert badge_problems(CSS.read_text(encoding="utf-8")) == []


def test_the_striped_states_put_their_text_on_a_solid_fill() -> None:
    assert striped_problems(CSS.read_text(encoding="utf-8")) == []


def test_the_rules_as_they_stood_are_refused() -> None:
    css = CSS.read_text(encoding="utf-8")
    error_as_it_stood = re.sub(r"(\.badge-state\.error\s*\{\s*background:\s*var\(--danger\);\s*color:\s*)[^;]+;",
                               r"\1#fff;", css)
    assert any("error" in p and "3.03" in p for p in badge_problems(error_as_it_stood))
    striped_as_it_stood = re.sub(
        r"\.badge-state\.pausing,\s*\.badge-state\.stopping\s*\{[^}]*\}",
        ".badge-state.pausing, .badge-state.stopping {\n"
        "  background: repeating-linear-gradient(135deg, var(--accent) 0 6px, var(--border) 6px 12px);\n"
        "  color: var(--text);\n}", css)
    assert any("padding-box" in p for p in striped_problems(striped_as_it_stood))
