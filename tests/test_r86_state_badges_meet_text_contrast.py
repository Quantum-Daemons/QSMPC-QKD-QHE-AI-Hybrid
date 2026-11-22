"""Every state badge's text meets WCAG's 4.5:1 against its own background (Round 86, 2026-11-22).

WHY THIS EXISTS

`.badge-state.error` drew white on `--danger` (#f56565), 3.03:1 at 12 px. No page showed that state in a normal run until
the ResNet-18 card's published comparison turned from "reproduced" to "differs" on 2026-11-22, and CI's text-layout check
failed the medical page on it. The page check sees only the states a run reaches; this test reads the stylesheet, so a state
no page shows yet is held to the same ratio. The other badges already drew #03070f on their colour, and "error" does too now.

WHAT IS ASSERTED

- Each `.badge-state.<state>` rule with a solid background resolves, through `:root`, to a text colour of at least 4.5:1
  against it (12 px bold is not large text under WCAG 2.2, so 3:1 does not apply).
- The two striped transitional states are listed, not passed: their text sits on alternating stripes of very different
  luminance, so no single text colour meets 4.5:1 on both.
- Seeded: the rule as it stood is refused.
"""

from __future__ import annotations

import re
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "web" / "src" / "styles.css"
#: Text over alternating `--accent` and `--border` stripes: about 2.3:1 on the first, so recorded rather than passed.
STRIPED = {"pausing", "stopping"}


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


def badge_problems(css: str) -> list[str]:
    root = dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{3,6})\s*;", css[css.index(":root"):css.index("}")]))

    def resolve(value: str) -> str:
        m = re.fullmatch(r"var\(--([a-z0-9-]+)\)", value.strip())
        return root[m.group(1)] if m else value.strip()

    rules = re.findall(r"\.badge-state\.([a-z]+)\s*\{\s*background:\s*([^;]+);\s*color:\s*([^;]+);\s*\}", css)
    assert len(rules) >= 6, f"found {len(rules)} badge-state rules; the pattern is reading nothing"
    problems = []
    for state, background, colour in rules:
        if state in STRIPED:
            continue
        ratio = contrast(resolve(colour), resolve(background))
        if ratio < 4.5:
            problems.append(f".badge-state.{state}: {resolve(colour)} on {resolve(background)} is {ratio:.2f}:1")
    return problems


def test_every_solid_state_badge_meets_4_5_to_1() -> None:
    assert badge_problems(CSS.read_text(encoding="utf-8")) == []


def test_the_striped_states_are_the_only_exception() -> None:
    css = CSS.read_text(encoding="utf-8")
    striped = set(re.findall(r"\.badge-state\.([a-z]+)[\s,]", css[css.index(".badge-state.pausing"):][:80]))
    assert striped == STRIPED, striped


def test_the_rule_as_it_stood_is_refused() -> None:
    css = CSS.read_text(encoding="utf-8")
    as_it_stood = re.sub(r"(\.badge-state\.error\s*\{\s*background:\s*var\(--danger\);\s*color:\s*)[^;]+;",
                         r"\1#fff;", css)
    assert any("error" in p and "3.03" in p for p in badge_problems(as_it_stood)), badge_problems(as_it_stood)
