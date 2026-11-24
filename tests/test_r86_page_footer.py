"""Every page ends with the site footer, and the footer says only what the owner's draft says (Round 86, 2026-11-22).

WHY THIS EXISTS

The pages had no footer: nothing on them linked the source, the licence, the models or the publisher, and the version a
visitor saw was nowhere. The owner's draft of 2026-11-13 gave one line, added on the owner's choice of 2026-11-22, with
its rules: the version from the release, no commit links, no price, roadmap or contact line, and no AI-tools line until
the owner decides it (the policy of 2026-11-06).

WHAT IS ASSERTED

- `App.tsx` renders `SiteFooter` inside `<main>`, after the routes, so every route ends with it.
- The footer carries each part of the draft, and its version is `web/package.json`'s, imported by name, which
  `tests/test_r86_the_version_has_one_source.py` holds equal to `qsh.__version__`.
- The review date is a real date, not in the future.
- It carries no commit hash, no price, roadmap, TRL or contact line, and no AI line; seeded with each.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import qsh

REPO = Path(__file__).resolve().parents[1]
FOOTER = REPO / "web" / "src" / "components" / "SiteFooter.tsx"
PARTS = ("Source:", "https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid", "(Apache-2.0)", "Models:",
         "https://huggingface.co/Daemons-Q", "v{version}", "By Quantum Daemons:", "https://quantum.daemons.jp/",
         "QKD and QSDC are simulated", "Page last reviewed: {PAGES_LAST_REVIEWED}")
FORBIDDEN = {
    "a commit hash": re.compile(r"\b[0-9a-f]{40}\b|/commit/"),
    "a price or tier": re.compile(r"\$|EUR|USD|price|pricing|per month|subscription", re.I),
    "a roadmap or TRL": re.compile(r"roadmap|\bTRL\b", re.I),
    "a contact line": re.compile(r"contact|mailto:|@[a-z0-9-]+\.[a-z]", re.I),
    "an AI line": re.compile(r"\bAI\b|generated|Claude|LLM", re.I),
}


def footer_problems(src: str) -> list[str]:
    problems = [f"missing {p!r}" for p in PARTS if p not in src]
    if not re.search(r'import \{ version \} from "\.\./\.\./package\.json";', src):
        problems.append("the version is not package.json's, imported by name")
    m = re.search(r'PAGES_LAST_REVIEWED = "([^"]+)"', src)
    if not m:
        problems.append("no review date")
    else:
        try:
            day = dt.datetime.strptime(m.group(1), "%d %B %Y").date()
            if day > dt.date.today():
                problems.append(f"the review date {m.group(1)} is in the future")
        except ValueError:
            problems.append(f"the review date {m.group(1)!r} is not a date")
    # The repository's own name contains "-AI-", which is not an AI line.
    body = src[src.index("export default function SiteFooter"):].replace("QSMPC-QKD-QHE-AI-Hybrid", "QSH")
    for what, pattern in FORBIDDEN.items():
        if pattern.search(body):
            problems.append(f"it carries {what}")
    return problems


def test_every_route_ends_with_the_footer() -> None:
    app = (REPO / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    main = app[app.index('<main className="main"'):app.index("</main>")]
    assert "</Routes>" in main and main.index("<SiteFooter />") > main.index("</Routes>"), "the footer is not after the routes in <main>"
    assert 'import SiteFooter from "./components/SiteFooter";' in app


def test_the_footer_says_what_the_draft_says() -> None:
    assert footer_problems(FOOTER.read_text(encoding="utf-8")) == []
    package = json.loads((REPO / "web" / "package.json").read_text(encoding="utf-8"))
    assert package["version"] == qsh.__version__


def test_the_check_refuses_what_the_footer_must_not_carry() -> None:
    src = FOOTER.read_text(encoding="utf-8")
    seeds = {
        "a commit hash": ' · <a href="https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid/commit/c854b8e">c854b8e</a>',
        "a price or tier": " · Reports from $800 per month",
        "a roadmap or TRL": " · Roadmap: TRL 4",
        "a contact line": " · Contact: <a href=\"mailto:someone@example.org\">mail</a>",
        "an AI line": " · Use of AI tools: drafted with an LLM",
    }
    for what, extra in seeds.items():
        seeded = src.replace("· QKD and QSDC are simulated", extra + " · QKD and QSDC are simulated")
        assert f"it carries {what}" in footer_problems(seeded), what
    assert "the review date 1 January 2999 is in the future" in footer_problems(
        src.replace('PAGES_LAST_REVIEWED = "25 November 2026"', 'PAGES_LAST_REVIEWED = "1 January 2999"'))
