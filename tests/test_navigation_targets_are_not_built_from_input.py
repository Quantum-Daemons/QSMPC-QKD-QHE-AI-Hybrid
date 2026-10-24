"""Every navigation target is built from the app's own lists, never from input.

WHY THIS EXISTS

`npm audit` reports two react-router advisories that only a major upgrade (v7.18 or later) clears: an
open redirect when a `<Link to>` or `useNavigate` target containing a backslash is built from untrusted
input, and constructor injection through `deserializeErrors()` in data-router hydration. Round 59
upgraded react-router-dom to 6.30.6, which clears the advisory scoped to that package, and assessed the
other two as unreachable here: the app calls no `useNavigate`, renders no data router (the
`deserializeErrors` code is not in the bundle), and every link target is either a string literal or
built from the static NAV and USECASES lists. The one URL parameter the app reads, `?usecase=`, is
validated against USECASES and never becomes a link.

That assessment is a property of the code, and code changes. This test holds it, so that the day a
link target is built from a query parameter, the residual advisory stops being unreachable and this
fails before the change ships.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"

#: The only expression targets the app has, each from a static list: NAV (`lib/nav.ts`) and USECASES
#: (`lib/useCases.ts`). A new expression must be added here deliberately, with its source checked.
ALLOWED_EXPRESSIONS = {"n.to", "`/usecase/${u.id}`"}


def _strip(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?<![:\"'`])//[^\n]*", " ", src)


def _sources():
    for p in sorted(WEB.rglob("*.ts*")):
        yield p, _strip(p.read_text())


def test_no_imperative_navigation():
    hits = [f"{p.relative_to(REPO)}" for p, s in _sources() if re.search(r"\buseNavigate\b|\bredirect\(", s)]
    assert not hits, (
        "useNavigate / redirect() appeared. The residual react-router open-redirect advisory applies to "
        f"navigation targets built from input; check where these targets come from: {hits}")


def _braced_values(src: str, attr: str):
    """(offset, expression) for every `attr={...}`, with braces balanced - a template literal's `${}`
    inside the value must not end it, and the next prop must not be swallowed into it."""
    for m in re.finditer(rf"\b{attr}=\{{", src):
        i, depth = m.end(), 1
        while i < len(src) and depth:
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        yield m.start(), src[m.end(): i - 1].strip()


def test_every_link_target_is_a_literal_or_from_a_static_list():
    found = set()
    offenders = []
    for p, s in _sources():
        for off, expr in _braced_values(s, "to"):
            found.add(expr)
            if expr not in ALLOWED_EXPRESSIONS:
                line = s[:off].count("\n") + 1
                offenders.append(f"{p.relative_to(REPO)}:{line} to={{{expr}}}")
    assert not offenders, (
        "a link target is built from an expression not on the allowlist:\n  " + "\n  ".join(offenders)
        + "\n\nIf it comes from a static list, add it to ALLOWED_EXPRESSIONS with that list named. If it "
        "comes from a URL parameter or any other input, the react-router open-redirect advisory (fixed "
        "only in v7.18+) becomes reachable: upgrade first, or validate the target against a fixed list.")
    # Anti-tautology: the scan must actually see the targets it allows.
    assert ALLOWED_EXPRESSIONS <= found, f"the scan no longer sees {ALLOWED_EXPRESSIONS - found}; re-derive"


def test_the_one_url_parameter_is_validated_and_not_a_link():
    src = _strip((WEB / "pages" / "Orchestration.tsx").read_text())
    assert "useSearchParams" in src and re.search(r"USECASES\w*\.(some|find)\(", src), (
        "`?usecase=` is no longer validated against the use-case list")
    assert not re.search(r"to=\{[^}]*searchParams", src), "a link target is built from the query string"
