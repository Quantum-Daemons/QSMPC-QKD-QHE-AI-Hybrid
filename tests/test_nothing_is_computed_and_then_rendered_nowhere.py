"""A component that is defined and never rendered is a feature that shipped as dead weight.

WHY THIS EXISTS

Twice in one round:

  * `metric_note` is written by six trainers, checked by the prose gate, serialised into
    `config.defaults.json` and shipped in the bundle - and had **no render path anywhere in
    `web/src/`**. Six models carried a sentence explaining how to read their own metrics and no
    visitor could see one. Worse, its absence let the companion caption assert a direction the note
    contradicted, so the missing render was load-bearing rather than merely wasteful.

  * `ReliabilityChart` was written in the SAME round that fixed `metric_note`, wired into
    `onnxInference.ts` so the data was computed on every evaluation, committed, and **never called**.
    The commit message described a plot that did not render. Caught by grepping for the call site
    after the commit, which is later than it should have been found.

The two are one defect: work that produces a value and stops short of the surface that shows it.
Neither `tsc` nor the bundler complains - an unused module-scope function is legal TypeScript and
gets tree-shaken out, so the build gets SMALLER and greener the more of this there is.

WHAT IS ASSERTED

Every component defined under `web/src/components/` is referenced somewhere other than its own
definition. "Component" is a capitalised function or const arrow returning JSX, which is the naming
convention React requires and this codebase follows without exception.

WHAT IS DELIBERATELY NOT ASSERTED

Not that every FIELD reaches a surface. `plot.reliability` being computed and unrendered is the same
defect one level down, but a field is read through property access that a static scan cannot follow
with confidence, and a check that is right most of the time about something this cheap to verify by
hand is worse than a check with a sharp edge.

Not that a component is reachable at runtime. A component rendered only behind a condition nothing
satisfies is dead too; that is what the deployed-site walk is for.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"

#: `export function Foo(` / `function Foo(` / `const Foo = (` / `const Foo: React.FC = (`
_DEF = re.compile(r"^\s*(?:export\s+)?(?:function\s+(?P<f>[A-Z]\w*)\s*\(|"
                  r"const\s+(?P<c>[A-Z]\w*)\s*(?::[^=]+)?=\s*(?:\([^)]*\)|\w+)\s*(?::[^=]+)?=>)", re.M)


def _tracked(*globs: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--", *globs], cwd=REPO,
                         check=True, capture_output=True, text=True).stdout.split()
    return [REPO / f for f in out]


def _defined() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in _tracked("web/src/components/*.tsx"):
        for m in _DEF.finditer(path.read_text(errors="ignore")):
            name = m.group("f") or m.group("c")
            if name:
                found.setdefault(name, path)
    return found


def test_the_scan_finds_the_components() -> None:
    """Non-vacuity: a pattern matching nothing would make the assertion below pass."""
    defined = _defined()
    assert len(defined) >= 10, (
        f"found only {len(defined)} components under web/src/components (11 at the time this was written); the pattern is "
        "not matching the code it was written against")
    for expected in ("VerificationChart", "ModelPanel", "ReliabilityChart"):
        assert expected in defined, f"{expected} was not found by the scan"


def test_every_component_defined_is_rendered_somewhere() -> None:
    """The property. Red on the tree that added `ReliabilityChart` without calling it."""
    defined = _defined()
    sources = _tracked("web/src/**/*.tsx", "web/src/**/*.ts")
    unused = []
    for name, home in sorted(defined.items()):
        uses = 0
        for path in sources:
            text = path.read_text(errors="ignore")
            hits = len(re.findall(rf"(?<![\w.]){re.escape(name)}(?![\w])", text))
            if path == home:
                # Its own file contains the definition; one occurrence there is the definition itself.
                hits -= 1
            uses += max(0, hits)
        if uses == 0:
            unused.append(f"{name}  ({home.relative_to(REPO)})")
    assert not unused, (
        "these components are defined and never rendered:\n  " + "\n  ".join(unused)
        + "\n\nNeither tsc nor the bundler will tell you: an unused module-scope function is legal "
        "and gets tree-shaken, so the build gets smaller and greener the more of this there is. "
        "Either render it or delete it - a component nobody calls is a feature that shipped as dead "
        "weight, and the commit message will describe something a visitor cannot see."
    )
