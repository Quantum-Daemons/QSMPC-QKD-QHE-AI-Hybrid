"""Markdown mathematics uses the house form, and published documents carry no special symbols.

WHY THIS EXISTS

Round 55 set the house style for GitHub-rendered Markdown: display equations in `$$...$$`, inline
in `` $`...`$ ``, the code-span form. Inside the backticks Markdown leaves the content alone, while in
a plain `$...$` it may first read `\\{`, `\\\\` or a pair of underscores as its own syntax. Nothing
enforced it, and by Round 72 it had drifted: the Round 70 and 72 sections of the register used plain
inline math, and so did README, `docs/architecture.md` and `references/REFERENCES.md`.
`$L_{\\tilde V} = 15n^2 - 43n + 14 + (2n-3)L_R$` carried two underscores, the case the form exists
for. Round 73 converted 22 plain sites, and 30 pieces of mathematics that had been typed as prose.

Separately, the user asked that README and the other Markdown documents avoid special symbols where
possible. They are ASCII except for the cases allowed below, each with its reason.

WHAT IS ASSERTED

- No plain single-dollar math in any tracked Markdown that GitHub renders. Fenced blocks, indented
  code blocks and code spans of any length are skipped. A shell `"$n"` inside an indented block is
  not mathematics, and neither is a `$` quoted inside a double-backtick span.
- No GitHub-only `` $`...`$ `` in the Markdown that Hugging Face renders (model cards and the Space
  card), where it would show literally.
- No non-ASCII character in tracked Markdown except the allowed cases.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Rendered by Hugging Face, not GitHub.
HF_RENDERED = ("models/", "spaces/")

#: (file, character) -> why it stays.
ALLOWED_NON_ASCII = {
    ("spaces/qsh-inference/README.md", "\U0001F510"):
        "the Space's `emoji:` field in its YAML header, which Hugging Face renders as the Space icon",
    ("docs/verification-log.md", "\u2014"):
        "quotes the page's own text, which uses an em dash",
}

_MATH_SPAN = re.compile(r"\$`[^`]*`\$")
_CODE_SPAN = re.compile(r"(`+)(.+?)\1")
_PLAIN_INLINE = re.compile(r"(?<![\$\\])\$(?!\$)([^\$\n]+?)(?<!\\)\$(?!\$)")


def _markdown() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO, check=True, capture_output=True,
                         text=True).stdout.split()
    assert len(out) > 40, f"only {len(out)} tracked Markdown files; the listing is looking at nothing"
    return out


def _prose_lines(text: str):
    """(line number, line with math spans and code spans removed), outside every code block."""
    fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
            continue
        if fence or line.startswith(("    ", "\t")):
            continue
        line = _MATH_SPAN.sub(" MATH ", line)
        yield n, _CODE_SPAN.sub(" CODE ", line)


def test_github_markdown_has_no_plain_inline_math() -> None:
    bad = []
    for f in _markdown():
        if f.startswith(HF_RENDERED):
            continue
        for n, line in _prose_lines((REPO / f).read_text(errors="replace")):
            for m in _PLAIN_INLINE.finditer(line):
                body = m.group(1)
                if re.fullmatch(r"\s*\d[\d,.]*\s*", body):
                    continue                     # an amount between two dollar signs, not math
                bad.append(f"{f}:{n}: ${body}$")
    assert not bad, (
        "plain inline math, which Markdown may rewrite before it is rendered; use the house form "
        "$`...`$:\n  " + "\n  ".join(bad[:30]))


def test_hugging_face_cards_carry_no_github_math() -> None:
    bad = [f for f in _markdown() if f.startswith(HF_RENDERED)
           and "$`" in (REPO / f).read_text(errors="replace")]
    assert not bad, f"GitHub-only $`...`$ math in Markdown that Hugging Face renders: {bad}"


def test_markdown_carries_no_special_symbols() -> None:
    found: dict[tuple[str, str], int] = {}
    for f in _markdown():
        for ch in (REPO / f).read_text(errors="replace"):
            if ord(ch) > 127:
                found[(f, ch)] = found.get((f, ch), 0) + 1
    bad = {k: v for k, v in found.items() if k not in ALLOWED_NON_ASCII}
    assert not bad, (
        "non-ASCII characters in tracked Markdown, which the user asked to avoid:\n  "
        + "\n  ".join(f"{f}: {ch!r} (U+{ord(ch):04X}) x{n}" for (f, ch), n in sorted(bad.items())))
    stale = [k for k in ALLOWED_NON_ASCII if k not in found]
    assert not stale, f"allowed cases that no longer occur; remove them from ALLOWED_NON_ASCII: {stale}"
