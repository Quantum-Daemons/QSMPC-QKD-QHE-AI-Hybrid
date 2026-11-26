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

import hashlib
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Rendered by Hugging Face, not GitHub.
HF_RENDERED = ("models/", "spaces/")

#: (file, character) -> why it stays.
ALLOWED_NON_ASCII = {
    ("docs/verification-log.md", "\u2014"):
        "quotes the page's own text, which uses an em dash",
}

_MATH_SPAN = re.compile(r"\$`[^`]*`\$")
_CODE_SPAN = re.compile(r"(`+)(.+?)\1")
_PLAIN_INLINE = re.compile(r"(?<![\$\\])\$(?!\$)([^\$\n]+?)(?<!\\)\$(?!\$)")


def _ls(*flags: str) -> list[str]:
    return subprocess.run(["git", "ls-files", *flags, "*.md"], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout.split()


def _markdown() -> list[str]:
    # Tracked files the working tree still has. A file deleted but not yet committed stays in the
    # index, and reading it raised FileNotFoundError: Round 84 deleted the two 9-wire QCNN model
    # cards, and both tests below failed on a file that can carry no symbol at all. The floor is
    # unchanged, so an empty listing still fails.
    gone = set(_ls("--deleted"))
    out = [f for f in _ls() if f not in gone]
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


#: A formula typed as prose (Round 86): a power, a one-letter subscript, a root or a logarithm written as
#: a call, an inequality, a plus-minus or an absolute value, outside every math span. The test above
#: catches math in the wrong delimiters; these are formulas with none, which Round 86 found 88 of in the
#: tracked documents, most in the register. A quotation keeps its source's own typing, so text in double
#: quotes and blockquote lines are not read, nor are code, math spans and display blocks.
_PROSE_FORMULA = {
    "power": re.compile(r"(?<![\w/.-])(?:\d+|[A-Za-z]|\))\^[-+]?(?:\d+|[A-Za-z]\w*|\{[^}]*\}|\([^)]*\))"),
    "subscript": re.compile(r"(?<![\w/.`-])[A-Za-z]_(?:[a-z0-9]|\{[^}]*\})(?![\w/])"),
    "root or logarithm": re.compile(r"\b(?:sqrt|log2|log10|ln)\("),
    "inequality": re.compile(r"(?<=\s)(?:<=|>=|!=)(?=\s)"),
    "plus-minus": re.compile(r"(?<!\w)\+/?-\s?\d"),
    "absolute value": re.compile(r"(?<![|\w])\|[a-zA-Z][\w ]{0,12}\|(?![|\w])"),
}
_QUOTED = re.compile(r'"[^"]*"')


def _prose_paragraphs(text: str):
    """(first line, paragraph) of prose: code, math spans, display blocks and quotations removed."""
    fence = display = False
    buf: list[str] = []
    start = 0

    def flush():
        para = "\n".join(buf)
        para = _MATH_SPAN.sub(" MATH ", para)
        para = re.sub(r"(`+)(.+?)\1", " CODE ", para, flags=re.S)
        return start, _QUOTED.sub(" QUOTE ", para)

    for n, line in enumerate(text.splitlines(), 1):
        bare = line.strip()
        if bare.startswith(("```", "~~~")):
            fence = not fence
        elif bare == "$$":
            display = not display
        elif not (fence or display or line.startswith(("    ", "\t")) or bare.startswith((">", "|---"))
                  or (bare.startswith("$$") and bare.endswith("$$"))) and bare:
            if not buf:
                start = n
            buf.append(line)
            continue
        if buf:
            yield flush()
            buf = []
    if buf:
        yield flush()


def _prose_formulas(text: str) -> list[str]:
    return [f"{n}: {name} {m.group(0)!r}" for n, para in _prose_paragraphs(text)
            for name, pat in _PROSE_FORMULA.items() for m in pat.finditer(para)]


def test_the_prose_formula_detector_sees_one_and_skips_quotes_and_code() -> None:
    """Without this, a detector that matches nothing would pass the test below."""
    sample = ("A mask of 2^30 hides x_k, and S = 2 sqrt(2) holds when p <= 1, within +/- 0.4 of |B|.\n\n"
              'Quoted: "rises with n >= 15". Code: `2^30` and $`2^{30}`$.\n\n> A quote: 2^30.\n\n'
              "$$\nC_s = 10^{-5}\n$$\n")
    names = sorted({hit.split()[1] for hit in _prose_formulas(sample)})
    assert names == ["absolute", "inequality", "plus-minus", "power", "root", "subscript"], names
    assert all(hit.startswith("1:") for hit in _prose_formulas(sample)), _prose_formulas(sample)


#: Frozen texts, each with the SHA-256 it is frozen at. A pre-registration's bytes are pinned by its trainer
#: (`PREREG_SHA256`) and were committed before its run, so retypesetting one after the run would be an edit to a
#: pre-registration with its results known. The exemption covers that file at that hash only and lapses, failing below,
#: the moment the file changes.
FROZEN = {
    # "clipped to +-10": committed 2026-11-25 (8f70aa5) before this test read it, the finance square student's run
    # under it the same day.
    "references/finance-square-student-preregistration.md":
        "9ac9039309cbf6bfa53b76fcbe749f817a3d25839419670e99413eb14d49759e",
    # F7b: committed 2026-11-25 before any of its seeds was trained. It needs no exemption (it passes the checks above)
    # and is listed so that no edit reaches it before its trainer pins the same hash.
    "references/finance-square-student-page-preregistration.md":
        "388a1668b82b8e9011539511efcbc3887b053a2e0e60022040c4054eee60492f",
    # The finance QCNN on AMLworld: committed 2026-11-25 before any input was scored for selection or any circuit trained.
    # The satellite threshold comparison: committed 2026-11-25 before any threshold was fitted or any test window scored.
    "references/satellite-threshold-calibration-preregistration.md":
        "a2436cde71d939c32b7d67f895f892fd0012262f1c7a1fb3a39505d85a153d65",
    # The 5G-NIDD CKKS student: committed 2026-11-25 before any student was trained.
    "references/telecom-nidd-ckks-student-preregistration.md":
        "b356193c073606ffa2ef8aa66b2cfa5ac3b9dcf679bd3b558ce4d0c44de5d34d",
    "references/finance-qcnn-amlworld-preregistration.md":
        "5a411b05d7e1984306d08397d95b147ab2a09796a4bfa4a6b624cf515253edd0",
    # Federated learning across 5G-NIDD's two base stations: committed 2026-11-26 before any of its models was trained.
    "references/telecom-nidd-federated-preregistration.md":
        "5e3a137bfb4c0382ce626bb733531f94b9a505b7ec93e8500bdad8122affff0d",
    # The downlink against Jinan-1's measured pass: committed 2026-11-26 (054aaab) before any model figure was set beside
    # the data, its addendum (df2bae4) before the comparison ran, and addendum 2 after it, saying the windows failed and
    # naming the post-hoc analysis as post hoc.
    "references/satellite-jinan1-pass-preregistration.md":
        "11665e2b800dd3f921d9b4bf5122252354e95d85fe773e64130c4ae284ae1984",
    # The Jinan-1 pass's second analysis: committed 2026-11-26 before any of its figures was computed.
    "references/satellite-jinan1-pass-v2-preregistration.md":
        "ddaea49f1f277ef457af9d41c1a775b4134daf0c8aec0dec73ee40b8b338292e",
    # The finance QCNN's fair comparison: committed 2026-11-26 before any of its arms ran.
    "references/finance-qcnn-amlworld-comparison-preregistration.md":
        "dc56325772e004894575ba59e71faf7e50d40f8250ac46c81f0d27a6b8322d2c",
    # The tensor-network checks of the medical, telecom and satellite QCNNs: committed 2026-11-26 (362cfc6) before any
    # ran, and its addendum the same day after the first run, which traced in single precision (see the addendum).
    "references/qcnn-tensor-network-checks-preregistration.md":
        "07647cb60c7d3b8430eec6fd516c75ca4ec27e2885f3ce1abebd6e623c81aebf",
}


def test_each_frozen_text_is_still_the_frozen_one() -> None:
    changed = [f for f, h in FROZEN.items() if hashlib.sha256((REPO / f).read_bytes()).hexdigest() != h]
    assert not changed, f"frozen texts that changed, so their exemption no longer applies: {changed}"


def test_github_markdown_types_no_formula_as_prose() -> None:
    bad = []
    for f in _markdown():
        if f.startswith(HF_RENDERED):
            continue
        if f in FROZEN and hashlib.sha256((REPO / f).read_bytes()).hexdigest() == FROZEN[f]:
            continue
        bad += [f"{f}:{hit}" for hit in _prose_formulas((REPO / f).read_text(errors="replace"))]
    assert not bad, ("formulas typed as prose; write them in the house form $`...`$ (or quote the source "
                     "they come from):\n  " + "\n  ".join(bad[:30]))


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


def test_no_math_span_holds_a_control_character() -> None:
    # A backslash command written through a Python string without an r prefix loses its backslash to an escape:
    # "\times" became a TAB and "imes", "\tau_0" a TAB and "au_0". Both reached main (threat-model.md, from the
    # aggregation write-up of 2026-11-13, and models-and-benchmarks.md) and rendered as a gap and a stray word, because
    # nothing looked inside a math span for a character no formula contains. Found and repaired on 2026-11-13.
    bad = []
    for f in _markdown():
        for n, line in enumerate((REPO / f).read_text(errors="replace").splitlines(), 1):
            for span in _MATH_SPAN.findall(line):
                if any(ord(ch) < 32 for ch in span):
                    bad.append(f"{f}:{n}: {span!r}")
    assert not bad, "math spans holding a control character (a lost backslash escape?):\n  " + "\n  ".join(bad)
