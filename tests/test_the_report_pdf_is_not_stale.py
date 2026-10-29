"""A committed PDF that contradicts its own generated inputs is worse than no PDF.

WHY THIS EXISTS

`report/technical-report.pdf` is a fixed artefact someone keeps - `report/build_data.py`'s own
docstring says so: *"a report is a worse place for that than a README because it is a fixed artefact
someone keeps."* It was committed at `f02862f` on 2026-08-14 and prints

    383 rows at this revision
    14 August 2026

while `report/generated/macros.tex` says `\nchecklistrows{391}` and `15 August 2026`, and the
checklist is 4,240 lines. The PDF was 12 commits behind the inputs it is generated from, and nothing
could see it: the generated `.tex` files are regenerated and committed while the PDF is rebuilt by
hand, so the two drift apart silently and only the PDF is the thing a reader downloads.

TWO ASSERTIONS OF DIFFERENT KINDS, ON PURPOSE

If both checks read the same evidence, one passing tells you nothing about the other.

- **Commit time** compares git metadata: no tracked input may be newer than the PDF. This catches a
  rebuild that was skipped entirely.
- **Content** reads the bytes of the PDF through `pdftotext` and requires every value in
  `macros.tex` to appear in it. This catches a rebuild that ran against stale inputs, or a `\input`
  that silently resolved to nothing - neither of which moves a timestamp.

A rebuild has to satisfy both. Round 42 got the ORDERING of this wrong in the other direction: it
regenerated `macros.tex` before the commit that changed the checklist, so the counts were briefly
right about a tree that no longer existed. Regenerate LAST.

WHAT THIS GATE CANNOT DO

It cannot tell whether the PDF says anything TRUE - only that it is not older than its inputs and
carries their values. The independent check that belongs beside it is to `pdftotext` the rebuilt PDF
and grep for the licence qualifiers that `test_published_licence_carries_its_qualifier.py` governs;
that reads meaning rather than freshness, and no timestamp can stand in for it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PDF = REPO / "report" / "technical-report.pdf"
MACROS = REPO / "report" / "generated" / "macros.tex"

#: Everything the PDF is built from. `build_figures.py` is included because the generated figures are
#: `\includegraphics`-ed, so a figure regenerated after the PDF is the same staleness in another form.
INPUTS = [
    "report/technical-report.tex",
    "report/build_data.py",
    "report/build_figures.py",
    # What the tables and figures are drawn FROM (Round 80). The baselines figure showed August's
    # numbers for two months while metrics.json moved on: the PDF was newer than every file listed
    # here, so nothing was stale by this test's own definition.
    "models/metrics.json",
    "src/qsh/models/registry.py",
]


def _tracked_inputs() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "report/generated"], cwd=REPO, capture_output=True, text=True, check=True
    )
    generated = [f for f in out.stdout.split() if f.endswith((".tex", ".pdf"))]
    files = [f for f in INPUTS + generated if (REPO / f).exists()]
    assert len(files) >= 5, f"only {len(files)} report inputs found; has report/ moved?"
    return files


def _committed_at(rel: str) -> int:
    """Author time of the last commit touching `rel`. 0 if never committed."""
    out = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", rel], cwd=REPO, capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip() or 0)


def test_the_pdf_is_committed_no_earlier_than_its_inputs() -> None:
    """Freshness, from git metadata."""
    assert PDF.exists(), "report/technical-report.pdf is missing"
    pdf_at = _committed_at("report/technical-report.pdf")
    assert pdf_at, "the PDF is untracked; this gate governs the COMMITTED artefact"
    newer = [
        f"{f} (committed {_committed_at(f) - pdf_at}s after the PDF)"
        for f in _tracked_inputs()
        if _committed_at(f) > pdf_at
    ]
    assert not newer, (
        "these report inputs were committed AFTER the PDF that is built from them:\n  "
        + "\n  ".join(newer)
        + "\n\nRebuild it: `python report/build_data.py` then two `pdflatex` passes, and commit the "
        "PDF in the same change. Do it LAST - anything that edits the checklist moves "
        "\\nchecklistrows and \\nchecklistlines."
    )


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext (poppler) not installed")
def test_the_pdf_carries_the_values_its_macros_declare() -> None:
    """Content, from the PDF's own bytes - independent of every timestamp above."""
    text = subprocess.run(
        ["pdftotext", str(PDF), "-"], capture_output=True, text=True, check=True
    ).stdout
    assert len(text) > 2000, "pdftotext returned almost nothing; is the PDF readable?"

    # `\newcommand{\name}{value}` -> the value a reader should be able to find in the document.
    # BALANCED braces, not `[^}]*`. LaTeX writes a thousands separator as `4{,}240`, so the naive
    # form stopped at the inner `}` and compared the PDF against the string "4{,". It reported the
    # right file for the wrong reason - a truncated expectation, in the round about truncation.
    src = MACROS.read_text()
    values: dict[str, str] = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{", src):
        i, depth = m.end(), 1
        while i < len(src) and depth:
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        values[m.group(1)] = src[m.end(): i - 1]
    assert values, "macros.tex declared nothing; the parse has stopped working"

    missing = []
    for name, raw in sorted(values.items()):
        if name == "gitsha":
            continue  # structurally names the PARENT commit; see plan39's finding, not a staleness signal
        plain = raw.replace("{,}", ",").replace("\\", "").strip()
        if plain and plain not in text:
            missing.append(f"\\{name} = {plain!r}")
    assert not missing, (
        "these generated values do not appear anywhere in the committed PDF:\n  "
        + "\n  ".join(missing)
        + "\n\nThe PDF was built from different inputs than the ones committed beside it. A timestamp "
        "check cannot see this, which is why this assertion reads the bytes instead."
    )
