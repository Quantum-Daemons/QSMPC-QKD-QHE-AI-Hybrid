"""A PDF this repository hands out must have a licence that permits handing it out.

WHY THIS EXISTS

`references/` held 11 tracked PDFs, 14.5 MB, about 45% of everything tracked - the largest single
redistribution category in the project - and nothing had ever adjudicated them. `DATA_LICENSES.md`,
whose title is "licences, attribution, and what may be published", did not mention `references/` or
`.pdf` once. Its preamble named three buckets (training corpora, bundled demo samples, trained
weights) and none of them was "a third-party document we redistribute verbatim", so this was a
structural blind spot rather than a missing row: the question had never been in scope.

What the exclusions ran on instead was SIZE. `.gitignore` dropped three PDFs for being "~35 MB
total" and one as a copyrighted book. The result was an inversion - the two most permissively
licensed papers in the project, both CC BY 4.0, were excluded for being ~17 MB each, while a
CC BY-NC-ND camera-ready shipped to every clone.

THE TRAP THIS CHECK ENCODES

arXiv's Atom API - `export.arxiv.org/api/query`, the obvious thing to reach for - carries **no
licence field at all**. It reports nothing even for a CC0 paper. An adjudication run through it reads
the entire corpus as unlicensed and would have excluded three papers that are in fact CC0 or CC BY.
The OAI-PMH interface returns a real `<license>` element. Recorded in the failure message, because
the next person to add a PDF will reach for the same wrong API.

WHAT IS AND IS NOT ASSERTED

This does not re-fetch licences at test time; a gate that needs the network is a gate that goes
yellow on a train. It asserts the LOCAL invariant: every tracked PDF has a row in section 8 of
`DATA_LICENSES.md` and appears under "Kept", and every file listed as excluded is genuinely not
tracked. The adjudication itself is dated evidence in that file, which is where a human can check it.

The corpus comes from `git ls-files`, never a glob. A glob would describe the author's disk; the
question here is what a CLONE contains, and those differ by exactly the files this round removed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LICENCES = REPO / "DATA_LICENSES.md"

#: Licences that permit redistributing a verbatim copy to anyone who clones. Deliberately a
#: whitelist: arXiv's `nonexclusive-distrib/1.0` grants arXiv the right to distribute and third
#: parties nothing, and reads like a licence to anyone skimming, so an "everything except" rule
#: would admit it by default.
GRANTS = ("CC0 1.0", "CC BY 4.0")


def _tracked_pdfs() -> list[str]:
    out = subprocess.run(["git", "ls-files", "references/*.pdf"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.splitlines() if p.strip())


def _section_8() -> str:
    text = LICENCES.read_text()
    start = text.index("## 8. Bundled reference PDFs")
    end = text.index("\n## ", start + 10)
    return text[start:end]


def test_section_8_exists_and_is_not_empty() -> None:
    """Non-vacuity first: every assertion below passes trivially against a missing section."""
    body = _section_8()
    assert "### Kept" in body and "### Excluded" in body, (
        "section 8 of DATA_LICENSES.md no longer has both a Kept and an Excluded table; the "
        "adjudication cannot be checked in one direction only"
    )
    assert len(_tracked_pdfs()) >= 5, (
        f"only {len(_tracked_pdfs())} tracked PDFs found under references/. If they were all "
        "removed this check has no subject; it must not read as green."
    )


def test_every_tracked_pdf_is_adjudicated_and_kept() -> None:
    """The direction that matters: nothing ships without a recorded grant."""
    kept = _section_8().split("### Excluded")[0]
    missing = [p for p in _tracked_pdfs() if Path(p).name not in kept]
    assert not missing, (
        "these PDFs are committed - every clone receives them - and no row in section 8 of "
        f"DATA_LICENSES.md records a licence permitting that:\n  " + "\n  ".join(missing)
        + "\n\nAdjudicate at PRIMARY SOURCE and add a row. For an arXiv paper use OAI-PMH:\n"
        "  https://oaipmh.arxiv.org/oai?verb=GetRecord&metadataPrefix=arXiv"
        "&identifier=oai:arXiv.org:NNNN.NNNNN\n"
        "NOT export.arxiv.org/api/query - the Atom API carries no licence field and reports "
        "'not present' even for CC0 papers, so it reads every paper as unlicensed."
    )


def test_every_kept_row_names_a_licence_that_permits_redistribution() -> None:
    """A row is not a grant. The row has to say something that actually allows this."""
    kept = _section_8().split("### Excluded")[0]
    rows = [ln for ln in kept.splitlines() if ln.startswith("| `") and ".pdf" in ln]
    assert len(rows) >= 5, f"only {len(rows)} rows parsed from the Kept table; the shape changed"
    ungranted = [r.split("|")[1].strip() for r in rows
                 if not any(g in r for g in GRANTS)]
    assert not ungranted, (
        f"these Kept rows name no redistribution-permitting licence: {ungranted}\n\n"
        f"Accepted: {GRANTS}. arXiv's default 'nonexclusive-distrib/1.0' is NOT one - it grants "
        "arXiv the right to distribute and grants third parties nothing. CC BY-NC-ND is not one "
        "either: section 7 of this same file already rejects a dataset on that licence."
    )


def test_every_excluded_pdf_is_actually_gone() -> None:
    """The claim 'excluded' has to be true of the repository, not just of the table.

    Checked against `git ls-files` rather than the filesystem, because the files still exist on the
    author's machine - they are gitignored, not deleted. The question is what a clone gets.
    """
    excluded = _section_8().split("### Excluded")[1]
    tracked = {Path(p).name for p in _tracked_pdfs()}
    still_here = [ln.split("|")[1].strip().strip("`")
                  for ln in excluded.splitlines()
                  if ln.startswith("| `") and ".pdf" in ln
                  and any(Path(ln.split("|")[1].strip().strip("`").replace("...", "")).name
                          and t.startswith(ln.split("|")[1].strip().strip("`").split("...")[0].lstrip("`"))
                          for t in tracked)]
    assert not still_here, (
        f"section 8 lists these as excluded, but they are still tracked: {still_here}. A table "
        "saying a file was removed, beside a repository that still ships it, is worse than no "
        "table - it is a licence finding that reads as discharged."
    )


def test_the_gitignore_rules_state_a_licence_reason() -> None:
    """Excluding on size is what produced the inversion this round removed.

    The three older rules exclude on ~35 MB and on one copyright call, which is why two CC BY 4.0
    papers were dropped while a CC BY-NC-ND camera-ready shipped. Those rules are left alone - the
    files are genuinely large and one is genuinely a book - but any rule added for a licence reason
    must say so, or the next reader will assume size again.
    """
    lines = (REPO / ".gitignore").read_text().splitlines()
    block = "\n".join(lines)
    assert "NO REDISTRIBUTION GRANT" in block, (
        "the licence-based exclusion block is gone from .gitignore. Without it the only stated "
        "reason for excluding a reference PDF is size, which is the rule that let a CC BY-NC-ND "
        "PDF ship while two CC BY 4.0 papers were dropped."
    )
    assert re.search(r"nonexclusive-distrib", block), (
        "the .gitignore comment no longer names arXiv's default licence. It is the one that looks "
        "like a grant and is not, so naming it is the whole point of the comment."
    )


def test_every_R_marker_names_a_file_a_clone_actually_gets() -> None:
    """The marker IS the claim, and Round 45 falsified it four times.

    `references/REFERENCES.md:14` defines `[R]` as "the PDF is committed to this folder and you get
    it when you clone". Round 45 untracked four PDFs on licence grounds, wrote the adjudication
    table, listed "leaving `REFERENCES.md` marking them `[R]`" in its own plan under "where this
    round most likely ships its own defect" - and shipped exactly that.

    The file had already been broken this way once: `:19` records that "an earlier revision marked
    the `[R-local]` entries `[R]`, which told a reader they had files that a clone never contained".
    So the licence half of the boundary got a new table and a new gate, and the reader-facing half
    got neither.

    Checked in BOTH directions, because either alone is satisfiable by deleting the other side.
    """
    import re

    text = (REPO / "references" / "REFERENCES.md").read_text()
    tracked = {Path(p).name for p in _tracked_pdfs()}
    assert tracked, "no tracked reference PDFs; this check has no subject"

    claimed, orphans = set(), []
    for m in re.finditer(r"`\[R\]`", text):
        # The entry's own `File:` line. The legend and the prose about markers have none, and are
        # deliberately not entries.
        block = text[m.start():m.start() + 1400]
        fm = re.search(r"File:\s*`?([^`\n]+\.pdf)", block)
        if not fm:
            continue
        stem = fm.group(1).strip().split("...")[0].strip()
        hit = next((x for x in tracked if x.startswith(stem[:40])), None)
        if hit is None:
            orphans.append(f"REFERENCES.md:{text.count(chr(10), 0, m.start()) + 1}: {stem[:70]}")
        else:
            claimed.add(hit)

    assert not orphans, (
        "these entries are marked `[R]` - which promises a clone contains the PDF - and the file is "
        "not tracked:\n  " + "\n  ".join(orphans)
        + "\n\nUse `[link]` (never bundled) or `[R-local]` (present here, gitignored). Excluding a "
        "PDF and leaving its marker tells a reader they have a file they do not have."
    )
    missing = sorted(tracked - claimed)
    assert not missing, (
        f"these PDFs ARE tracked but no `[R]` entry claims them: {missing}\n\n"
        "A bundled paper with no entry is a file a clone receives and no reader can place."
    )
