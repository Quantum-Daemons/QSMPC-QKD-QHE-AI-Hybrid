"""A licence qualified in the registry must be qualified everywhere it is restated.

WHY THIS EXISTS

On 2026-08-10 the BreastMNIST licence was downgraded after checking it: MedMNIST licenses ITSELF
CC BY 4.0, but BreastMNIST is a resize of BUSI, where the CC BY attaches to the ARTICLE and the
dataset host asks only "if you use this dataset, please cite". A citation request is not a grant.
The registry records that as::

    licence="CC-BY-4.0 asserted by MedMNIST; the upstream BUSI grant is NOT verifiable"

**That downgrade reached one of four places the licence is stated.** `DATA_LICENSES.md` line 36
carried it. Line 37 - the row whose own text says it is "the same corpus as the ResNet-18 row above"
- said a flat "CC BY 4.0 / redistributable: Yes". The redistributed-eval-sample paragraph said a flat
"BreastMNIST under CC BY 4.0". And `scripts/prep_model_inputs.py` hardcoded the flat string into the
`provenance` field of every published `eval_sample.json`, which is the surface where the claim
actually binds, because that file is redistributed inside three Hugging Face repos.

The word "unverifiable" appeared exactly once in a 428-line licence document.

WHAT IS ASSERTED

That for every dataset whose registry licence carries a hedge, the hedge also appears wherever that
corpus's licence is restated - in `DATA_LICENSES.md` and in any generator that writes a licence
string into a published artefact.

WHAT IS DELIBERATELY NOT ASSERTED

Not that any licence determination is right; that is a reading of the upstream terms and
`DATA_LICENSES.md` carries the argument. Only that one determination does not exist in two versions,
with the confident one on the artefact that leaves the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

from qsh.models.registry import specs

REPO = Path(__file__).resolve().parents[1]
LICENCES = REPO / "DATA_LICENSES.md"
PREP = REPO / "scripts" / "prep_model_inputs.py"
README = REPO / "README.md"

#: Words that mark a licence as qualified rather than granted. Matched case-insensitively.
HEDGES = ("unverifiable", "not verifiable", "asserted by", "unresolved")

#: Corpus name -> the token that identifies its rows in the prose. Kept explicit so a rename fails
#: here loudly rather than silently reducing the scan's subject to nothing.
CORPUS_TOKENS = {
    "BreastMNIST": "BreastMNIST",
    "ESA-ADB": "ESA-ADB",
    "AMLworld": "AMLworld",
    "OPS-SAT": "OPS-SAT",
}


def _hedged(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in HEDGES)


def _registry_licences() -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in specs():
        ds = getattr(spec, "dataset", None)
        if ds is None:
            continue
        out.setdefault(ds.name, "")
        out[ds.name] = out[ds.name] + " " + (ds.licence or "")
    return out


def test_the_scan_has_a_subject():
    """Anti-tautology: if nothing in the registry is hedged, the gate below proves nothing."""
    hedged = [n for n, lic in _registry_licences().items() if _hedged(lic)]
    assert hedged, (
        "no dataset in the registry carries a hedged licence, so this gate has nothing to check. If "
        "every licence is now cleanly granted that is good news and this test should be re-derived "
        "rather than left passing on an empty set.")


def test_every_hedged_licence_is_hedged_in_the_public_document():
    """The property: one determination, not two, and not a confident one in the shop window.

    SCOPED TO THE LICENCE CELL, not to any line mentioning the corpus. The first version of this
    check flagged every line containing the token, and fired on three lines that merely REFER to
    BreastMNIST while stating a different corpus's licence - the PneumoniaMNIST row saying it is "a
    different subset from BreastMNIST", the WDBC overlay row saying "the published medical model
    trains on BreastMNIST", and the eval-sample paragraph where the corpus and its qualifier sit on
    adjacent wrapped lines. A gate that cries wolf on correct text gets disabled, so the scan takes
    the table row whose DATASET cell names the corpus and reads that row's LICENCE cell.
    """
    doc = LICENCES.read_text()
    missing = []
    for name, lic in _registry_licences().items():
        if not _hedged(lic):
            continue
        token = next((t for c, t in CORPUS_TOKENS.items() if c.lower() in name.lower()), None)
        if token is None:
            continue
        for i, line in enumerate(doc.splitlines(), 1):
            cells = [c.strip() for c in line.split("|")]
            # A markdown row here is `| area | dataset | licence | redistributable | notes |`, so a
            # real row splits into at least six pieces counting the empty ends.
            if len(cells) < 6 or token not in cells[2]:
                continue
            if not re.search(r"CC[ -]?BY|CDLA", cells[3], re.I):
                continue
            if not _hedged(cells[3]):
                missing.append(
                    f"DATA_LICENSES.md:{i} - dataset cell names {token}, licence cell reads "
                    f"{cells[3][:60]!r} with no qualifier")
    assert not missing, (
        "the registry qualifies these licences and the public document does not:\n  "
        + "\n  ".join(missing)
        + "\n\nThe registry is the determination; DATA_LICENSES.md is where a reader checks it. A "
        "downgrade that reaches one and not the other leaves the confident version in the place "
        "people actually read - which is how the BreastMNIST row that calls itself a duplicate of "
        "the downgraded row kept saying 'CC BY 4.0 / Yes'."
    )


def test_the_string_that_ships_to_the_hub_carries_the_qualifier():
    """The surface where the licence claim actually binds, and the one the downgrade missed.

    `prep_model_inputs.py` writes `provenance` into every published `eval_sample.json`, which is
    redistributed inside the model repos. Fixing the registry and the document leaves that artefact
    still asserting the un-downgraded grant to everyone who downloads it.
    """
    src = PREP.read_text()
    for m in re.finditer(r"BreastMNIST", src):
        window = src[max(0, m.start() - 400): m.start() + 700]
        if re.search(r"CC BY 4\.0", window):
            assert _hedged(window), (
                "`scripts/prep_model_inputs.py` states BreastMNIST's licence as a flat CC BY 4.0 in a "
                "string that ships to the Hub inside eval_sample.json. The registry qualifies it; "
                "this copy is the one third parties receive.")
            return
    raise AssertionError(
        "no BreastMNIST licence string found in prep_model_inputs.py - if the generator moved, "
        "re-derive this test rather than deleting it; the point is that the SHIPPED string is "
        "checked, not only the ones in this repository.")


def test_the_readme_table_carries_the_qualifier_too():
    """The fourth surface, and this gate could not see it when it was written.

    The Round 55 fix reached `registry.py`, `DATA_LICENSES.md` and `prep_model_inputs.py`. README
    said `MedMNIST v2 **BreastMNIST** (CC-BY-4.0)` - a proper PREFIX of the registry string, cut
    exactly where the hedge begins - and carried ZERO occurrences of "BUSI", "unverifiable" or
    "asserted by MedMNIST". The gate written to stop a downgrade reaching only some copies was itself
    hardcoded to two files, so it reached three of four surfaces and reported success.

    That is the same shape as the defect it exists for, one level up: a fix that covers the places
    the fixer happened to be looking at. README is the file most readers see first.

    SCOPED to the row whose dataset cell names the corpus, for the reason recorded in the docstring
    above: a whole-file scan for the corpus name fires on passing mentions and gets switched off.
    """
    # The README and the documents its long sections moved to in Round 80 - the dataset table is in
    # docs/models-and-benchmarks.md now - so the move did not take the rows out of scope.
    bad = []
    for path in (README, *(REPO / "docs" / n for n in ("models-and-benchmarks.md", "use-cases.md",
                                                         "findings.md", "development.md"))):
      for i, line in enumerate(path.read_text().splitlines(), 1):
          cells = [c.strip() for c in line.split("|")]
          if len(cells) < 4:
              continue
          # A row that NAMES a hedged corpus and STATES a licence in the same cell.
          for name, lic in _registry_licences().items():
              if not _hedged(lic):
                  continue
              token = next((t for c, t in CORPUS_TOKENS.items() if c.lower() in name.lower()), None)
              if token is None:
                  continue
              for cell in cells:
                  if token in cell and re.search(r"CC[ -]?BY|CDLA", cell, re.I) and not _hedged(cell):
                      bad.append(f"{path.name}:{i} - names {token} and states its licence unqualified: "
                                 f"{cell[:80]!r}")
    assert not bad, (
        "README states a hedged licence without its qualifier:\n  " + "\n  ".join(bad)
        + "\n\nThe registry qualifies it and README is where most readers look first. Note the "
        "truncation shape: the unqualified form is a proper PREFIX of the qualified one, cut exactly "
        "where the hedge starts, so it reads as complete."
    )
