"""The PRS path uses REAL 1000 Genomes dosages, and says so by not falling back.

WHY THIS EXISTS. ``load_genomic_prs`` wraps its dosage read in ``except Exception`` and, on any
failure, samples genotypes from Hardy-Weinberg equilibrium with a warning. That is the right design -
offline CI has to work - but it means a broken read produces a full, plausible, correctly-shaped result
whose numbers are synthetic. Round 29 tripped it: a dosage file carrying ``population`` and
``super_population`` columns made ``int("GBR")`` raise, and the whole genomics use case quietly moved
to sampled genotypes while the docstring went on saying "real 1000 Genomes phase-3".

The first check written for that bug was itself worthless and is worth recording as the reason this
file asserts what it asserts: it tested that the dosages were all in ``{0, 1, 2}``. Hardy-Weinberg
sampling via ``rng.binomial(2, freq)`` also produces exactly ``{0, 1, 2}``, so the assertion passed in
both worlds. **The only signal that distinguishes them is the warning**, so that is what is asserted
here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from qsh.data.loaders import _PRS_FILES, load_genomic_prs, load_prs

REPO = Path(__file__).resolve().parents[1]

#: Scores whose dosages are available as REAL genotypes on any checkout, full file or committed slice.
#:
#: Named for what it means. It used to be `SCORES_WITH_REAL_DOSAGES`, computed from `.exists()` on the
#: full file only - so PRS313 was in the list on a maintainer's machine, where the gitignored 2,504-row
#: matrix is present, and out of it in CI. Both runs printed green while testing different things, and
#: the path that mattered was the one CI skipped.
#:
#: PRS313's full matrix is deliberately not in the repository: `DATA_LICENSES` section 4 justifies
#: about 30,800 genotype cells and 2,504 x 313 is 24x that. A 97 x 313 slice at the same cell budget IS
#: committed, so this list is now the same in both places and the assertions below mean the same thing
#: wherever they run.
def _dosage_path(name: str) -> Path | None:
    """The file `load_genomic_prs` will actually read: full matrix, else the committed slice."""
    _, dosage_file = _PRS_FILES[name]
    d = REPO / "data" / "genomics"
    for cand in (d / dosage_file, d / f"{Path(dosage_file).stem}_slice.tsv"):
        if cand.exists():
            return cand
    return None


def _n_individuals(name: str) -> int:
    """How many real genotype rows that file holds - the number the loader can return, at most."""
    path = _dosage_path(name)
    assert path is not None
    return sum(1 for ln in path.read_text().splitlines()
               if ln.strip() and not ln.startswith("#")) - 1     # minus the header row


def _real_dosages_available(name: str) -> bool:
    return _dosage_path(name) is not None


SCORES_WITH_REAL_DOSAGES = sorted(n for n in _PRS_FILES if _real_dosages_available(n))


class _CatchFallback(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.mark.parametrize("name", SCORES_WITH_REAL_DOSAGES)
def test_dosages_are_real_not_sampled(name: str) -> None:
    handler = _CatchFallback()
    log = logging.getLogger("qsh")
    log.addHandler(handler)
    try:
        dosages, weights, keys, eff, freq = load_genomic_prs(n_individuals=400, seed=0, name=name)
    finally:
        log.removeHandler(handler)

    assert not handler.messages, (
        f"{name} fell back to Hardy-Weinberg sampling instead of reading the bundled genotypes, so "
        f"every genomics number derived from it would be synthetic while labelled real: "
        f"{handler.messages[0][:160]}"
    )
    assert dosages.shape[1] == len(weights) == len(keys) == len(eff)
    # Not a hardcoded 400. PRS77 ships 400 individuals and PRS313 ships a 97-row slice at the same
    # genotype-cell budget, so the count is read from the file the loader will use. A literal 400 here
    # asserted a property of one score's bundled file and failed on the other for a reason that had
    # nothing to do with whether the genotypes were real - which is what this test is about.
    expected = min(400, _n_individuals(name))
    assert dosages.shape[0] == expected, (
        f"asked for 400 individuals and got {dosages.shape[0]}; {_dosage_path(name).name} holds "
        f"{_n_individuals(name)}, so {expected} was expected. A short read means the loader dropped "
        "real rows rather than the file being smaller than the request."
    )


@pytest.mark.parametrize("name", SCORES_WITH_REAL_DOSAGES)
def test_every_score_variant_has_a_dosage_column(name: str) -> None:
    """The join is by column NAME, so a renamed or missing variant must fail loudly, not silently.

    Positional slicing is what broke: it assumed exactly one metadata column before the variants.
    """
    keys, _, weights = load_prs(name)
    assert len(keys) == len(weights)
    # Both bundled scores are joined against their own dosage file; a mismatch raises inside
    # load_genomic_prs and is caught by the test above. Here, just assert the shapes agree.
    dosages, w, joined_keys, _, _ = load_genomic_prs(n_individuals=50, seed=0, name=name)
    assert joined_keys == keys, "the join reordered or dropped variants relative to the score file"


def test_the_two_bundled_scores_are_keyed_differently_on_purpose() -> None:
    """PRS77 is keyed by rsID; PRS313 by chr:pos:effect_allele, because it publishes no rsIDs.

    Pinned because the natural "tidy-up" is to make both look the same, and doing that to PRS313 means
    inventing identifiers: 28% of its positions carry several rsIDs and about 15% cannot be resolved
    at all, so a guess would score a different variant's genotypes.
    """
    rs_keys, _, _ = load_prs("breast_cancer")
    pos_keys, _, _ = load_prs("breast_cancer_prs313")
    assert all(k.startswith("rs") for k in rs_keys)
    assert len(rs_keys) == 77
    assert all(k.count(":") == 2 and not k.startswith("rs") for k in pos_keys)
    assert len(pos_keys) == 313


def test_at_least_one_score_has_committed_dosages() -> None:
    """The parametrised tests above must not be able to pass by collecting nothing.

    `SCORES_WITH_REAL_DOSAGES` is computed from the filesystem, so deleting the bundled dosage file
    would empty it and every test in this module would pass vacuously - the exact shape of failure
    this module was written to catch. PRS77's dosages are committed by design (400 individuals x 77
    SNPs, justified in DATA_LICENSES section 4), so at least one entry must always be present.
    """
    assert "breast_cancer" in SCORES_WITH_REAL_DOSAGES, (
        "data/genomics/g1k_dosages.tsv is missing. It is committed deliberately, and without it the "
        "real-genotype assertions in this file collect nothing and pass without checking anything."
    )
