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

import pytest

from qsh.data.loaders import _PRS_FILES, load_genomic_prs, load_prs


class _CatchFallback(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.mark.parametrize("name", sorted(_PRS_FILES))
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
    assert dosages.shape[0] == 400


@pytest.mark.parametrize("name", sorted(_PRS_FILES))
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
