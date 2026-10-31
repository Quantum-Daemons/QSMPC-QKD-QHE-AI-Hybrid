"""The genomics secure sum returns the pooled allele count itself, not the count mod a tiny field.

WHY THIS EXISTS

QSMS picks the paper's field by default, d prime with n <= d <= 2n: GF(5) for the four cohorts. The
pooled effect-allele count at the score's first variant is 45 on the bundled slice, and came back
as 0. `run_demo` reduced its "ground truth" mod the same prime, so it reported 0 == 0 and the test
of exactness passed by construction. The browser twin chose its field from the ANSWER instead,
nextPrime(total + 1), which is the same "true by construction" from the other side
(scratchpad audit82/factcheck-finance-genomics.md, A5; web/scripts/check_qsms_field.mjs).

WHAT IS ASSERTED

- The pooled count equals the UNREDUCED sum of the cohorts' counts, recomputed here from the data.
- The field comes from the largest total the inputs can reach (two alleles per person), not from
  the sum, and the run records that it therefore leaves the paper's bound.
- A pinned field too small for the counts is refused rather than allowed to wrap.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsh.qsmpc.secure_summation import QSMS
from qsh.qsmpc.shamir import is_probable_prime


def _cohort_counts(seed: int = 1) -> tuple[list[int], int]:
    """The per-cohort counts the use case pools, recomputed from the loader, and the cohort size."""
    from qsh.config import get_config
    from qsh.data.loaders import load_genomic_prs

    d = get_config().data
    dosages, *_ = load_genomic_prs(n_individuals=max(d.genomics_n // 3, 200), seed=seed,
                                   name=d.genomics_prs_score)
    assert set(np.unique(dosages[:, 0])) <= {0.0, 1.0, 2.0}, "dosages are not allele counts"
    return [int(s.sum()) for s in np.array_split(dosages[:, 0], d.genomics_biobanks)], int(dosages.shape[0])


def test_the_pooled_count_is_the_unreduced_total():
    from qsh.usecases import genomics

    counts, people = _cohort_counts()
    r = genomics.run_demo({"seed": 1})
    assert r["ground_truth_total"] == sum(counts), "the ground truth is reduced or recomputed wrongly"
    assert r["pooled_allele_count"] == sum(counts), (
        f"QSMS returned {r['pooled_allele_count']} for counts summing to {sum(counts)}")
    f = r["qsms_field"]
    assert f["max_possible_total"] == 2 * people and f["prime"] > f["max_possible_total"]
    # Recorded, not hidden: a field that holds the counts is far outside n <= d <= 2n.
    assert f["field_within_paper_bound"] is False and f["honest_statevector"] is False


def test_the_field_comes_from_the_range_not_the_answer():
    from qsh.qsmpc.secure_summation import field_for_total

    assert field_for_total(194, 4) == 197
    assert field_for_total(3, 7) == 11                  # the paper's own example field for n = 7
    for max_total in (10, 59, 60, 194, 1000):
        p = field_for_total(max_total, 4)
        assert is_probable_prime(p) and p > max_total
        # Every total the inputs can reach survives the sum, including the largest.
        for total in (0, max_total // 2, max_total):
            assert QSMS(n=4, t=3, prime=p).run([total, 0, 0, 0], seed=0).total == total


def test_a_field_too_small_for_the_counts_is_refused():
    from qsh.usecases import genomics

    with pytest.raises(ValueError, match="would wrap"):
        genomics.build({"seed": 1, "prime": 5})
