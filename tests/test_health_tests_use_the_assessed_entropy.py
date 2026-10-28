"""The continuous health tests are computed from the source's ASSESSED entropy, as SP 800-90B sets them.

WHY THIS EXISTS

SP 800-90B section 4.4 computes both cutoffs "Given the assessed min-entropy H of a noise source",
and section 4.4.2 says the adaptive-proportion test is "designed to detect a large loss of entropy".
Until Round 75 both implementations (`src/qsh/qrng/health.py`, `web/src/lib/clientQRNG.ts`) took H
from the most-common-value estimate of the very stream being tested. A degrading source therefore
loosened its own test: at p(1) = 0.9 the estimate fell to about 0.15 bit, the adaptive-proportion
cutoff rose to 969 of 1024, and a source emitting 90 percent ones passed. A comment beside the
browser code argued that the tests only catch a source that has gone stuck; the standard says the
second one exists precisely for the large-but-not-total failure.

The cutoff formula was also one off the standard's at three of the five entropies of its Table 2.

WHAT IS ASSERTED

- The adaptive-proportion cutoffs reproduce SP 800-90B Table 2 exactly, binary and non-binary.
- The cutoffs do not move with the stream: a fair and a ruined stream are tested against the same
  cutoffs, those of the assessed H.
- A source emitting 90 percent ones fails; a fair one passes; the estimate is still reported, and the
  report says when it fell below the assessment.
- The seedless extractor's ceiling is Giestinhas and Spiller's Eq. 7, which charges a weak seed's
  shortfall, not Eq. 5, which assumes a uniform one.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsh.qrng.health import adaptive_proportion_cutoff, health_suite, repetition_count_cutoff
from qsh.qrng.simulator import lhl_output_length, lhl_output_length_weak_seed

#: SP 800-90B (2018) Table 2, alpha = 2^-20: (entropy per sample, window, cutoff).
TABLE_2 = [(0.2, 1024, 941), (0.4, 1024, 840), (0.6, 1024, 748), (0.8, 1024, 664), (1.0, 1024, 589),
           (0.5, 512, 410), (1.0, 512, 311), (2.0, 512, 177), (4.0, 512, 62), (8.0, 512, 13)]

ASSESSED = 0.9


@pytest.mark.parametrize("h,w,cutoff", TABLE_2)
def test_adaptive_proportion_cutoffs_reproduce_table_2(h: float, w: int, cutoff: int) -> None:
    assert adaptive_proportion_cutoff(h, w) == cutoff


def _stream(p_one: float, n: int = 8192, seed: int = 7) -> np.ndarray:
    return (np.random.default_rng(seed).random(n) < p_one).astype(np.int64)


def test_the_cutoffs_do_not_move_with_the_stream() -> None:
    fair = health_suite(_stream(0.5), raw=_stream(0.5), assessed_h=ASSESSED)
    ruined = health_suite(_stream(0.9), raw=_stream(0.9), assessed_h=ASSESSED)
    assert fair.adaptive_proportion_cutoff == ruined.adaptive_proportion_cutoff \
        == adaptive_proportion_cutoff(ASSESSED)
    assert fair.repetition_count_cutoff == ruined.repetition_count_cutoff \
        == repetition_count_cutoff(ASSESSED)
    assert fair.assessed_min_entropy_per_bit == ASSESSED


def test_a_source_emitting_ninety_percent_ones_fails() -> None:
    ruined = health_suite(_stream(0.9), raw=_stream(0.9), assessed_h=ASSESSED)
    assert not ruined.adaptive_proportion_passed and not ruined.passed, ruined.failures
    assert ruined.estimate_below_assessment and ruined.min_entropy_per_bit < 0.2


def test_a_fair_source_passes() -> None:
    fair = health_suite(_stream(0.5), raw=_stream(0.5), assessed_h=ASSESSED)
    assert fair.passed and not fair.estimate_below_assessment, fair.failures


def test_the_default_assessment_is_the_configured_one() -> None:
    """A caller that passes nothing still gets the assessment, never the stream's estimate."""
    from qsh.config import get_config
    report = health_suite(_stream(0.9), raw=_stream(0.9))
    assert report.assessed_min_entropy_per_bit == get_config().qrng.min_entropy_per_raw_bit
    assert not report.passed


def test_the_seedless_ceiling_is_eq_7_not_eq_5() -> None:
    n, h, eps = 8208, 0.9, -40
    assert lhl_output_length(h * n, eps) == 7309                      # Eq. 5, a uniform seed
    assert lhl_output_length_weak_seed(h * n, n, h, eps) == 5898      # Eq. 7, a 0.9-entropy seed
    out = lhl_output_length_weak_seed(h * n, n, h, eps)
    seed_len = out + n - 1
    # Eq. 7 itself, with the seed the extractor actually uses: out <= H_in + H_seed - |seed| - 80 + 2.
    assert out <= h * n + h * seed_len - seed_len - 2 * 40 + 2
    assert out + 1 > h * n + h * (seed_len + 1) - (seed_len + 1) - 2 * 40 + 2
