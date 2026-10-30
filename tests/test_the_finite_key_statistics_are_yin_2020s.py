"""The finite key's statistics: Yin et al. 2020's Chernoff-type bounds, and (A16) as Rusca prints it.

WHY THIS EXISTS (Round 82, the user's choice)

Both published bounds the finite key rests on - Lim et al. 2014 (vacuum + weak decoy, the satellite)
and Rusca et al. 2018 (one decoy, the fibre) - move every count's expectation by the Hoeffding
deviation of its basis TOTAL, sqrt(n/2 ln(1/eps)). The model's key on Liao et al. 2017's pass was
0.55 of the published one, and the same fraction, 0.56, of Khmelev et al. 2023's, whose key was
computed with Chernoff bounds; Round 79 read that as the gap being the analysis, and this round tests
it with Yin, Zhou, Gu, Xie, Lu and Chen, Sci. Rep. 10, 14312 (2020), arXiv:2002.06530, which bounds
the expectation of a sum of independent Bernoulli variables from its OBSERVED value in closed form,
read on the rendered page:

  (9)   x* <= x + beta + sqrt(2 beta x + beta^2)
  (10)  x* >= x - beta/2 - sqrt(2 beta x + beta^2/4),        beta = ln(1/eps)

Neither statistics is tighter everywhere: Chernoff's deviation, about sqrt(2 beta x), beats
Hoeffding's sqrt(n beta / 2) only for a count under a quarter of its basis. So the default bounds each
count both ways at eps/2 and keeps the tighter, a union bound. And reading (A16) again for this found
the third transcription error in the 1-decoy bound: tau0 on delta(n_Z) as well as on the error term,
which the paper does not print and its derivation does not allow - the unsafe direction.

WHAT IS ASSERTED

- The closed forms are the paper's, and they HOLD: drawn many times from sums of Bernoulli variables
  with unequal probabilities, the expectation falls outside each bound less often than eps.
- Where each statistics wins: Chernoff for a count that is a small share of its basis, Hoeffding for
  a large share, as the deviations' forms say.
- The default is never worse than either alone, by more than the price of halving eps, on the fibre's
  threshold, the fibre's default block and Liao's pass; and it is the default everywhere.
- (A16) as printed: the single-photon bound the module returns is recomputed here from the counts by
  the paper's (A16)-(A17), independently, and the old placement is shown to give a different one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qsh.config import get_config
from qsh.qkd.channel import ChannelParams
from qsh.qkd.finite_key import (
    DEVIATIONS,
    FiniteKeyParams,
    VacuumDecoyParams,
    _bounds,
    _chernoff_lower,
    _chernoff_upper,
    _delta,
    expected_counts,
    fibre_block_pulses,
    finite_key_length,
    finite_key_over,
    minimum_pulses_for_positive_key,
    one_decoy_key_length,
)


def test_the_closed_forms_are_the_papers() -> None:
    """Eqs. (9) and (10) at one point, by hand: x = 100, eps = 1e-10."""
    beta = math.log(1e10)
    assert _chernoff_upper(100.0, 1e-10) == pytest.approx(100 + beta + math.sqrt(200 * beta + beta ** 2), rel=1e-12)
    assert _chernoff_lower(100.0, 1e-10) == pytest.approx(100 - beta / 2 - math.sqrt(200 * beta + beta ** 2 / 4),
                                                          rel=1e-12)
    # Floored at zero, as the paper requires, and an observed zero still bounds the expectation above.
    assert _chernoff_lower(5.0, 1e-10) == 0.0
    assert _chernoff_upper(0.0, 1e-10) == pytest.approx(2 * beta, rel=1e-12)


@pytest.mark.parametrize("mean", [3.0, 40.0, 400.0])
def test_the_bounds_hold_at_their_failure_probability(mean) -> None:
    """The point of a bound is that it holds. 20,000 draws of a sum of 2,000 Bernoulli variables with
    unequal probabilities (the case Yin et al. state, "not necessarily equal"), at eps = 0.01: the
    expectation is above the upper bound, and below the lower, in under 1 percent of draws each."""
    rng = np.random.default_rng(82)
    p = rng.uniform(0.2, 1.8, 2000)
    p *= mean / p.sum()
    trials, eps = 20_000, 0.01
    draws = (rng.random((trials, p.size)) < p).sum(axis=1).astype(float)
    above = sum(mean > _chernoff_upper(x, eps) for x in draws) / trials
    below = sum(mean < _chernoff_lower(x, eps) for x in draws) / trials
    assert above <= eps and below <= eps, (above, below)


def test_chernoff_wins_for_a_small_share_and_hoeffding_for_a_large() -> None:
    """At a basis of 1e6 detections and eps = 1e-10: a count of 1 percent of it is bounded far more
    tightly by its own Chernoff deviation, one of 70 percent more loosely."""
    n, eps = 1e6, 1e-10
    for share, chernoff_wins in ((0.01, True), (0.7, False)):
        x = share * n
        lo, hi = _bounds((x, n - x), eps, "chernoff")
        h_lo, h_hi = _bounds((x, n - x), eps, "hoeffding")
        assert ((hi[0] - x) < (h_hi[0] - x)) is chernoff_wins, (share, hi[0] - x, h_hi[0] - x)
        assert ((x - lo[0]) < (x - h_lo[0])) is chernoff_wins, (share, x - lo[0], x - h_lo[0])
    # The crossover is near a quarter: 2 beta x = n beta / 2.
    assert _delta(n, eps) == pytest.approx(math.sqrt(2 * math.log(1 / eps) * n / 4), rel=1e-12)


def test_each_link_has_its_own_default() -> None:
    """The fibre's union of two, the satellite's Chernoff (the #78 fix): each the best on its own link,
    which the next test measures."""
    q = get_config().qkd
    assert q.finite_key_deviation == "tighter" and DEVIATIONS[0] == "tighter"
    assert q.free_space_finite_key_deviation == "chernoff"
    assert FiniteKeyParams().deviation == VacuumDecoyParams().deviation == "tighter"
    with pytest.raises(ValueError):
        FiniteKeyParams(deviation="gaussian")


def test_the_union_wins_on_the_fibre_and_chernoff_on_a_pass() -> None:
    """Halving eps costs each bound a little (beta grows by ln 2). On the fibre's threshold the union is
    far better than both, since there the signal is best bounded by Hoeffding and the vacuum's errors by
    Chernoff. On Liao's pass Chernoff already wins nearly every count, so the halving is pure cost: the
    union comes out a few bits BELOW Chernoff alone, which is why the satellite's default is Chernoff
    (the #78 fix; Round 82 had called the union the tightest on every block)."""
    ch = ChannelParams()
    thresholds = {d: minimum_pulses_for_positive_key(ch, FiniteKeyParams(deviation=d)) for d in DEVIATIONS}
    assert thresholds["tighter"] < 0.8 * min(thresholds["chernoff"], thresholds["hoeffding"]), thresholds
    # The shipped fibre block, n_Z = 1e7.
    keys = {}
    for d in DEVIATIONS:
        p = FiniteKeyParams(deviation=d)
        pulses = fibre_block_pulses(ch, p, kind="pa_bits", run_pulses=0, seconds=0, pa_bits=1e7)
        keys[d] = finite_key_length(int(pulses), ch, p).key_bits
    assert keys["tighter"] >= max(keys["chernoff"], keys["hoeffding"]), keys
    # Liao's pass (the test oracle's own segments).
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_the_vacuum_decoy_bound_is_lim_2014 import _liao_pass

    seg = _liao_pass(1000)
    liao = {d: finite_key_over(seg, VacuumDecoyParams(deviation=d)).key_bits for d in DEVIATIONS}
    assert liao["chernoff"] > liao["tighter"] >= 0.995 * liao["chernoff"], liao
    assert liao["chernoff"] > 1.3 * liao["hoeffding"], liao


def _s_z1_as_printed(n_pulses: float, ch: ChannelParams, p: FiniteKeyParams, *, tau0_on_delta: bool) -> float:
    """Rusca et al.'s single-photon bound in Z from the counts, by (3), (A4), (A16) and (A17) as the
    paper prints them, under Hoeffding - written out here, not taken from the module. With
    `tau0_on_delta` the vacuum bound takes the form this file had until Round 82."""
    z, _ = expected_counts(n_pulses, ch, p)
    eps = p.eps_sec / 20.0
    (mu1, p1), (mu2, p2) = p.intensities
    t0 = p1 * math.exp(-mu1) + p2 * math.exp(-mu2)
    t1 = p1 * mu1 * math.exp(-mu1) + p2 * mu2 * math.exp(-mu2)
    d_n = math.sqrt(sum(z.n) / 2 * math.log(1 / eps))
    d_m = math.sqrt(sum(z.m) / 2 * math.log(1 / eps))
    n_plus_1 = math.exp(mu1) / p1 * (z.n[0] + d_n)
    n_minus_2 = math.exp(mu2) / p2 * (z.n[1] - d_n)
    errors = t0 * math.exp(mu2) / p2 * (z.m[1] + d_m)
    s0_upper = 2 * t0 * (errors / t0 + d_n) if tau0_on_delta else 2 * (errors + d_n)       # (A16)
    return t1 * mu1 / (mu2 * (mu1 - mu2)) * (n_minus_2 - mu2 ** 2 / mu1 ** 2 * n_plus_1    # (A17)
                                            - (mu1 ** 2 - mu2 ** 2) / mu1 ** 2 * s0_upper / t0)


@pytest.mark.parametrize("n_pulses", [6e6, 2e7, 1e9])
def test_a16_is_as_the_paper_prints_it(n_pulses) -> None:
    ch, p = ChannelParams(), FiniteKeyParams(deviation="hoeffding")
    r = one_decoy_key_length(*expected_counts(n_pulses, ch, p), p)
    printed = _s_z1_as_printed(n_pulses, ch, p, tau0_on_delta=False)
    assert r.s_z1 == pytest.approx(printed, rel=1e-12), (r.s_z1, printed)
    # The check can fail: the old placement gives a larger single-photon bound, by more than any
    # rounding.
    old = _s_z1_as_printed(n_pulses, ch, p, tau0_on_delta=True)
    assert old > printed * (1 + 1e-6), (old, printed)
