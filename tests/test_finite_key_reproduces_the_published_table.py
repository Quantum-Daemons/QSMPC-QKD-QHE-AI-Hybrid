"""`finite_key.py` against Rusca et al.'s own published numbers, and against its own limits.

WHY THIS EXISTS

Round 47 replaced a cited threshold with a computed one. That is only an improvement if the
computation is checked against the paper it implements, so this is the sibling of
`tests/test_decoy_reference.py`, which checks the asymptotic decoy bound against Ma et al.'s
published curves.

WHAT IS ASSERTED, AND AT WHAT ACCURACY

The paper's Table I gives the 1-decoy secret key rate at fixed privacy-amplification block sizes
n_Z for four attenuations, on a 1 GHz source, at parameters the authors optimised and did not print;
so each row is optimised here the way the paper says it optimises them. Since Round 80 the match is
close, and the reason it was not is a finding about this file, not about the bound:

  THE CHANNEL HERE APPLIED THE DETECTOR TWICE. Rusca et al.'s attenuation is "global": "the detector
  and the internal losses of Bob's apparatus are included in the global attenuation". Until Round 80
  this file set the fibre's loss to the column's decibels AND a 50 percent detector on top, 3 dB more
  than the paper's channel. That was the "factor of about 2" every round since Round 47 recorded as
  this implementation's conservatism, and Round 79's 2-decoy test inherited it.

With the attenuation applied once:

  - UNDER THE PAPER'S OWN DEAD-TIME CONVENTION - their Eq. (B3) sums the Z-basis detections, where
    `channel.dead_time_factor` charges each detector its own clicks - the 26 dB row is reproduced to
    a few parts in a thousand. That also settles Round 79's reading of (A16): with the signal's
    errors bounding the vacuum the same row reads 13 percent low.
  - UNDER THIS PROJECT'S CONVENTION the 26 dB and 46 dB rows sit within 5 percent at both block sizes
    (the 26 dB one 3 percent above: at the paper's optimised basis bias, near 0.9, the Z-basis sum
    charges more dead time than per-detector clicks do), and the 56 dB row within 25 percent below.
  - The TIME to fill a block, Rusca et al.'s Eq. (B7), within 10 percent of the table's 14 s and
    20 min.
  - The 64 dB row is not asserted: there the dark counts set the key, and this model's background
    (two detectors at 1e-8 each) and the paper's differ by an assumption the paper does not state.

Plus MONOTONICITY in block size and in loss, and the SIGN at this project's own operating point:
Round 46's two-million-pulse block carries no composable key on the shipped fibre.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scipy.optimize import differential_evolution  # noqa: E402

from qsh.qkd.channel import ChannelParams  # noqa: E402
from qsh.qkd.finite_key import (  # noqa: E402
    FiniteKeyParams,
    finite_key_length,
    minimum_pulses_for_positive_key,
)

#: Rusca et al. Table I, 1-decoy: {dB: (km, {n_Z: (SKR in Hz, time to fill the block in s)})}.
PAPER_TABLE_I = {
    26: (100, {1e7: (243_000.0, 14.0), 1e9: (357_000.0, 17 * 60.0)}),
    46: (200, {1e7: (2627.0, 20 * 60.0), 1e9: (3970.0, 23 * 3600.0)}),
    56: (250, {1e7: (227.0, 3.4 * 3600.0), 1e9: (356.0, 10 * 86400.0)}),
    64: (290, {1e7: (11.3, 26 * 3600.0), 1e9: (25.5, 67 * 86400.0)}),
}

#: The paper's stated device parameters: p_DC = 1e-8, a 1% error floor, a 100 ns dead time - and the
#: detector efficiency INSIDE the global attenuation, as the paper defines it, so 1.0 here.
PAPER_DEVICE = dict(detector_efficiency=1.0, dark_count_prob=1e-8, misalignment=0.01)


def _paper_channel(att_db: float, dist_km: float, dead_time_s: float | None = None) -> ChannelParams:
    """The paper's channel: the column's global attenuation, once."""
    return ChannelParams(distance_km=dist_km, fiber_loss_db_per_km=att_db / dist_km,
                         dead_time_s=dead_time_s, **PAPER_DEVICE)


def _best(att_db: float, n_z: float, *, papers_dead_time: bool = False) -> tuple[float, float]:
    """The optimised SKR in Hz at a block of `n_z` and the seconds the block takes (Eq. B7).

    `papers_dead_time` emulates Eq. (B3) as printed - D the Z-basis detection probability, p_Z^2
    of every click - against this project's per-detector D, half of every click: the dead time
    scaled by 2 p_Z^2 for each candidate.
    """
    dist = PAPER_TABLE_I[att_db][0]

    def run(v):
        mu1, ratio, p1, p_z = v
        ch = _paper_channel(att_db, dist, 100e-9 * 2 * p_z * p_z if papers_dead_time else None)
        p = FiniteKeyParams(mu_signal=mu1, mu_decoy=mu1 * ratio, p_signal=p1, p_z=p_z)
        pulses = n_z / (finite_key_length(10 ** 9, ch, p).n_z / 1e9)
        k = finite_key_length(pulses, ch, p).key_bits
        return (k / pulses * ch.source_rate_hz if math.isfinite(k) else -1e30), pulses / ch.source_rate_hz

    res = differential_evolution(lambda v: -run(v)[0], bounds=[(0.05, 1.0), (0.02, 0.9), (0.05, 0.95),
                                                               (0.05, 0.98)],
                                 seed=0, maxiter=200, popsize=25, tol=1e-9, polish=True)
    return run(res.x)


def test_the_26_db_row_under_the_papers_dead_time() -> None:
    skr, _ = _best(26, 1e7, papers_dead_time=True)
    assert skr / 243_000.0 == pytest.approx(1.0, abs=0.01), f"{skr:,.0f} Hz against 243,000"


@pytest.mark.parametrize("att,band", [(26, (0.95, 1.05)), (46, (0.95, 1.05)), (56, (0.75, 1.0))])
@pytest.mark.parametrize("n_z", (1e7, 1e9))
def test_table_i_under_this_projects_dead_time(att, band, n_z) -> None:
    published = PAPER_TABLE_I[att][1][n_z][0]
    skr, _ = _best(att, n_z)
    lo, hi = band
    assert lo <= skr / published <= hi, (
        f"{att} dB, n_Z = {n_z:.0e}: {skr:,.1f} Hz against the paper's {published:,.1f} Hz, a ratio "
        f"of {skr / published:.3f}, outside {band}")


@pytest.mark.parametrize("att", (26, 46))
def test_the_block_takes_the_time_the_paper_says(att) -> None:
    """Eq. (B7): the pulses a block of n_Z key-basis detections needs, at the 1 GHz clock."""
    _, seconds = _best(att, 1e7)
    published = PAPER_TABLE_I[att][1][1e7][1]
    assert seconds == pytest.approx(published, rel=0.10), f"{seconds:,.1f} s against {published:,.0f} s"


def test_the_key_grows_with_the_block_and_shrinks_with_loss() -> None:
    """Monotonicity, which needs no agreement with anyone's table to be meaningful."""
    ch = _paper_channel(26, 100)
    lengths = [finite_key_length(n, ch).key_bits for n in (10 ** 8, 10 ** 9, 10 ** 10)]
    assert lengths[0] < lengths[1] < lengths[2], f"key does not grow with block size: {lengths}"

    at_fixed_block = [finite_key_length(10 ** 10, _paper_channel(a, d)).key_bits
                      for a, (d, _) in sorted(PAPER_TABLE_I.items())]
    assert at_fixed_block == sorted(at_fixed_block, reverse=True), (
        f"key does not fall monotonically with loss: {at_fixed_block}"
    )


def test_the_shipped_block_carries_no_composable_key() -> None:
    """THE CLAIM THE REPOSITORY MAKES, at this project's own operating point.

    Round 46 shipped 2,000,000 pulses. On the channel of that time the single-photon lower bound
    `s^l_{Z,1}` was NEGATIVE there - undefined, not merely small. On Round 76's Rusca set it is
    positive (about 440 counts since Round 79's two corrections, 225 before) and the key is still
    negative (about -607 bits): the block carries no composable key either way, now because the
    finite-size terms outweigh it. The threshold moved from 1.41e7 pulses to 5.88e6 with the Rusca
    set and to 4.78e6 with Round 79's corrections, and the shipped 2e7 clears it 4.2x.
    """
    from qsh.config import get_config

    q = get_config().qkd
    ch = ChannelParams(distance_km=q.distance_km, fiber_loss_db_per_km=q.fiber_loss_db_per_km,
                       detector_efficiency=q.detector_efficiency,
                       dark_count_prob=q.dark_count_prob, misalignment=q.misalignment)
    at_two_million = finite_key_length(2_000_000, ch)
    assert at_two_million.key_bits <= 0, (
        f"2e6 pulses now yield {at_two_million.key_bits:.1f} bits of composable key; the claim that "
        "Round 46's block carried none needs re-deriving before it is quoted again")

    threshold = minimum_pulses_for_positive_key(ch)
    assert threshold is not None, "no block size up to 1e11 pulses yields a key on the shipped fibre"
    assert 2e6 <= threshold <= 2e8, (
        f"the computed threshold moved to {threshold:,} pulses. The repository quotes it as an "
        "order-of-magnitude figure (about 5e6 pulses on the Rusca set); a value outside this range "
        "means the documented number needs re-deriving, not that this bound should be widened."
    )
    assert q.pulses >= 2 * threshold, f"the shipped {q.pulses:,} pulses no longer clear {threshold:,} twice"


def test_the_refuted_band_is_refuted() -> None:
    """The specific number three rounds of documents carried, asserted false.

    5e3 to 1.1e4 SIFTED bits was stated as the positive-key threshold. If a block yielding that
    many sifted bits produced a key, the old sentence would have been right and this whole
    correction wrong - so the refutation is asserted rather than assumed.
    """
    from qsh.config import get_config

    q = get_config().qkd
    ch = ChannelParams(distance_km=q.distance_km, fiber_loss_db_per_km=q.fiber_loss_db_per_km,
                       detector_efficiency=q.detector_efficiency,
                       dark_count_prob=q.dark_count_prob, misalignment=q.misalignment)
    import math

    for target_sifted in (5e3, 1.1e4):
        lo, hi = 1e4, 1e10
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if finite_key_length(int(mid), ch).sifted_bits > target_sifted:
                hi = mid
            else:
                lo = mid
        r = finite_key_length(int(hi), ch)
        assert r.key_bits <= 0, (
            f"a block of {hi:,.0f} pulses giving {r.sifted_bits:,.0f} sifted bits produced "
            f"{r.key_bits:,.1f} bits of composable key. That would vindicate the band this round "
            "removed."
        )
