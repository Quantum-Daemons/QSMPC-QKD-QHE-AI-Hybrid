"""`finite_key.py` against Rusca et al.'s own published numbers, and against its own limits.

WHY THIS EXISTS

Round 47 replaced a cited threshold with a computed one. That is only an improvement if the
computation is checked against the paper it implements, so this is the sibling of
`tests/test_decoy_reference.py`, which checks the asymptotic decoy bound against Ma et al.'s
published curves.

WHAT IS ASSERTED, AND AT WHAT ACCURACY

The paper's Table I gives the 1-decoy secret key rate at fixed privacy-amplification block sizes
for four attenuations, assuming a 1 GHz source. Reproducing it exactly is not possible from the
paper alone: Table I is quoted at parameters the authors OPTIMISED and did not print, and their
split of the security budget across the individual epsilons in Eq. (A24) is not stated. Their
detector model's 100 ns dead time is modelled here since Round 76 (`qkd.channel.dead_time_factor`),
counting every click rather than their Z-basis sum; until then this file listed it as a third reason. Measured against that
table, with the intensities and basis bias optimised here the way the paper says it optimises
them, this implementation comes out **conservative by a factor of about 2 at 26 dB and 46 dB**,
and by more at higher loss.

So this file asserts the things that are robust to a factor of a few, and deliberately does not
assert agreement to a digit:

  - the ORDER OF MAGNITUDE against Table I, within a factor of 5 and always on the low side;
  - MONOTONICITY in block size and in loss;
  - the SIGN at this project's own operating point, which is the claim the repository makes.

The last one is why the accuracy caveat does not weaken the finding: Round 46's two-million-pulse
block carries no composable key on the shipped fibre, by a margin no factor of a few closes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.qkd.channel import ChannelParams                                    # noqa: E402
from qsh.qkd.finite_key import (                                             # noqa: E402
    FiniteKeyParams, finite_key_length, minimum_pulses_for_positive_key,
)

#: Rusca et al. Table I, 1-decoy row, SKR in Hz at n_Z = 1e7 on a 1 GHz source. The distances are
#: the paper's own column headings for these attenuations.
PAPER_TABLE_I = {26: (100, 243_000.0), 46: (200, 2627.0), 56: (250, 227.0), 64: (290, 11.3)}

#: The paper's stated device parameters: p_DC = 1e-8, eta_det ~ 50%, p_Err = 1%.
PAPER_DEVICE = dict(detector_efficiency=0.5, dark_count_prob=1e-8, misalignment=0.01)


def _paper_channel(att_db: float, dist_km: float) -> ChannelParams:
    """A channel with the paper's device parameters and the stated total attenuation."""
    return ChannelParams(distance_km=dist_km, fiber_loss_db_per_km=att_db / dist_km,
                         **PAPER_DEVICE)


def _skr_hz_at_block(att_db: float, dist_km: float, target_n_z: float = 1e7,
                     params: FiniteKeyParams | None = None) -> float:
    """Key rate in Hz at a 1 GHz source, at the block size giving `target_n_z` Z-basis detections."""
    import math

    ch = _paper_channel(att_db, dist_km)
    p = params or FiniteKeyParams()
    lo, hi = 1e6, 1e17
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if finite_key_length(int(mid), ch, p).n_z > target_n_z:
            hi = mid
        else:
            lo = mid
    r = finite_key_length(int(hi), ch, p)
    return r.key_bits / (hi / 1e9)


def test_the_order_of_magnitude_matches_the_published_table_and_errs_low() -> None:
    """Against Table I at the two attenuations the paper's own model is least sensitive at.

    56 dB and 64 dB are deliberately excluded: at those losses the paper's dead-time and
    afterpulsing terms dominate its detector model, and this implementation - which has neither -
    diverges by more than an order of magnitude. Asserting there would be asserting agreement
    about a model this project does not implement.
    """
    # p_Z is optimised because Table I is: at p_Z = 0.5 only a quarter of pulses land in the Z
    # basis, which alone accounts for a factor of ~4 and is not a disagreement about physics.
    tuned = FiniteKeyParams(mu_signal=0.427, mu_decoy=0.210, p_signal=0.648, p_z=0.885)
    for att in (26, 46):
        dist, published = PAPER_TABLE_I[att]
        got = _skr_hz_at_block(att, dist, params=tuned)
        assert got > 0, f"no key at {att} dB, where the paper reports {published} Hz"
        ratio = got / published
        assert 0.2 <= ratio <= 1.0, (
            f"at {att} dB this implementation gives {got:,.1f} Hz against the paper's "
            f"{published:,.1f} Hz - a ratio of {ratio:.2f}. It is expected to be CONSERVATIVE, so "
            "a ratio above 1 means it is claiming more key than the published analysis and is the "
            "serious direction of failure; below 0.2 means it has drifted too far to be quoted as "
            "the same bound."
        )


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
    positive (about 225 counts) and the key is still negative (about -607 bits): the block carries no
    composable key either way, now because the finite-size terms outweigh it. The threshold moved
    from 1.41e7 pulses to 5.88e6, and the shipped 2e7 clears it 3.4x.
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
        "order-of-magnitude figure (about 6e6 pulses on the Rusca set); a value outside this range "
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
