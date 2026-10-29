"""The vacuum+weak finite-key bound (`finite_key.vacuum_decoy_*`) against published numbers.

WHY THIS EXISTS (Round 79, the user's choice). The satellite card's key was asymptotic only, and the
1-decoy finite-key analysis this repository had is not the protocol Micius ran: Micius sent a vacuum
state as well (Liao et al., Nature 549, 43 (2017): "mu_s=0.8, mu_1=0.1, mu_2=0", "sent with
probabilities of 50%, 25%, and 25%"). By the user's choice the bound for that source is Lim, Curty,
Walenta, Xu and Zbinden, Phys. Rev. A 89, 022307 (2014), validated against published figures and
then checked on Liao's 19 December 2016 pass without fitting.

WHAT IS ASSERTED

- Against Rusca et al.'s Table I (arXiv:1801.03443), whose 2-decoy rows are Lim et al.'s analysis on
  Rusca's detector model: optimised over the variables Rusca et al. say they optimise, with mu3 =
  1e-6 as their Fig. 4 states, the bound comes out within 20 percent below the 26 dB and 46 dB rows
  at both block sizes the table gives, and never above. (Round 79 wrote "within a factor of 5":
  until Round 80 this file and the 1-decoy one applied the detector on top of the paper's global
  attenuation, which already includes it, 3 dB too much; see that file.)
- What the shared channel model cannot explain away: the RATIO of the two protocols, which cancels
  it. Rusca et al.'s headline is that the 1-decoy protocol wins at these losses; here it wins too,
  by a ratio within 15% of the paper's at all four cells. And the optimiser keeps more than 11% of
  the pulses on the vacuum at n_Z = 1e7, which Rusca et al. report of their own optimum.
- On Liao's pass, from the model's own counts (within 1% of Liao's detections and sifted bits;
  `test_satellite_network_is_checked.py`): a key, below the paper's 300,939 bits and within a factor
  of about two of it, with nothing fitted; and above what the 1-decoy bound makes of the same pulses
  with the vacuum's quarter unused, which is what the vacuum is for. (Before this round's two
  corrections to the 1-decoy bound it found no key there at all.)
- The algebra: a block split into segments on one channel is the same block; the phase-error
  constant is 21 squared, as both papers print it; the source defaults are the config's Micius
  source; the intensity conditions Lim et al. state are enforced.
"""

from __future__ import annotations

import math

import pytest
from scipy.optimize import differential_evolution

from qsh.config import get_config
from qsh.qkd.channel import EARTH_RADIUS_KM, ChannelParams, FreeSpaceParams
from qsh.qkd.finite_key import (
    FiniteKeyParams,
    VacuumDecoyParams,
    _gamma,
    block_counts,
    blocks_for_positive_key,
    expected_counts,
    finite_key_length,
    finite_key_over,
    vacuum_decoy_key_length,
)

#: Rusca et al. Table I, SKR in Hz on a 1 GHz source: {dB: (km, {n_Z: (1-decoy, 2-decoy)})}.
TABLE_I = {
    26: (100, {1e7: (243_000.0, 236_000.0), 1e9: (357_000.0, 355_000.0)}),
    46: (200, {1e7: (2627.0, 2503.0), 1e9: (3970.0, 3881.0)}),
}
#: The detector inside the global attenuation, as the paper defines it (Round 80).
PAPER_DEVICE = dict(detector_efficiency=1.0, dark_count_prob=1e-8, misalignment=0.01)
LIAO_FINAL_KEY = 300_939


def _channel(att_db: float) -> ChannelParams:
    km = TABLE_I[att_db][0]
    return ChannelParams(distance_km=km, fiber_loss_db_per_km=att_db / km, **PAPER_DEVICE)


def _two_decoy_hz(v, ch: ChannelParams, n_z: float) -> float:
    mu1, ratio, p1, p2_share, p_z = v
    try:
        p = VacuumDecoyParams(mu_signal=mu1, mu_decoy=mu1 * ratio, mu_vacuum=1e-6, p_signal=p1,
                              p_decoy=(1 - p1) * p2_share, p_z=p_z)
    except ValueError:
        return -1e30
    pulses = n_z / expected_counts(1.0, ch, p)[0].detections     # Rusca et al. Eq. (B7)
    r = vacuum_decoy_key_length(*expected_counts(pulses, ch, p), p)
    return r.key_bits / pulses * ch.source_rate_hz if math.isfinite(r.key_bits) else -1e30


def _one_decoy_hz(v, ch: ChannelParams, n_z: float) -> float:
    mu1, ratio, p1, p_z = v
    p = FiniteKeyParams(mu_signal=mu1, mu_decoy=mu1 * ratio, p_signal=p1, p_z=p_z)
    pulses = n_z / (finite_key_length(10 ** 9, ch, p).n_z / 1e9)
    r = finite_key_length(pulses, ch, p)
    return r.key_bits / pulses * ch.source_rate_hz if math.isfinite(r.key_bits) else -1e30


def _best(f, bounds, ch, n_z):
    """Rusca et al.: "we optimize the SKR over the different decoy state variables"."""
    res = differential_evolution(lambda v: -f(v, ch, n_z), bounds=bounds, seed=0, maxiter=200,
                                 popsize=25, tol=1e-9, polish=True)
    return -res.fun, res.x


@pytest.fixture(scope="module")
def optimised() -> dict:
    out = {}
    for att in TABLE_I:
        ch = _channel(att)
        for n_z in (1e7, 1e9):
            two, x = _best(_two_decoy_hz, [(0.05, 1.0), (0.02, 0.9), (0.05, 0.95), (0.05, 0.95),
                                           (0.05, 0.98)], ch, n_z)
            one, _ = _best(_one_decoy_hz, [(0.05, 1.0), (0.02, 0.9), (0.05, 0.95), (0.05, 0.98)],
                           ch, n_z)
            out[att, n_z] = dict(two=two, one=one, p_vacuum=(1 - x[2]) * (1 - x[3]))
    return out


@pytest.mark.parametrize("att", sorted(TABLE_I))
@pytest.mark.parametrize("n_z", (1e7, 1e9))
def test_the_two_decoy_rows_within_twenty_percent_and_below(optimised, att, n_z) -> None:
    published = TABLE_I[att][1][n_z][1]
    ratio = optimised[att, n_z]["two"] / published
    assert 0.8 <= ratio <= 1.0, (
        f"{att} dB, n_Z = {n_z:.0e}: {optimised[att, n_z]['two']:,.1f} Hz against the paper's "
        f"{published:,.1f} Hz, a ratio of {ratio:.3f}. Above 1 claims more key than the published "
        "analysis, the serious failure; below 0.8 has drifted from the bound the paper evaluates.")


@pytest.mark.parametrize("att", sorted(TABLE_I))
@pytest.mark.parametrize("n_z", (1e7, 1e9))
def test_the_protocol_ratio_is_the_papers(optimised, att, n_z) -> None:
    one_paper, two_paper = TABLE_I[att][1][n_z]
    ours = optimised[att, n_z]["two"] / optimised[att, n_z]["one"]
    paper = two_paper / one_paper
    assert ours < 1.0, f"the 1-decoy protocol should win at {att} dB, as it does in the paper"
    assert ours == pytest.approx(paper, rel=0.15), f"{ours:.3f} against the paper's {paper:.3f}"


def test_the_optimum_keeps_the_vacuum_share_rusca_reports(optimised) -> None:
    """Rusca et al., Appendix C, at n_Z = 1e7: "even at its minimum value it remains always greater
    than 11%" - the probability of sending the vacuum, over every attenuation."""
    for att in TABLE_I:
        assert optimised[att, 1e7]["p_vacuum"] > 0.11, (att, optimised[att, 1e7]["p_vacuum"])


def _liao_pass(n: int = 4000) -> list[tuple[float, ChannelParams]]:
    """Liao's 19 December 2016 pass as (pulses, channel) segments, the geometry of
    `test_the_satellite_channel_is_sourced._pass`: 645 km at the top, data while under 1200 km."""
    q = get_config().qkd
    a, b = EARTH_RADIUS_KM + q.free_space_station_altitude_km, EARTH_RADIUS_KM + q.free_space_orbit_altitude_km
    w = math.sqrt(398600.4418 / b ** 3)
    psi = lambda r: math.acos((a * a + b * b - r * r) / (2 * a * b))  # noqa: E731
    psi0 = psi(645.0)
    half = math.acos(math.cos(psi(1200.0)) / math.cos(psi0)) / w
    h, out = 2 * half / n, []
    for i in range(n + 1):
        t = -half + i * h
        r = math.sqrt(a * a + b * b - 2 * a * b * math.cos(psi0) * math.cos(w * t))
        out.append(((h / 2 if i in (0, n) else h) * q.free_space_source_rate_hz,
                    ChannelParams(distance_km=r, free_space=FreeSpaceParams())))
    return out


def test_liaos_pass_carries_a_key_below_the_papers() -> None:
    segments = _liao_pass()
    r = finite_key_over(segments)
    assert 0.4 < r.key_bits / LIAO_FINAL_KEY < 1.0, (
        f"{r.key_bits:,.0f} bits against Liao's {LIAO_FINAL_KEY:,}. Liao et al. do not name their "
        "analysis; they state the failure probability (1e-9), an intensity fluctuation (<5%) and a "
        "Hamming code, none of which is fitted here.")
    assert r.qber == pytest.approx(0.0114, abs=0.002)         # Liao: "an average of 1.1%"
    # The 1-decoy bound on the same pulses: the signal and decoy shares only, the vacuum's unused.
    q = get_config().qkd
    used = q.free_space_p_signal + q.free_space_p_decoy
    one = finite_key_over([(n * used, ch) for n, ch in segments],
                          FiniteKeyParams(mu_signal=q.free_space_mu_signal, mu_decoy=q.free_space_mu_decoy,
                                          p_signal=q.free_space_p_signal / used, p_z=0.5))
    assert 0.0 < one.key_bits < r.key_bits, (one.key_bits, r.key_bits)
    assert r.key_bits / one.key_bits == pytest.approx(1.076, abs=0.02)


def test_a_block_split_on_one_channel_is_the_same_block() -> None:
    ch = ChannelParams(distance_km=800.0, free_space=FreeSpaceParams())
    whole = finite_key_over([(3e10, ch)])
    parts = finite_key_over([(1e10, ch)] * 3)
    assert parts.key_bits == pytest.approx(whole.key_bits, rel=1e-9)
    assert whole.key_bits > 0.0


def test_the_key_grows_with_the_block_and_falls_with_range() -> None:
    def key(km: float, seconds: float) -> float:
        ch = ChannelParams(distance_km=km, free_space=FreeSpaceParams())
        return finite_key_over([(seconds * ch.source_rate_hz, ch)]).key_bits
    assert key(800, 100) < key(800, 300) < key(800, 1000)
    assert key(600, 300) > key(800, 300) > key(1000, 300) > key(1200, 300)
    assert key(800, 1) == float("-inf")                 # no single-photon bound in one second


def test_pooling_blocks_finds_the_fewest_that_carry_a_key() -> None:
    """`blocks_for_positive_key` on a block that carries none: the count it returns carries a key
    and one fewer does not."""
    ch = ChannelParams(distance_km=1200.0, free_space=FreeSpaceParams())
    p = VacuumDecoyParams()
    z, x = block_counts([(10 * ch.source_rate_hz, ch)], p)          # ten seconds: no key
    assert vacuum_decoy_key_length(z, x, p).key_bits <= 0.0
    m = blocks_for_positive_key(z, x, p)
    assert m is not None and m > 1
    assert vacuum_decoy_key_length(z.scaled(m), x.scaled(m), p).key_bits > 0.0
    assert vacuum_decoy_key_length(z.scaled(m - 1), x.scaled(m - 1), p).key_bits <= 0.0
    assert blocks_for_positive_key(z, x, p, hi=2) is None


def test_the_phase_error_constant_is_21_squared() -> None:
    """Lim et al. Eq. (5) and Rusca et al. (A21) both print 21^2 / a^2; this file had 2^21."""
    a, b, c, d = 1e-9, 0.02, 4e5, 6e5
    expected = math.sqrt((c + d) * (1 - b) * b / (c * d * math.log(2))
                         * math.log2((c + d) / (c * d * (1 - b) * b) * 441 / a ** 2))
    assert _gamma(a, b, c, d) == pytest.approx(expected, rel=1e-15)


def test_the_defaults_are_the_micius_source_and_the_conditions_hold() -> None:
    q = get_config().qkd
    p = VacuumDecoyParams()
    assert (p.mu_signal, p.mu_decoy, p.mu_vacuum) == (q.free_space_mu_signal, q.free_space_mu_decoy, 0.0)
    assert (p.p_signal, p.p_decoy) == (q.free_space_p_signal, q.free_space_p_decoy)
    assert p.p_vacuum == pytest.approx(0.25) and p.eps_sec == 1e-9
    with pytest.raises(ValueError, match="mu1 > mu2"):
        VacuumDecoyParams(mu_signal=0.3, mu_decoy=0.2, mu_vacuum=0.15)
    with pytest.raises(ValueError, match="share"):
        VacuumDecoyParams(p_signal=0.6, p_decoy=0.4)
