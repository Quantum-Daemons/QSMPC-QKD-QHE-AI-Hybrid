"""Composable finite-key length for 1-decoy BB84 (Rusca et al. 2018).

WHY THIS EXISTS

Until Round 47 this repository stated, in `config/defaults.py` and five other places, that

    "the positive-key threshold for these device parameters is ~5e3 to 1.1e4 sifted bits
     (Rusca, Boaron, Grunenfelder, Martin & Zbinden, Appl. Phys. Lett. 112, 171104 (2018),
     1-decoy finite-key bound)"

Two things were wrong with that sentence, and the second is worse than the first.

  1. THE NUMBER. Evaluated at this project's own 50 km channel, the paper's equations give a
     threshold of **~1e5 sifted bits, about 4e7 pulses** - not 5e3 to 1.1e4. The shipped block of
     2e6 pulses yields 12,442 sifted bits and NO KEY AT ALL: the single-photon lower bound
     `s^l_{Z,1}` comes out NEGATIVE there, and no setting of the intensities, their probabilities
     or the basis bias makes it positive (searched; see `optimise_parameters`).

     STATED TO ONE FIGURE ON PURPOSE. This code returns 1.436e5 sifted bits and 3.84e7 pulses at
     the shipped channel, and those digits are not earned - see HOW ACCURATE THIS IS below. Every
     site that quotes the threshold quotes the order of magnitude.

  2. THE ATTRIBUTION. The paper does not state a positivity threshold anywhere. It compares
     1-decoy against 2-decoy for privacy-amplification block sizes n_Z from 1e5 to 1e11, and its
     own words are that the 1-decoy approach wins "unless a really big (> 10^11) or really small
     (< 10^5) PA block size is applied". A number was attributed to a paper that does not contain
     it, and the citation made it look checked.

The giveaway was sitting in the same comment. Directly after the band, the config quoted - as a
"corroborating primary datum" - Lucamarini et al., Opt. Express 21, 24550 (2013): "1.4x10^5 counts
... is the smallest sample size providing a positive rate" at 50 km. That is 13 to 28 times LARGER
than the band it was quoted to support, and it agrees with what these equations compute. The
corroboration refuted the claim it was attached to, in the next sentence, for three rounds.

So the repository now OWNS the computation instead of citing a band for it. This module is the
sibling of `decoy_reference.py`, whose docstring says it "exists to be CHECKED": the asymptotic
decoy bound is checked against Ma et al.'s published curves there, and the finite-key length is
checked against Rusca et al.'s published Table I here, in
`tests/test_finite_key_reproduces_the_published_table.py`.

HOW ACCURATE THIS IS

Until Round 80 this paragraph said the implementation came out CONSERVATIVE by a factor of 2-3
against Rusca et al.'s Table I. The factor was the test's, not the bound's: it applied a 50 percent
detector on top of the paper's attenuation, which is "global" - "the detector and the internal
losses of Bob's apparatus are included" - so its channel had 3 dB more loss than the paper's. With
the attenuation applied once, and (A16) as the paper prints it since Round 82, Table I's 26 dB row is
reproduced to within 2 percent below under the paper's own dead-time convention, the 26 dB row within
5 percent and the 46 dB row within 7 percent under this project's, and the 56 dB row within 25 percent
below, all under the published Hoeffding statistics
(`tests/test_finite_key_reproduces_the_published_table.py`). Thresholds are still quoted to one or
two figures, since the paper's own optimisation and epsilon split are not printed.

Three claims never depended on that accuracy:

  (a) the shipped 2e6-pulse block carries no composable key, because `s^l_{Z,1}` is NEGATIVE there
      and no factor can rescue a negative count;
  (b) the "~5e3 to 1.1e4" band was far too small, by an order of magnitude at minimum;
  (c) the citation was wrong regardless of any numerics, since the paper contains no such threshold.

WHAT IS AND IS NOT CLAIMED

This computes a composable secret key LENGTH in bits for a given number of pulses. It does not
change what `run_bb84` reports - that remains the asymptotic per-pulse decoy bound, which is a
different quantity and is labelled as one. The two are shown side by side so that a block size
which cannot carry a composable key says so, rather than publishing an asymptotic rate in silence.

EQUATIONS, by number in arXiv:1801.03443v2

  (1)   l <= s^l_{Z,0} + s^l_{Z,1}(1 - h(phi^u_Z)) - lambda_EC - 6 log2(19/eps_sec)
                                                                - log2(2/eps_cor)
  (3)   n^pm_{Z,k} = (e^k / p_k)(n_{Z,k} pm delta(n_Z, eps1))
  (4)   s^l_{Z,1} = tau1 mu1 / (mu2(mu1 - mu2)) [ n^-_{Z,mu2} - (mu2^2/mu1^2) n^+_{Z,mu1}
                                                  - ((mu1^2 - mu2^2)/mu1^2)(s^u_{Z,0}/tau0) ]
  (A16) s^u_{Z,0} = 2 ( tau0 (e^k/p_k)(m_{Z,k} + delta(m_Z, eps)) + delta(n_Z, eps) )
  (A19) s^l_{Z,0} = tau0/(mu1 - mu2) ( mu1 n^-_{Z,mu2} - mu2 n^+_{Z,mu1} )
  (A21) gamma(a,b,c,d) = sqrt( (c+d)(1-b)b/(c d ln2) * log2( (c+d)/(c d (1-b) b) * 21^2/a^2 ) )
  (A22) v^u_{X,1} = tau1/(mu1 - mu2) ( m^+_{X,mu1} - m^-_{X,mu2} )
  (A23) phi^u_Z = v^u_{X,1}/s^l_{X,1} + gamma(eps_sec, v^u_{X,1}/s^l_{X,1}, s^l_{Z,1}, s^l_{X,1})

`a = 6, b = 19` in (1) are the paper's own constants for the 1-decoy protocol.

TWO CORRECTIONS IN ROUND 79, both in the safe direction before and after:

  - THE CONSTANT IN (A21) IS 21 SQUARED, and this file had 2 to the 21st. Both papers print it as a
    superscript 2 on 21 (Rusca et al. (A21); Lim et al. Eq. (5), whose supplement derives it as
    1/eps^2 with eps_sec = 21 eps), and a text extraction of either PDF reads "212", which is how
    2 ** 21 got in. It sat inside a log2 and inflated gamma by about a tenth.
  - (A16) BOUNDS THE VACUUM BY "the errors relative to one intensity" and does not say which. This
    file used the signal's, with no reason recorded; it now uses the decoy's, the tighter bound at
    every row of Table I (see the comment at that line).

Together they move the threshold on the shipped fibre from 5.88e6 pulses to 4.78e6, and this file
from 0.45 / 0.39 / 0.16 of Table I at 26 / 46 / 56 dB to 0.53 / 0.47 / 0.26 - on the test channel of
that time, which Round 80 found applied the detector twice; on the paper's channel the same code
reads 1.03 / 0.96 / 0.80, and 1.00 at 26 dB under the paper's dead time. The 2e6-pulse block still
carries no key. The figures above at 3.84e7 pulses are the pre-Round-76
channel's and are left as the record of Round 47.

THE VACUUM+WEAK PROTOCOL (Round 79) is the second half of this module: Lim, Curty, Walenta, Xu and
Zbinden, Phys. Rev. A 89, 022307 (2014), arXiv:1311.7129, the analysis Rusca et al. compare against
and the protocol Micius ran (signal, weak decoy and vacuum). See `VacuumDecoyParams`.

THE STATISTICS, THE TIGHTER OF TWO BY DEFAULT (Round 82, the user's choice). Both papers bound each count's
expectation by the Hoeffding deviation of the basis TOTAL, delta(n_B, eps) = sqrt(n_B/2 ln(1/eps)),
whatever the count: the vacuum's few hundred detections are moved by the same three thousand as the
signal's eight hundred thousand on Liao's pass. Yin, Zhou, Gu, Xie, Lu and Chen, Sci. Rep. 10, 14312
(2020), arXiv:2002.06530, give a variant of the Chernoff bound for exactly this use - the expected
value of a sum of independent Bernoulli variables from its OBSERVED value - in closed form, read on
the rendered page:

  (9)   x* <= x + beta + sqrt(2 beta x + beta^2)
  (10)  x* >= x - beta/2 - sqrt(2 beta x + beta^2/4),        beta = ln(1/eps)

Each replaces one one-sided Hoeffding use, at the same eps, so the security budgets (Lim's 21 terms,
Rusca's (A24)) are unchanged. Zhang, Zhao, Razavi and Ma, Phys. Rev. A 95, 012333 (2017), whose
Chernoff bounds Khmelev et al. 2023 used, invert the multiplicative bound with the observed value in
place of the expectation, which Yin et al. call not rigorous; on these passes the two differ by under
0.3 percent.

NEITHER IS TIGHTER EVERYWHERE. Chernoff's deviation, about sqrt(2 beta x), beats Hoeffding's
sqrt(n beta / 2) only for a count x under a quarter of its basis total n. The satellite's weak decoy
(6 percent of detections) and vacuum gain; the fibre's 1-decoy signal and decoy, about 70 and 30
percent, lose, and Chernoff alone moved the fibre's threshold from 5.35e6 pulses to 5.71e6. So the
default, `deviation="tighter"`, bounds every count both ways at eps/2 each and keeps the tighter: a
union bound, the failure probability of each use still at most eps, so the budgets are unchanged.
It is the tightest of the three on every block measured. "chernoff" and "hoeffding" alone remain;
"hoeffding" is the published form, which Rusca's Table I was computed with and is checked against.

AND A THIRD TRANSCRIPTION ERROR, in (A16), found reading it again for this (Round 82): tau0 multiplies
the error term only, 2 (tau0 (e^k/p_k)(m_{Z,k} + delta) + delta(n_Z)), as the paper prints it and as
its (A13)-(A15) derive it; this file had tau0 outside the bracket, scaling delta(n_Z) down by about a
fifth. That made the vacuum's upper bound too small and the single-photon bound too large: the unsafe
direction, unlike the two Round 79 found. Corrected, the shipped fibre's threshold under Hoeffding is
5.35e6 pulses, not 4.78e6; under the default statistics it is 3.60e6.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from .bb84 import e_det_for_bound
from .channel import ChannelParams, dead_time_factor

__all__ = [
    "DEVIATIONS",
    "DEVIATION_LABELS",
    "DecoyCounts",
    "FiniteKeyResult",
    "FiniteKeyParams",
    "VacuumDecoyParams",
    "block_counts",
    "blocks_for_positive_key",
    "expected_counts",
    "fibre_block_pulses",
    "finite_key_length",
    "finite_key_over",
    "minimum_pulses_for_positive_key",
    "one_decoy_key_length",
    "optimise_parameters",
    "params_for_source",
    "vacuum_decoy_key_length",
]


def _h2(x: float) -> float:
    """Binary entropy, with the endpoints returning 0 rather than a NaN from log(0)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def _qkd_cfg():
    """The shipped QKD config, imported lazily so this module stays importable standalone.

    Mirrors `bb84._qkd_cfg` deliberately rather than importing it: same shape, same reason, and a
    reader comparing the two dataclasses sees the same mechanism in both.
    """
    from ..config import get_config

    return get_config().qkd


@dataclass(frozen=True)
class FiniteKeyParams:
    """The variables Rusca et al. optimise over, plus the security parameters.

    The paper's own sentence: "we optimize the SKR over the different decoy state variables, i.e.
    mu_i and the associated probability p_mu_i, and the probability to choose the Z basis for
    Alice (pZa) and Bob (pZb)". `optimise_parameters` below searches exactly that set, so the
    optimisation has the paper's procedure behind it rather than a hand-picked grid.
    """

    # READ FROM CONFIG, not spelled here - the same fix `QKDParams` took in Round 47 and this
    # sibling did not follow, which is the one-site-short pattern this project has recorded in eight
    # consecutive rounds. These were `0.5` and `0.1` while the shipped pair moved to the finite-key
    # optimum `0.3416` / `0.1417`, so every bare `FiniteKeyParams()` - which is how the refuted-band
    # test constructs it - evaluated the paper's bound at intensities the demo does not run.
    mu_signal: float = field(default_factory=lambda: _qkd_cfg().mu_signal)
    mu_decoy: float = field(default_factory=lambda: _qkd_cfg().mu_decoy)
    p_signal: float = 0.5          # p_mu1, probability of sending the signal intensity
    p_z: float = 0.5               # basis bias, applied to Alice and Bob alike
    eps_sec: float = 1e-9          # the paper's own value
    eps_cor: float = 1e-15         # the paper's own value
    f_ec: float = 1.16             # error-correction leakage factor
    # The statistics (Round 82): "chernoff" or "hoeffding", the paper's own. See the module docstring.
    deviation: str = field(default_factory=lambda: _qkd_cfg().finite_key_deviation)

    def __post_init__(self) -> None:
        if self.deviation not in DEVIATIONS:
            raise ValueError(f"deviation {self.deviation!r}: one of {DEVIATIONS}")

    @property
    def intensities(self) -> tuple[tuple[float, float], ...]:
        """(mu, p) for the signal and the decoy, the order `DecoyCounts` keeps."""
        return ((self.mu_signal, self.p_signal), (self.mu_decoy, 1.0 - self.p_signal))


@dataclass(frozen=True)
class FiniteKeyResult:
    """A composable key length, and every intermediate needed to see why it is what it is."""

    key_bits: float                # l in Eq. (1). NEGATIVE or zero means no key exists.
    n_z: float                     # privacy-amplification block: Z-basis detections
    n_x: float
    sifted_bits: float             # n_z + n_x, the quantity the rest of this repo calls "sifted"
    s_z0: float                    # vacuum events, lower bound
    s_z1: float                    # single-photon events, lower bound. Negative => no bound holds.
    phi_z: float                   # phase error rate, upper bound
    lambda_ec: float               # bits disclosed in error correction
    qber: float

    @property
    def positive(self) -> bool:
        return self.key_bits > 0.0


def _delta(x: float, eps: float) -> float:
    """Hoeffding correction, Eq. (3). Guarded at zero because a count cannot be negative."""
    return math.sqrt(max(x, 0.0) / 2.0 * math.log(1.0 / eps))


#: The statistics a finite key can use; `QKDConfig.finite_key_deviation` picks one.
DEVIATIONS = ("tighter", "chernoff", "hoeffding")
#: How each is named where a key is reported, so a page says which statistics its number rests on.
DEVIATION_LABELS = {
    "tighter": "the tighter of Chernoff and Hoeffding per count",
    "chernoff": "Chernoff-type bounds, Yin et al. 2020",
    "hoeffding": "Hoeffding, as published",
}


def _chernoff_upper(x: float, eps: float) -> float:
    """The largest expectation consistent with an observed count `x`: Yin et al. 2020 Eq. (9)."""
    b, x = math.log(1.0 / eps), max(x, 0.0)
    return x + b + math.sqrt(2.0 * b * x + b * b)


def _chernoff_lower(x: float, eps: float) -> float:
    """The smallest expectation consistent with an observed count `x`: Yin et al. 2020 Eq. (10),
    floored at zero as the paper requires - an expected count cannot be negative."""
    b, x = math.log(1.0 / eps), max(x, 0.0)
    return max(0.0, x - b / 2.0 - math.sqrt(2.0 * b * x + b * b / 4.0))


def _bounds(counts: tuple[float, ...], eps: float, deviation: str) -> tuple[list[float], list[float]]:
    """Lower and upper bounds on each count's expectation, at failure probability `eps` each.

    "hoeffding": the published form, each count moved by the deviation of the basis TOTAL (Lim et
    al.'s n^pm, Rusca et al.'s (3) and (A4)). "chernoff": each count by its own, Yin et al.'s Eqs.
    (9) and (10). "tighter": both at eps/2, the tighter kept - a union bound, so each still fails
    with probability at most eps.
    """
    if deviation == "hoeffding":
        d = _delta(sum(counts), eps)
        return [c - d for c in counts], [c + d for c in counts]
    if deviation == "chernoff":
        return [_chernoff_lower(c, eps) for c in counts], [_chernoff_upper(c, eps) for c in counts]
    if deviation == "tighter":
        h_lo, h_hi = _bounds(counts, eps / 2.0, "hoeffding")
        c_lo, c_hi = _bounds(counts, eps / 2.0, "chernoff")
        return ([max(a, b) for a, b in zip(h_lo, c_lo, strict=True)],
                [min(a, b) for a, b in zip(h_hi, c_hi, strict=True)])
    raise ValueError(f"deviation {deviation!r}: one of {DEVIATIONS}")


def _vacuum_errors_expectation(v0: float, n_tot: float, eps: float, deviation: str) -> float:
    """An upper bound on the expected vacuum errors, given `v0` bounding the errors themselves:
    Rusca et al.'s (A13) step, delta(s_{Z,0}) <= delta(n_Z) under Hoeffding, Yin et al.'s Eq. (9)
    under Chernoff - increasing in its argument, so the bound on the errors may stand in for them -
    and the tighter of the two at eps/2 each."""
    if deviation == "hoeffding":
        return v0 + _delta(n_tot, eps)
    if deviation == "chernoff":
        return _chernoff_upper(v0, eps)
    return min(v0 + _delta(n_tot, eps / 2.0), _chernoff_upper(v0, eps / 2.0))


def _tau(n: int, p: FiniteKeyParams) -> float:
    """Total probability that Alice sends an n-photon state, summed over the intensities."""
    return sum(prob * math.exp(-mu) * mu ** n / math.factorial(n)
               for mu, prob in ((p.mu_signal, p.p_signal), (p.mu_decoy, 1.0 - p.p_signal)))


def _gamma(a: float, b: float, c: float, d: float) -> float:
    """The sampling correction to the phase error rate: Rusca et al. (A21), Lim et al. Eq. (5).

    One function for both protocols, so the constant cannot differ between them again: 21 squared,
    as both papers print it (see the module docstring for the 2 ** 21 it replaced).
    """
    return math.sqrt((c + d) * (1.0 - b) * b / (c * d * math.log(2))
                     * math.log2((c + d) / (c * d * (1.0 - b) * b) * 21 ** 2 / a ** 2))


def _gain_and_error(eta: float, mu: float, e_det: float, y0: float) -> tuple[float, float]:
    """Observed gain Q_mu and error rate E_mu for a weak coherent pulse of intensity mu.

    The same weak-coherent model `bb84.py::_decoy_secure_fraction` uses, so the finite-key length
    and the asymptotic rate describe one channel rather than two.
    """
    q = y0 + 1.0 - math.exp(-eta * mu)
    if q <= 0.0:
        return 0.0, 0.5
    e = (e_det * (1.0 - math.exp(-eta * mu)) + 0.5 * y0) / q
    return q, min(0.5, e)


@dataclass(frozen=True)
class DecoyCounts:
    """Detections `n` and errors `m` in one basis, per intensity, in the order of the parameters'
    `intensities`: signal and decoy for the 1-decoy protocol, then the vacuum for Lim et al.'s."""

    n: tuple[float, ...]
    m: tuple[float, ...]

    def __add__(self, other: DecoyCounts) -> DecoyCounts:
        return DecoyCounts(tuple(a + b for a, b in zip(self.n, other.n, strict=True)),
                           tuple(a + b for a, b in zip(self.m, other.m, strict=True)))

    @property
    def detections(self) -> float:
        return sum(self.n)

    @property
    def errors(self) -> float:
        return sum(self.m)

    def scaled(self, k: float) -> DecoyCounts:
        """The counts of `k` such blocks: expected counts are linear in the pulses sent."""
        return DecoyCounts(tuple(k * v for v in self.n), tuple(k * v for v in self.m))


def expected_counts(n_pulses: float, channel: ChannelParams,
                    params: FiniteKeyParams | VacuumDecoyParams) -> tuple[DecoyCounts, DecoyCounts]:
    """The expected (key basis Z, test basis X) counts from `n_pulses` channel uses on one channel.

    One counting for both protocols, so the two bounds always see the same channel: each
    intensity's gain and error from `_gain_and_error`, both parties in a basis with probability p_B
    squared, and the dead-time factor from every click. Linear in `n_pulses`, which is what lets a
    pass be summed segment by segment (`finite_key_over`).
    """
    eta = channel.transmittance()
    # THE MISALIGNMENT, not `intrinsic_qber(channel)`. `_gain_and_error` adds the background back
    # as its own `0.5 * y0` term, so passing the observed QBER charges dark counts twice - the
    # identical defect Round 47 found in `bb84.py` and `quantumSim.ts`, and it was written into
    # this file first time through before that fix was made. Eavesdropping IS included:
    # `e_det_for_bound` composes the intercept-resend disturbance with the misalignment, and has
    # since Round 47 - a comment here said the opposite until 2026-10-24. Every caller passes Eve 0,
    # so no figure depended on the discrepancy.
    e_det, y0 = e_det_for_bound(channel), 2.0 * channel.dark_count_prob
    gains = [_gain_and_error(eta, mu, e_det, y0) for mu, _ in params.intensities]
    # THE DEAD TIME (Rusca et al. Eq. (B2)-(B3)): every click, in either basis and at any
    # intensity, leaves a detector recovering, so the correction is computed from all of them and
    # scales each count. Not modelled before Round 76, which the published-table test named as one
    # reason this implementation and Table I differ.
    c_dt = dead_time_factor(channel, sum(p_k * q for (_, p_k), (q, _) in
                                         zip(params.intensities, gains, strict=True)))
    out = []
    for p_b in (params.p_z, 1.0 - params.p_z):
        n = tuple(n_pulses * p_k * p_b * p_b * c_dt * q   # Alice and Bob both choose this basis
                  for (_, p_k), (q, _) in zip(params.intensities, gains, strict=True))
        out.append(DecoyCounts(n, tuple(n_k * e for n_k, (_, e) in zip(n, gains, strict=True))))
    return out[0], out[1]


def one_decoy_key_length(z: DecoyCounts, x: DecoyCounts,
                         params: FiniteKeyParams | None = None) -> FiniteKeyResult:
    """Rusca et al.'s composable key length from the key-basis counts `z` and test-basis counts `x`.

    Non-positive means no key; -inf means no single-photon bound exists at all.
    """
    p = params or FiniteKeyParams()
    # The security budget is split across the statistical terms. eps_sec/20 for each of the
    # Hoeffding corrections is the conservative reading of Eq. (A24), which sums 6*eps1 + 4*eps2
    # among other terms; taking each individual epsilon at eps_sec/20 keeps the sum below eps_sec.
    eps1 = eps2 = p.eps_sec / 20.0
    mu1, mu2 = p.mu_signal, p.mu_decoy
    p1, p2 = p.p_signal, 1.0 - p.p_signal
    t0, t1 = _tau(0, p), _tau(1, p)
    per_basis: dict[str, dict] = {}
    for basis, c in (("Z", z), ("X", x)):
        n_tot, m_tot = c.detections, c.errors
        n_lo, n_hi = _bounds(c.n, eps1, p.deviation)
        m_lo, m_hi = _bounds(c.m, eps2, p.deviation)
        n_plus_1 = math.exp(mu1) / p1 * n_hi[0]
        n_minus_2 = math.exp(mu2) / p2 * n_lo[1]
        m_plus_1 = math.exp(mu1) / p1 * m_hi[0]
        m_minus_2 = math.exp(mu2) / p2 * m_lo[1]
        # (A19), CLAMPED AT ZERO. This clamp is load-bearing: without it a negative vacuum count
        # propagates through the `- ((mu1^2-mu2^2)/mu1^2)(s^u_{Z,0}/tau0)` term of Eq. (4) with the
        # wrong sign and INFLATES the single-photon bound, moving the computed threshold by a
        # factor of several. A count of events cannot be negative.
        s0 = max(0.0, t0 / (mu1 - mu2) * (mu1 * n_minus_2 - mu2 * n_plus_1))
        # (A16) bounds the vacuum by "the errors relative to one intensity" and does not say which.
        # THE DECOY'S, since Round 79; the signal's before, with no reason recorded. The paper says
        # the approach it chose "proved to give the best SKR", and the decoy's bound is the tighter
        # one at every row of Table I and on the shipped fibre. Round 80 settled it on the paper's own
        # channel and dead-time convention: with the decoy's errors Table I's 26 dB row is reproduced
        # to within 2 percent (since Round 82's correction below), with the signal's it reads 13
        # percent low. It is fixed here
        # rather than picked per block, so the concentration inequalities Eq. (A24) counts are the
        # same four.
        #
        # tau0 MULTIPLIES THE ERRORS ONLY (Round 82): the vacuum's errors are at most tau0 (e^k/p_k)
        # m*_{Z,k} by (A15), and their expectation, half the vacuum events by (A12), exceeds them by
        # at most delta(s_{Z,0}) <= delta(n_Z) by (A13). This line had tau0 on the delta too.
        v0 = t0 * math.exp(mu2) / p2 * m_hi[1]                                    # (A15), the decoy's
        s0_upper = 2.0 * _vacuum_errors_expectation(v0, n_tot, eps1, p.deviation)  # (A16)
        s1 = (t1 * mu1 / (mu2 * (mu1 - mu2))) * (                                   # (4)
            n_minus_2 - (mu2 ** 2 / mu1 ** 2) * n_plus_1
            - ((mu1 ** 2 - mu2 ** 2) / mu1 ** 2) * (s0_upper / t0))
        v1_upper = t1 / (mu1 - mu2) * (m_plus_1 - m_minus_2)                        # (A22)
        per_basis[basis] = dict(n=n_tot, m=m_tot, s0=s0, s1=s1, v1=v1_upper,
                                qber=(m_tot / n_tot if n_tot > 0 else 0.5))

    z_, x_ = per_basis["Z"], per_basis["X"]
    n_z, n_x, qber = z_["n"], x_["n"], z_["qber"]
    lambda_ec = p.f_ec * n_z * _h2(qber)

    if z_["s1"] <= 0.0 or x_["s1"] <= 0.0:
        # No positive lower bound on single-photon events exists, so Eq. (1) has no meaning here.
        # Reported as -inf rather than 0 so a caller cannot mistake "no bound" for "a small key".
        return FiniteKeyResult(float("-inf"), n_z, n_x, n_z + n_x, z_["s0"], z_["s1"],
                               float("nan"), lambda_ec, qber)

    b = min(0.5, max(0.0, x_["v1"] / x_["s1"]))
    phi = 0.0 if b <= 0.0 else min(0.5, b + _gamma(p.eps_sec, b, z_["s1"], x_["s1"]))  # (A23)

    key = (z_["s0"] + z_["s1"] * (1.0 - _h2(phi)) - lambda_ec                         # (1)
           - 6.0 * math.log2(19.0 / p.eps_sec) - math.log2(2.0 / p.eps_cor))
    return FiniteKeyResult(key, n_z, n_x, n_z + n_x, z_["s0"], z_["s1"], phi, lambda_ec, qber)


def finite_key_length(n_pulses: int, channel: ChannelParams | None = None,
                      params: FiniteKeyParams | None = None) -> FiniteKeyResult:
    """The composable secret key length in bits, or a non-positive number if none exists.

    `n_pulses` is the total number of channel uses, not the block size after sifting - the
    conversion is part of what this computes, and stating the threshold in pulses is what makes it
    comparable to `QKDConfig.pulses`.
    """
    channel = channel or ChannelParams()
    p = params or FiniteKeyParams()
    return one_decoy_key_length(*expected_counts(n_pulses, channel, p), p)


def minimum_pulses_for_positive_key(channel: ChannelParams | None = None,
                                    params: FiniteKeyParams | None = None,
                                    *, hi: int = 10 ** 11) -> int | None:
    """Fewest channel uses that yield a positive composable key, or None if `hi` does not.

    Bisection on a monotone quantity: adding pulses adds statistics, and every finite-size
    correction shrinks relative to the counts it corrects.
    """
    channel = channel or ChannelParams()
    params = params or FiniteKeyParams()
    if finite_key_length(hi, channel, params).key_bits <= 0.0:
        return None
    lo = 1000
    while hi - lo > max(1000, lo // 1000):
        mid = (lo + hi) // 2
        if finite_key_length(mid, channel, params).key_bits > 0.0:
            hi = mid
        else:
            lo = mid
    return hi


def optimise_parameters(channel: ChannelParams | None = None,
                        base: FiniteKeyParams | None = None,
                        *, seed: int = 0) -> tuple[FiniteKeyParams, int | None]:
    """Minimise the block size needed for a positive key over the paper's own variable set.

    THE SCIENTIFIC GROUNDING IS THE PAPER'S PROCEDURE, not a grid chosen here. Rusca et al. state
    that they "optimize the SKR over the different decoy state variables, i.e. mu_i and the
    associated probability p_mu_i, and the probability to choose the Z basis"; that is exactly
    `FiniteKeyParams`'s first four fields and exactly what is searched.

    `scipy.optimize.differential_evolution` is used rather than a hand-rolled sweep because the
    objective is non-smooth - it is a bisection over a bound that goes undefined when `s^l_{Z,1}`
    turns negative - so gradient methods and coordinate descent both stall on the flat -inf
    region. Differential evolution is a population method that does not need derivatives and does
    not assume the objective is finite everywhere. Seeded, so the result is reproducible.

    Returns the best parameters found and the block size they need.
    """
    import numpy as np
    from scipy.optimize import differential_evolution

    channel = channel or ChannelParams()
    base = base or FiniteKeyParams()
    penalty = float(10 ** 12)

    def objective(v: "np.ndarray") -> float:
        mu1, ratio, p_signal, p_z = v
        mu2 = mu1 * ratio                      # parameterised as a RATIO so mu2 < mu1 by construction
        cand = FiniteKeyParams(mu_signal=float(mu1), mu_decoy=float(mu2),
                               p_signal=float(p_signal), p_z=float(p_z),
                               eps_sec=base.eps_sec, eps_cor=base.eps_cor, f_ec=base.f_ec,
                               deviation=base.deviation)
        n = minimum_pulses_for_positive_key(channel, cand)
        return penalty if n is None else float(n)

    res = differential_evolution(
        objective,
        bounds=[(0.05, 1.0),      # mu_signal: above 1 the multiphoton fraction dominates
                (0.02, 0.9),      # mu_decoy / mu_signal
                (0.05, 0.95),     # p_signal
                (0.05, 0.95)],    # p_z
        seed=seed, maxiter=60, popsize=20, tol=1e-6, polish=False)
    mu1, ratio, p_signal, p_z = res.x
    best = FiniteKeyParams(mu_signal=float(mu1), mu_decoy=float(mu1 * ratio),
                           p_signal=float(p_signal), p_z=float(p_z),
                           eps_sec=base.eps_sec, eps_cor=base.eps_cor, f_ec=base.f_ec,
                           deviation=base.deviation)
    return best, minimum_pulses_for_positive_key(channel, best)


# --- The vacuum+weak protocol: Lim et al. 2014 -----------------------------------------------------
#
# WHY THIS EXISTS (Round 79, the user's choice). The 1-decoy analysis above is not the protocol
# Micius ran. Micius sent a VACUUM state as well - "mu_s=0.8, mu_1=0.1, mu_2=0", "sent with
# probabilities of 50%, 25%, and 25%" (Liao et al., Nature 549, 43 (2017)) - and the 1-decoy bound
# has no use for the vacuum's quarter of the pulses. Lim et al.'s analysis uses all three
# intensities, and it is the one Rusca et al.'s Table I sets the 1-decoy bound against, so it is
# checked against that table's 2-decoy rows the way the 1-decoy bound is checked against its own.
#
# On Liao's 19 December 2016 pass (the model's counts; `tests/test_the_vacuum_decoy_bound_is_lim_2014.py`)
# it finds 164,183 bits under the published Hoeffding statistics, where the paper reports 300,939, and
# 226,056 under the default since Round 82. The 1-decoy bound on the same pulses, the vacuum's quarter
# unused, finds 140,400 under Hoeffding - 152,598 before Round 82 corrected its (A16), and NONE
# (-7,172) before Round 79's two corrections, which is what first suggested the vacuum was needed. It
# was not needed for a key; it adds 17 percent to it under Hoeffding and 21 under the default.
#
# EQUATIONS, by number in arXiv:1311.7129v2, written with Z as the key basis and X as the test basis
# (Rusca et al.'s naming; Lim et al. key on X), and eps = eps_sec / 21:
#
#   (1)  l = s_Z0 + s_Z1 (1 - h(phi_Z)) - leak_EC - 6 log2(21/eps_sec) - log2(2/eps_cor)
#        n^pm_{B,k} = (e^k / p_k)(n_{B,k} pm sqrt(n_B / 2 ln(1/eps))), and m^pm_{B,k} alike
#   (2)  s_B0 >= tau0 (mu2 n^-_{B,mu3} - mu3 n^+_{B,mu2}) / (mu2 - mu3)
#   (3)  s_B1 >= tau1 mu1 [n^-_{B,mu2} - n^+_{B,mu3} - (mu2^2 - mu3^2)/mu1^2 (n^+_{B,mu1} - s_B0/tau0)]
#                / (mu1 (mu2 - mu3) - mu2^2 + mu3^2)
#   (4)  v_X1 <= tau1 (m^+_{X,mu2} - m^-_{X,mu3}) / (mu2 - mu3)
#   (5)  phi_Z <= v_X1/s_X1 + gamma(eps_sec, v_X1/s_X1, s_X1, s_Z1)
#
# leak_EC = f_EC n_Z h(m_Z / n_Z), Lim et al.'s own model. They require mu1 > mu2 + mu3 and
# mu2 > mu3 >= 0, and the vacuum here is mu3 = 0 exactly, where (2) is tau0 n^-_{B,mu3}.
#
# THE BOUND TAKES COUNTS, NOT A CHANNEL. A satellite pass is not one channel - the loss changes by
# about ten decibels between the top of Liao's pass and its end - and the bound never needed one:
# it needs the detections and errors per basis and per intensity, which over a pass are SUMS over
# its seconds. `expected_counts` gives them for one channel, `DecoyCounts` adds, and
# `finite_key_over` runs either bound on a list of (pulses, channel) segments taken as one block.
# A block at one range is the one-segment case.


@dataclass(frozen=True)
class VacuumDecoyParams:
    """Alice's three intensities and their probabilities, the basis bias and the security budget.

    The defaults are Micius's source as Liao et al. state it, read from the config, which carries it
    as `free_space_*`; p_z = 0.5 because Micius chose bases passively, at a beam splitter. eps_sec =
    1e-9 is Liao's "statistical failure probability"; eps_cor and f_EC are Lim et al.'s values.
    """

    mu_signal: float = field(default_factory=lambda: _qkd_cfg().free_space_mu_signal)
    mu_decoy: float = field(default_factory=lambda: _qkd_cfg().free_space_mu_decoy)
    mu_vacuum: float = 0.0
    p_signal: float = field(default_factory=lambda: _qkd_cfg().free_space_p_signal)
    p_decoy: float = field(default_factory=lambda: _qkd_cfg().free_space_p_decoy)
    p_z: float = 0.5
    eps_sec: float = 1e-9
    eps_cor: float = 1e-15
    f_ec: float = 1.16
    deviation: str = field(default_factory=lambda: _qkd_cfg().finite_key_deviation)

    def __post_init__(self) -> None:
        if self.deviation not in DEVIATIONS:
            raise ValueError(f"deviation {self.deviation!r}: one of {DEVIATIONS}")
        mu1, mu2, mu3 = self.mu_signal, self.mu_decoy, self.mu_vacuum
        if not (mu1 > mu2 + mu3 and mu2 > mu3 >= 0.0):
            raise ValueError(f"Lim et al. need mu1 > mu2 + mu3 and mu2 > mu3 >= 0, not {mu1}, {mu2}, {mu3}")
        if min(self.p_signal, self.p_decoy, self.p_vacuum) <= 0.0:
            raise ValueError(f"every intensity needs a share of the pulses, not {self.intensities}")

    @property
    def p_vacuum(self) -> float:
        return 1.0 - self.p_signal - self.p_decoy

    @property
    def intensities(self) -> tuple[tuple[float, float], ...]:
        """(mu, p) for the signal, the decoy and the vacuum, the order `DecoyCounts` keeps."""
        return ((self.mu_signal, self.p_signal), (self.mu_decoy, self.p_decoy),
                (self.mu_vacuum, self.p_vacuum))

    def tau(self, n: int) -> float:
        """The probability that Alice sends an n-photon state, summed over the three intensities."""
        return sum(p * math.exp(-mu) * mu ** n / math.factorial(n) for mu, p in self.intensities)


def _events(c: DecoyCounts, p: VacuumDecoyParams, eps: float) -> tuple[float, float]:
    """Lower bounds on the vacuum and single-photon events in one basis, Eqs. (2) and (3).

    The vacuum bound is clamped at zero, which only removes a bound weaker than the trivial one: a
    count of events cannot be negative, and (3) increases with s_B0, so the clamp keeps (3) valid.
    """
    (mu1, p1), (mu2, p2), (mu3, p3) = p.intensities
    n_lo, n_hi = _bounds(c.n, eps, p.deviation)

    def lo(k: int) -> float:
        mu, p_k = p.intensities[k]
        return math.exp(mu) / p_k * n_lo[k]

    def hi(k: int) -> float:
        mu, p_k = p.intensities[k]
        return math.exp(mu) / p_k * n_hi[k]

    t0, t1 = p.tau(0), p.tau(1)
    s0 = max(0.0, t0 * (mu2 * lo(2) - mu3 * hi(1)) / (mu2 - mu3))                       # (2)
    s1 = (t1 * mu1 * (lo(1) - hi(2) - (mu2 ** 2 - mu3 ** 2) / mu1 ** 2 * (hi(0) - s0 / t0))  # (3)
          / (mu1 * (mu2 - mu3) - mu2 ** 2 + mu3 ** 2))
    return s0, s1


def vacuum_decoy_key_length(z: DecoyCounts, x: DecoyCounts,
                            params: VacuumDecoyParams | None = None) -> FiniteKeyResult:
    """Lim et al.'s composable key length from the key-basis counts `z` and test-basis counts `x`.

    Non-positive means no key; -inf means no single-photon bound exists at all, as in
    `finite_key_length`, so a caller cannot read "no bound" as "a small key".
    """
    p = params or VacuumDecoyParams()
    eps = p.eps_sec / 21.0
    s_z0, s_z1 = _events(z, p, eps)
    _, s_x1 = _events(x, p, eps)
    n_z, n_x = z.detections, x.detections
    qber = z.errors / n_z if n_z > 0 else 0.5
    lambda_ec = p.f_ec * n_z * _h2(qber)
    if s_z1 <= 0.0 or s_x1 <= 0.0:
        return FiniteKeyResult(float("-inf"), n_z, n_x, n_z + n_x, s_z0, s_z1, float("nan"),
                               lambda_ec, qber)
    (_, _), (mu2, p2), (mu3, p3) = p.intensities
    m_lo, m_hi = _bounds(x.m, eps, p.deviation)
    v_x1 = p.tau(1) * (math.exp(mu2) / p2 * m_hi[1]                                     # (4)
                       - math.exp(mu3) / p3 * m_lo[2]) / (mu2 - mu3)
    b = min(0.5, max(0.0, v_x1 / s_x1))
    phi = 0.0 if b <= 0.0 else min(0.5, b + _gamma(p.eps_sec, b, s_x1, s_z1))            # (5)
    key = (s_z0 + s_z1 * (1.0 - _h2(phi)) - lambda_ec                                    # (1)
           - 6.0 * math.log2(21.0 / p.eps_sec) - math.log2(2.0 / p.eps_cor))
    return FiniteKeyResult(key, n_z, n_x, n_z + n_x, s_z0, s_z1, phi, lambda_ec, qber)


def params_for_source(source, *, eps_sec: float = 1e-9, f_ec: float | None = None,
                      deviation: str | None = None) -> FiniteKeyParams | VacuumDecoyParams:
    """The finite-key bound a BB84 source supports, with its own intensities and shares.

    `source` is a `bb84.QKDParams`. Lim et al. 2014 when it sends the vacuum, Rusca et al. 2018's
    1-decoy bound when it does not; the basis bias is the source's, and f_EC defaults to the config's
    error-correction efficiency, the one the asymptotic rate uses. Round 80: one place decides it for
    every caller, fibre and satellite. The statistics are the config's unless `deviation` names one.
    """
    f = _qkd_cfg().error_correction_eff if f_ec is None else float(f_ec)
    common = dict(mu_signal=float(source.mu_signal), mu_decoy=float(source.mu_decoy),
                  p_signal=float(source.p_signal), p_z=float(source.basis_prob), eps_sec=float(eps_sec),
                  f_ec=f, deviation=str(deviation or _qkd_cfg().finite_key_deviation))
    if source.vacuum_decoy:
        return VacuumDecoyParams(p_decoy=float(source.p_decoy), **common)
    return FiniteKeyParams(**common)


def fibre_block_pulses(channel: ChannelParams, params: FiniteKeyParams | VacuumDecoyParams, *,
                       kind: str, run_pulses: float, seconds: float, pa_bits: float) -> float:
    """The channel uses in a fibre key's block (Round 80), by the block's definition.

    "pulses": the run's own; "seconds": that long at the channel's clock; "pa_bits": as many as it
    takes for the key basis to collect `pa_bits` detections, the privacy-amplification block size
    finite-key analyses state (Rusca et al.'s n_Z). Expected counts are linear in the pulses, so the
    last is a division, exact.
    """
    if kind == "pulses":
        return float(run_pulses)
    if kind == "seconds":
        return float(seconds) * float(channel.source_rate_hz)
    if kind == "pa_bits":
        per_pulse = expected_counts(1.0, channel, params)[0].detections
        return float(pa_bits) / per_pulse if per_pulse > 0.0 else math.inf
    raise ValueError(f"block kind {kind!r}: one of 'pulses', 'seconds', 'pa_bits'")


def _bound(params: FiniteKeyParams | VacuumDecoyParams):
    return vacuum_decoy_key_length if isinstance(params, VacuumDecoyParams) else one_decoy_key_length


def block_counts(segments: Iterable[tuple[float, ChannelParams]],
                 params: FiniteKeyParams | VacuumDecoyParams) -> tuple[DecoyCounts, DecoyCounts]:
    """The expected (Z, X) counts of several (pulses, channel) segments, summed."""
    z = x = None
    for n_pulses, channel in segments:
        dz, dx = expected_counts(n_pulses, channel, params)
        z, x = (dz, dx) if z is None else (z + dz, x + dx)
    if z is None:
        raise ValueError("no segments: a block needs at least one")
    return z, x


def blocks_for_positive_key(z: DecoyCounts, x: DecoyCounts,
                            params: FiniteKeyParams | VacuumDecoyParams, *, hi: int = 10_000) -> int | None:
    """The fewest copies of this block, pooled into one, that carry a positive key, or None if `hi`
    copies do not. For a pass, how many passes; for one second at a range, how many seconds.

    Bisection on a monotone quantity, as `minimum_pulses_for_positive_key`: pooled counts only
    tighten every statistical correction relative to the counts they correct.
    """
    bound = _bound(params)
    if bound(z, x, params).key_bits > 0.0:
        return 1
    if bound(z.scaled(hi), x.scaled(hi), params).key_bits <= 0.0:
        return None
    lo, top = 1, hi
    while top - lo > 1:
        mid = (lo + top) // 2
        if bound(z.scaled(mid), x.scaled(mid), params).key_bits > 0.0:
            top = mid
        else:
            lo = mid
    return top


def finite_key_over(segments: Iterable[tuple[float, ChannelParams]],
                    params: FiniteKeyParams | VacuumDecoyParams | None = None) -> FiniteKeyResult:
    """Either bound on several (pulses, channel) segments taken as ONE block: a pass, or part of one.

    The bound follows the source: Lim et al.'s for `VacuumDecoyParams` (the default, Micius's
    source), Rusca et al.'s 1-decoy one for `FiniteKeyParams`.
    """
    p = params or VacuumDecoyParams()
    return _bound(p)(*block_counts(segments, p), p)
