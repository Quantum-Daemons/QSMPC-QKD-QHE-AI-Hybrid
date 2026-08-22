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

HOW ACCURATE THIS IS, and it is not accurate enough to quote to two figures

Checked against that Table I - secret key rate at n_Z = 1e7 for 26 / 46 / 56 / 64 dB - this
implementation comes out CONSERVATIVE by roughly a factor of 2-3 at moderate loss, and by more at
high loss, even after optimising the intensities, their probabilities and the basis bias the way the
paper says it optimises them. The gap is in the safe direction, but it is a gap: the threshold
computed here is an UPPER bound on the block size really required, not the block size really
required. So it is quoted everywhere as an order of magnitude - "~1e5 sifted bits, about 4e7 pulses"
- and never as 1.436e5.

Three claims are robust to that factor and are the only ones the repository makes:

  (a) the shipped 2e6-pulse block carries no composable key, because `s^l_{Z,1}` is NEGATIVE there
      and a factor of 2-3 cannot rescue a negative count;
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
  (A16) s^u_{Z,0} = 2 tau0 [ (e^k/p_k)(m_{Z,k} + delta(m_Z, eps)) + delta(n_Z, eps) ]
  (A19) s^l_{Z,0} = tau0/(mu1 - mu2) ( mu1 n^-_{Z,mu2} - mu2 n^+_{Z,mu1} )
  (A21) gamma(a,b,c,d) = sqrt( (c+d)(1-b)b/(c d ln2) * log2( (c+d)/(c d (1-b) b) * 2^21/a^2 ) )
  (A22) v^u_{X,1} = tau1/(mu1 - mu2) ( m^+_{X,mu1} - m^-_{X,mu2} )
  (A23) phi^u_Z = v^u_{X,1}/s^l_{X,1} + gamma(eps_sec, v^u_{X,1}/s^l_{X,1}, s^l_{Z,1}, s^l_{X,1})

`a = 6, b = 19` in (1) are the paper's own constants for the 1-decoy protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .bb84 import e_det_for_bound
from .channel import ChannelParams

__all__ = [
    "FiniteKeyResult",
    "FiniteKeyParams",
    "finite_key_length",
    "minimum_pulses_for_positive_key",
    "optimise_parameters",
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


def _tau(n: int, p: FiniteKeyParams) -> float:
    """Total probability that Alice sends an n-photon state, summed over the intensities."""
    return sum(prob * math.exp(-mu) * mu ** n / math.factorial(n)
               for mu, prob in ((p.mu_signal, p.p_signal), (p.mu_decoy, 1.0 - p.p_signal)))


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


def finite_key_length(n_pulses: int, channel: ChannelParams | None = None,
                      params: FiniteKeyParams | None = None) -> FiniteKeyResult:
    """The composable secret key length in bits, or a non-positive number if none exists.

    `n_pulses` is the total number of channel uses, not the block size after sifting - the
    conversion is part of what this computes, and stating the threshold in pulses is what makes it
    comparable to `QKDConfig.pulses`.
    """
    channel = channel or ChannelParams()
    p = params or FiniteKeyParams()
    eta = channel.transmittance()
    # THE MISALIGNMENT, not `intrinsic_qber(channel)`. `_gain_and_error` adds the background back
    # as its own `0.5 * y0` term, so passing the observed QBER charges dark counts twice - the
    # identical defect this round found in `bb84.py` and `quantumSim.ts`, and it was written into
    # this file first time through before that fix was made. Any eavesdropping is deliberately not
    # folded in here either: `e_det` is a device property, and an intercept-resend fraction belongs
    # in the observed statistics rather than in the detector model.
    e_det = e_det_for_bound(channel)
    y0 = 2.0 * channel.dark_count_prob

    # The security budget is split across the statistical terms. eps_sec/20 for each of the
    # Hoeffding corrections is the conservative reading of Eq. (A24), which sums 6*eps1 + 4*eps2
    # among other terms; taking each individual epsilon at eps_sec/20 keeps the sum below eps_sec.
    eps1 = eps2 = p.eps_sec / 20.0

    intensities = ((p.mu_signal, p.p_signal), (p.mu_decoy, 1.0 - p.p_signal))
    per_basis: dict[str, dict] = {}
    for basis, p_basis in (("Z", p.p_z), ("X", 1.0 - p.p_z)):
        counts = {}
        for mu, p_mu in intensities:
            q, e = _gain_and_error(eta, mu, e_det, y0)
            n = n_pulses * p_mu * p_basis * p_basis * q      # Alice and Bob both choose this basis
            counts[mu] = (n, n * e)
        n_tot = sum(n for n, _ in counts.values())
        m_tot = sum(m for _, m in counts.values())
        d_n, d_m = _delta(n_tot, eps1), _delta(m_tot, eps2)

        mu1, mu2 = p.mu_signal, p.mu_decoy
        p1, p2 = p.p_signal, 1.0 - p.p_signal
        n_plus_1 = math.exp(mu1) / p1 * (counts[mu1][0] + d_n)
        n_minus_2 = math.exp(mu2) / p2 * (counts[mu2][0] - d_n)
        m_plus_1 = math.exp(mu1) / p1 * (counts[mu1][1] + d_m)
        m_minus_2 = math.exp(mu2) / p2 * (counts[mu2][1] - d_m)

        t0, t1 = _tau(0, p), _tau(1, p)
        # (A19), CLAMPED AT ZERO. This clamp is load-bearing: without it a negative vacuum count
        # propagates through the `- ((mu1^2-mu2^2)/mu1^2)(s^u_{Z,0}/tau0)` term of Eq. (4) with the
        # wrong sign and INFLATES the single-photon bound, moving the computed threshold by a
        # factor of several. A count of events cannot be negative.
        s0 = max(0.0, t0 / (mu1 - mu2) * (mu1 * n_minus_2 - mu2 * n_plus_1))
        s0_upper = 2.0 * t0 * (math.exp(mu1) / p1 * (counts[mu1][1] + d_m) + d_n)   # (A16)
        s1 = (t1 * mu1 / (mu2 * (mu1 - mu2))) * (                                   # (4)
            n_minus_2 - (mu2 ** 2 / mu1 ** 2) * n_plus_1
            - ((mu1 ** 2 - mu2 ** 2) / mu1 ** 2) * (s0_upper / t0))
        v1_upper = t1 / (mu1 - mu2) * (m_plus_1 - m_minus_2)                        # (A22)
        per_basis[basis] = dict(n=n_tot, m=m_tot, s0=s0, s1=s1, v1=v1_upper,
                                qber=(m_tot / n_tot if n_tot > 0 else 0.5))

    z, x = per_basis["Z"], per_basis["X"]
    n_z, n_x, qber = z["n"], x["n"], z["qber"]
    lambda_ec = p.f_ec * n_z * _h2(qber)

    if z["s1"] <= 0.0 or x["s1"] <= 0.0:
        # No positive lower bound on single-photon events exists, so Eq. (1) has no meaning here.
        # Reported as -inf rather than 0 so a caller cannot mistake "no bound" for "a small key".
        return FiniteKeyResult(float("-inf"), n_z, n_x, n_z + n_x, z["s0"], z["s1"],
                               float("nan"), lambda_ec, qber)

    b = min(0.5, max(0.0, x["v1"] / x["s1"]))
    if b <= 0.0:
        phi = 0.0
    else:
        c, d = z["s1"], x["s1"]
        gamma = math.sqrt((c + d) * (1.0 - b) * b / (c * d * math.log(2))               # (A21)
                          * math.log2((c + d) / (c * d * (1.0 - b) * b)
                                      * (2 ** 21) / p.eps_sec ** 2))
        phi = min(0.5, b + gamma)

    key = (z["s0"] + z["s1"] * (1.0 - _h2(phi)) - lambda_ec                             # (1)
           - 6.0 * math.log2(19.0 / p.eps_sec) - math.log2(2.0 / p.eps_cor))
    return FiniteKeyResult(key, n_z, n_x, n_z + n_x, z["s0"], z["s1"], phi, lambda_ec, qber)


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
                               eps_sec=base.eps_sec, eps_cor=base.eps_cor, f_ec=base.f_ec)
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
                           eps_sec=base.eps_sec, eps_cor=base.eps_cor, f_ec=base.f_ec)
    return best, minimum_pulses_for_positive_key(channel, best)
