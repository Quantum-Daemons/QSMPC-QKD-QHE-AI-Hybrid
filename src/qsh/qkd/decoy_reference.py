"""Published decoy-state BB84 reference curves, for reproducing literature values.

This module exists to be CHECKED, not to be trusted. Every equation below is transcribed from a
named paper and the test suite asserts the paper's own published reference values, so a reader can
verify the implementation against the source rather than against this project's say-so.

Primary source for the simulator equations:
    X. Ma, B. Qi, Y. Zhao, H.-K. Lo, "Practical decoy state for quantum key distribution",
    Phys. Rev. A 72, 012326 (2005). arXiv:quant-ph/0503005v2.

ROUND 46 RE-CHECKED EVERY EQUATION NUMBER IN THIS FILE AGAINST THAT PDF, and four of them were
wrong: the key rate is Eq. (1) and was cited as (33), Y_i is Eq. (7) and was cited as (6), e_i is
Eq. (9) and was cited as (7), and e_0 = 1/2 is stated in prose rather than as Eq. (4). A module whose
docstring says it "exists to be CHECKED" is exactly the one where a reader follows the citation, so a
wrong number here costs more than elsewhere. The FORMULAS were correct throughout - only the pointers
were wrong, which is the harder version to notice.

The GLLP secure fraction for the NON-decoy case:
    D. Gottesman, H.-K. Lo, N. Luetkenhaus, J. Preskill, Quantum Inf. Comput. 4, 325 (2004),
    arXiv:quant-ph/0212066.

Why the non-decoy branch is worth having at all: this project's whole QKD claim rests on decoy
states, and the asymptotic model used at runtime already assumes the decoy method has bounded the
single-photon contribution. That assumption is invisible until you compute what the link would do
WITHOUT it. The sharpest published statement of the difference, and the one the tests assert:
with the GYS parameter set and mu = 0.1, the tagged fraction exceeds 1 at ZERO distance - a
non-decoy link yields no key even back-to-back.

Nothing here produces a score. Each function returns a physical quantity with a citation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

E0 = 0.5
"""Error rate of a dark count: a background click is uncorrelated with the prepared state, so it is
wrong half the time. Ma et al. state this in the prose following Eq. (9) rather than as a
numbered equation; the Eq. (4) citation this line carried until Round 46 was wrong."""


@dataclass(frozen=True)
class ChannelSpec:
    """A fibre BB84 channel.

    Defaults are the GYS parameter set from C. Gobby, Z. L. Yuan, A. J. Shields, Appl. Phys. Lett.
    84, 3762 (2004), as tabulated in Ma et al. (2005) Table 1. They are defaults so the published
    curves can be reproduced without restating them at each call site.

    **FOUR of the five are Table 1; `f_ec` is NOT, and saying "as tabulated in Table 1" of all of
    them was wrong.** Table 1 gives alpha = 0.21 dB/km, e_detector = 3.3%, Y0 = 1.7e-6 and
    eta_Bob = 0.045 - which are `alpha_db_per_km`, `misalignment`, `dark_count` and `detector_eff`
    here. Its remaining column, `f`, is a REPETITION RATE: 2 MHz for GYS, 0.1 MHz for KTH.

    `f_ec = 1.22` comes from the paper's prose instead - "use f(e) = 1.22, which is the upper bound
    of f(e) in secure distance for this experiment", cited onward to its own reference [12], and
    also at "mu_optimal ~ 0.48 for f(e) = 1.22". A correct number under a wrong provenance, which is
    the harder kind to notice: nothing computed from it is wrong, and a reader checking Table 1 for
    it finds a megahertz.
    """

    alpha_db_per_km: float = 0.21
    detector_eff: float = 0.045
    dark_count: float = 1.7e-6
    misalignment: float = 0.033
    f_ec: float = 1.22
    q_sift: float = 0.5
    """Sifting factor.

    0.5 is standard BB84: Alice and Bob choose bases independently and agree half the time.

    **THIS DOCSTRING ASSERTED THE OPPOSITE OF THE PAPER AND IS CORRECTED IN ROUND 53.** It said the
    published curves are plotted for the EFFICIENT BB84 of Lo, Chau and Ardehali "and Ma et al. say
    so in the text". Read at source (arXiv quant-ph/0503005), the text says the reverse. Eq. (42) is
    what Figure 2 plots, and the sentence introducing it reads:

        "where q = 1/2 with standard BB84"

    The paper does define both conventions earlier - "1/2 for the BB84 protocol ... and if one uses
    the efficient BB84 protocol, q ~ 1" - but for the figure this module reproduces it states 1/2.

    **What follows from that, stated as an open question rather than silently repaired.** The
    reference values this module is tested against, R/eta ~ 0.11 and R/eta^2 ~ 0.15, appear NOWHERE
    in the paper's prose - searched, and they are not there. `tests/test_decoy_reference.py` passes
    `q_sift=1.0` to reach them, which doubles the rate; at the q the paper names for that figure this
    model produces 0.0568 and 0.0755. So one of three things is true and the record should not
    pretend to know which: the figures were read off Figure 2's plotted curve rather than its text
    and the curve is at q ~ 1 despite the caption; or they were read correctly and this model is 2x
    low, with `q_sift=1.0` closing the gap by coincidence; or they were derived from this model in
    the first place and then attributed to the paper.

    **Not resolved here, because resolving it requires reading values off a plotted figure**, which
    is not something to do from an extracted text layer. Until then the 0.11 / 0.15 assertions are
    this module's own outputs at `q_sift=1.0`, and describing them as "the published values" is the
    claim that needs evidence. The default stays 0.5 because that is what this project runs and what
    the paper names.
    """


def h2(x: float) -> float:
    """Binary Shannon entropy, 0 outside (0, 1)."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def transmittance(distance_km: float, ch: ChannelSpec) -> float:
    """Overall transmittance eta = eta_Bob * 10^(-alpha L / 10). Ma et al. Eq. (2)."""
    return ch.detector_eff * 10.0 ** (-ch.alpha_db_per_km * distance_km / 10.0)


def gain(mu: float, eta: float, ch: ChannelSpec) -> float:
    """Overall gain Q_mu = Y_0 + 1 - exp(-eta mu). Ma et al. Eq. (10)."""
    return ch.dark_count + 1.0 - math.exp(-eta * mu)


def qber(mu: float, eta: float, ch: ChannelSpec) -> float:
    """E_mu, from E_mu Q_mu = e_0 Y_0 + e_det (1 - exp(-eta mu)). Ma et al. Eq. (11)."""
    q = gain(mu, eta, ch)
    if q <= 0.0:
        return E0
    return (E0 * ch.dark_count + ch.misalignment * (1.0 - math.exp(-eta * mu))) / q


def single_photon_yield(eta: float, ch: ChannelSpec) -> float:
    """Y_1 = Y_0 + eta - Y_0 eta. Ma et al. Eq. (7) at i = 1.

    (Was cited as Eq. (6) until Round 46; checked against arXiv:quant-ph/0503005v2.)"""
    return ch.dark_count + eta - ch.dark_count * eta


def single_photon_error(eta: float, ch: ChannelSpec) -> float:
    """e_1 = (e_0 Y_0 + e_det eta) / Y_1. Ma et al. Eq. (9) at i = 1.

    (Was cited as Eq. (7) - which is the Y_i equation - until Round 46.)"""
    y1 = single_photon_yield(eta, ch)
    if y1 <= 0.0:
        return E0
    return (E0 * ch.dark_count + ch.misalignment * eta) / y1


def tagged_fraction(mu: float, eta: float, ch: ChannelSpec) -> float:
    """GLLP tagged fraction Delta = p_multi / Q_mu.

    Without decoy states the worst case must be assumed: EVERY detection originating from a
    multi-photon pulse is known to Eve, because she can split one photon off and keep it. So the
    fraction of detections that cannot be counted as secure is the multi-photon emission
    probability divided by the total detection probability.

    Delta >= 1 means no detection can be certified single-photon and the key rate is zero
    regardless of distance or error rate.
    """
    q = gain(mu, eta, ch)
    if q <= 0.0:
        return 1.0
    p_multi = 1.0 - math.exp(-mu) * (1.0 + mu)      # P(n >= 2) for a Poissonian source
    return p_multi / q


def key_rate_decoy(mu: float, distance_km: float, ch: ChannelSpec) -> float:
    """Decoy-state key rate per PULSE.

        R >= q { -Q_mu f h2(E_mu) + Q_1 [1 - h2(e_1)] },  q = ch.q_sift.

    Ma et al. Eq. (1). Decoy states let Y_1 and e_1 be estimated directly, so the single-photon
    contribution Q_1 is credited rather than discarded.
    """
    eta = transmittance(distance_km, ch)
    q_mu = gain(mu, eta, ch)
    e_mu = qber(mu, eta, ch)
    q_1 = single_photon_yield(eta, ch) * mu * math.exp(-mu)
    e_1 = single_photon_error(eta, ch)
    return max(0.0, ch.q_sift * (-q_mu * ch.f_ec * h2(e_mu) + q_1 * (1.0 - h2(e_1))))


def key_rate_nondecoy(mu: float, distance_km: float, ch: ChannelSpec) -> float:
    """GLLP key rate per PULSE without decoy states.

        R >= q Q_mu { (1 - Delta) - f h2(E_mu) - (1 - Delta) h2(E_mu / (1 - Delta)) }

    GLLP Eq. (50). The single-photon fraction is bounded only by the multi-photon emission
    probability, so the rate collapses as soon as Delta approaches the detection probability -
    which, at long distance, it does quadratically rather than linearly in eta.
    """
    eta = transmittance(distance_km, ch)
    q_mu = gain(mu, eta, ch)
    e_mu = qber(mu, eta, ch)
    delta = tagged_fraction(mu, eta, ch)
    if delta >= 1.0:
        return 0.0
    untagged = 1.0 - delta
    inner = untagged - ch.f_ec * h2(e_mu) - untagged * h2(min(0.5, e_mu / untagged))
    return max(0.0, ch.q_sift * q_mu * inner)


def _optimise_mu(rate_fn, distance_km: float, ch: ChannelSpec,
                 lo: float = 1e-4, hi: float = 1.0, steps: int = 4000) -> tuple[float, float]:
    """Best mu for a given rate function at one distance, by scan.

    A scan rather than a solver because the non-decoy objective is zero over a wide region and a
    gradient method parks in it. Cheap enough: this runs once per plotted point.
    """
    best_mu, best_r = lo, 0.0
    for i in range(steps + 1):
        mu = lo + (hi - lo) * i / steps
        r = rate_fn(mu, distance_km, ch)
        if r > best_r:
            best_mu, best_r = mu, r
    return best_mu, best_r


def optimal_decoy(distance_km: float, ch: ChannelSpec | None = None) -> tuple[float, float]:
    """(mu, rate) maximising the decoy-state rate at this distance."""
    return _optimise_mu(key_rate_decoy, distance_km, ch or ChannelSpec())


def optimal_nondecoy(distance_km: float, ch: ChannelSpec | None = None) -> tuple[float, float]:
    """(mu, rate) maximising the GLLP non-decoy rate at this distance."""
    return _optimise_mu(key_rate_nondecoy, distance_km, ch or ChannelSpec())


def max_distance_km(rate_fn, ch: ChannelSpec | None = None,
                    hi_km: float = 400.0, tol_km: float = 0.5) -> float:
    """Longest distance at which an optimised rate is still positive, by bisection."""
    ch = ch or ChannelSpec()
    lo, hi = 0.0, hi_km
    if _optimise_mu(rate_fn, lo, ch)[1] <= 0.0:
        return 0.0
    while hi - lo > tol_km:
        mid = 0.5 * (lo + hi)
        if _optimise_mu(rate_fn, mid, ch)[1] > 0.0:
            lo = mid
        else:
            hi = mid
    return lo
