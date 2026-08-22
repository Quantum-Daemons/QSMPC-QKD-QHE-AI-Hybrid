"""Quantum Secure Direct Communication (QSDC) — block-transmission simulation.

QSDC (the report §6: QSDC / QT-QSS) transmits a *message* over the quantum channel
directly, without first distributing a key. Security comes from a two-step / DL04-style
block transmission with an interspersed **eavesdrop check**: a random subset of carriers
is sampled and their error rate estimated *before* the sensitive block is released. If the
SECRECY CAPACITY of the link is not positive the transmission is aborted, so an eavesdropper
learns nothing about the message (it was never sent on a compromised channel).

This is the natural primitive for **satellite communications**: a free-space downlink where
re-keying round-trips are expensive, so sending the message directly (with an inline
eavesdrop check) is attractive. We simulate the *information structure* faithfully on a
:class:`~qsh.qkd.channel.ChannelParams` link; a real system adds decoy states, frame error
correction (LDPC) and privacy amplification.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, replace

import numpy as np

from .channel import ChannelParams, intrinsic_qber


def _h2(x: float) -> float:
    """Binary entropy H2(x)."""
    if x <= 0 or x >= 1:
        return 0.0
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


@dataclass
class QSDCResult:
    """Outcome of a QSDC block transmission."""

    delivered: bytes | None        # the message if the channel passed the eavesdrop check, else None
    eavesdrop_detected: bool       # abort ATTRIBUTABLE to Eve: this link would have closed without her
    error_rate: float              # measured error on the sampled check carriers
    # Cs = 1 - h(e) - g h(e). REPORTED SIGNED. This used to be replaced by a literal 0.0 on the
    # abort path, so a link that missed by 51 bit/carrier and one that missed by 0.001 reported the
    # same number -- and 0.0 also reads as "marginal but not negative", the one thing it was not.
    # How badly a link fails is the whole quantity here: it is the difference between "widen the
    # aperture" and "QSDC does not close over this class of link".
    secrecy_capacity: float
    message_bits: int
    check_bits: int
    loss_db: float                 # one-way channel loss backing g
    eve_gain_g: float              # g = Q_Eve / Q_Bob = 10^(loss_dB/10)
    link_budget_limited: bool      # Cs <= 0 even with Eve removed -- the link, not an attack
    #: Serfling upper confidence bound on the MESSAGE carriers' error rate, at ``EPS_PE``. This is
    #: the quantity the abort is decided on; ``error_rate`` above is the point estimate and is kept
    #: only so the gap between the two is visible.
    error_rate_upper: float = 0.0
    #: The block is too short for the check subset to certify ANY useful error rate, so the abort is
    #: attributable to sample size rather than to Eve or to the link budget. Distinct from both:
    #: a statistics-limited abort says nothing about the channel.
    statistics_limited: bool = False


#: Failure probability of the parameter estimation. Matches the QSDC precedent that exists - Wu, Long
#: & Hayashi, Phys. Rev. Applied 17, 064011 (2022), Fig. 4, evaluate their finite-length rate at
#: "B <= 1e-9, E <= 1e-9" - and standard QKD practice.
EPS_PE = 1e-9


def serfling_upper(observed: float, n_check: int, n_message: int, eps: float = EPS_PE) -> float:
    """One-sided upper confidence bound on the error rate of the UNMEASURED carriers.

    Sampling is WITHOUT REPLACEMENT - the check carriers are drawn from the same finite block as the
    message carriers and then discarded - so the right tool is Serfling's inequality, not Hoeffding's.
    Curras-Lorenzo, Navarrete, Pereira & Curty, arXiv:2410.04095, Eq. (15), after Lemma 6 of
    Tomamichel, Lim, Gisin & Renner, Nat. Commun. 3, 634 (2012), after Serfling, Ann. Statist. 2, 39
    (1974). Guarantees ``Pr[e_message >= e_up] <= eps``.

    NOT A QSDC CITATION, AND THAT MATTERS. The QSDC literature does not specify a finite-size
    parameter-estimation procedure, and says so: Pan et al., Photonics Res. 8, 1522 (2020) - "the
    finite-size analysis of QKD cannot be directly invoked for QSDC ... would be an interesting
    direction for future research"; the IEEE COMST survey (arXiv:2311.13974) - "the finite-length
    security analysis of QSDC is currently under investigation"; and Lei et al., Entropy 27(4), 332
    (2025), from Long's own group, which adheres to "the asymptotic limit, assuming an infinite block
    length". The two papers that do carry finite-size QSDC analysis are about other layers entirely -
    Wu-Long-Hayashi on the wiretap CODE, Sun et al. (IEEE Trans. Commun. 72, 4633) on DECOY
    INTENSITIES - and citing either for a check-subset bound would misrepresent them.

    So this bound is imported from QKD and is THIS PROJECT'S OWN CONSERVATIVE CHOICE. Recorded as
    such rather than dressed in a citation that does not cover it.
    """
    if n_check <= 0 or n_message <= 0:
        return 0.5
    n_total = n_check + n_message
    dev = math.sqrt(n_total * (n_check + 1) * math.log(1.0 / eps)
                    / (2.0 * n_message * n_check * n_check))
    return float(min(0.5, observed + dev))


def quantum_secure_direct_comm(
    message: bytes,
    params: ChannelParams | None = None,
    *,
    check_fraction: float = 0.25,
    seed: int | None = None,
) -> QSDCResult:
    """Send ``message`` directly over the quantum channel with an inline eavesdrop check.

    The abort criterion is **secrecy capacity**, not a fixed QBER. Qi et al., Light Sci. Appl. 8,
    22 (2019), protocol step 2: if ``Cs`` is zero the channel is insecure and the transfer stops.

        Cs = 1 - h(e) - g h(e_chk),    g = Q_Eve / Q_Bob = 10^(loss_dB / 10)

    This replaces a ``threshold=0.11`` parameter that was the g = 1 (LOSS-FREE) root of the same
    expression -- which is why it coincided exactly with the Shor-Preskill BB84 bound and looked
    right. Real links are lossy and ``g`` grows with loss, so the true tolerance collapses: at the
    24.5 dB of this project's own free-space downlink it is ~3e-4, three orders of magnitude below
    0.11 and wrong in the PERMISSIVE direction. The browser path was corrected first; this is the
    reference implementation catching up, and until now the two disagreed about whether the same
    nominal link delivers.
    """
    params = params or ChannelParams()
    rng = np.random.default_rng(seed)

    msg_bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8)) if message else np.array([], np.uint8)
    n_msg = int(msg_bits.size)
    # check carriers interspersed so that they are ~check_fraction of all carriers sent
    n_check = int(round(check_fraction / max(1e-9, 1.0 - check_fraction) * max(n_msg, 1)))

    e = intrinsic_qber(params)                       # true channel error (includes Eve)
    check_errors = rng.random(n_check) < e
    error_rate = float(check_errors.mean()) if n_check else float(e)

    # `params.loss_db()`, not the fibre product: a free-space channel carries a diffraction spec and
    # this expression silently ignored it, so the same downlink was 28.0 dB here and 24.48 dB in the
    # browser - a factor 2.7 in g, and so in the secrecy capacity the abort is decided on.
    # `g` IS Q_Eve/Q_Bob, AND IT INCLUDES DETECTOR EFFICIENCY. Qi, Long et al. define it twice - "g
    # represents the gap between QEve and QBob, depending on the back channel loss and the efficiency
    # of detector", and again numerically: "the loss of the back channel, INCLUDES THE EFFICIENCY OF
    # THE DETECTOR and channel loss, is about 4.1 dB, so that g = 2.57". 10^(4.1/10) = 2.5704
    # reproduces their figure exactly, so the functional form was right and what went into L was not.
    #
    # Omitting eta_d makes g too SMALL, which understates I(A:E) and OVERSTATES secrecy capacity - an
    # optimistic-security error, the same direction as the two this project has already corrected.
    #
    # Measured on the shipped configuration, and the factor is NOT one number:
    #
    #     fibre 50 km   eta_d 0.25   g 10.00 -> 40.00    4.00x   QBER tolerance 0.011551 -> 0.002405
    #     downlink      eta_d 0.40   g 280.43 -> 701.06  2.50x   tolerance 2.669e-4 -> 9.635e-5
    #
    # The 4.0x quoted in the sweep is exactly 1/0.25 from this repo's own default - it is NOT Qi's
    # number, whose SNSPDs at 70% would give 1.43x. On the satellite downlink, which is the path QSDC
    # actually serves here, the factor is 2.5x. Quoting a single "4.0x" would be wrong for satellite.
    #
    # CAVEAT CARRIED: Qi et al. do not itemise their 4.1 dB - the components they state sum to
    # 2.15 dB, leaving ~1.95 dB unaccounted, presumably return-path optics. A fully defensible `g`
    # would be an explicit return-path budget with its terms named, rather than channel loss plus
    # detector efficiency. This is the honest floor, not the final answer.
    loss_db = params.loss_db() - 10.0 * math.log10(params.detector_efficiency)
    g = 10.0 ** (loss_db / 10.0)

    # THE ABORT IS DECIDED ON THE UPPER BOUND, NOT ON THE POINT ESTIMATE.
    #
    # `error_rate` is the frequency of errors on a HANDFUL of check carriers, and it was fed straight
    # into the secrecy capacity. Measured before this change, on the SHIPPED default channel: an
    # 11-byte message gives 29 check carriers, the true QBER is 0.01520 against a tolerance of
    # 0.00241 - so the link provably cannot carry QSDC - and the transfer DELIVERED 118 times in 200.
    # Not because the physics was wrong; because 0.9848^29 = 0.64, so a zero-error sample is the
    # common case at that block length. The check reported success on a channel it had no power to
    # judge, which is worse than a wrong threshold: it is a threshold applied to a number that does
    # not mean what the comparison assumes.
    #
    # With the bound in place the same call aborts, and so does every message this demo can send.
    # That is the correct answer and it is worth stating plainly rather than tuning around: at 29
    # check carriers with ZERO observed errors, epsilon = 1e-9 certifies only e < 0.701, and
    # h(0.701) is near 1, so C_s is deeply negative. There is no epsilon that rescues it - even a
    # laughable eps = 1e-2 certifies only e < 0.331. Certifying the fibre tolerance of 2.405e-3 needs
    # about 1.8e6 check carriers; the satellite tolerance of 9.635e-5 needs about 1.1e9. Qi et al.'s
    # own experiment runs blocks of 1,312 x 830 = 1,088,960 pulses, which is the regime this bound
    # says is required - so the arithmetic is not an artefact of being too strict, it is what the
    # published experiments already do.
    # AND THE ENTROPY ARGUMENT IS THE SUM OF THE TWO BASIS ERROR RATES, NOT A POOLED ONE.
    #
    # Eq. (8) reads `Cs = Q_Bob * [ 1 - h(e) - g * h(e_x + e_z) ]`, and the paper defines `e_x`,
    # `e_z` as "the bit-error rates in the X-basis and the Z-basis in the error-check". This is
    # structural rather than notational: Eq. (5) gives Eve's term as `h(xi)` with
    # `xi = (1 - sqrt((1-2p)^2 + (1-2e_x-2e_z)^2 [1 - (1-2p)^2]))/2`, and at the maximising `p = 1/2`
    # the `(1-2p)^2` terms vanish and `xi` collapses to exactly `e_x + e_z`.
    #
    # Their own arithmetic settles it. The paper states `I(A:E) = g * Q_Bob * h(e_x + e_z) = 9.1e-4`
    # at `g = 2.57`, `Q_Bob = 0.3%`, `e_x = e_z = 0.8%`. Reproducing that number:
    #
    #     h(e_x + e_z) = h(0.016)  ->  9.1248e-4   MATCHES to the stated precision
    #     h(e)         = h(0.006)  ->  4.0798e-4   off by 2.2x
    #     h(e_x)       = h(0.008)  ->  5.1828e-4   off by 1.8x
    #
    # Only the sum reproduces it. A pooled single rate understates Eve's information by ~2.24x and
    # OVERSTATES the secrecy capacity by 26% at the paper's own operating point - the third
    # optimistic-security error found in this one function, after the 0.11 threshold and the omitted
    # detector efficiency. All three erred in the same direction, which is worth noticing: a
    # simplification that makes a demo work is not a neutral coin flip.
    #
    # `2 * e_check` IS JUSTIFIED HERE AND WOULD NOT BE IN GENERAL. The paper does not assume
    # `e_x == e_z`; it measures and plots them separately. Doubling is legitimate in this simulator
    # only because nothing in `intrinsic_qber` distinguishes the bases - misalignment, dark counts
    # and Eve's intercept-resend all act symmetrically - so the two basis rates have the same
    # expectation by construction. A channel model with basis-dependent asymmetry would have to
    # carry both rates. Note also that `e` in the FIRST term is a different measurement: Bob's
    # bit-error rate on the return leg (0.6% in the paper), against 0.8% per basis on the check.
    error_upper = serfling_upper(error_rate, n_check, n_msg)
    eve_arg = min(0.5, 2.0 * error_upper)          # h(e_x + e_z); the sum saturates at 1/2
    secrecy = 1.0 - _h2(error_upper) - g * _h2(eve_arg)

    # Attribute the abort, now THREE ways. A statistics-limited abort says nothing about the channel:
    # the sample was too small to certify anything, and reporting that as either "eavesdropper" or
    # "link budget" would be an accusation the data does not support. Checked first because it
    # dominates - if the block cannot certify the CLEAN channel's own error rate, nothing downstream
    # of the check is evidence about anything.
    clean = replace(params, eve_intercept_fraction=0.0)
    e_clean = intrinsic_qber(clean)
    secrecy_clean = 1.0 - _h2(e_clean) - g * _h2(min(0.5, 2.0 * e_clean))
    # Would this block certify the clean channel, given a PERFECT sample of it? If even that fails,
    # the binding constraint is the number of carriers rather than the physics.
    e_clean_up = serfling_upper(e_clean, n_check, n_msg)
    stats_limited = (1.0 - _h2(e_clean_up)
                     - g * _h2(min(0.5, 2.0 * e_clean_up))) <= 0.0 < secrecy_clean
    link_limited = secrecy_clean <= 0.0

    common = dict(message_bits=n_msg, check_bits=n_check, loss_db=round(loss_db, 3),
                  eve_gain_g=round(g, 3), link_budget_limited=link_limited,
                  error_rate_upper=round(error_upper, 5), statistics_limited=stats_limited)

    if secrecy <= 0.0:
        # Eve is blamed only for an abort that is neither link-limited nor statistics-limited.
        return QSDCResult(None, not (link_limited or stats_limited), round(error_rate, 5),
                          round(secrecy, 5), **common)

    # above zero secrecy capacity: LDPC frame correction recovers the block exactly
    return QSDCResult(message, False, round(error_rate, 5), round(secrecy, 5), **common)
