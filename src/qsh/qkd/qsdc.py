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
    # `g` IS Q_Eve/Q_Bob, AND IT INCLUDES DETECTOR EFFICIENCY. Qi, Lo et al. define it twice - "g
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
    secrecy = 1.0 - _h2(error_rate) - g * _h2(error_rate)

    # Attribute the abort. Re-evaluate the identical link with Eve removed: if it still cannot
    # close, the limit is the link budget and calling it "eavesdrop detected" accuses a clean
    # channel. Only an abort the link would NOT have suffered on its own is Eve's.
    clean = replace(params, eve_intercept_fraction=0.0)
    secrecy_clean = 1.0 - (1.0 + g) * _h2(intrinsic_qber(clean))
    link_limited = secrecy_clean <= 0.0

    if secrecy <= 0.0:
        return QSDCResult(None, not link_limited, round(error_rate, 5), round(secrecy, 5),
                          n_msg, n_check, round(loss_db, 3), round(g, 3), link_limited)

    # above zero secrecy capacity: LDPC frame correction recovers the block exactly
    return QSDCResult(message, False, round(error_rate, 5), round(secrecy, 5),
                      n_msg, n_check, round(loss_db, 3), round(g, 3), link_limited)
