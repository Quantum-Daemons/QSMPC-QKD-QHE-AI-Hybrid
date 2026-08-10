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
    secrecy_capacity: float        # Cs = 1 - h(e) - g h(e), clamped at 0 for reporting
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

    loss_db = params.distance_km * params.fiber_loss_db_per_km
    g = 10.0 ** (loss_db / 10.0)
    secrecy = 1.0 - _h2(error_rate) - g * _h2(error_rate)

    # Attribute the abort. Re-evaluate the identical link with Eve removed: if it still cannot
    # close, the limit is the link budget and calling it "eavesdrop detected" accuses a clean
    # channel. Only an abort the link would NOT have suffered on its own is Eve's.
    clean = replace(params, eve_intercept_fraction=0.0)
    secrecy_clean = 1.0 - (1.0 + g) * _h2(intrinsic_qber(clean))
    link_limited = secrecy_clean <= 0.0

    if secrecy <= 0.0:
        return QSDCResult(None, not link_limited, round(error_rate, 5), 0.0,
                          n_msg, n_check, round(loss_db, 3), round(g, 3), link_limited)

    # above zero secrecy capacity: LDPC frame correction recovers the block exactly
    return QSDCResult(message, False, round(error_rate, 5), round(secrecy, 5),
                      n_msg, n_check, round(loss_db, 3), round(g, 3), link_limited)
