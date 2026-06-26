"""Quantum Secure Direct Communication (QSDC) — block-transmission simulation.

QSDC (the report §6: QSDC / QT-QSS) transmits a *message* over the quantum channel
directly, without first distributing a key. Security comes from a two-step / DL04-style
block transmission with an interspersed **eavesdrop check**: a random subset of carriers
is sampled and their error rate estimated *before* the sensitive block is released. If the
error rate exceeds the secure threshold the transmission is aborted, so an eavesdropper
learns nothing about the message (it was never sent on a compromised channel).

This is the natural primitive for **satellite communications**: a free-space downlink where
re-keying round-trips are expensive, so sending the message directly (with an inline
eavesdrop check) is attractive. We simulate the *information structure* faithfully on a
:class:`~qsh.qkd.channel.ChannelParams` link; a real system adds decoy states, frame error
correction (LDPC) and privacy amplification.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    eavesdrop_detected: bool
    error_rate: float              # measured error on the sampled check carriers
    secrecy_capacity: float        # Wyner secure bits per carrier, max(0, 1 - 2 H2(e))
    message_bits: int
    check_bits: int
    threshold: float


def quantum_secure_direct_comm(
    message: bytes,
    params: ChannelParams | None = None,
    *,
    check_fraction: float = 0.25,
    threshold: float = 0.11,
    seed: int | None = None,
) -> QSDCResult:
    """Send ``message`` directly over the quantum channel with an inline eavesdrop check.

    The channel error rate (intrinsic + any eavesdropper) is estimated on a random
    ``check_fraction`` of carriers. Below ``threshold`` the message block is released and
    recovered (frame error correction modelled as exact); above it the transfer aborts.
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

    eavesdrop = error_rate > threshold
    if eavesdrop:
        return QSDCResult(None, True, round(error_rate, 5), 0.0, n_msg, n_check, threshold)

    secrecy = max(0.0, 1.0 - 2.0 * _h2(error_rate))  # Wyner wiretap secure fraction
    # below threshold: LDPC frame correction recovers the block exactly
    return QSDCResult(message, False, round(error_rate, 5), round(secrecy, 5),
                      n_msg, n_check, threshold)
