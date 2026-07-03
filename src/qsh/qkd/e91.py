"""E91 entanglement-based QKD (CHSH) — simulation.

An entanglement-based alternative to prepare-and-measure BB84 (the OptiQKD / AML-QKD papers evaluate BB84,
E91 and COW side by side): a shared entangled source distributes pairs, Alice and Bob each measure in a
randomly chosen basis, a subset of the outcomes tests the CHSH inequality (S > 2 certifies genuine
entanglement / no eavesdropper), and the matching-basis outcomes form the key. Modelled with a Werner state
of visibility ``V`` degraded by channel loss + noise: the CHSH value is ``S = 2*sqrt(2)*V`` (Tsirelson bound
scaled by V) and the intrinsic QBER is ``(1 - V)/2``.

Honest label: a simulation, not quantum hardware. Reuses :class:`~qsh.qkd.bb84.QKDResult` + ``_h2`` and
:mod:`~qsh.qkd.channel` so the AI optimizer (:mod:`~qsh.qkd.optimize`) can treat BB84 and E91 uniformly.
"""

from __future__ import annotations

import numpy as np

from .bb84 import QKDResult, _h2
from .channel import ChannelParams, intrinsic_qber

_TSIRELSON = 2.0 * np.sqrt(2.0)   # maximum CHSH value (a fixed physical constant, not a tunable)


def chsh_s(visibility: float) -> float:
    """CHSH S-parameter for a Werner state of the given visibility (``S = 2*sqrt(2)*V``; S > 2 => Bell violation)."""
    return float(_TSIRELSON * float(np.clip(visibility, 0.0, 1.0)))


def run_e91(n_pairs: int = 100_000, params: ChannelParams | None = None, *, visibility: float = 0.97,
            basis_prob: float = 0.5, error_correction_eff: float = 1.16, seed: int | None = None,
            pp_threshold: float = 0.11) -> QKDResult:
    """Simulate an E91 session over ``n_pairs`` entangled pairs.

    The state visibility is degraded by the channel's intrinsic error (loss/misalignment/dark counts) to an
    effective ``V_eff``; ``S = 2*sqrt(2)*V_eff`` and ``QBER = (1 - V_eff)/2``. Matching-basis coincidences
    (ratio ``p^2 + (1-p)^2`` for basis bias ``p``) form the sifted key; the BBM92/GLLP secure fraction
    ``1 - H2(e) - f*H2(e)`` gives the SKR. A CHSH ``S <= 2`` (no Bell violation) or a QBER above
    ``pp_threshold`` aborts with no secure key.
    """
    params = params or ChannelParams()
    rng = np.random.default_rng(seed)
    eta = params.transmittance()
    p = float(np.clip(basis_prob, 0.01, 0.99))
    v_eff = float(np.clip(visibility * (1.0 - intrinsic_qber(params)), 0.0, 1.0))   # channel-degraded visibility
    s = chsh_s(v_eff)
    qber = float(min(0.5, (1.0 - v_eff) / 2.0))
    detected = rng.random(n_pairs) < eta                                   # coincidence detection (lossy arm)
    same_basis = rng.random(n_pairs) < (p * p + (1.0 - p) * (1.0 - p))     # matching-basis sift
    sift = detected & same_basis
    n_sift = int(np.count_nonzero(sift))
    key = (rng.random(max(n_sift, 0)) < 0.5).astype(int)                   # the sifted key bits (for demos)
    if s <= 2.0 or qber > pp_threshold or n_sift == 0:
        return QKDResult(n_sift, qber, 0.0, 0, key, params)                # no Bell violation / aborted
    secure_fraction = max(0.0, 1.0 - _h2(qber) - error_correction_eff * _h2(qber))
    skr = secure_fraction * (n_sift / n_pairs)
    return QKDResult(sifted_bits=n_sift, qber=qber, secret_key_rate=skr,
                     secure_key_bits=int(secure_fraction * n_sift), sifted_key=key, params=params)
