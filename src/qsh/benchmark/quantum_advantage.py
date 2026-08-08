"""Quantum-advantage metric contrasting the QSMPC-FHE path with classical MPC-FHE.

The standout quantum win in this PoC is **communication**: Rate-1 QFHE has a
ciphertext/plaintext ratio ~1.3 vs classical CKKS ~10^3, alongside an information-theoretic
(vs computational) security model. This module turns those into a single tiered 0-10 score
so the benchmark can headline "quantum advantage" quantitatively.

It is a transparent weighted geometric mean of three ratios, not a hardware quantum-supremacy
claim (the quantum path is simulated).
"""

from __future__ import annotations

import math

from .metrics import BenchResult


def quantum_advantage(classical: BenchResult, quantum: BenchResult,
                      *, w_comm: float | None = None, w_latency: float | None = None,
                      w_security: float | None = None) -> dict:
    """Composite quantum-advantage score (0-10) of ``quantum`` over ``classical``.

    Weights, the log-scale, and the tier boundaries come from the central config
    (config/defaults.py: QuantumAdvantageConfig); explicit kwargs override per call.
    """
    from ..config import get_config

    qa = get_config().quantum_advantage
    w_comm = qa.w_comm if w_comm is None else w_comm
    w_latency = qa.w_latency if w_latency is None else w_latency
    w_security = qa.w_security if w_security is None else w_security

    c_ct = classical.ciphertext_plaintext_ratio
    q_ct = quantum.ciphertext_plaintext_ratio
    comm_ratio = (c_ct / q_ct) if (c_ct and q_ct and q_ct > 0) else 1.0

    lat_ratio = (classical.latency_ms / quantum.latency_ms) if (
        classical.latency_ms and quantum.latency_ms > 0) else 1.0

    sec_mult = qa.info_theoretic_multiplier if "information-theoretic" in (
        quantum.security_model or "").lower() else 1.0

    # weighted geometric mean -> log-scaled to 0-10 (≈10^3 comm gain ⇒ high score)
    composite = (comm_ratio ** w_comm) * (max(lat_ratio, 1e-6) ** w_latency) * (sec_mult ** w_security)
    raw = math.log10(composite + 1.0) * qa.log_scale + qa.log_offset
    score = max(0.0, min(10.0, raw))

    tier = ("dominant" if score >= qa.tier_dominant else "strong" if score >= qa.tier_strong
            else "moderate" if score >= qa.tier_moderate else "marginal")
    return {
        "score": round(score, 2),
        # Uncapped composite. A reader seeing 10.0 cannot tell a run that just reached the ceiling
        # from one far past it; this says which. Reported, never used for the tier.
        "score_raw": round(raw, 2),
        "clamped": raw > 10.0 or raw < 0.0,
        "tier": tier,
        "communication_ratio": round(comm_ratio, 1),
        "latency_ratio": round(lat_ratio, 2),
        "security_multiplier": sec_mult,
        "interpretation": (
            f"Quantum path is ~{comm_ratio:.0f}x more communication-compact"
            + (f" and {lat_ratio:.1f}x faster" if lat_ratio > 1 else "")
            + f"; {tier} advantage."
        ),
    }
