"""RETIRED composite "quantum-advantage" score (QSMPC-FHE path against classical MPC-FHE).

Retired from every page of the demo on 2026-10-23 and displayed nowhere. It is still computed in
the full deployment: ``GET /api/benchmark`` returns it (``api/main.py``) and the comparison stage
writes it into the run artifact (``qsh.orchestration.stages``). Removing it from both is deferred.

Why it is retired, so nobody quotes it. Its communication ratio divides a CKKS byte ratio by the
byte ratio of a classically SIMULATED state vector, which are not the same quantity. The Rate-1
QFHE paper it was motivated by (Chardouvelis, Doettling and Malavolta, TCC 2021, IACR ePrint
2020/1454) reports no concrete ciphertext/plaintext ratio, and its security rests on the quantum
hardness of LWE, so it is computational, not information-theoretic; classical FHE also reaches
rate 1 (Gentry and Halevi, TCC 2019; Brakerski, Doettling, Garg and Malavolta, TCC 2019). The
1.5 security multiplier below still fires on the QSMPC row's label, which keeps the substring the
scorer tests so no unrendered score moves, although no run here is information-theoretically
secure. The ``interpretation`` string it returns ("more communication-compact") is therefore not
a finding.

It is a weighted geometric mean of three ratios, not a hardware quantum-supremacy claim (the
quantum path is simulated).
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
