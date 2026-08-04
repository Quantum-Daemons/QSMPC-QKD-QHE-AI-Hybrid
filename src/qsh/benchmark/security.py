"""Numeric security scoring for private-inference paths (0-100).

Turns the qualitative `security_model` string into a comparable score so the benchmark
can rank paths on security, not just speed. The components are grounded in:
  - NIST FIPS 203 ML-KEM categories (768 = cat 3 ~AES-192, 1024 = cat 5 ~AES-256);
  - HNDL resistance from the PQC-first hybrid (ML-KEM + X25519) + optional QKD key material;
  - information-theoretic (QOTP perfect hiding) vs computational (RLWE) security;
  - eavesdropper detection via QBER (BB84 secure threshold ~11%);
  - the ~0.1 bit/weight information leakage bound from the PRX "Quantum-Secure Multiparty
    Deep Learning" paper (``references/Quantum-Secure_Multiparty_Deep_Learning.pdf``).

The score is a transparent weighted sum, not a certification — it is for comparison/UX.
"""

from __future__ import annotations

# ML-KEM parameter set -> NIST security category.
_KEM_CATEGORY = {
    "ml-kem-512": 1, "ml-kem-768": 3, "ml-kem-1024": 5,
    "ml-kem-768+x25519": 3, "ml-kem-1024+x25519": 5,
}


def kem_category(kem_name: str | None) -> int:
    if not kem_name:
        return 3
    return _KEM_CATEGORY.get(kem_name.lower(), 3)


def compute_security_score(
    *,
    security_model: str,
    pqc_hybrid: bool = True,
    kem: str | None = "ml-kem-768+x25519",
    qkd_key_bits: int | None = None,
    qber: float | None = None,
    leakage_bits_per_weight: float | None = None,
) -> dict:
    """Return a 0-100 security score with a per-component breakdown.

    Component weights + thresholds come from the central config (config/defaults.py:
    SecurityScoringConfig) — single source of truth. Components (max): HNDL/PQC 30,
    security-model 30, key-strength 20, eavesdropper-detection 10, leakage-bound 10.
    """
    from ..config import get_config

    s = get_config().security_scoring
    it = "information-theoretic" in (security_model or "").lower()
    c: dict[str, float] = {}

    # HNDL resistance: PQC-first hybrid, with a real bonus for mixed-in QKD key material.
    #
    # This used to read `min(hndl_pqc_hybrid, hndl_pqc_hybrid + qkd_bonus)`, which is just
    # `hndl_pqc_hybrid` -- the bonus could never apply whenever pqc_hybrid was true, i.e. in every
    # normal run. Selecting bb84-decoy versus pqc-only changed the summary label and nothing else,
    # while docs/crypto-agility.md claimed every selector "genuinely changes the demo run's
    # metrics". A PQC-only hybrid now scores the cap minus the bonus and QKD-mixed reaches the cap,
    # so the ceiling is unchanged and the control is no longer inert.
    if not pqc_hybrid:
        c["hndl_pqc_hybrid"] = 0.0
    elif qkd_key_bits and qkd_key_bits >= 256:
        c["hndl_pqc_hybrid"] = s.hndl_pqc_hybrid
    else:
        c["hndl_pqc_hybrid"] = max(0.0, s.hndl_pqc_hybrid - s.qkd_bonus)

    # Security model: information-theoretic beats computational.
    c["security_model"] = s.info_theoretic if it else s.computational

    # Key strength from NIST category (cat5 -> 20, cat3 -> 16, cat1 -> 10).
    cat = kem_category(kem)
    c["key_strength"] = s.key_strength_by_category.get(cat, 16.0)

    # Eavesdropper detection: QBER below the BB84 secure threshold (~11%).
    if qber is None:
        c["eavesdropper_detection"] = 0.0
    elif qber < s.qber_secure_threshold:
        c["eavesdropper_detection"] = s.eavesdrop_secure_pts
    elif qber < s.qber_danger_threshold:
        c["eavesdropper_detection"] = s.eavesdrop_warn_pts
    else:
        c["eavesdropper_detection"] = 0.0  # attack detected -> not a secure key

    # Information-leakage bound (lower is better; PRX ~0.1 bit/weight is strong).
    if leakage_bits_per_weight is None:
        c["leakage_bound"] = s.leakage_pts_default_it if it else s.leakage_pts_default_comp
    elif leakage_bits_per_weight <= s.leakage_low:
        c["leakage_bound"] = s.leakage_pts_low
    elif leakage_bits_per_weight <= s.leakage_mid:
        c["leakage_bound"] = s.leakage_pts_mid
    else:
        c["leakage_bound"] = s.leakage_pts_high

    total = round(min(100.0, sum(c.values())), 1)
    return {
        "total": total,
        "components": {k: round(v, 1) for k, v in c.items()},
        "kem_category": cat,
        "summary": (
            f"{'information-theoretic' if it else 'computational'} "
            f"security, NIST cat-{cat} KEM, HNDL-resistant hybrid"
            + (", QKD-mixed" if qkd_key_bits and qkd_key_bits >= 256 else "")
        ),
    }
