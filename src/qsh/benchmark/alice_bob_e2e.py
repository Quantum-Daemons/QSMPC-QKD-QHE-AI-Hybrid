"""Alice <-> Bob two-party end-to-end orchestration benchmark.

The canonical two-party setting from the report (Zhang et al., *Experimental SMPC from
Quantum Oblivious Transfer with Bit Commitment*, npj Quantum Information 2026): two
organisations compute over each other's private data without a trusted third party. This
wires the real primitives end to end with per-party timing:

    Bob   : publishes a hybrid-KEM public key (responder keypair)
    QKD   : BB84 over a 50 km fibre channel -> shared QKD key (eavesdrop check via QBER)
    Alice : initiates the PQC handshake (ML-KEM + X25519), QKD key mixed in
    Bob   : derives the same PQC+QKD session
    Alice : keyed-hash PSI of her blacklist against Bob's suspicious set (no raw data shared)
    Bob   : private (CKKS) inference on a flagged record

Reuses :mod:`~qsh.pqc.hybrid_kem`, :mod:`~qsh.pqc.transport`, :mod:`~qsh.qkd.bb84`,
:mod:`~qsh.qkd.channel`, :mod:`~qsh.qsmpc.qot`, :mod:`~qsh.fhe.ckks_inference`, and scores
the run with :mod:`~qsh.benchmark.security`. No raw data crosses the wire — only the PSI
result and an encrypted inference output.
"""

from __future__ import annotations

import time

import numpy as np


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


def run_alice_bob_e2e(*, seed: int = 0, n_shared: int = 8) -> dict:
    """Run the two-party protocol and return per-party timing + results + scores."""
    from ..fhe.ckks_inference import LinearModel, encrypted_linear_inference
    from ..pqc.hybrid_kem import HybridKEMRecipient
    from ..pqc.transport import establish_initiator, establish_responder
    from ..qkd.bb84 import QKDParams, derive_shared_key, run_bb84
    from ..qkd.channel import ChannelParams
    from ..qsmpc.qot import private_set_intersection
    from .security import compute_security_score

    rng = np.random.default_rng(seed)
    timing: dict[str, float] = {}

    # --- Phase 1: Bob publishes a hybrid-KEM public key (responder keygen) ---
    t = time.perf_counter()
    bob = HybridKEMRecipient()
    timing["bob_keygen_ms"] = _ms(t)

    # --- Phase 2: QKD (BB84, 50 km fibre) -> shared QKD key, eavesdrop check via QBER ---
    t = time.perf_counter()
    # `qkd_params=QKDParams()` - the benchmark must measure the model the demo ships, not the
    # legacy branch it reaches by omission.
    qkd = run_bb84(40_000, ChannelParams(distance_km=50), seed=seed,
                   qkd_params=QKDParams())
    qkd_key = derive_shared_key(qkd, 32) if qkd.secure_key_bits >= 256 else None
    timing["qkd_negotiation_ms"] = _ms(t)

    # --- Phase 3: PQC handshake (Alice initiates, Bob responds); QKD mixed in ---
    t = time.perf_counter()
    ct, chan_alice = establish_initiator(bob.public_key, qkd_key=qkd_key)
    timing["alice_pqc_handshake_ms"] = _ms(t)
    t = time.perf_counter()
    chan_bob = establish_responder(bob, ct, qkd_key=qkd_key)
    timing["bob_pqc_handshake_ms"] = _ms(t)
    session_agree = chan_alice.session_key == chan_bob.session_key
    session_key = chan_alice.session_key

    # --- Phase 4: keyed-hash PSI (Alice blacklist ∩ Bob suspicious set), no raw share ---
    shared = {f"ACCT-{int(x)}" for x in rng.integers(10**6, 10**7, n_shared)}
    alice_blacklist = sorted(shared | {f"A-{int(x)}" for x in rng.integers(10**6, 10**7, 200)})
    bob_suspicious = sorted(shared | {f"B-{int(x)}" for x in rng.integers(10**6, 10**7, 180)})
    t = time.perf_counter()
    psi = private_set_intersection(alice_blacklist, bob_suspicious, session_key)
    timing["psi_ms"] = _ms(t)

    # --- Phase 5: Bob runs private (CKKS) inference on a flagged record ---
    t = time.perf_counter()
    w = rng.normal(size=(2, 8)); b = rng.normal(size=2)
    out = encrypted_linear_inference(LinearModel(w, b), rng.normal(size=8))
    timing["private_inference_ms"] = _ms(t)

    sec = compute_security_score(
        security_model="computational + QKD (two-party, TTP-free)",
        pqc_hybrid=True, kem=bob.public_key.kem_name.lower(),
        qkd_key_bits=qkd.secure_key_bits, qber=qkd.qber,
    )
    bob.free()

    total_ms = round(sum(timing.values()), 3)
    return {
        "parties": ["Alice", "Bob"],
        "phases": [
            {"name": "Bob key generation", "party": "Bob", "ms": timing["bob_keygen_ms"]},
            {"name": "QKD (BB84)", "party": "Alice+Bob", "ms": timing["qkd_negotiation_ms"]},
            {"name": "PQC handshake (initiate)", "party": "Alice",
             "ms": timing["alice_pqc_handshake_ms"]},
            {"name": "PQC handshake (respond)", "party": "Bob",
             "ms": timing["bob_pqc_handshake_ms"]},
            {"name": "Private set intersection", "party": "Alice", "ms": timing["psi_ms"]},
            {"name": "Private inference (CKKS)", "party": "Bob",
             "ms": timing["private_inference_ms"]},
        ],
        "timing_ms": timing,
        "total_ms": total_ms,
        "session_key_agreement": bool(session_agree),
        "kem": bob.public_key.kem_name,
        "qber": round(qkd.qber, 4),
        "qkd_secure_key_bits": int(qkd.secure_key_bits),
        "qkd_mixed": qkd_key is not None,
        "psi_intersection_size": psi.intersection_size,
        "psi_ground_truth": len(shared),
        "psi_correct": psi.intersection_size == len(shared),
        "private_inference_class": out["prediction"],
        "private_inference_error": out["max_abs_error"],
        "security_score": sec["total"],
        "security_components": sec["components"],
        "security_summary": sec["summary"],
        "ttp_free": True,
        "raw_data_shared": False,
    }
