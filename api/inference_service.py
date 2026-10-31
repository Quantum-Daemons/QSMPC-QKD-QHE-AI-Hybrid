"""Lightweight inference + crypto demo endpoints backing the SPA.

The public demo pushes heavy compute to the browser (ONNX/WASM); this service offers
server-side counterparts for the full local deployment: a trained-model prediction
and a live PQC hybrid-KEM handshake the SPA can visualise.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from qsh.config import get_config


@lru_cache(maxsize=4)
def _fraud_model():
    from sklearn.linear_model import LogisticRegression

    from qsh.data.loaders import load_finance_fraud

    from qsh.models.registry import PAYSIM_MODEL_FEATURES, PAYSIM_SAMPLE_COLUMNS

    X, y = load_finance_fraud(n=3000, seed=0)
    # The four columns the page's model reads (Round 84): device_risk and geo_mismatch are built from
    # PaySim's balance columns, which the dataset's description says must not be used for fraud detection.
    X = X[:, [PAYSIM_SAMPLE_COLUMNS.index(n) for n in PAYSIM_MODEL_FEATURES]]
    return LogisticRegression(max_iter=500).fit(X, y)


def predict_fraud(features: list[float]) -> dict:
    """Score a transaction with the trained fraud model.

    Takes the model's four columns, or all eight sample columns (the four are then selected by name).
    Any other width is refused: padding or truncating, as this did before Round 84, would silently read
    the wrong columns now that the model takes four of the eight."""
    from qsh.models.registry import PAYSIM_MODEL_FEATURES, PAYSIM_SAMPLE_COLUMNS

    clf = _fraud_model()
    x = np.asarray(features, dtype=float).reshape(1, -1)
    if x.shape[1] == len(PAYSIM_SAMPLE_COLUMNS):
        x = x[:, [PAYSIM_SAMPLE_COLUMNS.index(n) for n in PAYSIM_MODEL_FEATURES]]
    elif x.shape[1] != len(PAYSIM_MODEL_FEATURES):
        raise ValueError(f"expected {len(PAYSIM_MODEL_FEATURES)} features ({', '.join(PAYSIM_MODEL_FEATURES)}) "
                         f"or all {len(PAYSIM_SAMPLE_COLUMNS)} sample columns, got {x.shape[1]}")
    proba = float(clf.predict_proba(x)[0, 1])
    return {"fraud_probability": round(proba, 4), "prediction": int(proba >= 0.5)}


def pqc_handshake_demo() -> dict:
    """Run a hybrid ML-KEM-768 + X25519 handshake; return sizes + agreement proof."""
    from qsh.pqc.hybrid_kem import HybridKEMRecipient, encapsulate

    with HybridKEMRecipient() as bob:
        pub = bob.public_key
        ct, ss_sender = encapsulate(pub)
        ss_recipient = bob.decapsulate(ct)
    return {
        "kem": pub.kem_name,
        "classical": "X25519",
        "mlkem_public_bytes": len(pub.mlkem_public),
        "ciphertext_bytes": len(ct.mlkem_ciphertext),
        "shared_secret_bytes": len(ss_sender),
        "agreement": ss_sender == ss_recipient,
        "shared_secret_prefix": ss_sender.hex()[:16],
    }


def qkd_demo(distance_km: float = 50.0, eve: float = 0.0) -> dict:
    """Run a small SIMULATED BB84 session (no quantum channel); return QBER/SKR (and Eve detection)."""
    from qsh.qkd.bb84 import QKDParams, run_bb84
    from qsh.qkd.channel import ChannelParams

    # `qkd_params=QKDParams()`: without it this endpoint returns the legacy non-decoy figure while
    # /validations labels it a per-pulse decoy bound. Round 45 fixed the browser's copy of this
    # payload and not the server's.
    # The block size is READ, not spelled, for the same reason as
    # `benchmark/alice_bob_e2e.py`: `demo_alice_bob_pulses` was raised to 400_000 in Round 46
    # because 40_000 leaves the decoy bound at **79** secure bits, and the raise reached the browser
    # only. (94 was the count at Round 46's mu = 0.5 / nu = 0.1; Round 47's finite-key optimum
    # 0.3416 / 0.1417 makes it 79. Still far below the 256 a session key needs.) This endpoint
    # backs the same panel, so a literal here reintroduces the split it fixed.
    r = run_bb84(get_config().web.demo_alice_bob_pulses,
                 ChannelParams(distance_km=distance_km, eve_intercept_fraction=eve),
                 qkd_params=QKDParams(),
                 seed=1)
    return {
        "distance_km": distance_km, "eve_fraction": eve,
        "qber": round(r.qber, 4),
        # NINE decimals, matching `web/src/lib/api.ts` which feeds the same panel. This was
        # five, so one panel showed 0.00198 from Python and 0.001977 from the browser for
        # one quantity; then six, until Round 75, which held the satellite downlink's 1.723e-5
        # to two significant figures.
        "skr_per_pulse": round(r.secret_key_rate, 9),
        "secure_key_bits": r.secure_key_bits,
        # The block that count is the composable finite key of (Round 80), so the page can say so;
        # six significant figures, as the browser's twin emits it. 0 when the run computed no
        # finite key - an abort above `pp_threshold`, or nothing sifted - where `run_bb84` returns
        # `block_pulses=None`, as `api.ts` maps the browser's absent block (`?? 0`). Formatting the
        # None raised TypeError, an HTTP 500 for every intercept fraction from 0.45 up at 50 km.
        "block_pulses": float(f"{r.block_pulses:.6g}") if r.block_pulses is not None else 0.0,
        # Config, not a literal. This was 0.08 while the browser used
        # security_scoring.qber_secure_threshold (0.11), so full mode and demo mode returned
        # OPPOSITE eavesdropper verdicts for any QBER in the 0.08-0.11 band.
        "eavesdropper_likely": r.qber > get_config().security_scoring.qber_secure_threshold,
    }
