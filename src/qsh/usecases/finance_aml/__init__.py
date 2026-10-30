"""Finance AML / cross-bank fraud detection use case (deep).

Two banks want to find accounts that *both* have flagged as suspicious, then privately
score a transaction against a shared fraud model -- over a PQC channel with simulated QKD
key material folded in.

What the matching does NOT do is keep the lists private from each other: it is a keyed-hash
PSI under a session key both banks hold, so over this small ID space either bank can hash
every candidate ID and recover the other's list (see :func:`qsh.qsmpc.qot.private_set_intersection`).
What holds is that no third party on the channel sees either list. A production PSI would use an
OPRF-based protocol.

Maps to: QSMPC×QKD (private set intersection) + QHE/FHE (private inference) on a
PQC-first hybrid transport. Data: synthetic PaySim/AMLSim-style transactions
(privacy-safe; swap in real CSVs for production).

Run:  ``python -m qsh.orchestration.pipeline --usecase finance_aml``
"""

from __future__ import annotations

import numpy as np

from ...fhe.ckks_inference import LinearModel
from ...orchestration.pipeline import Pipeline, standard_stages
from ...orchestration.stages import Context


def _fraud_model(seed: int = 0, n_tx: int | None = None) -> tuple[LinearModel, np.ndarray, dict, np.ndarray]:
    """Train the REAL 2026-industry fraud model (XGBoost) and **distil it to a CKKS-friendly linear
    surrogate** for the encrypted inference.

    XGBoost/GBM is the de-facto AML/fraud-detection standard, but a tree ensemble is not FHE-feasible
    (trees = secure comparisons). So the full deployment trains the real XGBoost (the "real model"),
    then fits a logistic **distillation** (`LinearModel`) to XGBoost's own decisions — the encrypted
    CKKS inference runs the distilled linear model, faithfully approximating the real GBM while staying
    FHE-feasible + keeping exact linear-SHAP. Returns ``(surrogate, flagged_tx, model_meta)``.
    """
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from ...config import get_config
    from ...data.loaders import load_finance_fraud

    X, y = load_finance_fraud(n=n_tx or get_config().data.aml_transactions, seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    gbm = xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.2, subsample=0.9,
                            eval_metric="logloss", n_jobs=2, random_state=seed)
    gbm.fit(Xtr, ytr)
    xgb_acc = float((gbm.predict(Xte) == yte).mean())
    # distillation: the linear student mimics XGBoost's decisions (train on the teacher's labels)
    teacher = gbm.predict(X)
    student = LogisticRegression(max_iter=500, class_weight="balanced").fit(X, teacher)
    fidelity = float((student.predict(X) == teacher).mean())     # how faithfully the surrogate tracks XGBoost
    w = np.vstack([np.zeros_like(student.coef_[0]), student.coef_[0]])
    b = np.array([0.0, float(student.intercept_[0])])
    # a known-fraud transaction that BOTH the real model and the surrogate flag (so the encrypted run flags it)
    flag_mask = (y == 1) & (gbm.predict(X) == 1) & (student.predict(X) == 1)
    flagged_tx = X[flag_mask][0] if flag_mask.any() else X[y == 1][0]
    meta = {"real_model": "XGBoost (n_estimators=80, depth=4)", "xgb_test_accuracy": round(xgb_acc, 4),
            "encrypted_surrogate": "logistic distillation", "distillation_fidelity": round(fidelity, 4)}
    # E[x] over the training matrix - the SHAP reference point. Returned rather than stuffed into
    # `meta`, which is surfaced as a JSON artifact and would not survive a numpy array.
    return (LinearModel(weight=w.astype(float), bias=b), flagged_tx.astype(float), meta,
            X.mean(axis=0).astype(float))


def _bank_blacklists(seed: int = 0) -> tuple[list[str], list[str], set[str]]:
    """Two banks' flagged-account sets with a known overlap (shared fraud ring). Sizes from config."""
    from ...config import get_config

    d = get_config().data
    rng = np.random.default_rng(seed)
    shared = {f"ACCT-{rng.integers(10**6, 10**7)}" for _ in range(d.aml_shared_accounts)}
    a = set(shared) | {f"A-{rng.integers(10**6, 10**7)}" for _ in range(d.aml_bank_a_accounts)}
    b = set(shared) | {f"B-{rng.integers(10**6, 10**7)}" for _ in range(d.aml_bank_b_accounts)}
    return sorted(a), sorted(b), shared


def build(config: dict | None = None) -> tuple[Pipeline, Context]:
    config = dict(config or {})
    seed = config.get("seed", 0)
    model, flagged, model_meta, baseline = _fraud_model(seed, n_tx=config.get("aml_transactions"))
    bank_a, bank_b, shared = _bank_blacklists(seed)

    ctx = Context(
        usecase="finance_aml",
        config={"qsmpc_op": "psi", "inference": "ckks", "seed": seed, **config},
    )
    ctx.data["set_a"] = bank_a
    ctx.data["set_b"] = bank_b
    ctx.data["linear_model"] = model
    ctx.data["features"] = flagged
    # SHAP reference point E[x], NOT the zero vector. `explain_baseline` was READ by stages.py and
    # written by nothing, so every Python-side attribution was measured against a transaction that
    # does not occur: PaySim's log-scale `amount` has mean ~11.5, so its bar was inflated against a
    # reference of log-amount zero and the true top driver reported a contribution of exactly zero.
    # Completeness cannot catch this - it holds for ANY baseline - so the fix is to supply the right
    # one, not to add another self-check.
    ctx.data["explain_baseline"] = baseline
    ctx.data["_finance_model"] = model_meta        # real XGBoost + distillation fidelity (for surfacing)
    # human-readable AML feature names for the XAI / SHAP attribution chart (8 features)
    ctx.data["feature_names"] = [
        "amount", "velocity_24h", "cross_border", "night_tx",
        "new_payee", "amount_zscore", "device_risk", "geo_mismatch",
    ]
    ctx.data["_ground_truth_shared"] = len(shared)
    return Pipeline(name="finance-aml", stages=standard_stages()), ctx


def run_demo(config: dict | None = None) -> dict:
    """Run the use case and return a compact result summary."""
    pipe, ctx = build(config)
    pipe.run(ctx, out_dir="results/")
    m = pipe.bus.metrics()
    return {
        "shared_fraud_accounts_found": m.get("psi_intersection_size"),
        "ground_truth_shared": ctx.data["_ground_truth_shared"],
        # Canonical key, not one backend's - see stages.py. Reading `fhe_prediction`
        # alone meant a QCNN run computed a class and this surfaced None.
        "private_fraud_prediction": ctx.artifacts.get("private_prediction",
                 ctx.artifacts.get("fhe_prediction")),
        "private_fraud_prediction_backend": ctx.artifacts.get("private_prediction_backend"),
        "model": ctx.data.get("_finance_model"),     # real XGBoost + distillation fidelity
        "explanation": ctx.artifacts.get("explanation"),
        "qber": m.get("qber"),
        "metrics": m,
    }
