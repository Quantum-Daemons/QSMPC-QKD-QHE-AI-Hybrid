"""Finance AML / cross-bank fraud detection use case (deep).

Two banks want to find accounts that *both* have flagged as suspicious, then privately
score a transaction against a shared fraud model -- over a PQC channel with simulated QKD
key material folded in.

The matching is a semi-honest DH-OPRF PSI on RFC 9497's OPRF over ristretto255
(:mod:`qsh.qsmpc.psi`): Bank A blinds its IDs, Bank B evaluates them under a key fresh to the run and
sends its own outputs, and Bank A learns the intersection and the size of Bank B's list while Bank B
learns the size of Bank A's. Neither can test IDs against the other's list offline. It has no
malicious security, it is not post-quantum (a discrete-log assumption), the two banks are notional
parties in one process, and its messages are sealed under the PQC session key.

Maps to: QSMPC×QKD (private set intersection) + QHE/FHE (private inference) on a
PQC-first hybrid transport. Data: the bundled PaySim simulator sample (synthetic transactions,
CC BY-SA 4.0, ``data/finance/paysim_sample.tsv``). The fraud model reads four of its eight columns,
the ones ``registry.PAYSIM_MODEL_FEATURES`` names and defines; none of them is built from PaySim's
balance columns, which the dataset's description says must not be used for fraud detection because
fraudulent transactions are cancelled (``registry.PAYSIM_EXCLUDED_FEATURES``).

Run:  ``python -m qsh.orchestration.pipeline --usecase finance_aml``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...fhe.ckks_inference import LinearModel
from ...orchestration.pipeline import Pipeline, standard_stages
from ...orchestration.stages import Context


def _fraud_model(seed: int = 0, n_tx: int | None = None
                 ) -> tuple[LinearModel, np.ndarray, dict, np.ndarray, np.ndarray]:
    """Train the XGBoost fraud model and **distil it to a CKKS-friendly linear
    surrogate** for the encrypted inference.

    A tree ensemble is widely used for AML and fraud scoring, but it is not CKKS-native:
    every split is a comparison, which CKKS evaluates only through polynomial approximation. Under
    TFHE trees do run - Effendi and Chattopadhyay ran a quantized XGBoost with graph features on
    downsized AMLworld HI-Small samples under TFHE (Concrete ML), with encrypted inference 15,038 to
    165,917 times slower than plaintext on their CPU (SPACE 2024, LNCS, pp. 80-105,
    doi:10.1007/978-3-031-80408-3_6; arXiv:2411.02926, Tables 4 and 6). So this deployment trains the
    real XGBoost (the "real model"), then fits a logistic **distillation** (`LinearModel`) to
    XGBoost's own decisions; the encrypted CKKS inference runs the distilled linear model, which keeps
    the CKKS step a single packed dot product and the attribution exact linear-SHAP.

    Both models read the four columns of ``registry.PAYSIM_MODEL_FEATURES``, selected by name from
    the loader's eight. Returns ``(surrogate, flagged_tx, model_meta, baseline, flagged_raw)``:
    ``flagged_tx`` is the scored transaction in those four columns, and ``flagged_raw`` the SAME
    transaction in all eight of the loader's columns. The finance QCNN reads AMLworld since Round 86's
    batch 6 and takes its row from ``_qcnn_features`` instead.
    """
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from ...config import get_config
    from ...data.loaders import load_finance_fraud
    from ...models.registry import PAYSIM_MODEL_FEATURES, PAYSIM_SAMPLE_COLUMNS

    X_all, y = load_finance_fraud(n=n_tx or get_config().data.aml_transactions, seed=seed)
    keep = [PAYSIM_SAMPLE_COLUMNS.index(name) for name in PAYSIM_MODEL_FEATURES]
    X = X_all[:, keep]
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
    flagged_at = int(np.flatnonzero(flag_mask)[0]) if flag_mask.any() else int(np.flatnonzero(y == 1)[0])
    meta = {"real_model": "XGBoost (n_estimators=80, depth=4)", "xgb_test_accuracy": round(xgb_acc, 4),
            "encrypted_surrogate": "logistic distillation", "distillation_fidelity": round(fidelity, 4),
            "features": list(PAYSIM_MODEL_FEATURES)}
    # E[x] over the training matrix - the SHAP reference point. Returned rather than stuffed into
    # `meta`, which is surfaced as a JSON artifact and would not survive a numpy array.
    return (LinearModel(weight=w.astype(float), bias=b), X[flagged_at].astype(float), meta,
            X.mean(axis=0).astype(float), X_all[flagged_at].astype(float))


#: The finance QCNN's held-out sample, a byte-identical copy of `web/public/data/finance/qcnn_heldout.json` (the full
#: deployment's image ships `data/` and not `web/public/`): 8,192 AMLworld test-window transactions at the circuit's eight
#: inputs, raw, with their labels. Written by `scripts/export_finance_qcnn_amlworld.py`; a test fails the moment the two
#: copies differ by a byte.
QCNN_HELDOUT = Path(__file__).resolve().parents[4] / "data" / "finance" / "qcnn_heldout.json"


def _qcnn_features(seed: int) -> tuple[np.ndarray, str] | None:
    """An AMLworld test-window transaction in the finance QCNN artifact's own columns, raw, and which one it is.

    Since Round 86's batch 6 (the owner's choice of 2026-11-26) the finance QCNN is the AMLworld circuit of the
    pre-registered comparison, which reads eight of AMLworld's 114 inputs; this deployment's other finance models read
    PaySim's four columns, so the QCNN scores a transaction of its own, drawn by the run's seed from the held-out sample,
    and the run says which. Until then it read the PaySim transaction the surrogate flagged, in PaySim's four columns.
    The artifact's `feature_names` must be the sample's, in order, and its `input_space` raw; otherwise it is refused.
    None when no artifact ships.
    """
    import json

    from ...qhe.qcnn_trained import load_weights

    w = load_weights("finance_aml")
    if not w:
        return None
    held = json.loads(QCNN_HELDOUT.read_text())
    if w.get("input_space") != held["input_space"] or list(w["feature_names"]) != list(held["feature_names"]):
        raise ValueError(f"the finance QCNN reads {w['feature_names']} ({w.get('input_space')}); the held-out sample "
                         f"ships {held['feature_names']} ({held['input_space']})")
    i = int(np.random.default_rng(seed).integers(len(held["rows"])))
    note = (f"AMLworld test-window transaction {i + 1} of {len(held['rows'])} of the QCNN's held-out sample "
            f"({held['licence']}), label {held['labels'][i]}: the QCNN reads AMLworld's inputs, so it scores a "
            "transaction of its own, not the PaySim one the surrogate flagged")
    return np.asarray(held["rows"][i], dtype=float), note


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
    model, flagged, model_meta, baseline, flagged_raw = _fraud_model(
        seed, n_tx=config.get("aml_transactions"))
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
    # The model's four columns, by their AML-style names, for the XAI / SHAP attribution chart; what
    # each one is in PaySim's terms is in registry.PAYSIM_MODEL_FEATURES.
    from ...models.registry import PAYSIM_MODEL_FEATURES

    ctx.data["feature_names"] = list(PAYSIM_MODEL_FEATURES)
    # The trained QCNN's row, only where it will be read: a run that selects the QCNN (as `usecases/medical_fl` does for its
    # own artifact). Since Round 86's batch 6 an AMLworld transaction of its own, in its own raw columns (`_qcnn_features`).
    if ctx.config.get("inference") in ("qcnn", "qcnn-ckks"):
        qcnn_row = _qcnn_features(seed)
        if qcnn_row is not None:
            ctx.data["qcnn_features"], ctx.data["qcnn_row_note"] = qcnn_row
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
