"""Medical federated learning + private inference use case (deep).

Several hospitals jointly train a diagnostic model without sharing patient data
(federated learning + quantum-safe secure aggregation), then a clinician runs a
private inference that never exposes the patient's record to the server.

Maps to: QHE/FHE (private inference) + QSMPC-style secure aggregation on a PQC+QKD
transport. Default data: Wisconsin breast-cancer (bundled, fast). Set
``config['dataset']='medmnist'`` for the heavier MedMNIST CNN variant locally.

Run:  ``python -m qsh.orchestration.pipeline --usecase medical_fl``
"""

from __future__ import annotations

import numpy as np

from ...orchestration.crypto_agility import CryptoRegistry
from ...orchestration.events import EventBus
from ...orchestration.pipeline import Pipeline
from ...orchestration.stages import (
    Context,
    ExplainabilityStage,
    PQCHandshakeStage,
    PrivateInferenceStage,
    QKDKeyAgreementStage,
    QuantumComparisonStage,
    ResultStage,
    SignatureStage,
    Stage,
)


class FederatedTrainingStage(Stage):
    """Train across virtual hospitals with quantum-safe secure aggregation."""

    name = "federated_training"

    def run(self, ctx: Context, bus: EventBus, reg: CryptoRegistry) -> None:
        from ...ai.federated import federated_train, partition

        X, y = ctx.data["X"], ctx.data["y"]
        n_hosp = ctx.config.get("hospitals", 4)
        is_cnn = ctx.config.get("model") == "cnn" and ctx.data.get("image") is not None
        use_qfl = bool(ctx.config.get("qfl")) and not is_cnn   # Quantum Federated Learning (hybrid VQC)
        rounds = ctx.config.get("rounds", 4 if use_qfl else (2 if is_cnn else 6))
        cut = int(len(X) * 0.8)
        parts = partition(X[:cut], y[:cut], n_hosp, iid=ctx.config.get("iid", False),
                          seed=ctx.config.get("seed", 0))
        if is_cnn:
            img = ctx.data["image"]
            bus.log(f"federated training a diagnostic CNN across {n_hosp} hospitals on real "
                    f"MedMNIST images (secure aggregation on)", self.name)
            mk = {"in_ch": int(img.shape[0]), "img_size": int(img.shape[1]),
                  "channels": (int(ctx.config.get("cnn_channels", 4)),), "hidden": 16,
                  "one_block": True, "pool": "avg"}
            res, model = federated_train(
                parts, kind="cnn", rounds=rounds, local_epochs=1, lr=0.05,
                eval_data=(X[cut:], y[cut:]), secure_aggregation=True, model_kwargs=mk,
                seed=ctx.config.get("seed", 0), return_model=True,
            )
            ctx.data["cnn_model"] = model               # the genuine model the FHE run will infer
        elif use_qfl:
            # Quantum Federated Learning: a hybrid quantum-classical model (PennyLane VQC head +
            # classical body) federated across hospitals under quantum-safe secure aggregation.
            bus.log(f"Quantum Federated Learning across {n_hosp} hospitals — hybrid quantum-classical "
                    f"VQC (PennyLane), secure aggregation on", self.name)
            res = federated_train(
                parts, kind="qfl", rounds=rounds, local_epochs=1, lr=0.1,
                eval_data=(X[cut:], y[cut:]), secure_aggregation=True,
                model_kwargs={"n_qubits": 4, "n_layers": 2}, seed=ctx.config.get("seed", 0),
            )
        else:
            bus.log(f"federated training across {n_hosp} hospitals "
                    f"(secure aggregation on)", self.name)
            res = federated_train(
                parts, kind="mlp", rounds=rounds, eval_data=(X[cut:], y[cut:]),
                secure_aggregation=True, seed=ctx.config.get("seed", 0),
            )
        ctx.artifacts["federated_accuracy"] = round(res.global_acc, 4)
        ctx.artifacts["federated_round_acc"] = [round(a, 4) for a in res.round_acc]
        bus.metric("federated_accuracy", round(res.global_acc, 4), self.name)
        bus.metric("hospitals", n_hosp, self.name)
        bus.log(f"global model accuracy = {res.global_acc:.3f} "
                f"(no raw patient data shared)", self.name)


def _data(config: dict) -> tuple[np.ndarray, np.ndarray, object, np.ndarray]:
    """Default: Wisconsin breast-cancer tabular (bundled, fast — keeps the default + tests light)."""
    from sklearn.linear_model import LogisticRegression

    from ...data.loaders import load_breast_cancer
    from ...fhe.ckks_inference import LinearModel

    X, y = load_breast_cancer()
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    w = np.vstack([np.zeros_like(clf.coef_[0]), clf.coef_[0]])
    b = np.array([0.0, float(clf.intercept_[0])])
    patient = X[y == 1][0].astype(float)            # a malignant case to score privately
    return X, y.astype(np.int64), LinearModel(w.astype(float), b), patient


def _medmnist_data(config: dict):
    """Real MedMNIST images for the genuine CNN-under-FHE path.

    Loads a real MedMNIST 2D set (baked into the full Docker image at build time → no runtime
    network; the synthetic generator in ``load_medmnist`` is an offline-CI fallback only, logged as
    such — never a silent substitute for real data). Downsamples to a FHE-feasible resolution, trains
    a pixel-logistic surrogate (for the CKKS linear inference + faithful pixel-level SHAP), and picks
    a positive patient image. The genuine diagnostic CNN itself is federated-trained in
    ``FederatedTrainingStage`` and run under FHE+MPC in the benchmark row.

    Returns ``(X images (N,1,s,s), y, pixel_logistic LinearModel, patient_flat (s*s,), image (1,s,s))``.
    """
    from sklearn.linear_model import LogisticRegression

    from ...ai.cnn_layers import downsample
    from ...data.loaders import load_medmnist
    from ...fhe.ckks_inference import LinearModel

    name = config.get("medmnist_name", "pneumoniamnist")
    img_size = int(config.get("img_size", 14))
    X, y = load_medmnist(name, split="train")
    y = np.asarray(y).reshape(-1)
    if len(np.unique(y)) > 2:                        # collapse to binary for the small 2-class head
        y = (y > 0).astype(np.int64)
    y = y.astype(np.int64)
    Xs = downsample(X, img_size).astype(np.float32)  # (N, 1, s, s)
    n = min(int(config.get("medmnist_n", 512)), len(Xs))
    Xs, y = Xs[:n], y[:n]
    flat = Xs.reshape(len(Xs), -1)
    clf = LogisticRegression(max_iter=300).fit(flat, y)
    w = np.vstack([np.zeros_like(clf.coef_[0]), clf.coef_[0]])
    b = np.array([0.0, float(clf.intercept_[0])])
    pos = np.where(y == 1)[0]
    idx = int(pos[0]) if len(pos) else 0
    return Xs, y, LinearModel(w.astype(float), b), flat[idx].astype(float), Xs[idx]


def build(config: dict | None = None) -> tuple[Pipeline, Context]:
    config = dict(config or {})
    # the medical CNN path is opt-in: dataset='medmnist' or workload='cnn' (the WebUI's CNN selector)
    use_cnn = config.get("dataset") == "medmnist" or config.get("workload") == "cnn"
    image = None
    feature_names = None
    if use_cnn:
        config.setdefault("dataset", "medmnist")
        config["workload"] = "cnn"
        config["model"] = "cnn"
        X, y, model, patient, image = _medmnist_data(config)
        feature_names = [f"px({r},{c})" for r in range(image.shape[1]) for c in range(image.shape[2])]
    else:
        X, y, model, patient = _data(config)
    ctx = Context(
        usecase="medical_fl",
        config={"inference": "ckks", "qkd": config.get("qkd", "bb84-decoy"),
                "hospitals": 4, "rounds": 6, **config},
    )
    ctx.data.update(X=X, y=y, linear_model=model, features=patient)
    if image is not None:
        ctx.data["image"] = image
        ctx.data["feature_names"] = feature_names
    else:
        try:                                        # human-readable names for the XAI chart
            from sklearn.datasets import load_breast_cancer as _skbc

            ctx.data["feature_names"] = list(_skbc().feature_names)
        except Exception:                           # pragma: no cover
            pass
    stages = [
        PQCHandshakeStage(),
        QKDKeyAgreementStage(),
        FederatedTrainingStage(),
        PrivateInferenceStage(),
        ExplainabilityStage(),
        QuantumComparisonStage(),
        ResultStage(),
        SignatureStage(),
    ]
    return Pipeline(name="medical-fl", stages=stages), ctx


def run_demo(config: dict | None = None) -> dict:
    pipe, ctx = build(config)
    pipe.run(ctx, out_dir="results/")
    m = pipe.bus.metrics()
    return {
        "federated_accuracy": ctx.artifacts.get("federated_accuracy"),
        "private_diagnosis": ctx.artifacts.get("fhe_prediction"),
        "explanation": ctx.artifacts.get("explanation"),
        "hospitals": m.get("hospitals"),
        "metrics": m,
    }
