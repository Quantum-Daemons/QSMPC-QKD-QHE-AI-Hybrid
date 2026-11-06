"""Medical federated learning + private inference use case (deep).

Several hospitals jointly train a diagnostic model without sharing patient data
(federated learning with pairwise-mask aggregation), then a clinician runs a
private inference that never exposes the patient's record to the server.

Each pair of hospitals agrees its mask seed by X25519 for the run, so an aggregator holding only the
public keys and the masked updates cannot remove a mask; the masks are reals added in float64 and
hide an update only partially, every hospital runs in this process, and there is no dropout
recovery or key authentication (see :mod:`qsh.ai.federated`).
Where the CNN variant runs the masked FHE-to-MPC
conversion, every CKKS slot the client can decode is masked since Round 84 (independent masks in
each real and imaginary part on TenSEAL), so the client's whole view is bounded per run by
``view_statistical_distance``; it is semi-honest with an honest majority, and the class index it
receives is still an output (see :mod:`qsh.mpc.secret_inference`).

Maps to: QHE/FHE (private inference) + secure-aggregation-style masking on a PQC
transport with simulated QKD key material. Default data: Wisconsin breast-cancer (bundled, fast). Set
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
    """Train across virtual hospitals with pairwise-mask aggregation, each pair's seed agreed by X25519."""

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
            bus.log(f"federated training an image-classifier CNN across {n_hosp} hospitals on real "
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
            # classical body) federated across hospitals with pairwise-mask aggregation.
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


def _data(config: dict) -> tuple[np.ndarray, np.ndarray, object, np.ndarray, int]:
    """Default: Wisconsin breast-cancer tabular (bundled, fast - keeps the default + tests light).

    Returns the patient's row index too, so the QCNN's raw row (:func:`_qcnn_features`) is the SAME
    patient's rather than a second copy of the selection rule below.
    """
    from sklearn.linear_model import LogisticRegression

    from ...data.loaders import load_breast_cancer
    from ...fhe.ckks_inference import LinearModel

    X, y = load_breast_cancer()
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    w = np.vstack([np.zeros_like(clf.coef_[0]), clf.coef_[0]])
    b = np.array([0.0, float(clf.intercept_[0])])
    idx = int(np.flatnonzero(y == 0)[0])            # a malignant case to score privately
    #                                ^ WDBC (scikit-learn) labels malignant as 0, not 1
    patient = X[idx].astype(float)
    return X, y.astype(np.int64), LinearModel(w.astype(float), b), patient, idx


def _qcnn_features(row_index: int) -> np.ndarray:
    """One WDBC patient's RAW values in the trained QCNN's own column order (Round 84).

    The artifact names its 8 columns (`feature_names`: mean radius ... worst symmetry) and declares
    `input_space: "raw"`, which it standardises with its own training moments. The stage used to hand
    it `features`, the CKKS model's 30-column z-score row, whose first 8 columns are different
    measurements (mean compactness sat where it expects worst radius) in different units. A space
    other than raw is refused rather than guessed: this function has raw values and nothing else.
    """
    from sklearn.datasets import load_breast_cancer as _skbc

    from ...qhe.qcnn_trained import load_weights

    w = load_weights("medical_fl")
    if not w:
        raise ValueError("no trained medical QCNN artifact to take the columns from")
    if w.get("input_space") != "raw":
        raise ValueError(f"the medical QCNN declares input_space {w.get('input_space')!r}; "
                         "this use case supplies raw WDBC values only")
    ds = _skbc()
    names = list(ds.feature_names)
    return ds.data[row_index, [names.index(n) for n in w["feature_names"]]].astype(float)


def _medmnist_data(config: dict):
    """Real MedMNIST images for the genuine CNN-under-FHE path.

    Loads the configured MedMNIST 2D set (``benchmark.medical_cnn``: BreastMNIST, the published student's
    corpus), baked into the full Docker image at build time, so no runtime network. A set that cannot be
    loaded is an error; synthetic images stand in only where a caller opts in (``load_medmnist``). Downsamples to a FHE-feasible resolution, trains
    a pixel-logistic surrogate (for the CKKS linear inference + faithful pixel-level SHAP), and picks
    a positive patient image. The image-classifier CNN itself is federated-trained in
    ``FederatedTrainingStage`` and run under FHE+MPC in the benchmark row.

    Returns ``(X images (N,1,s,s), y, pixel_logistic LinearModel, patient_flat (s*s,), image (1,s,s))``.
    """
    from sklearn.linear_model import LogisticRegression

    from ...ai.cnn_layers import downsample
    from ...config import get_config
    from ...data.loaders import load_medmnist
    from ...fhe.ckks_inference import LinearModel

    # The corpus, resolution and subset the published medical student and the benchmark's encrypted CNN use. These were
    # the literals "pneumoniamnist", 14 and 512 until Round 86, so the full deployment trained and scored a chest X-ray
    # model on a page whose browser tier and published student are BreastMNIST (a reader found it).
    mc = get_config().benchmark.medical_cnn
    name = config.get("medmnist_name", mc["medmnist_name"])
    img_size = int(config.get("img_size", mc["img_size"]))
    X, y = load_medmnist(name, split="train")
    y = np.asarray(y).reshape(-1)
    if len(np.unique(y)) > 2:                        # collapse to binary for the small 2-class head
        y = (y > 0).astype(np.int64)
    y = y.astype(np.int64)
    Xs = downsample(X, img_size).astype(np.float32)  # (N, 1, s, s)
    n = min(int(config.get("medmnist_n", mc["subset"])), len(Xs))
    Xs, y = Xs[:n], y[:n]
    flat = Xs.reshape(len(Xs), -1)
    clf = LogisticRegression(max_iter=300).fit(flat, y)
    w = np.vstack([np.zeros_like(clf.coef_[0]), clf.coef_[0]])
    b = np.array([0.0, float(clf.intercept_[0])])
    # WHICH index is the abnormal class is per-dataset, and the two MedMNIST corpora this project
    # uses disagree: PneumoniaMNIST is {0: normal, 1: pneumonia} while BreastMNIST is
    # {0: malignant, 1: "normal, benign"} - opposite conventions. Hardcoding 1 happens to be right
    # for the default and would silently pick a HEALTHY patient to "score privately" on breastmnist.
    # Read it from MedMNIST's own metadata instead. The same inversion, hardcoded in four other
    # places, had the demo reporting the opposite diagnosis to what the model meant (Round 24).
    pos = np.where(y == _abnormal_class(name))[0]
    idx = int(pos[0]) if len(pos) else 0
    return Xs, y, LinearModel(w.astype(float), b), flat[idx].astype(float), Xs[idx]


#: Words MedMNIST uses on the healthy side of a binary label map.
_HEALTHY_WORDS = ("normal", "benign")


def _abnormal_class(dataset: str) -> int:
    """The label index MedMNIST assigns to the abnormal class, read from its own metadata.

    Returns 1 only when there is no binary MedMNIST metadata to read - the convention used by the
    synthetic offline-CI generator. It deliberately does NOT guess when metadata is present but
    ambiguous: raising there is the point, because a wrong guess is invisible and picks the wrong
    patient, which is the failure mode this function exists to remove.
    """
    try:
        from medmnist import INFO

        labels = INFO[dataset]["label"]
    except Exception:
        return 1                                   # synthetic fallback corpus; no metadata exists
    if len(labels) != 2:
        return 1
    healthy = [int(k) for k, v in labels.items()
               if any(word in v.lower() for word in _HEALTHY_WORDS)]
    if len(healthy) != 1:
        raise ValueError(
            f"cannot tell which class of {dataset} is abnormal from its labels {labels}; "
            f"name it explicitly rather than assuming 1"
        )
    return 1 - healthy[0]


def _class_labels(dataset: str | None) -> dict[int, str]:
    """MedMNIST's own names for a dataset's class indices, or {} without its metadata (the synthetic test images)."""
    from ...config import get_config

    name = dataset or get_config().benchmark.medical_cnn["medmnist_name"]
    try:
        from medmnist import INFO
    except ImportError:
        return {}
    labels = (INFO.get(name) or {}).get("label") or {}
    return {int(k): str(v) for k, v in labels.items()}


def build(config: dict | None = None) -> tuple[Pipeline, Context]:
    config = dict(config or {})
    # The medical CNN path: dataset='medmnist' (which the full deployment's page sends since Round 86) or workload='cnn'.
    use_cnn = config.get("dataset") == "medmnist" or config.get("workload") == "cnn"
    image = None
    class_labels: dict[int, str] = {}
    feature_names = None
    if use_cnn:
        config.setdefault("dataset", "medmnist")
        config["workload"] = "cnn"
        config["model"] = "cnn"
        X, y, model, patient, image = _medmnist_data(config)
        feature_names = [f"px({r},{c})" for r in range(image.shape[1]) for c in range(image.shape[2])]
        patient_index = None
        # What each class index MEANS, from MedMNIST's metadata: BreastMNIST's 0 is malignant where PneumoniaMNIST's 1
        # was pneumonia, so a bare index reads the wrong way round the moment the corpus changes (Round 86).
        class_labels = _class_labels(config.get("medmnist_name"))
    else:
        X, y, model, patient, patient_index = _data(config)
    ctx = Context(
        usecase="medical_fl",
        config={"inference": "ckks", "qkd": config.get("qkd", "bb84-decoy"),
                "hospitals": 4, "rounds": 6, **config},
    )
    ctx.data.update(X=X, y=y, linear_model=model, features=patient)
    # The trained QCNN's own row, in its declared columns and units, only where it will be read: a
    # run that selects QCNN on the tabular data. The CNN variant has no QCNN artifact.
    if patient_index is not None and ctx.config.get("inference") == "qcnn":
        ctx.data["qcnn_features"] = _qcnn_features(patient_index)
        # THE TWO PATHS COUNT MALIGNANT DIFFERENTLY. The QCNN trainer labels malignant 1
        # (`scripts/train/qcnn_variational.py`: y = target == 0), while this use case keeps
        # scikit-learn's WDBC convention, malignant 0, which the CKKS branch reports. Without the
        # map, `private_diagnosis` meant opposite things depending on the backend selected.
        ctx.data["qcnn_class_to_case"] = {0: 1, 1: 0}
    # SHAP reference point E[x], not the zero vector. Round 22 supplied this for finance,
    # genomics and satellite and MISSED medical, so the Python attribution here was measured
    # against a patient whose every feature is zero - not a patient - while the browser path
    # (realData.ts) already used E[x]. The two disagreed on the reference point for the same
    # use case, which is worse than either choice alone: the signed artifact and the rendered
    # chart described different comparisons. `stages.py` records which reference point was
    # used, so this is checkable from the artifact rather than from this comment.
    ctx.data["explain_baseline"] = X.mean(axis=0)
    if image is not None:
        ctx.data["image"] = image
        ctx.data["feature_names"] = feature_names
        if class_labels:
            ctx.data["class_labels"] = class_labels
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
        # Read the canonical key, not one backend's. This used to be `fhe_prediction` only,
        # so a QCNN run produced a class and reported None here.
        "private_diagnosis": ctx.artifacts.get("private_prediction",
                                              ctx.artifacts.get("fhe_prediction")),
        "private_diagnosis_backend": ctx.artifacts.get("private_prediction_backend"),
        "explanation": ctx.artifacts.get("explanation"),
        "hospitals": m.get("hospitals"),
        "metrics": m,
    }
