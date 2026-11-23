#!/usr/bin/env python3
"""Membership-inference audit of a model's training procedure: LiRA and RMIA over shadow models.

On the owner's choice of 2026-11-22 (references/pets-evaluation-2026-11.md), each model trained here is audited the way
the attacks are evaluated in their papers: K shadow models are fitted by the model's own training procedure, each on a
random half of an audit pool, so every record is a member of about half of them; each shadow model in turn is the
target and the others are its references. A record's score says how much more likely the target's output is if the
record was in its training set. The pooled scores of members and non-members give the ROC, and what is reported is the
true-positive rate at low false-positive rates, where membership inference matters (Carlini et al., *Membership
Inference Attacks From First Principles*, IEEE S&P 2022, arXiv:2112.03570), with the AUC beside it.

- **LiRA**, online, with per-record means and a global variance over the reference models' logit-confidences
  (Carlini et al., section IV), implemented here.
- **RMIA** (Zarifzadeh, Liu and Shokri, *Low-Cost High-Power Membership Inference Attacks*, ICML 2024,
  arXiv:2312.03262), offline, from ML Privacy Meter (MIT) at a pinned commit, imported from a checkout outside this
  repository: `--mlpm <checkout>`. Its population is a fifth of the pool that no shadow model trains on.

The audit pool is the model's public training data (OPS-SAT-AD's train and test partitions; BreastMNIST's train and
test splits, under the terms DATA_LICENSES.md records). Only aggregate figures are written; no record is.

    python scripts/audit_membership_inference.py --target opssat --mlpm <ml_privacy_meter checkout> \
        -o docs/measurements/mia-opssat-<date>.json
    CUDA_VISIBLE_DEVICES= python scripts/audit_membership_inference.py --target breastmnist_students \
        --teacher <the published ResNet-18 graph> --mlpm <checkout> -o docs/measurements/mia-breastmnist-students-<date>.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
MLPM_COMMIT = "e384af8fd931"
FPRS = (1e-3, 1e-2, 1e-1)
EPS = 1e-12


# --------------------------------------------------------------------------------------------- targets


def _opssat(args=None):
    """OPS-SAT-AD's teacher (scripts/train/satellite_opssat.py's settings) and its degree-2 CKKS student."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    from qsh.models import datasets, distil

    xtr, ytr = datasets.opssat_ad("train")
    xte, yte = datasets.opssat_ad("test")
    X = np.vstack([xtr, xte]).astype(float)
    y = np.concatenate([ytr, yte]).astype(int)

    def fit(Xs, ys, seed):
        teacher = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                                 l2_regularization=1.0, early_stopping=True,
                                                 validation_fraction=0.15, random_state=seed)
        teacher.fit(Xs, ys)
        student = distil.distil_linear(teacher.predict_proba(Xs), Xs, seed=seed, degree=2)
        return {"teacher": teacher, "student": student}

    return X, y, fit, {"data": "OPS-SAT-AD, train and test partitions (CC BY 4.0, Zenodo 12588359)",
                       "models": {"teacher": "satellite_comms.opssat: HistGradientBoostingClassifier, the trainer's settings",
                                  "student": "satellite_comms.opssat_student: soft logistic on the degree-2 expansion"}}


def _telecom_qml(args=None):
    """The QKD-attack random forest (scripts/train/telecom_ids.py's settings) on the rows its generator writes."""
    from sklearn.ensemble import RandomForestClassifier

    from qsh.qml_ids.attacks import generate

    X, y, _ = generate(n_per_class=300, seed=42)

    def fit(Xs, ys, seed):
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=4)
        rf.fit(Xs, ys)
        return {"forest": rf}

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), fit, {
        "data": "QKD-attack telemetry this repository generates (qsh.qml_ids.attacks, 8 classes x 300, seed 42; Apache-2.0)",
        "models": {"forest": "telecom_ids.qml: RandomForestClassifier(n_estimators=200), the trainer's settings"}}


def _breastmnist_students(args):
    """The two BreastMNIST students the encrypted tier runs, each distilled from the published ResNet-18's labels.

    The teacher is held fixed: it is the published graph (its sha256 is checked against the pin), trained on the train
    split, and each shadow student is distilled from its labels on that shadow's members by the students' own recipes
    (scripts/train/medical_breastmnist.py and medical_square_student.py): class-weighted cross-entropy, at most 120
    epochs, the epoch kept by AUC on the val split's teacher labels, and for the CNN student the net-benefit cut folded
    into its head. So this measures what a student's own training adds. Whether a record was in the teacher's training
    set is not audited here: that needs the teacher retrained for every shadow, a GPU job.
    """
    import hashlib
    import importlib.util
    import tempfile

    import onnxruntime as ort
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, TensorDataset

    from qsh.ai.cnn_layers import downsample, small_medmnist_cnn
    from qsh.models import datasets, distil, metrics, train

    if torch.cuda.is_available():
        raise SystemExit("this target runs on the CPU (the owner approves each GPU job): set CUDA_VISIBLE_DEVICES=")
    if args is None or args.teacher is None:
        raise SystemExit("--teacher (the published ResNet-18 graph) is required for this target")
    torch.set_num_threads(args.threads)
    pin = json.loads((REPO / "models" / "hub_file_hashes.json").read_text(encoding="utf-8"))["medical_fl.resnet18"]
    digest = hashlib.sha256(args.teacher.read_bytes()).hexdigest()
    if digest != pin["files"]["onnx/model.onnx"]:
        raise SystemExit(f"{args.teacher} is not the published teacher at {pin['revision'][:10]} (sha256 {digest[:12]})")
    spec = importlib.util.spec_from_file_location("square", REPO / "scripts" / "train" / "medical_square_student.py")
    square = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(square)
    MALIGNANT, BENIGN = 0, 1                     # MedMNIST's label order (scripts/train/medical_breastmnist.py)

    def split(name):
        imgs, labels = datasets.breastmnist(name, size=28)
        x = (np.asarray(imgs, dtype=np.float32)[:, None] / 255.0 - 0.5) / 0.5
        return x, np.asarray(labels).ravel().astype(np.int64)

    (xtr, _), (xva, yva), (xte, _) = split("train"), split("val"), split("test")
    sess = ort.InferenceSession(str(args.teacher), providers=["CPUExecutionProvider"])
    teach = lambda x: sess.run(None, {"input": x})[0].argmax(1).astype(np.int64)      # noqa: E731
    to14 = lambda x: np.asarray(downsample(x, 14), dtype=np.float32)                    # noqa: E731
    pool28 = np.concatenate([xtr, xte])
    X, t_pool = to14(pool28), teach(pool28)
    x14va, t_va = to14(xva), teach(xva)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(x14va), torch.from_numpy(t_va)), batch_size=256)

    class Probs:
        """predict_proba over the trained student, in eval mode, softmax in float64."""

        def __init__(self, model):
            self.model = model.eval()

        def predict_proba(self, x):
            with torch.no_grad():
                z = self.model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).double()
            return torch.softmax(z, 1).numpy()

    def cnn():
        return small_medmnist_cnn(in_ch=1, n_classes=2, img_size=14, channels=(4,), hidden=16)

    def fit(Xs, ys, seed):
        counts = np.bincount(ys, minlength=2).astype(np.float64)
        weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)
        out = {}
        with tempfile.TemporaryDirectory(prefix="qsh-mia-") as tmp:
            for name, make in (("student", cnn), ("square_student", square.square_student)):
                train.seed_everything(seed)
                model = make()
                res = train.train_torch(
                    f"mia.{name}", model,
                    DataLoader(TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)), batch_size=32, shuffle=True),
                    val_dl, metric_fn=lambda y, p: metrics.auroc(y, p), metric_name="auc",
                    loss_fn=torch.nn.CrossEntropyLoss(weight=weight),
                    cfg=train.TrainConfig(epochs=120, batch_size=32, lr=2e-3, patience=25, seed=seed),
                    out_dir=Path(tmp) / name)
                model.load_state_dict(load_file(res.checkpoint))
                model.eval()
                if name == "student":
                    # The trainer's operating point: the net-benefit cut on the val split's true labels, folded into
                    # the head unless no cut beats both defaults or it collapses validation to one class.
                    scores = train.predict_loader(model, DataLoader(
                        TensorDataset(torch.from_numpy(x14va), torch.from_numpy(yva)), batch_size=256))[1]
                    p_mal = torch.softmax(torch.tensor(scores), 1).numpy()[:, MALIGNANT]
                    q, nb, nb_all, nb_none = metrics.best_threshold_by_net_benefit(yva == MALIGNANT, p_mal, lo=0.05,
                                                                                   hi=0.5)
                    if nb - max(nb_all, nb_none) > 1e-9 and len(np.unique(np.where(p_mal >= q, MALIGNANT, BENIGN))) == 2:
                        distil.fold_threshold_into_torch_head(model, q, positive_class=MALIGNANT)
                out[name] = Probs(model)
        return out

    return X, t_pool, fit, {
        "data": "MedMNIST v2 BreastMNIST at 28 px, train and test splits (CC BY 4.0 as MedMNIST states it; the BUSI "
                "authors publish no licence; DATA_LICENSES.md), downsampled to 14 px as the students see them; the val "
                "split selects each student's epoch, as in training",
        "teacher": {"key": "medical_fl.resnet18", "hf_revision": pin["revision"], "sha256": digest,
                    "held_fixed": "every shadow student is distilled from the published teacher, which was trained on "
                                  "the train split, so membership in the teacher's training set is not audited here"},
        "label": "the teacher's label, which is each student's training target and which anyone can compute from the "
                 "published teacher",
        "models": {"student": "medical_fl.student: conv 1->4, ReLU, avg-pool, dense 196->16, ReLU, dense 16->2, the "
                              "net-benefit cut folded into the head",
                   "square_student": "medical_fl.square_student: conv 1->4, batch norm, x^2, avg-pool, dense 196->2"}}


def _breastmnist_pipeline(args):
    """BreastMNIST's whole pipeline for every shadow: the ResNet-18 teacher retrained on the shadow's members, then both
    students distilled from it.

    On the owner's approval of 2026-11-23 (a GPU job). Each shadow trains the teacher by scripts/train/medical_breastmnist.py's
    settings (random initialisation, class-weighted cross-entropy, AdamW at 3e-4 with weight decay 1e-4, at most 60
    epochs, the epoch kept by AUC on the val split with patience 15) on its members at 28 px, labels its members and the
    val split with it, and distils both students by their recipes, as `breastmnist_students` does from the published
    teacher. All three models are scored on the true label, so membership in the teacher's own training set is audited,
    and each student's figure is what the whole pipeline leaks through it.
    """
    import importlib.util
    import tempfile

    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, TensorDataset

    from qsh.ai.cnn_layers import downsample, small_medmnist_cnn
    from qsh.models import datasets, distil, metrics, train

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "train" / path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    square, trainer = load("square", "medical_square_student.py"), load("breast", "medical_breastmnist.py")
    MALIGNANT, BENIGN = 0, 1
    dev = train._device()

    def split(name):
        imgs, labels = datasets.breastmnist(name, size=28)
        x = (np.asarray(imgs, dtype=np.float32)[:, None] / 255.0 - 0.5) / 0.5
        return x, np.asarray(labels).ravel().astype(np.int64)

    (xtr, ytr), (xva, yva), (xte, yte) = split("train"), split("val"), split("test")
    to14 = lambda x: np.asarray(downsample(x, 14), dtype=np.float32)                    # noqa: E731
    x14va = to14(xva)
    pool28, y = np.concatenate([xtr, xte]), np.concatenate([ytr, yte])

    class Probs:
        """predict_proba over a trained torch model, in eval mode, softmax in float64; `small` downsamples to 14 px."""

        def __init__(self, model, small: bool):
            self.model, self.small = model.eval(), small

        def predict_proba(self, x):
            x = to14(x) if self.small else np.ascontiguousarray(x, dtype=np.float32)
            out = []
            with torch.no_grad():
                for i in range(0, len(x), 256):
                    out.append(torch.softmax(self.model(torch.from_numpy(x[i:i + 256]).to(dev)).double(), 1).cpu().numpy())
            return np.concatenate(out)

    def seq_argmax(model, x):
        return Probs(model, small=False).predict_proba(x).argmax(1).astype(np.int64)

    def weighted(labels):
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        return torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32).to(dev)

    def fit(Xs, ys, seed):
        out = {}
        with tempfile.TemporaryDirectory(prefix="qsh-mia-") as tmp:
            train.seed_everything(seed)
            teacher = trainer.build_resnet("resnet18")
            res = train.train_torch(
                "mia.teacher", teacher,
                DataLoader(TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys)), batch_size=32, shuffle=True),
                DataLoader(TensorDataset(torch.from_numpy(xva), torch.from_numpy(yva)), batch_size=256),
                metric_fn=lambda yy, p: metrics.auroc(yy, p), metric_name="auc",
                cfg=train.TrainConfig(epochs=60, batch_size=32, lr=3e-4, weight_decay=1e-4, patience=15, seed=seed),
                loss_fn=torch.nn.CrossEntropyLoss(weight=weighted(ys)), out_dir=Path(tmp) / "teacher")
            teacher.load_state_dict(load_file(res.checkpoint))
            teacher.eval()
            out["teacher"] = Probs(teacher, small=False)
            t_members, t_val = seq_argmax(teacher, Xs), seq_argmax(teacher, xva)
            val_dl = DataLoader(TensorDataset(torch.from_numpy(x14va), torch.from_numpy(t_val)), batch_size=256)
            for name, make in (("student", lambda: small_medmnist_cnn(in_ch=1, n_classes=2, img_size=14, channels=(4,),
                                                                      hidden=16)),
                               ("square_student", square.square_student)):
                train.seed_everything(seed)
                model = make()
                r = train.train_torch(
                    f"mia.{name}", model,
                    DataLoader(TensorDataset(torch.from_numpy(to14(Xs)), torch.from_numpy(t_members)), batch_size=32,
                               shuffle=True),
                    val_dl, metric_fn=lambda yy, p: metrics.auroc(yy, p), metric_name="auc",
                    loss_fn=torch.nn.CrossEntropyLoss(weight=weighted(t_members)),
                    cfg=train.TrainConfig(epochs=120, batch_size=32, lr=2e-3, patience=25, seed=seed),
                    out_dir=Path(tmp) / name)
                model.load_state_dict(load_file(r.checkpoint))
                model.eval()
                if name == "student":
                    scores = train.predict_loader(model, DataLoader(
                        TensorDataset(torch.from_numpy(x14va), torch.from_numpy(yva)), batch_size=256))[1]
                    p_mal = torch.softmax(torch.tensor(scores), 1).numpy()[:, MALIGNANT]
                    q, nb, nb_all, nb_none = metrics.best_threshold_by_net_benefit(yva == MALIGNANT, p_mal, lo=0.05,
                                                                                   hi=0.5)
                    if nb - max(nb_all, nb_none) > 1e-9 and len(np.unique(np.where(p_mal >= q, MALIGNANT, BENIGN))) == 2:
                        distil.fold_threshold_into_torch_head(model, q, positive_class=MALIGNANT)
                out[name] = Probs(model, small=True)
        return out

    return pool28, y, fit, {
        "data": "MedMNIST v2 BreastMNIST at 28 px, train and test splits (CC BY 4.0 as MedMNIST states it; the BUSI "
                "authors publish no licence; DATA_LICENSES.md); the val split selects every model's epoch, as in training",
        "label": "the true label, for all three models",
        "device": str(dev),
        "models": {"teacher": "medical_fl.resnet18's recipe, retrained on each shadow's members (random init, 60 epochs "
                              "at most, AUC selection on val)",
                   "student": "medical_fl.student's recipe, distilled from that shadow's teacher, the net-benefit cut "
                              "folded into the head",
                   "square_student": "medical_fl.square_student's recipe, distilled from that shadow's teacher"}}


#: The heart federation's DP arms audited: clipping alone, and the two ends of the page's sweep.
FEDHEART_DP_EPSILONS = (math.inf, 8.0, 1.0)


def fedheart_parts(Xs: np.ndarray, ys: np.ndarray, n_sites: int) -> list[tuple[list, list]]:
    """Each hospital's rows and labels; the hospital is the last column, which the models do not see."""
    return [([r[:-1] for r in Xs[Xs[:, -1] == k].tolist()], ys[Xs[:, -1] == k].tolist()) for k in range(n_sites)]


def fedheart_federated(parts, seed: int) -> list[float]:
    """`qsh.ai.fed_heart.run`'s FedAvg arm: its initial weights, its streams and its order of operations."""
    from qsh.ai import fed_heart
    from qsh.ai.federated import dequantise, quantise

    sizes = [len(yk) for _, yk in parts]
    total = sum(sizes)
    streams = [fed_heart.mulberry32(seed * 1000 + 200 + k) for k in range(len(parts))]
    glob = list(fed_heart.init_theta(seed))
    for _ in range(fed_heart.ROUNDS):
        acc = np.zeros(len(glob), dtype=np.uint64)
        for k, (Xk, yk) in enumerate(parts):
            local = fed_heart.train_sgd_steps(glob, Xk, yk, streams[k])
            acc += quantise(np.asarray(local, dtype=np.float64) * (sizes[k] / total), len(parts))
        glob = dequantise((acc % (1 << 32)).astype(np.uint32)).tolist()
    return glob


def fedheart_pooled(parts, seed: int) -> list[float]:
    """`run`'s pooled arm: every hospital's rows together, Adam for 50 epochs."""
    from qsh.ai import fed_heart

    Xp = [r for Xk, _ in parts for r in Xk]
    yp = [v for _, yk in parts for v in yk]
    return fed_heart.train_adam(fed_heart.init_theta(seed), Xp, yp, fed_heart.mulberry32(seed * 1000 + 300))


def fedheart_dp(parts, seed: int, epsilon: float, rb) -> list[float]:
    """`run_dp`'s loop at one epsilon: full-batch descent on clipped per-patient gradients, each hospital's sum rounded to
    2^-16 with its share of the discrete Gaussian noise added, the shares summed."""
    from qsh.ai import dp, fed_heart

    n_sites, dim = len(parts), len(parts[0][0][0]) + 1
    unit = 2 ** fed_heart.DP_FIXED_POINT_BITS
    sensitivity = fed_heart.DP_CLIP * unit + math.sqrt(dim)
    V = dp.variance_for(epsilon, fed_heart.DP_ROUNDS, sensitivity, n_sites, dim, fed_heart.DP_DELTA)
    total = sum(len(yk) for _, yk in parts)
    theta = fed_heart.init_theta(seed, dim - 1)
    for _ in range(fed_heart.DP_ROUNDS):
        sums = [0] * dim
        for Xk, yk in parts:
            g = fed_heart.clipped_gradient_sum(theta, Xk, yk, fed_heart.DP_CLIP)
            for j, x in enumerate(g):
                sums[j] += math.floor(x * unit + 0.5) + dp.discrete_gaussian(V, rb)
        theta = [t - fed_heart.DP_LR * (sums[j] / unit) / total for j, t in enumerate(theta)]
    return theta


class _Logit:
    """predict_proba of a logistic model's weights; the last column of the input (the hospital) is not a feature."""

    def __init__(self, theta: list[float]) -> None:
        self.w, self.b = np.asarray(theta[:-1], dtype=float), float(theta[-1])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = 1.0 / (1.0 + np.exp(-(np.asarray(x)[:, :-1] @ self.w + self.b)))
        return np.column_stack([1.0 - p1, p1])


def _fedheart(args=None):
    """FLamby's Fed-Heart-Disease federation the medical page runs, through its Python twin `qsh.ai.fed_heart`.

    Each shadow federation trains on its members among each hospital's rows (train and test splits pooled within the
    hospital) by the page's procedures and random streams: FedAvg under the secure sum's quantisation, the pooled model,
    and the record-level DP federation at epsilon infinity (clipping alone), 8 and 1 (delta 1e-6), whose noise here comes
    from a seeded stream: NOT private, the same distribution as the page's CSPRNG.
    """
    from qsh.ai import dp, fed_heart

    s = json.loads(fed_heart.SAMPLE.read_text(encoding="utf-8"))
    cs = s["centres"]
    rows, labels = [], []
    for k, c in enumerate(cs):
        for part in ("train", "test"):
            for x, yy in zip(c[part]["X"], c[part]["y"], strict=True):
                rows.append([*x, k])
                labels.append(int(yy))

    def fit(Xs, ys, seed):
        parts = fedheart_parts(Xs, ys, len(cs))
        out = {"federated": _Logit(fedheart_federated(parts, seed)), "pooled": _Logit(fedheart_pooled(parts, seed))}
        for eps in FEDHEART_DP_EPSILONS:
            name = "dp_clip_only" if not math.isfinite(eps) else f"dp_epsilon_{eps:g}"
            out[name] = _Logit(fedheart_dp(parts, seed, eps, dp.seeded_rand_below(f"mia-fedheart-{seed}-{eps:g}")))
        return out

    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), fit, {
        "data": f"{s['corpus']}: UCI Heart Disease's four hospitals as FLamby prepares them ({s['licence']}), train and "
                f"test rows pooled within each hospital, {len(rows)} rows",
        "models": {"federated": "the page's FedAvg: 15 rounds of 100 local SGD steps at lr 0.001, the secure sum's ring",
                   "pooled": "the four hospitals' rows together, Adam (lr 0.001, batches of 4, 50 epochs)",
                   "dp_clip_only": "record-level DP federation, 50 rounds of full-batch descent, clipping at 1, no noise",
                   "dp_epsilon_8": "the same with the page's discrete Gaussian noise for epsilon 8, delta 1e-6",
                   "dp_epsilon_1": "the same for epsilon 1"},
        "noise": "the DP arms' noise is drawn from dp.seeded_rand_below (not private; the CSPRNG's distribution)",
        "dp_delta": fed_heart.DP_DELTA}


def dp_bound_check(results: dict, delta: float) -> dict:
    """Each DP arm's measured rates against its guarantee: an (epsilon, delta)-DP training procedure lets no test of a
    record's membership reach a true-positive rate above exp(epsilon) * FPR + delta (Kairouz, Oh and Viswanath, ICML
    2015). An attack's rate is a lower bound on what is possible, so a rate above the bound would mean the accounting
    or the implementation is wrong; one below it shows nothing more."""
    out = {}
    for name, res in results.items():
        if not name.startswith("dp_epsilon_"):
            continue
        eps = float(name.removeprefix("dp_epsilon_"))
        rows = []
        for attack in ("lira_online", "rmia_offline"):
            for f in FPRS:
                tpr = res[attack].get(f"tpr_at_fpr_{f:g}")
                if tpr is None:
                    continue
                bound = min(1.0, math.exp(eps) * f + delta)
                rows.append({"attack": attack, "fpr": f, "tpr": tpr, "bound": round(bound, 6), "holds": tpr <= bound})
        out[name] = {"epsilon": eps, "delta": delta, "rows": rows, "all_hold": all(r["holds"] for r in rows)}
    return out


TARGETS = {"opssat": _opssat, "telecom_qml": _telecom_qml, "breastmnist_students": _breastmnist_students,
           "breastmnist_pipeline": _breastmnist_pipeline, "fedheart": _fedheart}


# --------------------------------------------------------------------------------------------- attacks


def _true_prob(model, X, y) -> np.ndarray:
    p = model.predict_proba(X)
    return np.clip(p[np.arange(len(y)), y], EPS, 1 - EPS)


def lira_online(phi_t: np.ndarray, phi_ref: np.ndarray, m_ref: np.ndarray) -> np.ndarray:
    """Per-record means of the logit-confidence over IN and OUT references, a global variance for each; the score is the
    log-likelihood ratio of the target's logit-confidence. NaN where a record has no IN or no OUT reference."""
    inn = np.where(m_ref, phi_ref, np.nan)
    out = np.where(~m_ref, phi_ref, np.nan)
    mu_in, mu_out = np.nanmean(inn, axis=1), np.nanmean(out, axis=1)
    sd_in = np.nanstd(inn - mu_in[:, None]) + 1e-6
    sd_out = np.nanstd(out - mu_out[:, None]) + 1e-6
    ll_in = -0.5 * ((phi_t - mu_in) / sd_in) ** 2 - np.log(sd_in)
    ll_out = -0.5 * ((phi_t - mu_out) / sd_out) ** 2 - np.log(sd_out)
    return ll_in - ll_out


def rmia_offline(p_t: np.ndarray, p_ref: np.ndarray, m_ref: np.ndarray, pz_t: np.ndarray, pz_ref: np.ndarray,
                 a: float, gamma: float = 1.0) -> np.ndarray:
    """RMIA, offline (Zarifzadeh, Liu and Shokri, ICML 2024): Pr(x) is approximated from the OUT references as
    ((1 + a) P_OUT(x) + (1 - a)) / 2; every population record z is OUT of every reference. A record's score is the share of
    z whose likelihood ratio it beats by more than gamma."""
    out = np.where(~m_ref, p_ref, np.nan)
    px = (1 + a) / 2 * np.nanmean(out, axis=1) + (1 - a) / 2
    pz = (1 + a) / 2 * pz_ref.mean(axis=1) + (1 - a) / 2
    rx, rz = p_t / px, pz_t / pz
    return (rx[:, None] / rz[None, :] > gamma).mean(axis=1)


def roc_summary(scores: np.ndarray, member: np.ndarray) -> dict:
    """AUC, and the best TPR among the ROC's points at or below each FPR (how the attacks' papers read it). Ties count
    once, as roc_curve treats them: RMIA's scores are fractions of a finite population, so many tie at the top."""
    from sklearn.metrics import roc_auc_score, roc_curve

    ok = np.isfinite(scores)
    s, m = scores[ok], member[ok]
    fpr, tpr, _ = roc_curve(m, s)
    out = {"auc": round(float(roc_auc_score(m, s)), 4), "members": int(m.sum()), "non_members": int((~m).sum())}
    for f in FPRS:
        if f * (~m).sum() < 1:
            out[f"tpr_at_fpr_{f:g}"] = None     # not resolvable with this many non-members
            continue
        if fpr[1] > f:
            # The top score is shared by more than this share of non-members (RMIA's score is a fraction of a finite
            # population), so no threshold reaches this FPR: the reading is absent, not zero.
            out[f"tpr_at_fpr_{f:g}"] = None
            out.setdefault("not_resolvable", []).append(f"fpr {f:g}: {fpr[1]:.4f} of non-members tie at the top score")
            continue
        out[f"tpr_at_fpr_{f:g}"] = round(float(tpr[fpr <= f].max()), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--models", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--population", type=float, default=0.2)
    ap.add_argument("--offline-a", type=float, default=0.3)
    ap.add_argument("--mlpm", required=True, type=Path, help="an ml_privacy_meter checkout at commit " + MLPM_COMMIT)
    ap.add_argument("--teacher", type=Path, default=None, help="breastmnist_students: the published ResNet-18 graph")
    ap.add_argument("--threads", type=int, default=4, help="breastmnist_students: torch's CPU threads")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    head = subprocess.run(["git", "-C", str(args.mlpm), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if head != MLPM_COMMIT:
        raise SystemExit(f"{args.mlpm} is at {head!r}, not the pinned {MLPM_COMMIT}")
    sys.path.insert(0, str(args.mlpm))
    from modules.mia.attacks.rmia import run_rmia

    X, y, fit, about = TARGETS[args.target](args)
    rng = np.random.default_rng(args.seed)
    n = len(y)
    order = rng.permutation(n)
    pop_idx = order[: int(round(args.population * n))]
    aud_idx = order[int(round(args.population * n)):]
    K = args.models - args.models % 2
    # Complementary pairs: models 2s and 2s+1 split the audit records between them, so every record is a member of
    # exactly half of the models; a target's references exclude itself and its partner, so every record is IN for
    # exactly (K - 2) / 2 of them, the balanced reference set the attacks assume.
    M = np.zeros((len(aud_idx), K), dtype=bool)
    for s in range(K // 2):
        half = rng.permutation(len(aud_idx))[: len(aud_idx) // 2]
        M[half, 2 * s] = True
        M[:, 2 * s + 1] = ~M[:, 2 * s]

    t0 = time.time()
    names = None
    P_aud, P_pop = {}, {}
    for k in range(K):
        members = aud_idx[M[:, k]]
        models = fit(X[members], y[members], args.seed * 1000 + k)
        if names is None:
            names = list(models)
            for name in names:
                P_aud[name] = np.zeros((len(aud_idx), K))
                P_pop[name] = np.zeros((len(pop_idx), K))
        for name, model in models.items():
            P_aud[name][:, k] = _true_prob(model, X[aud_idx], y[aud_idx])
            P_pop[name][:, k] = _true_prob(model, X[pop_idx], y[pop_idx])
    fit_seconds = time.time() - t0

    results = {}
    for name in names:
        phi = np.log(P_aud[name]) - np.log1p(-P_aud[name])
        lira, rmia, member = [], [], []
        for t in range(K):
            ref = [j for j in range(K) if j not in (t, t ^ 1)]
            lira.append(lira_online(phi[:, t], phi[:, ref], M[:, ref]))
            rmia.append(rmia_offline(P_aud[name][:, t], P_aud[name][:, ref], M[:, ref], P_pop[name][:, t],
                                     P_pop[name][:, ref], args.offline_a))
            member.append(M[:, t])
        # ML Privacy Meter's run_rmia keeps each record's top `num_reference_models` OUT signals. Where that number is
        # exactly a record's OUT count, and the population's reference set is that size too, it is the formula above:
        # checked once, at target 0, with half of the references standing in for the population's.
        ref0 = [j for j in range(K) if j not in (0, 1)]
        half_refs = ref0[: len(ref0) // 2]
        theirs = run_rmia(P_aud[name][:, 0], P_aud[name][:, ref0], M[:, ref0], P_pop[name][:, 0],
                          P_pop[name][:, half_refs], args.offline_a, num_reference_models=len(ref0) // 2)
        mine = rmia_offline(P_aud[name][:, 0], P_aud[name][:, ref0], M[:, ref0], P_pop[name][:, 0],
                            P_pop[name][:, half_refs], args.offline_a)
        crosscheck = float(np.max(np.abs(np.asarray(theirs) - mine)))
        lira, rmia, member = np.concatenate(lira), np.concatenate(rmia), np.concatenate(member)
        lira_sum, rmia_sum = roc_summary(lira, member), roc_summary(rmia, member)
        # A record counts as exposed if, as a member, LiRA puts it above the threshold at FPR 1e-3 in most targets.
        neg = np.sort(lira[~member & np.isfinite(lira)])[::-1]
        thr = neg[max(int(np.floor(1e-3 * len(neg))) - 1, 0)]
        per = np.zeros(len(aud_idx))
        cnt = np.zeros(len(aud_idx))
        for t in range(K):
            s = lira[t * len(aud_idx):(t + 1) * len(aud_idx)]
            mm = M[:, t]
            per += (mm & (s > thr))
            cnt += mm
        exposed = int(((per / np.maximum(cnt, 1)) > 0.5).sum())
        results[name] = {"lira_online": lira_sum, "rmia_offline": rmia_sum,
                         "rmia_matches_ml_privacy_meter_max_abs_diff": crosscheck,
                         "records_exposed_in_most_targets_at_fpr_1e-3": exposed, "audit_records": int(len(aud_idx))}
        print(name, json.dumps(results[name]))

    record = {
        "what": "membership inference against the training procedure, over shadow models",
        "target": args.target, **about,
        "design": {"pool": int(n), "audit_records": int(len(aud_idx)), "population_records": int(len(pop_idx)),
                   "shadow_models": K, "membership": "complementary pairs: each record a member of exactly half the models",
                   "references_per_target": K - 2,
                   "seed": args.seed, "rmia_offline_a": args.offline_a, "fit_seconds": round(fit_seconds, 1)},
        "attacks": {"lira_online": "Carlini et al., IEEE S&P 2022 (arXiv:2112.03570); implemented here",
                    "rmia_offline": "Zarifzadeh, Liu and Shokri, ICML 2024 (arXiv:2312.03262); implemented here and checked "
                                    f"against ML Privacy Meter's run_rmia at {MLPM_COMMIT}"},
        "results": results,
        **({"dp_bound_check": dp_bound_check(results, about["dp_delta"])} if "dp_delta" in about else {}),
        "reading":"TPR at a fixed low FPR is the measure; an AUC near 0.5 with TPR near the FPR means the attack learns "
                   "little about membership from this model's outputs. An attack gives a lower bound on leakage, not a proof "
                   "of its absence.",
        "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                              text=True).stdout.strip(),
                        "python": platform.python_version(), "numpy": np.__version__, "date": time.strftime("%Y-%m-%d")},
    }
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
