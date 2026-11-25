"""The pre-registered 3D nodule student (`references/medical-nodule3d-square-student-preregistration.md`).

Stages, each run on its own and each refusing to run if the pre-registration's SHA-256 is not the recorded one:

    python scripts/train/medical_nodule3d_student.py split    --crops DIR
    python scripts/train/medical_nodule3d_student.py teachers --crops DIR --work W     # GPU: asked for first
    python scripts/train/medical_nodule3d_student.py students --crops DIR --work W     # GPU: asked for first
    python scripts/train/medical_nodule3d_student.py ckks     --crops DIR --work W     # node-seal, all test crops

DIR holds `crops_28mm.npz` and `crops_28mm.json` from tools_external/lidc/make_crops.py (outside the repository: the
crops are LIDC-IDRI's voxels). `split` writes the split once, with its SHA-256, and refuses to write it again.
The teachers here are the 3D ResNet-18; CT-FM's probe and TotalSegmentator's size-only candidate write their
out-of-fold probabilities into W/teachers/ by their own scripts, in the same form, and `students` takes the best of
whatever candidates W/teachers/ holds only once all three are there.

The encrypted pass is the page's own module, `web/src/lib/clientFHESquare3d.ts`, bundled with esbuild and run under
node-seal, as tests/test_r86_nodule3d_square_ckks_in_node.py runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PREREG = REPO / "references" / "medical-nodule3d-square-student-preregistration.md"
PREREG_SHA256 = "9f6ec0871b5c3aa2c1d2064923c1f61fcf556d30485d0f35f6a01f1c97c8c7e5"
SPLIT = REPO / "docs" / "measurements" / "medical-nodule3d-split.json"
RECORD = REPO / "docs" / "measurements" / "medical-nodule3d-square-student.json"
OUT_DIR = REPO / "models" / "medical_lidc_nodule3d_square_student"
KEY = "medical_fl.nodule3d_square_student"
SEEDS = (1, 2, 3, 4, 5)
TEACHERS = ("resnet3d18", "ctfm_probe", "totalseg_size")
FOLDS = 5
#: The pre-registered parameter sets, with the HomomorphicEncryption.org standard's 128-bit post-quantum bound for a
#: ternary secret at that N.
PARAMS = {
    "A2": {"N": 8192, "coeff_mod_bits": [48, 34, 34, 34, 52], "scale_bits": 34, "pq_bound_bits": 202},
    "B": {"N": 16384, "coeff_mod_bits": [60, 40, 40, 40, 60], "scale_bits": 40, "pq_bound_bits": 411},
}
MARGIN, FLOOR, MAX_LOGIT_ERR = 0.03, 0.60, 1e-3        # criteria 1, 2 and 3
SIDE = 28


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_preregistration() -> None:
    got = sha256(PREREG)
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.relative_to(REPO)} has SHA-256 {got}, not the committed {PREREG_SHA256}: "
                         "the pre-registration changed, so nothing runs")


# ---------------------------------------------------------------------------------------------------------------- data

def load_crops(crops: Path) -> tuple[np.ndarray, list[dict]]:
    meta = json.loads((crops / "crops_28mm.json").read_text())
    cubes = np.load(crops / "crops_28mm.npz")["crops"]
    if cubes.shape[1:] != (SIDE, SIDE, SIDE) or len(cubes) != len(meta["records"]):
        raise SystemExit(f"crops {cubes.shape} do not match {len(meta['records'])} records")
    if SPLIT.exists():
        check_crops_match_split(meta["records"], json.loads(SPLIT.read_text()))
    return cubes, meta["records"]


def label_of(ratings: list[int]) -> int | None:
    """The pre-registered label: the median rating above 3 is malignant (1), below 3 benign (0), exactly 3 dropped."""
    m = float(np.median(ratings))
    return None if m == 3.0 else int(m > 3.0)


def make_split(records: list[dict]) -> dict:
    """By patient, stratified on the patient's majority label (a tie is malignant); benign stratum first; within
    a stratum the patients sorted by ID are permuted by NumPy's default generator seeded 0, one generator for both
    strata; the first (7n + 5) // 10 train, the next (n + 5) // 10 validation, the rest test: 0.7n and 0.1n
    rounded half up, computed exactly."""
    labels: dict[str, list[int]] = {}
    for r in records:
        y = label_of(r["malignancy"])
        if y is not None:
            labels.setdefault(r["patient"], []).append(y)
    strata = {0: [], 1: []}
    for patient in sorted(labels):
        ys = labels[patient]
        strata[int(2 * sum(ys) >= len(ys))].append(patient)
    rng = np.random.default_rng(0)
    parts = {"train": [], "val": [], "test": []}
    for s in (0, 1):
        order = [strata[s][i] for i in rng.permutation(len(strata[s]))]
        n = len(order)
        ntr, nva = (7 * n + 5) // 10, (n + 5) // 10
        parts["train"] += order[:ntr]
        parts["val"] += order[ntr:ntr + nva]
        parts["test"] += order[ntr + nva:]
    for k in parts:
        parts[k] = sorted(parts[k])
    return parts


def split_counts(records: list[dict], parts: dict) -> dict:
    where = {p: k for k, ps in parts.items() for p in ps}
    out = {k: {"patients": len(v), "nodules": 0, "malignant": 0, "benign": 0} for k, v in parts.items()}
    for r in records:
        y = label_of(r["malignancy"])
        if y is None or r["patient"] not in where:
            continue
        c = out[where[r["patient"]]]
        c["nodules"] += 1
        c["malignant" if y else "benign"] += 1
    return out


def labelled(cubes: np.ndarray, records: list[dict], parts: dict, part: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The labelled crops of one part, in record order, normalised as every network reads them."""
    keep = set(parts[part])
    idx = [i for i, r in enumerate(records) if r["patient"] in keep and label_of(r["malignancy"]) is not None]
    x = (cubes[idx].astype(np.float32) / 255.0 - 0.5) / 0.5
    y = np.array([label_of(records[i]["malignancy"]) for i in idx], dtype=np.int64)
    return x[:, None], y, [records[i]["patient"] for i in idx]


def folds_of(patients: list[str], y: np.ndarray) -> dict[str, int]:
    """Five folds over the training patients, stratified as the split is: each stratum's patients sorted by ID,
    permuted by the default generator seeded 0, and dealt round-robin."""
    by: dict[str, list[int]] = {}
    for p, v in zip(patients, y, strict=True):
        by.setdefault(p, []).append(int(v))
    strata = {0: [], 1: []}
    for p in sorted(by):
        strata[int(2 * sum(by[p]) >= len(by[p]))].append(p)
    rng = np.random.default_rng(0)
    fold, k = {}, 0
    for s in (0, 1):
        for i in rng.permutation(len(strata[s])):
            fold[strata[s][i]] = k % FOLDS
            k += 1
    return fold


# ------------------------------------------------------------------------------------------------------------ networks

def resnet3d18(n_classes: int = 2):
    """A ResNet-18 with 3D convolutions: basic blocks of 64, 128, 256 and 512 channels, a 3x3x3 stem at stride 1 and
    no max pool, since the crop is only 28 points a side (28 -> 28 -> 14 -> 7 -> 4)."""
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int, stride: int) -> None:
            super().__init__()
            self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
            self.b1 = nn.BatchNorm3d(cout)
            self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
            self.b2 = nn.BatchNorm3d(cout)
            self.act = nn.ReLU(inplace=True)
            self.skip = (nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))
                         if stride != 1 or cin != cout else nn.Identity())

        def forward(self, x):
            return self.act(self.b2(self.c2(self.act(self.b1(self.c1(x))))) + self.skip(x))

    class ResNet3d18(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Conv3d(1, 64, 3, 1, 1, bias=False), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
            chans, layers, cin = (64, 128, 256, 512), [], 64
            for i, c in enumerate(chans):
                layers += [Block(cin, c, 1 if i == 0 else 2), Block(c, c, 1)]
                cin = c
            self.layers = nn.Sequential(*layers)
            self.head = nn.Linear(512, n_classes)

        def forward(self, x):
            return self.head(self.layers(self.stem(x)).mean(dim=(2, 3, 4)))

    return ResNet3d18()


def student(arm: str):
    """conv 1->4 (3x3x3, pad 1), batch norm, x^2 (square) or ReLU (control), avg-pool 2, dense 4*14^3 -> 2."""
    import torch.nn as nn

    class Student3d(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(1, 4, 3, padding=1)
            self.bn = nn.BatchNorm3d(4)
            self.pool = nn.AvgPool3d(2)
            self.fc = nn.Linear(4 * 14 ** 3, 2)

        def forward(self, x):
            y = self.bn(self.conv(x))
            y = y * y if arm == "square" else y.clamp_min(0)
            return self.fc(self.pool(y).flatten(1))

    return Student3d()


def folded(model) -> dict:
    """The square arm's inference weights in float64, batch norm folded into the convolution (exact in eval mode)."""
    import torch

    with torch.no_grad():
        bn = model.bn
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = (model.conv.weight * s[:, None, None, None, None]).double().cpu().numpy()      # (4, 1, 3, 3, 3)
        b = ((model.conv.bias - bn.running_mean) * s + bn.bias).double().cpu().numpy()
        return {"conv_w": w, "conv_b": b,
                "fc_w": model.fc.weight.double().cpu().numpy(), "fc_b": model.fc.bias.double().cpu().numpy()}


def plain_logits(crop: np.ndarray, p: dict) -> np.ndarray:
    """The folded square network in float64 on one normalised (28, 28, 28) crop, as PyTorch lays it out."""
    padded = np.pad(crop, 1)
    out = np.empty((4, SIDE, SIDE, SIDE))
    for c in range(4):
        acc = np.full((SIDE, SIDE, SIDE), p["conv_b"][c])
        for dz in range(3):
            for dy in range(3):
                for dx in range(3):
                    acc += p["conv_w"][c, 0, dz, dy, dx] * padded[dz:dz + SIDE, dy:dy + SIDE, dx:dx + SIDE]
        out[c] = acc
    pooled = (out ** 2).reshape(4, 14, 2, 14, 2, 14, 2).mean(axis=(2, 4, 6)).reshape(-1)
    return p["fc_w"] @ pooled + p["fc_b"]


class Augmented:
    """Training crops with a random flip along each axis and a random multiple of 90 degrees in the axial plane
    (the last two axes), drawn from torch's generator, which train_torch's seed_everything seeds."""

    def __init__(self, x: np.ndarray, target: np.ndarray) -> None:
        import torch

        self.x, self.t = torch.tensor(x), torch.tensor(target)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        import torch

        v = self.x[i]
        flips = [d for d in (1, 2, 3) if torch.rand(()) < 0.5]
        if flips:
            v = torch.flip(v, flips)
        k = int(torch.randint(0, 4, ()))
        if k:
            v = torch.rot90(v, k, (2, 3))
        return v.contiguous(), self.t[i]


def logits_of(model, x: np.ndarray) -> np.ndarray:
    """The network's two logits on each crop, in order, in float64. Read directly, not as log-probabilities: a
    confident float32 softmax rounds to 0 and its logarithm would be infinite."""
    import torch

    from qsh.models import train

    dev = train._device()
    model.eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 128):
            out.append(model(torch.tensor(x[i:i + 128]).to(dev)).double().cpu().numpy())
    return np.concatenate(out)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def environment() -> dict:
    import torch

    env = {"python": platform.python_version(), "torch": torch.__version__, "kernel": platform.release(),
           "cpu": platform.processor() or platform.machine()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


# -------------------------------------------------------------------------------------------------------------- stages

def stage_split(crops: Path | None, records_file: Path | None = None) -> int:
    """The split, drawn once. From the crops' records, or (records_file) from scripts/data/lidc_records.py's records of
    every cluster, read from pylidc's database with no image: after the data drive failed on 2026-11-24 the series are
    streamed through the system disk and cut one patient at a time, so the split comes first and the crops must carry
    the same records (`check_crops_match_split`)."""
    if SPLIT.exists():
        raise SystemExit(f"{SPLIT.relative_to(REPO)} exists: the split is written once and never redrawn")
    if records_file is not None:
        records = json.loads(records_file.read_text())["records"]
        source = {"records": len(records), "patients": len({r["patient"] for r in records}),
                  "from": "scripts/data/lidc_records.py (pylidc's annotation database, no image read)",
                  "records_json_sha256": sha256(records_file)}
    else:
        cubes, records = load_crops(crops)
        source = {"records": len(records), "patients": len({r["patient"] for r in records}),
                  "json_sha256": sha256(crops / "crops_28mm.json"), "npz_sha256": sha256(crops / "crops_28mm.npz")}
    parts = make_split(records)
    seen = [p for k in parts for p in parts[k]]
    assert len(seen) == len(set(seen)), "a patient is in two parts"
    body = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
            "crops": source,
            "rule": make_split.__doc__.split("\n")[0].strip() + " (scripts/train/medical_nodule3d_student.py, make_split)",
            "counts": split_counts(records, parts), "parts": parts}
    text = json.dumps(body, indent=1) + "\n"
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(text)
    print(json.dumps(body["counts"], indent=1))
    print(f"wrote {SPLIT.relative_to(REPO)}, SHA-256 {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


def check_crops_match_split(records: list[dict], split: dict) -> None:
    """The crops cut after a records-based split must be those records: the same labelled clusters, patient by patient."""
    key = lambda r: (r["patient"], r["series"], r["cluster"], tuple(r["malignancy"]))       # noqa: E731
    want = split["crops"].get("records")
    if want is not None and len(records) != want:
        raise SystemExit(f"{len(records)} crops against {want} records the split was drawn from")
    if len({key(r) for r in records}) != len(records):
        raise SystemExit("two crops carry the same cluster")


def read_split() -> dict:
    if not SPLIT.exists():
        raise SystemExit("no split: run the split stage first")
    return json.loads(SPLIT.read_text())


def stage_teachers(crops: Path, work: Path) -> int:
    """The 3D ResNet-18 candidate, out of fold: for fold k, it trains on three folds, stops early on fold k + 1's AUC,
    and scores fold k, so no fold's score comes from a network that saw it or was chosen on it."""
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, TensorDataset

    from qsh.models import metrics, train

    split = read_split()
    cubes, records = load_crops(crops)
    x, y, patients = labelled(cubes, records, split["parts"], "train")
    fold = folds_of(patients, y)
    f = np.array([fold[p] for p in patients])
    oof = np.full(len(y), np.nan)
    runs = {}
    for k in range(FOLDS):
        inner = (k + 1) % FOLDS
        tr, va, te = (f != k) & (f != inner), f == inner, f == k
        train.seed_everything(k)
        model = resnet3d18()
        onehot = np.eye(2, dtype=np.float32)[y[tr]]
        res = train.train_torch(
            f"medical_nodule3d.teacher.resnet3d18.fold{k}", model,
            DataLoader(Augmented(x[tr], onehot), batch_size=32, shuffle=True),
            DataLoader(TensorDataset(torch.tensor(x[va]), torch.tensor(y[va])), batch_size=128),
            metric_fn=lambda a, b: metrics.auroc(a, b), metric_name="auc",
            loss_fn=torch.nn.CrossEntropyLoss(),
            cfg=train.TrainConfig(epochs=120, batch_size=32, lr=2e-3, patience=25, seed=k),
            out_dir=work / "teachers" / "resnet3d18" / f"fold{k}",
            extra_manifest={"role": f"teacher candidate resnet3d18, fold {k} out of fold, inner stop on fold {inner}"})
        model.load_state_dict(load_file(res.checkpoint))
        probs = train.predict_loader(model, DataLoader(TensorDataset(torch.tensor(x[te]), torch.tensor(y[te])),
                                                       batch_size=128))[1]
        oof[te] = np.asarray(probs)[:, 1]
        runs[k] = {"inner_auc": round(float(res.best_metric), 6), "best_epoch": int(res.best_epoch),
                   "fold_auc": round(float(metrics.auroc(y[te], np.asarray(probs))), 6)}
        print(f"  resnet3d18 fold {k}: inner {runs[k]['inner_auc']:.4f}, out of fold {runs[k]['fold_auc']:.4f}", flush=True)
    assert not np.isnan(oof).any()
    write_teacher(work, "resnet3d18", patients, y, oof, {"folds": runs, "environment": environment()})
    return 0


def write_teacher(work: Path, name: str, patients: list[str], y: np.ndarray, oof: np.ndarray, extra: dict) -> None:
    """One candidate's out-of-fold probabilities over the training crops, in `labelled(..., "train")` order."""
    from qsh.models import metrics

    out = work / "teachers" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    auc = float(metrics.auroc(y, np.stack([1 - oof, oof], 1)))
    out.write_text(json.dumps({"name": name, "oof_auc": round(auc, 6), "patients": patients,
                               "labels": y.tolist(), "oof": oof.tolist(), **extra}) + "\n")
    print(f"  {name}: out-of-fold AUC {auc:.4f} over {len(y)} training crops")


def platt_and_threshold(d_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float, float]:
    """Platt scaling of the logit difference on validation (an unpenalised logistic fit), and the probability
    threshold that maximises balanced accuracy there, the lowest of any tie."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, max_iter=10000).fit(d_val[:, None], y_val)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    p = 1 / (1 + np.exp(-(a * d_val + b)))
    best, best_t = -1.0, 0.5
    for t in np.unique(p):
        pred = p >= t
        bacc = 0.5 * (np.mean(pred[y_val == 1]) + np.mean(~pred[y_val == 0]))
        if bacc > best + 1e-12:
            best, best_t = bacc, float(t)
    return a, b, best_t


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(0.5 * (np.mean(pred[y == 1]) + np.mean(~pred[y == 0])))


def stage_students(crops: Path, work: Path) -> int:
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, TensorDataset

    from qsh.models import metrics, train

    split = read_split()
    have = {n: json.loads((work / "teachers" / f"{n}.json").read_text())
            for n in TEACHERS if (work / "teachers" / f"{n}.json").exists()}
    if set(have) != set(TEACHERS):
        raise SystemExit(f"teacher candidates missing: {sorted(set(TEACHERS) - set(have))}")
    cubes, records = load_crops(crops)
    xtr, ytr, ptr = labelled(cubes, records, split["parts"], "train")
    xva, yva, _ = labelled(cubes, records, split["parts"], "val")
    xte, yte, pte = labelled(cubes, records, split["parts"], "test")
    for n, t in have.items():
        assert t["patients"] == ptr and t["labels"] == ytr.tolist(), f"{n} was scored on other training crops"
    chosen = max(have, key=lambda n: have[n]["oof_auc"])
    soft = np.asarray(have[chosen]["oof"], dtype=np.float32)
    target = np.stack([1 - soft, soft], 1)
    record = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
              "split": {"file": str(SPLIT.relative_to(REPO)), "sha256": sha256(SPLIT), "counts": split["counts"]},
              "teachers": {n: {"oof_auc": t["oof_auc"]} for n, t in have.items()}, "teacher": chosen,
              "environment": environment(), "arms": {"control": {}, "square": {}},
              "majority_class_rate_test": round(float(max(np.mean(yte == 0), np.mean(yte == 1))), 6),
              "malignant_share": {k: round(float(np.mean(v)), 6) for k, v in (("train", ytr), ("val", yva), ("test", yte))}}
    for arm in ("control", "square"):
        for seed in SEEDS:
            train.seed_everything(seed)
            model = student(arm)
            res = train.train_torch(
                f"medical_nodule3d_student.{arm}", model,
                DataLoader(Augmented(xtr, target), batch_size=32, shuffle=True),
                DataLoader(TensorDataset(torch.tensor(xva), torch.tensor(yva)), batch_size=128),
                metric_fn=lambda a, b: metrics.auroc(a, b), metric_name="auc",
                loss_fn=torch.nn.CrossEntropyLoss(),
                cfg=train.TrainConfig(epochs=120, batch_size=32, lr=2e-3, patience=25, seed=seed),
                out_dir=work / f"{arm}_seed{seed}",
                extra_manifest={"role": f"pre-registered {arm} arm, seed {seed}, soft labels from {chosen}"})
            model.load_state_dict(load_file(res.checkpoint))
            lv, lt = logits_of(model, xva), logits_of(model, xte)
            pt = softmax(lt)
            d_va, d_te = lv[:, 1] - lv[:, 0], lt[:, 1] - lt[:, 0]
            a, b, t = platt_and_threshold(d_va, yva)
            q_va, q_te = 1 / (1 + np.exp(-(a * d_va + b))), 1 / (1 + np.exp(-(a * d_te + b)))
            pred = q_te >= t
            record["arms"][arm][str(seed)] = {
                "val_auc": round(float(res.best_metric), 6), "test_auc": round(float(metrics.auroc(yte, pt)), 6),
                "val_balanced_accuracy": round(balanced_accuracy(yva, q_va >= t), 6),
                "test_balanced_accuracy": round(balanced_accuracy(yte, pred), 6),
                "predicts_both_classes": bool(0 < pred.sum() < len(pred)),
                "platt": [a, b], "threshold": t, "best_epoch": int(res.best_epoch), "epochs_run": int(res.epochs_run),
                "checkpoint_sha256": sha256(Path(res.checkpoint))}
            if arm == "square":
                np.savez(work / f"square_seed{seed}_folded.npz", **folded(model))
            r = record["arms"][arm][str(seed)]
            print(f"  {arm} seed {seed}: val {r['val_auc']:.4f} test {r['test_auc']:.4f} "
                  f"bacc {r['test_balanced_accuracy']:.4f}", flush=True)
    means = {a: float(np.mean([r["test_auc"] for r in record["arms"][a].values()])) for a in ("control", "square")}
    sds = {a: float(np.std([r["test_auc"] for r in record["arms"][a].values()], ddof=1)) for a in ("control", "square")}
    sq = record["arms"]["square"]
    export = max(sq, key=lambda s: sq[s]["val_auc"])
    record["criteria"] = {
        "mean_test_auc": {k: round(v, 6) for k, v in means.items()},
        "sd_test_auc": {k: round(v, 6) for k, v in sds.items()},
        "1_accuracy": bool(means["square"] >= means["control"] - MARGIN),
        "2_no_collapse": bool(all(r["test_auc"] >= FLOOR and r["predicts_both_classes"] for r in sq.values())),
        "export_seed": int(export)}
    record["export_bootstrap"] = bootstrap_auc(work, int(export), xte, yte, pte)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["criteria"], indent=1))
    return 0


def bootstrap_auc(work: Path, seed: int, xte: np.ndarray, yte: np.ndarray, pte: list[str]) -> dict:
    """The exported seed's test AUC, from its folded float64 network, with a 95% percentile interval over 2,000
    resamples of the test patients (default generator seeded 0)."""
    from qsh.models import metrics

    p = dict(np.load(work / f"square_seed{seed}_folded.npz"))
    prob = softmax(np.array([plain_logits(c[0], p) for c in xte.astype(np.float64)]))
    patients = sorted(set(pte))
    rows = {q: [i for i, r in enumerate(pte) if r == q] for q in patients}
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(2000):
        idx = [i for q in rng.choice(patients, len(patients)) for i in rows[q]]
        if len(set(yte[idx].tolist())) == 2:
            aucs.append(metrics.auroc(yte[idx], prob[idx]))
    return {"auc": round(float(metrics.auroc(yte, prob)), 6), "ci95": [round(float(np.percentile(aucs, 2.5)), 6),
            round(float(np.percentile(aucs, 97.5)), 6)], "resamples": len(aucs)}


HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3], DIR = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export * from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquare3d"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const w = JSON.parse(readFileSync(DIR + "/weights.json", "utf8"));
const raw = readFileSync(DIR + "/crops.u8"), n = raw.length / m.VOXELS, set = process.argv[5];
const runs = [];
for (let i = 0; i < n; i++) {
  const v = Array.from(raw.subarray(i * m.VOXELS, (i + 1) * m.VOXELS), (u) => (u / 255 - 0.5) / 0.5);
  const r = await m.runSquare3d(v, w, set);
  runs.push({ logits: r.logits, plain: r.plain, err: r.maxAbsError, ms: r.ms, bytes: r.bytes, ciphertexts: r.ciphertexts });
}
writeFileSync(DIR + "/runs.json", JSON.stringify(runs));
"""


def weights_json(p: dict, sets: list[str], seed: int, platt: list[float], threshold: float) -> dict:
    return {"kind": "cnn3d-1conv-square-1dense",
            "conv": {"in_ch": 1, "out_ch": 4, "kernel": 3, "padding": 1, "activation": "square", "batch_norm": "folded",
                     "weight": p["conv_w"].tolist(), "bias": p["conv_b"].tolist()},
            "pool": {"type": "avg", "stride": 2},
            "dense": [{"in": 4 * 14 ** 3, "out": 2, "weight": p["fc_w"].tolist(), "bias": p["fc_b"].tolist(),
                       "activation": "none"}],
            "classes": ["benign", "malignant"],
            "ckks": {k: {kk: PARAMS[k][kk] for kk in ("N", "coeff_mod_bits", "scale_bits")} for k in sets},
            "decision": {"platt": platt, "threshold": threshold,
                         "rule": "malignant when 1 / (1 + exp(-(a * (logit1 - logit0) + b))) >= threshold"},
            "export_seed": seed}


def stage_ckks(crops: Path, work: Path) -> int:
    """Every test crop under each pre-registered set, through the page's module, decisions at the validation
    threshold. The machine's load is recorded before and after, since the times are read on it."""
    record = json.loads(RECORD.read_text())
    split = read_split()
    cubes, records = load_crops(crops)
    keep = set(split["parts"]["test"])
    idx = [i for i, r in enumerate(records) if r["patient"] in keep and label_of(r["malignancy"]) is not None]
    seed = record["criteria"]["export_seed"]
    arm = record["arms"]["square"][str(seed)]
    a, b = arm["platt"]
    t = arm["threshold"]
    p = dict(np.load(work / f"square_seed{seed}_folded.npz"))
    decide = lambda lg: 1 / (1 + np.exp(-(a * (lg[1] - lg[0]) + b))) >= t      # noqa: E731
    record["ckks"] = {}
    for name in PARAMS:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "weights.json").write_text(json.dumps(weights_json(p, list(PARAMS), int(seed), [a, b], t)))
            (d / "crops.u8").write_bytes(np.ascontiguousarray(cubes[idx]).tobytes())
            (d / "harness.mjs").write_text(HARNESS)
            load_before = os.getloadavg()
            subprocess.run(["node", str(d / "harness.mjs"), str(REPO / "web"), str(d / "bundle.mjs"), str(d), name],
                           check=True, cwd=REPO / "web", timeout=7200)
            runs = json.loads((d / "runs.json").read_text())
        errs = [r["err"] for r in runs]
        differ = sum(int(decide(r["logits"]) != decide(r["plain"])) for r in runs)
        times = [sum(r["ms"].values()) for r in runs]
        run = {**PARAMS[name], "modulus_bits": sum(PARAMS[name]["coeff_mod_bits"]), "crops": len(runs),
               "max_abs_logit_error": max(errs), "median_abs_logit_error": float(np.median(errs)),
               "decisions_differ": differ, "median_ms": float(np.median(times)), "max_ms": float(np.max(times)),
               "ciphertexts": runs[0]["ciphertexts"], "bytes": runs[0]["bytes"],
               "load_average_before": [round(x, 2) for x in load_before],
               "load_average_after": [round(x, 2) for x in os.getloadavg()]}
        run["criterion_3"] = bool(run["max_abs_logit_error"] <= MAX_LOGIT_ERR and differ == 0)
        record["ckks"][name] = run
        print(f"  CKKS {name}: max |err| {run['max_abs_logit_error']:.3e}, decisions differ {differ}, "
              f"median {run['median_ms']:.0f} ms", flush=True)
    record["criteria"]["3_fidelity"] = {k: v["criterion_3"] for k, v in record["ckks"].items()}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("split", "teachers", "students", "ckks"))
    ap.add_argument("--crops", type=Path)
    ap.add_argument("--records", type=Path, help="split only: scripts/data/lidc_records.py's records, no crops needed")
    ap.add_argument("--work", type=Path)
    args = ap.parse_args()
    check_preregistration()
    if args.stage == "split":
        if not (args.crops or args.records):
            raise SystemExit("split needs --crops or --records")
        return stage_split(args.crops, args.records)
    if not args.crops:
        raise SystemExit("--crops is required for this stage")
    if not args.work:
        raise SystemExit("--work is required for this stage")
    args.work.mkdir(parents=True, exist_ok=True)
    return {"teachers": stage_teachers, "students": stage_students, "ckks": stage_ckks}[args.stage](args.crops, args.work)


if __name__ == "__main__":
    raise SystemExit(main())
