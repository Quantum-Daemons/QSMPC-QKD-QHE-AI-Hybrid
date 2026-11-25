#!/usr/bin/env python3
"""Train the 5G-NIDD CKKS student exactly as pre-registered (Round 86, batch 6).

references/telecom-nidd-ckks-student-preregistration.md (and its addendum) fixed everything below before this ran: the
teacher's split, the 36 numeric features standardised on the training rows and clipped to [-20, 20], a distilled and a
direct multinomial logistic regression, each penalty chosen by validation log loss against the arm's own target, a
temperature for the direct arm, and the page's arm chosen by validation macro-F1. The teacher is the published graph
(`Daemons-Q/qsh-telecom-5g-nidd-gbm`, `onnx/model.onnx`) at its pinned revision, refused unless its SHA-256 is the pin.

CPU only. Writes docs/measurements/telecom-nidd-student-2026-11-25.json, and, when criterion 1 holds,
web/public/data/telecom_nidd/student_weights.json and student_heldout.json for the page.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/train/telecom_nidd_student.py --teacher <path to the pinned model.onnx>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets as D  # noqa: E402

PREREG = REPO / "references" / "telecom-nidd-ckks-student-preregistration.md"
PREREG_SHA256 = "20b710a2329a24c828e1c02b535a17bff7ef54e51abad8587a8127abe037a9d1"
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-student-2026-11-25.json"
WEB = REPO / "web" / "public" / "data" / "telecom_nidd"
L2_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
CLIP = 20.0
HELD_OUT_ROWS = 2048
ECE_BINS = 15


def softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def fit(Z: np.ndarray, T: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression to targets T (rows sum to 1), L2 on the weights, by L-BFGS-B."""
    from scipy.optimize import minimize

    n, d = Z.shape
    k = T.shape[1]

    def f(theta):
        W, b = theta[: k * d].reshape(k, d), theta[k * d:]
        P = softmax(Z @ W.T + b)
        loss = -np.sum(T * np.log(np.clip(P, 1e-12, None))) / n + l2 * np.sum(W * W)
        G = (P - T) / n
        return loss, np.concatenate([(G.T @ Z + 2 * l2 * W).ravel(), G.sum(axis=0)])

    r = minimize(f, np.zeros(k * d + k), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return r.x[: k * d].reshape(k, d), r.x[k * d:]


def log_loss(P: np.ndarray, T: np.ndarray) -> float:
    return float(-np.mean(np.sum(T * np.log(np.clip(P, 1e-12, None)), axis=1)))


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> tuple[float, list[float]]:
    per = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        per.append(float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0)
    return float(np.mean(per)), per


def ece(P: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    conf, pred = P.max(axis=1), P.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(float(np.mean(pred[m] == y[m])) - float(conf[m].mean()))
    return float(total)


def teacher_proba(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = []
    for i in range(0, len(x), 65_536):
        res = sess.run(None, {name: x[i:i + 65_536].astype(np.float32)})
        prob = next(r for r in res if isinstance(r, np.ndarray) and r.ndim == 2)
        outs.append(np.asarray(prob, dtype=np.float64))
    return np.concatenate(outs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    args = ap.parse_args()
    started = time.time()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    pins = json.loads((REPO / "models" / "hub_file_hashes.json").read_text())["telecom_ids.nidd"]
    tsha = hashlib.sha256(args.teacher.read_bytes()).hexdigest()
    if tsha != pins["files"]["onnx/model.onnx"]:
        raise SystemExit(f"the teacher graph hashes to {tsha}, not the pinned {pins['files']['onnx/model.onnx']}")

    from sklearn.model_selection import train_test_split

    x, y, names, classes = D.load_5g_nidd()
    k = len(classes)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    numeric = [i for i, n in enumerate(D.NIDD_FEATURES) if n not in D.NIDD_CATEGORICAL]
    centre, scale = xtr[:, numeric].mean(axis=0).astype(np.float64), xtr[:, numeric].std(axis=0).astype(np.float64)
    scale[scale == 0] = 1.0

    def z(a: np.ndarray) -> np.ndarray:
        return np.clip((a[:, numeric] - centre) / scale, -CLIP, CLIP)

    fi, vi = train_test_split(np.arange(len(ytr)), test_size=0.10, random_state=0, stratify=ytr)
    Zf, Zv, Zt = z(xtr[fi]), z(xtr[vi]), z(xte)
    yf, yv = ytr[fi], ytr[vi]
    print(f"  {len(fi):,} fit, {len(vi):,} validation, {len(yte):,} test rows; {len(numeric)} numeric features")
    Pt_f, Pt_v, Pt_t = (teacher_proba(args.teacher, a) for a in (xtr[fi], xtr[vi], xte))
    onehot = np.eye(k)

    arms = {}
    for arm, Tf, Tv in (("distilled", Pt_f, Pt_v), ("direct", onehot[yf], onehot[yv])):
        table, fits = [], {}
        for l2 in L2_GRID:
            W, b = fit(Zf, Tf, l2)
            table.append({"l2": l2, "validation_log_loss": round(log_loss(softmax(Zv @ W.T + b), Tv), 8)})
            fits[l2] = (W, b)
        best = min(table, key=lambda r: (r["validation_log_loss"], r["l2"]))["l2"]
        W, b = fits[best]
        temperature = 1.0
        if arm == "direct":
            from scipy.optimize import minimize_scalar

            lv = Zv @ W.T + b

            def val_loss(t: float, lv: np.ndarray = lv) -> float:
                return log_loss(softmax(lv / np.exp(t)), onehot[yv])

            temperature = float(np.exp(minimize_scalar(val_loss, bounds=(-4, 4), method="bounded").x))
        Pv = softmax((Zv @ W.T + b) / temperature)
        Pt = softmax((Zt @ W.T + b) / temperature)
        val_f1, _ = macro_f1(yv, Pv.argmax(1), k)
        test_f1, per = macro_f1(yte, Pt.argmax(1), k)
        arms[arm] = {"penalty_table": table, "l2": best, "temperature": temperature,
                     "validation_macro_f1": round(val_f1, 6), "test_macro_f1": round(test_f1, 6),
                     "test_per_class_f1": dict(zip(classes, [round(v, 6) for v in per], strict=True)),
                     "test_accuracy": round(float(np.mean(Pt.argmax(1) == yte)), 6),
                     "test_ece": round(ece(Pt, yte), 6),
                     "test_agreement_with_teacher": round(float(np.mean(Pt.argmax(1) == Pt_t.argmax(1))), 6),
                     "W": W, "b": b}
        print(f"  {arm:9s} l2 {best:g} T {temperature:.4f}  val macro-F1 {val_f1:.4f}  test {test_f1:.4f}  "
              f"ECE {arms[arm]['test_ece']:.4f}  agreement {arms[arm]['test_agreement_with_teacher']:.4f}")
    t_f1, t_per = macro_f1(yte, Pt_t.argmax(1), k)
    chosen = max(arms, key=lambda a: (arms[a]["validation_macro_f1"], a == "direct"))
    criterion_1 = arms[chosen]["test_macro_f1"] >= 0.95
    record = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
        "teacher": {"repo": "Daemons-Q/qsh-telecom-5g-nidd-gbm", "revision": pins["revision"], "sha256": tsha,
                    "test_macro_f1": round(t_f1, 6), "test_per_class_f1": dict(zip(classes, [round(v, 6) for v in t_per], strict=True)),
                    "test_ece": round(ece(Pt_t, yte), 6)},
        "rows": {"fit": int(len(fi)), "validation": int(len(vi)), "test": int(len(yte))},
        "features": [D.NIDD_FEATURES[i] for i in numeric], "clip": CLIP,
        "arms": {a: {kk: vv for kk, vv in v.items() if kk not in ("W", "b")} for a, v in arms.items()},
        "chosen": chosen, "criterion_1_test_macro_f1_at_least_0_95": bool(criterion_1),
        "wall_clock_s": round(time.time() - started, 1), "device": "cpu",
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)}; chosen {chosen}; criterion 1 {'holds' if criterion_1 else 'FAILS'}")
    if not criterion_1:
        return 0
    W, b, temperature = arms[chosen]["W"], arms[chosen]["b"], arms[chosen]["temperature"]
    rng = np.random.default_rng(0)
    take = []
    for c in range(k):                                   # stratified: each class in its share of the test split
        idx = np.flatnonzero(yte == c)
        take.extend(rng.choice(idx, size=max(1, round(HELD_OUT_ROWS * len(idx) / len(yte))), replace=False).tolist())
    take = np.sort(np.asarray(take[:HELD_OUT_ROWS]))
    WEB.mkdir(parents=True, exist_ok=True)
    licence = {"licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
               "attribution": ("5G-NIDD: Samarakoon et al., '5G-NIDD: A Comprehensive Network Intrusion Detection Dataset "
                               "Generated over 5G Wireless Network', Fairdata, doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361, CC BY 4.0"),
               "modifications": ("The flows published twice under contradictory labels lose their Benign copy; 36 numeric columns "
                                 f"are standardised by the training rows and clipped to [-{CLIP:g}, {CLIP:g}]; {len(take)} rows of the "
                                 "test split, stratified by class, ship.")}
    (WEB / "student_weights.json").write_text(json.dumps({
        "_generated_by": "scripts/train/telecom_nidd_student.py; do not edit by hand", "arm": chosen,
        "features": record["features"], "classes": list(classes), "centre": centre.round(9).tolist(),
        "scale": scale.round(9).tolist(), "clip": CLIP, "W": W.round(9).tolist(), "b": b.round(9).tolist(),
        "temperature": temperature, "record": str(RECORD.relative_to(REPO)), "preregistration": record["preregistration"],
        "licence": "Apache-2.0", "note": "fitted numbers; the corpus's credit is in student_heldout.json"}, indent=1) + "\n")
    (WEB / "student_heldout.json").write_text(json.dumps({
        "corpus": "5G-NIDD (real traffic from a 5G test network, University of Oulu)", "split": "the teacher's test split",
        "rows": np.round(Zt[take], 6).tolist(), "labels": yte[take].astype(int).tolist(),
        "teacher_class": Pt_t[take].argmax(1).astype(int).tolist(), **licence}) + "\n")
    print(f"wrote {WEB.relative_to(REPO)}/student_weights.json and student_heldout.json ({len(take)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
