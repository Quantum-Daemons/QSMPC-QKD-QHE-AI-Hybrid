#!/usr/bin/env python3
"""Train the 5G-NIDD CKKS student exactly as pre-registered (Round 86, batch 6).

references/telecom-nidd-ckks-student-preregistration.md (and its two addenda) fixed everything below before this ran:
the teacher's split, the 36 numeric features standardised on the training rows and clipped to [-20, 20], a direct
multinomial logistic regression with its penalty chosen by validation log loss and a temperature fitted on validation.
Addendum 2 (the owner's choice of 2026-11-26) dropped the distilled arm, which needed a download the owner declined: the
teacher is a reference row from its own record (models/metrics.json), not run here.

CPU only. Writes docs/measurements/telecom-nidd-student-2026-11-25.json, and, when criterion 1 holds,
web/public/data/telecom_nidd/student_weights.json and student_heldout.json for the page.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/train/telecom_nidd_student.py
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
from qsh.models.registry import DS_5G_NIDD  # noqa: E402

PREREG = REPO / "references" / "telecom-nidd-ckks-student-preregistration.md"
PREREG_SHA256 = "b356193c073606ffa2ef8aa66b2cfa5ac3b9dcf679bd3b558ce4d0c44de5d34d"
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


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    started = time.time()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    teacher = json.loads((REPO / "models" / "metrics.json").read_text())["telecom_ids.nidd"]

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
    onehot = np.eye(k)

    arms = {}
    for arm, Tf, Tv in (("direct", onehot[yf], onehot[yv]),):
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
                     "W": W, "b": b}
        print(f"  {arm:9s} l2 {best:g} T {temperature:.4f}  val macro-F1 {val_f1:.4f}  test {test_f1:.4f}  "
              f"ECE {arms[arm]['test_ece']:.4f}")
    chosen = "direct"
    criterion_1 = arms[chosen]["test_macro_f1"] >= 0.95
    record = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
        "teacher_reference": {"key": "telecom_ids.nidd", "from": "models/metrics.json", "not_run_here": True,
                              "test_macro_f1": teacher.get("macro_f1"), "test_per_class_f1": teacher.get("per_class_f1")},
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
               "attribution": DS_5G_NIDD.attribution,     # the registry's credit, the one every 5G-NIDD file carries
               "modifications": ("The flows published twice under contradictory labels lose their Benign copy; 36 numeric columns "
                                 f"are standardised by the training rows and clipped to [-{CLIP:g}, {CLIP:g}]; {len(take)} rows of the "
                                 "test split, stratified by class, ship.")}
    # Fitted on 5G-NIDD (CC BY 4.0), so the weights carry the corpus's licence and credit, as every set of weights fitted on a
    # CC BY corpus here does (the owner's choice of 2026-11-26; this file said Apache-2.0 and pointed to the held-out rows).
    weights_licence = {"licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                       "attribution": DS_5G_NIDD.attribution,
                       "modifications": ("Fitted on the training flows, after the flows published twice under contradictory labels "
                                         f"lose their Benign copy; 36 numeric columns standardised by the training rows and clipped "
                                         f"to [-{CLIP:g}, {CLIP:g}]. No flow ships in this file; the held-out rows are "
                                         "student_heldout.json's.")}
    (WEB / "student_weights.json").write_text(json.dumps({
        "_generated_by": "scripts/train/telecom_nidd_student.py; do not edit by hand", "arm": chosen,
        "features": record["features"], "classes": list(classes), "centre": centre.round(9).tolist(),
        "scale": scale.round(9).tolist(), "clip": CLIP, "W": W.round(9).tolist(), "b": b.round(9).tolist(),
        "temperature": temperature, "record": str(RECORD.relative_to(REPO)), "preregistration": record["preregistration"],
        **weights_licence}, indent=1) + "\n")
    (WEB / "student_heldout.json").write_text(json.dumps({
        "corpus": "5G-NIDD (real traffic from a 5G test network, University of Oulu)", "split": "the teacher's test split",
        "rows": np.round(Zt[take], 6).tolist(), "labels": yte[take].astype(int).tolist(), **licence}) + "\n")
    print(f"wrote {WEB.relative_to(REPO)}/student_weights.json and student_heldout.json ({len(take)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
