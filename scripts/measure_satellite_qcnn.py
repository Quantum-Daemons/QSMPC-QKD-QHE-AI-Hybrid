#!/usr/bin/env python3
"""The satellite QCNNs scored as their pre-registration says (references/satellite-qcnn-tn-preregistration.md).

For each corpus (A: OPS-SAT-AD, B: ESA-ADB Mission1) and each of the five seeds' artifacts: the circuit's readout on
the validation and test rows (by `qsh.qhe.qcnn_tree`, which equals the state vector), then on the test rows the
average precision, and the minority-class F1 and balanced accuracy at the threshold that maximises F1 on validation.
The comparators on the identical split and features: logistic regression on the eight z-scored inputs, and the
classical surrogate on the circuit's own feature map (the trainer's `classical_surrogate` estimator, scored the same
way). For A, the shipped OPS-SAT teacher and student are reference rows from `models/metrics.json`: nineteen features
against eight. For B, by addendum 2, average precision leads and the thresholded figures carry a note: the validation
months hold two positive windows.

    python scripts/measure_satellite_qcnn.py --artifacts DIR -o docs/measurements/satellite-qcnn-<date>.json
    (DIR holds opssat-s0 .. opssat-s4 and esa-s0 .. esa-s4, each with the trainer's weights.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

CORPORA = {"opssat": "satellite_opssat", "esa": "satellite_esa"}


def scored(y_val, s_val, y_te, s_te) -> dict:
    """Average precision on test; F1 and balanced accuracy at the validation threshold that maximises F1."""
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_recall_curve,
    )

    p, r, t = precision_recall_curve(y_val, s_val)
    f1 = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-300), 0.0)
    thr = float(t[int(np.argmax(f1))])
    pred = (s_te >= thr).astype(int)
    return {"average_precision": round(float(average_precision_score(y_te, s_te)), 6),
            "minority_f1": round(float(f1_score(y_te, pred, zero_division=0)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_te, pred)), 6),
            "validation_threshold": thr, "validation_f1": round(float(f1.max()), 6)}


def comparators(Xs: np.ndarray, Z: np.ndarray, y: np.ndarray, fit: slice, val: slice, te: slice, c_enc: float) -> dict:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(max_iter=2000).fit(Z[fit], y[fit])
    theta = c_enc * np.pi * Xs
    phi = np.ones((len(Xs), 1))
    for q in range(theta.shape[1]):
        phi = np.hstack([phi, phi * np.sin(theta[:, q:q + 1]), phi * np.cos(theta[:, q:q + 1])])
    sur = LogisticRegression(max_iter=2000).fit(phi[fit], y[fit])
    return {"logistic_regression_z": scored(y[val], lr.decision_function(Z[val]), y[te], lr.decision_function(Z[te])),
            "classical_surrogate": {"n_features": int(phi.shape[1]),
                                    **scored(y[val], sur.decision_function(phi[val]), y[te], sur.decision_function(phi[te]))}}


def main() -> int:
    from qcnn_variational import load, split_for

    from qsh.qhe import qcnn_tree

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out = {}
    for short, usecase in CORPORA.items():
        X, y, cols, prov = load(usecase)
        order, cut, vcut = split_for(usecase, y, 0)
        X, y = X[order], y[order]
        fit, val, te = slice(0, cut), slice(cut, vcut), slice(vcut, len(y))
        seeds = {}
        for seed in range(5):
            raw = (args.artifacts / f"{short}-s{seed}" / "weights.json").read_bytes()
            w = json.loads(raw)
            r = qcnn_tree.readout(w, X)
            logit = r * float(w["scale"]) + float(w["bias"])
            seeds[str(seed)] = {**scored(y[val], logit[val], y[te], logit[te]),
                                "trainer_rule_minority_f1": w.get("test_minority_f1"),
                                "artifact_sha256": hashlib.sha256(raw).hexdigest(), "command": w["training"]["command"]}
        w0 = json.loads((args.artifacts / f"{short}-s0" / "weights.json").read_text())
        mu, sd = X[fit].mean(0), X[fit].std(0)
        sd = np.where(sd == 0, 1.0, sd)
        Z = (X - mu) / sd
        comp = comparators(np.tanh(Z), Z, y, fit, val, te, float(w0.get("encoding_scale", 1.0)))
        spread = {k: {"mean": round(float(np.mean([s[k] for s in seeds.values()])), 6),
                      "sd": round(float(np.std([s[k] for s in seeds.values()], ddof=1)), 6),
                      "min": round(float(min(s[k] for s in seeds.values())), 6),
                      "max": round(float(max(s[k] for s in seeds.values())), 6)}
                  for k in ("average_precision", "minority_f1", "balanced_accuracy")}
        entry = {"usecase": usecase, "features": cols, "data": prov,
                 "rows": {"fit": cut, "validation": vcut - cut, "test": len(y) - vcut,
                          "test_positive": int(y[te].sum()), "validation_positive": int(y[val].sum())},
                 "qcnn_by_seed": seeds, "qcnn_spread": spread, **comp}
        if short == "opssat":
            m = json.loads((REPO / "models" / "metrics.json").read_text())
            entry["reference_rows"] = {
                k: {"auc_pr": m[k].get("auc_pr"), "minority_class_f1": m[k].get("minority_class_f1"),
                    "note": "nineteen features, the shipped model's own threshold: not like for like"}
                for k in ("satellite_comms.opssat", "satellite_comms.opssat_student") if k in m}
        else:
            entry["note"] = ("Addendum 2: average precision leads; the thresholded figures rest on the validation months' "
                             f"{int(y[val].sum())} positive windows.")
        out[short] = entry
        print(short, json.dumps({"qcnn": spread, "lr": comp["logistic_regression_z"], "surrogate": comp["classical_surrogate"]}), flush=True)
    record = {"what": "the satellite QCNNs and their comparators on the pre-registered splits and metrics",
              "preregistration": "references/satellite-qcnn-tn-preregistration.md (824c814 and its addenda)", **out,
              "seconds": round(time.time() - t0, 1),
              "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                                    capture_output=True, text=True).stdout.strip(),
                              "python": platform.python_version(), "numpy": np.__version__, "date": time.strftime("%Y-%m-%d")}}
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
