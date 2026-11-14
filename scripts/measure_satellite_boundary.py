#!/usr/bin/env python3
"""A decision boundary for the satellite page's segment classifier on the overlay's plane (Round 86, the owner's choice).

The page draws OPS-SAT-AD's 529 test segments on two of their 19 features, peaks in the segment (`n_peaks`) and peaks
in its second difference (`diff2_peaks`), the pair that separates the classes best. The published classifier
(`satellite_comms.opssat`) reads all 19, so its boundary has no single place on that plane. Following the 2026 check of
the practice (sklearn's DecisionBoundaryDisplay needs a model on exactly two features; Molnar's Interpretable ML and the
decision-boundary-map papers ask for the slice's anchor and its fidelity to be stated), this tries, in order, fixed
before any figure was seen:

  1. A SLICE of the published graph through one observed test segment, the medoid of the test partition in z-space
     (standardised by the train partition): the other 17 features held at its values, the two plane features on a grid.
     It passes when at least 20 test segments lie near the slice (each held feature within 1 SD of the medoid's) and
     the graph's class for at least 90% of them, at its 0.5 cut-off, is the class the slice gives at their two values.
  2. Otherwise a SURROGATE: a logistic regression on the two features and their squares and product, fitted to the
     published graph's probabilities on the train partition (each segment weighted by them), scored on the test
     partition: class agreement with the graph at 0.5, the probabilities' R^2, and both models' AUC-PR against the
     labels. It passes at 90% agreement.
  3. Otherwise no boundary, and the page says why.

Writes docs/measurements/satellite-boundary-2026-11-14.json and, for a passing figure, web/public/data/satellite/
boundary.json (a 60 x 60 grid of probabilities over the range of the points it draws, widened by a quarter of it, rounded
to 3 decimals, and for a slice the segments near it with their classes). The published graph must hash to the pinned Hub file.

Usage: QSH_DATA_ROOT=<root holding raw/opssat/dataset.csv> python -I scripts/measure_satellite_boundary.py
       --graph <models/satellite_opssat/onnx/model.onnx>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
RECORD = REPO / "docs" / "measurements" / "satellite-boundary-2026-11-14.json"
CARD = REPO / "web" / "public" / "data" / "satellite" / "boundary.json"
KEY = "satellite_comms.opssat"
PLANE = ("n_peaks", "diff2_peaks")
GRID = 60
MIN_NEAR, MIN_AGREEMENT = 20, 0.90


def medoid(z: np.ndarray) -> int:
    """The row with the least summed Euclidean distance to all rows."""
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    return int(d.sum(1).argmin())


def near_slice(z: np.ndarray, anchor: np.ndarray, held: np.ndarray, sd: float = 1.0) -> np.ndarray:
    """Rows whose every held feature lies within `sd` standard deviations of the anchor's."""
    return (np.abs(z[:, held] - anchor[held]) <= sd).all(axis=1)


def quad(x2: np.ndarray) -> np.ndarray:
    """The two plane features, their squares and their product."""
    a, b = x2[:, 0], x2[:, 1]
    return np.stack([a, b, a * a, b * b, a * b], axis=1)


def fit_surrogate(x2: np.ndarray, p: np.ndarray):
    """Logistic regression on quad(x2), standardised, fitted to probabilities by duplicating each row with weights p and
    1 - p (the exact encoding of a soft target)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    f = quad(x2)
    xx = np.concatenate([f, f])
    yy = np.concatenate([np.ones(len(f)), np.zeros(len(f))])
    w = np.concatenate([p, 1 - p])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000))
    model.fit(xx, yy, logisticregression__sample_weight=w)
    return lambda q: model.predict_proba(quad(q))[:, 1]


def grid_over(x2: np.ndarray, floor: np.ndarray, n: int = GRID, margin: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """A grid over the points' range widened by `margin` of it on each side, never below the data's minimum (`floor`):
    the slice speaks for the points near it, and a grid over every outlier draws mostly extrapolation."""
    lo, hi = x2.min(0), x2.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    lo, hi = np.maximum(lo - margin * span, floor), hi + margin * span
    return np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)


def main() -> int:
    import onnxruntime as ort
    from sklearn.metrics import average_precision_score, r2_score

    from qsh.models import datasets, registry

    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("--graph", type=Path, required=True)
    a = ap.parse_args()
    pinned = json.loads((REPO / "models" / "hub_file_hashes.json").read_text())[KEY]
    graph_sha = hashlib.sha256(a.graph.read_bytes()).hexdigest()
    if graph_sha != pinned["files"]["onnx/model.onnx"] or pinned["revision"] != registry.get(KEY).hf_revision:
        raise SystemExit(f"{a.graph} is not the pinned published graph of {KEY}")
    sess = ort.InferenceSession(str(a.graph), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]

    def prob(x: np.ndarray) -> np.ndarray:
        outs = sess.run(None, {sess.get_inputs()[0].name: np.asarray(x, np.float32)})
        p = np.asarray(outs[1] if len(outs) > 1 else outs[0], dtype=np.float64)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.ravel()

    xtr, ytr = datasets.opssat_ad("train")
    xte, yte = datasets.opssat_ad("test")
    feats = list(datasets.OPSSAT_FEATURES)
    i0, i1 = feats.index(PLANE[0]), feats.index(PLANE[1])
    held = np.array([k for k in range(len(feats)) if k not in (i0, i1)])
    mu, sd = xtr.mean(0), xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    zte = (xte - mu) / sd
    p_tr, p_te = prob(xtr), prob(xte)
    ap_model = float(average_precision_score(yte, p_te))

    # 1. the slice through the medoid
    m = medoid(zte)
    anchor = xte[m].copy()
    near = near_slice(zte, zte[m], held)
    at_near = np.repeat(anchor[None, :], int(near.sum()), axis=0)
    at_near[:, [i0, i1]] = xte[near][:, [i0, i1]]
    slice_cls, model_cls = prob(at_near) >= 0.5, p_te[near] >= 0.5
    slice_agree = float((slice_cls == model_cls).mean()) if near.any() else None
    slice_pass = bool(near.sum() >= MIN_NEAR and slice_agree is not None and slice_agree >= MIN_AGREEMENT)

    # 2. the surrogate
    sur = fit_surrogate(xtr[:, [i0, i1]], p_tr)
    s_te = sur(xte[:, [i0, i1]])
    sur_agree = float(((s_te >= 0.5) == (p_te >= 0.5)).mean())
    sur_pass = sur_agree >= MIN_AGREEMENT
    chosen = "slice" if slice_pass else "surrogate" if sur_pass else "none"

    # The grid spans what is drawn: the points near the slice, or every test point for the surrogate.
    shown = xte[near][:, [i0, i1]] if chosen == "slice" else xte[:, [i0, i1]]
    xs, ys = grid_over(shown, xte[:, [i0, i1]].min(0))
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    if chosen == "slice":
        full = np.repeat(anchor[None, :], len(pts), axis=0)
        full[:, [i0, i1]] = pts
        grid_p = prob(full)
    elif chosen == "surrogate":
        grid_p = sur(pts)
    else:
        grid_p = None

    record = {
        "_about": ("A decision boundary for the satellite segment classifier on the overlay's plane (Round 86): the slice "
                   "through the medoid, the two-feature surrogate, and which one passed. Written by "
                   "scripts/measure_satellite_boundary.py; aggregate figures only."),
        "model": KEY, "hf_revision": pinned["revision"], "graph_sha256": graph_sha, "graph_outputs": names,
        "plane": list(PLANE), "partitions": {"train": int(len(ytr)), "test": int(len(yte)), "test_anomalous": int(yte.sum())},
        "model_test_auc_pr": ap_model,
        "slice": {"anchor": "the test partition's medoid in z-space (train mean and SD)",
                  "held_values": {feats[k]: float(anchor[k]) for k in held},
                  "near": int(near.sum()), "near_rule": "every held feature within 1 SD of the medoid's",
                  "agreement": slice_agree, "passes": slice_pass},
        "surrogate": {"form": "logistic regression on n_peaks, diff2_peaks, their squares and product, standardised, fitted "
                              "to the published graph's probabilities on the train partition",
                      "test_agreement_at_0.5": sur_agree, "test_r2_of_probabilities": float(r2_score(p_te, s_te)),
                      "test_auc_pr": float(average_precision_score(yte, s_te)), "passes": sur_pass},
        "rules": {"min_near": MIN_NEAR, "min_agreement": MIN_AGREEMENT}, "chosen": chosen,
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    if grid_p is not None:
        CARD.write_text(json.dumps({
            "_about": "The satellite overlay's boundary; written by scripts/measure_satellite_boundary.py.",
            "model": KEY, "hf_revision": pinned["revision"], "kind": chosen, "plane": list(PLANE),
            "x": [round(float(v), 4) for v in xs], "y": [round(float(v), 4) for v in ys],
            "p": [[round(float(v), 3) for v in row] for row in grid_p.reshape(len(ys), len(xs))],
            "slice": record["slice"] if chosen == "slice" else None,
            # The segments the slice speaks for: their two plane values, true class, the full model's class and the
            # slice's class at their values (OPS-SAT-AD's test partition, CC BY 4.0, as the overlay already shows it).
            "near": ([{"x": round(float(xte[i, i0]), 4), "y": round(float(xte[i, i1]), 4), "label": int(yte[i]),
                       "model": int(p_te[i] >= 0.5), "slice": int(c)}
                      for i, c in zip(np.flatnonzero(near), slice_cls, strict=True)] if chosen == "slice" else None),
            "surrogate": record["surrogate"] if chosen == "surrogate" else None,
            "model_test_auc_pr": ap_model, "test_segments": int(len(yte))}, indent=0) + "\n")
    print(json.dumps({k: record[k] for k in ("model_test_auc_pr", "slice", "surrogate", "chosen")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
