#!/usr/bin/env python3
"""What an informed adversary reconstructs from a published linear student: one training record from the rest.

On the owner's choice of 2026-11-22 (references/pets-evaluation-2026-11.md), the published CKKS students are audited
against the informed adversary of Balle, Cherubin and Hayes, *Reconstructing Training Data with Informed Adversaries*
(IEEE S&P 2022, doi:10.1109/SP46214.2022.9833677), who knows every training record but one and asks what the published
artefacts give away about it. This is the worst case the threat model of differential privacy assumes; the reconstruction
is reimplemented here (the authors' companion code is CC BY-NC 4.0). Two routes, each over every training record in turn:

- **A, the published preprocessing.** The student's file carries the mean and scale its CKKS client standardises with
  (`ckks_client.mean`, `scale`): the training set's own statistics. The mean of the linear terms is the sum of the
  records over their count, so the missing record is the count times the mean less the known records' sum.
- **B, the weights at the optimum.** The soft-distilled logistic student sees each standardised, expanded row twice,
  as class 1 with weight p_i and as class 0 with weight 1 - p_i, and `class_weight="balanced"` multiplies them by a1 and
  a0, which scikit-learn 1.9 computes from the sample-weighted class counts (0.624 and 2.516 here, not 1). Its gradient
  for row i is c_i x_i with c_i = a1 p_i (s(z_i) - 1) + a0 (1 - p_i) s(z_i); at the optimum w = -C * sum_i c_i x_i and
  sum_i c_i = 0, so the missing row's term and its coefficient are both known from the others', and their quotient is the
  row. It is run on the model retrained with the trainer's settings, and on the same model solved to tolerance 1e-12 (C):
  the trainer's solver stops at 1e-4 on its normalised objective. The published weights differ from the retrained ones by
  at most 0.006 (the solver's stopping point).

OPS-SAT-AD's telemetry is public, so this shows the mechanism, not a disclosure. Only aggregate errors are written.

    python scripts/audit_reconstruction_informed.py --target opssat_student -o docs/measurements/reconstruction-opssat-<date>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
THRESHOLDS = (1e-3, 1e-2, 1e-1)


def _errors(x_hat: np.ndarray, x: np.ndarray) -> dict:
    rel = np.linalg.norm(x_hat - x, axis=1) / np.maximum(np.linalg.norm(x, axis=1), 1e-12)
    rel = rel[np.isfinite(rel)]
    out = {"records": int(rel.size), "relative_error_median": float(np.median(rel)),
           "relative_error_p90": float(np.quantile(rel, 0.9))}
    for t in THRESHOLDS:
        out[f"share_below_{t:g}"] = round(float((rel < t).mean()), 4)
    return out


def opssat_student() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from qsh.models import datasets, distil, train

    train.seed_everything(42)
    xtr, ytr = datasets.opssat_ad("train")
    X = np.asarray(xtr, dtype=float)
    n, d = X.shape
    published = json.loads((REPO / "models" / "satellite_opssat_student" / "weights.json").read_text(encoding="utf-8"))
    pub_mean = np.asarray(published["ckks_client"]["mean"], dtype=float)

    # A: the published mean of the linear terms is the training sum over n.
    total = X.sum(axis=0)
    x_hat_a = n * pub_mean[:d] - (total[None, :] - X)
    route_a = _errors(x_hat_a, X)

    # B: the weights at the optimum, on the model retrained with the trainer's settings.
    teacher = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                                             early_stopping=True, validation_fraction=0.15, random_state=42).fit(xtr, ytr)
    p = teacher.predict_proba(X)[:, 1]
    student = distil.distil_linear(teacher.predict_proba(X), X, seed=42, degree=2)
    poly, scale, lr = student.named_steps["poly"], student.named_steps["scale"], student.named_steps["lr"]
    Xt = scale.transform(poly.transform(X))
    from sklearn.utils.class_weight import compute_class_weight

    w, b, C = lr.coef_.ravel(), float(lr.intercept_[0]), float(lr.C)
    yd = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    wd = np.concatenate([p, 1.0 - p])
    a0, a1 = compute_class_weight("balanced", classes=np.array([0, 1]), y=yd, sample_weight=wd)

    def coefficient(w_, b_):
        s = 1 / (1 + np.exp(-(Xt @ w_ + b_)))
        return a1 * p * (s - 1) + a0 * (1 - p) * s

    r = coefficient(w, b)
    G, R = (r[:, None] * Xt).sum(axis=0), r.sum()
    g = -w / C - (G[None, :] - r[:, None] * Xt)      # each record's own term, from the others'
    c = -(R - r)                                      # its coefficient s(z_j) - p_j, from the others'
    with np.errstate(divide="ignore", invalid="ignore"):
        xt_hat = g / c[:, None]
    x_hat_b = xt_hat[:, :d] * scale.scale_[:d] + scale.mean_[:d]
    route_b = _errors(x_hat_b, X)
    route_b["coefficient_abs_median"] = float(np.median(np.abs(r)))
    route_b["class_weights"] = [round(float(a0), 6), round(float(a1), 6)]
    route_b["stationarity_residual"] = float(np.linalg.norm(w / C + G) / max(np.linalg.norm(w), 1e-12))
    route_b["published_weights_max_abs_diff"] = float(np.max(np.abs(w - np.asarray(published["W"]).ravel())))

    # C: the same student solved to the optimum the closed form assumes. The trainer's solver stops at a tolerance of
    # 1e-4 on its normalised objective, which leaves the stationarity residual above; refit with the same data, weights
    # and C to tolerance 1e-12, and the same closed form is run again.
    from sklearn.linear_model import LogisticRegression

    Xd = np.vstack([Xt, Xt])
    tight = LogisticRegression(C=C, class_weight="balanced", max_iter=100000, tol=1e-12, random_state=42).fit(
        Xd, yd, sample_weight=wd)
    w2, b2 = tight.coef_.ravel(), float(tight.intercept_[0])
    r2 = coefficient(w2, b2)
    G2, R2 = (r2[:, None] * Xt).sum(axis=0), r2.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        xt_hat2 = (-w2 / C - (G2[None, :] - r2[:, None] * Xt)) / (-(R2 - r2))[:, None]
    route_c = _errors(xt_hat2[:, :d] * scale.scale_[:d] + scale.mean_[:d], X)
    route_c["stationarity_residual"] = float(np.linalg.norm(w2 / C + G2) / max(np.linalg.norm(w2), 1e-12))
    route_c["solver_iterations"] = int(np.max(tight.n_iter_))

    return {"target": "satellite_comms.opssat_student", "training_records": int(n), "features": int(d),
            "data": "OPS-SAT-AD train partition (CC BY 4.0), public telemetry",
            "routes": {"A_published_preprocessing_mean": route_a, "B_weights_at_the_optimum": route_b,
                       "C_weights_solved_to_the_optimum": route_c}}


TARGETS = {"opssat_student": opssat_student}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    t0 = time.time()
    result = TARGETS[args.target]()
    record = {
        "what": "an informed adversary's reconstruction of each training record from the rest and the published student",
        "attack": "Balle, Cherubin and Hayes, IEEE S&P 2022 (doi:10.1109/SP46214.2022.9833677); reimplemented here",
        **result,
        "reading": "Relative L2 error of each reconstructed row against the true one. The informed adversary is the "
                   "worst case differential privacy is defined against; nothing here is a DP-trained model, so an exact "
                   "reconstruction is the expected outcome, and the training statistics a client must have are a release.",
        "seconds": round(time.time() - t0, 1),
        "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                              text=True).stdout.strip(),
                        "python": platform.python_version(), "numpy": np.__version__, "date": time.strftime("%Y-%m-%d")},
    }
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(result["routes"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
