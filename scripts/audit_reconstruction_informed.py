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


def _multinomial_rows(Z: np.ndarray, coef: np.ndarray, intercept: np.ndarray, soft: np.ndarray, C: float,
                      kept: np.ndarray) -> tuple[np.ndarray, float]:
    """Each row rebuilt from all the others and a multinomial soft-target logistic model at its optimum.

    The student is fitted on each row once per class, weighted by the teacher's probability for that class (copies under
    1e-6 dropped, as the trainer drops them). Grouped by row, the gradient for class k is sum_i c_ik z_i + W_k / C with
    c_ik = s_ik * (sum of row i's kept weights) - (row i's kept weight for k), and the unpenalised intercepts give
    sum_i c_ik = 0 for every k. So the missing row's coefficients are minus the others' sums, its term for each class is
    -W_k / C less the others' terms, and the row is their least-squares quotient over the classes.
    """
    logits = Z @ coef.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    s = np.exp(logits)
    s /= s.sum(axis=1, keepdims=True)
    w = np.where(kept, soft, 0.0)
    c = s * w.sum(axis=1, keepdims=True) - w                      # (n, K)
    G = c.T @ Z                                                   # (K, d): sum_i c_ik z_i
    R = c.sum(axis=0)                                             # (K,)
    g = -coef[None, :, :] / C - (G[None, :, :] - c[:, :, None] * Z[:, None, :])   # (n, K, d)
    cj = -(R[None, :] - c)                                        # (n, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_hat = np.einsum("nk,nkd->nd", cj, g) / (cj ** 2).sum(axis=1, keepdims=True)
    residual = float(np.linalg.norm(coef / C + G) / max(np.linalg.norm(coef), 1e-12))
    return z_hat, residual


def telecom_student() -> dict:
    """The QKD-attack student (scripts/train/telecom_ids.py's `distil_linear_student`), retrained by its own recipe."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from qsh.qml_ids.attacks import generate

    X, y, _ = generate(n_per_class=300, seed=42)
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    n, d = Xtr.shape
    published = json.loads((REPO / "models" / "telecom_ids_student" / "weights.json").read_text(encoding="utf-8"))
    pub_mean = np.asarray(published["input"]["mean"], dtype=float)

    route_a = _errors(n * pub_mean - (Xtr.sum(axis=0)[None, :] - Xtr), Xtr)
    route_a["published_mean_max_abs_diff"] = float(np.max(np.abs(pub_mean - Xtr.mean(axis=0))))

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4).fit(Xtr, ytr)
    soft = rf.predict_proba(Xtr)
    K = soft.shape[1]
    mean, scale = Xtr.mean(axis=0), Xtr.std(axis=0)
    scale[scale == 0] = 1.0
    Z = (Xtr - mean) / scale
    kept = soft > 1e-6
    rep_X, rep_y, rep_w = np.repeat(Z, K, axis=0), np.tile(np.arange(K), n), soft.reshape(-1)
    keep = rep_w > 1e-6
    out = {}
    for label, kw in (("B_weights_at_the_optimum", {"max_iter": 4000}),
                      ("C_weights_solved_to_the_optimum", {"max_iter": 100000, "tol": 1e-12})):
        lr = LogisticRegression(C=10.0, random_state=42, **kw).fit(rep_X[keep], rep_y[keep], sample_weight=rep_w[keep])
        z_hat, residual = _multinomial_rows(Z, lr.coef_, lr.intercept_, soft, 10.0, kept)
        route = _errors(z_hat * scale + mean, Xtr)
        route["stationarity_residual"] = residual
        route["solver_iterations"] = int(np.max(lr.n_iter_))
        if label.startswith("B"):
            route["published_weights_max_abs_diff"] = float(np.max(np.abs(lr.coef_ - np.asarray(published["W"]))))
        out[label] = route
    return {"target": "telecom_ids.student", "training_records": int(n), "features": int(d),
            "data": "QKD-attack telemetry this repository generates (qsh.qml_ids.attacks, 8 classes x 300, seed 42), the "
                    "trainer's 75 percent training split",
            "routes": {"A_published_preprocessing_mean": route_a, **out}}


def _logistic_rows(Z: np.ndarray, y: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, float]:
    """Each row rebuilt from the others and an unregularised logistic model at its optimum: sum_i (s_i - y_i) z_i = 0 and,
    from the intercept, sum_i (s_i - y_i) = 0, so the missing row's coefficient and term are minus the others'."""
    w, b = theta[:-1], theta[-1]
    c = 1 / (1 + np.exp(-(Z @ w + b))) - y
    G, R = c @ Z, c.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        z_hat = -(G[None, :] - c[:, None] * Z) / (-(R - c))[:, None]
    scale = max(float(np.abs(c).sum()), 1e-12)
    return z_hat, float(np.linalg.norm(np.append(G, R)) / scale)


def fedheart() -> dict:
    """The heart federation's models as the page trains them (`qsh.ai.fed_heart`, seed 0), on the hospitals' training rows.

    Each hospital publishes the mean and deviation of its training rows with the sample (the page standardises with
    them), so route A is per hospital; the models see the standardised rows, and a row is rebuilt in that space and
    mapped back with its hospital's statistics, which an informed adversary knows. The DP arms are not here: noise is
    what stands against this adversary, and it is audited by the canary game (scripts/audit_dp_one_run.py).
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from audit_membership_inference import fedheart_federated
    from scipy.optimize import minimize

    from qsh.ai import fed_heart

    sample = json.loads(fed_heart.SAMPLE.read_text(encoding="utf-8"))
    cs = sample["centres"]
    Z = np.array([r for c in cs for r in c["train"]["X"]], dtype=float)
    y = np.array([v for c in cs for v in c["train"]["y"]], dtype=float)
    site = np.array([k for k, c in enumerate(cs) for _ in c["train"]["y"]])
    mean = np.array([cs[k]["mean"] for k in site], dtype=float)
    std = np.array([cs[k]["std"] for k in site], dtype=float) + 1e-9
    X = Z * std + mean                                             # the rows as recorded, up to the sample's rounding

    hat_a = np.zeros_like(X)
    for k, c in enumerate(cs):
        rows = site == k
        n_k = int(rows.sum())
        hat_a[rows] = n_k * np.asarray(c["mean"], dtype=float) - (X[rows].sum(axis=0)[None, :] - X[rows])
    route_a = _errors(hat_a, X)

    theta0 = fed_heart.init_theta(0)
    pooled = np.asarray(fed_heart.train_adam(theta0, Z.tolist(), y.astype(int).tolist(), fed_heart.mulberry32(300)))
    parts = [(cs[k]["train"]["X"], cs[k]["train"]["y"]) for k in range(len(cs))]
    federated = np.asarray(fedheart_federated(parts, 0))

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        z = Z @ theta[:-1] + theta[-1]
        p = 1 / (1 + np.exp(-z))
        f = float(np.sum(np.logaddexp(0, z) - y * z))
        g = np.append(Z.T @ (p - y), (p - y).sum())
        return f, g

    opt = minimize(loss, pooled, jac=True, method="L-BFGS-B", options={"maxiter": 100000, "gtol": 1e-12, "ftol": 1e-15})
    routes = {"A_published_preprocessing_mean": route_a}
    for label, theta in (("B_pooled_as_the_page_trains_it", pooled), ("B_fedavg_as_the_page_trains_it", federated),
                         ("C_pooled_solved_to_the_optimum", opt.x)):
        z_hat, residual = _logistic_rows(Z, y, theta)
        r = _errors(z_hat * std + mean, X)
        r["stationarity_residual"] = residual
        routes[label] = r
    routes["C_pooled_solved_to_the_optimum"]["solver_iterations"] = int(opt.nit)
    return {"target": "medical_fl.fed_heart (the page's pooled and FedAvg models, seed 0)", "training_records": int(len(y)),
            "features": int(Z.shape[1]),
            "data": f"{sample['corpus']}: UCI Heart Disease's four hospitals as FLamby prepares them ({sample['licence']}), "
                    "the training rows",
            "routes": routes}


TARGETS = {"opssat_student": opssat_student, "telecom_student": telecom_student, "fedheart": fedheart}


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
