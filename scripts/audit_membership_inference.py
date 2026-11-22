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

The audit pool is the model's public training data (here OPS-SAT-AD's train and test partitions, CC BY 4.0). Only
aggregate figures are written; no record is.

    python scripts/audit_membership_inference.py --target opssat --mlpm <ml_privacy_meter checkout> \
        -o docs/measurements/mia-opssat-<date>.json
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
MLPM_COMMIT = "e384af8fd931"
FPRS = (1e-3, 1e-2, 1e-1)
EPS = 1e-12


# --------------------------------------------------------------------------------------------- targets


def _opssat():
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


TARGETS = {"opssat": _opssat}


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
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    head = subprocess.run(["git", "-C", str(args.mlpm), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if head != MLPM_COMMIT:
        raise SystemExit(f"{args.mlpm} is at {head!r}, not the pinned {MLPM_COMMIT}")
    sys.path.insert(0, str(args.mlpm))
    from modules.mia.attacks.rmia import run_rmia

    X, y, fit, about = TARGETS[args.target]()
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
        per = np.zeros(len(aud_idx)); cnt = np.zeros(len(aud_idx))
        for t in range(K):
            s = lira[t * len(aud_idx):(t + 1) * len(aud_idx)]
            mm = M[:, t]
            per += (mm & (s > thr)); cnt += mm
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
        "reading": "TPR at a fixed low FPR is the measure; an AUC near 0.5 with TPR near the FPR means the attack learns "
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
