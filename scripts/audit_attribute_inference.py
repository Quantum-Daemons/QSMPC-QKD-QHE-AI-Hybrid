#!/usr/bin/env python3
"""Attribute inference against the published heart models, held against imputation (Jayaraman and Evans).

On the owner's choice of 2026-11-22 (references/pets-evaluation-2026-11.md), the heart federation's published models
(seed 0, as the page trains them through `qsh.ai.fed_heart`) are asked a patient's sex, the attribute UCI records as 1
for male and 0 for female, given everything else about the patient: the other twelve features, the diagnosis and the
hospital. Jayaraman and Evans, *Are Attribute Inference Attacks Just Imputation?* (ACM CCS 2022, arXiv:2209.01292),
show the comparison that matters: an attack counts as disclosure from the model only where it beats an imputation
that uses the same knowledge without the model, and does better on the training records than on others.

- **Loss** (Yeom, Giacomelli, Fredrikson and Jha, IEEE CSF 2018): the value under which the model's loss on the known
  diagnosis is lower. For a logistic model this depends on the diagnosis alone: the loss falls as the logit moves toward
  the label, and the attribute moves the logit the way its weight's sign does, so one diagnosis always gets "male" and
  the other "female", whatever else the record says. The record checks this identity.
- **Model inversion** (Fredrikson et al., USENIX Security 2014): the value maximising the model's probability of the
  known diagnosis times the value's frequency at that hospital.
- **Imputation, label only**: the more frequent sex among patients of that hospital with that diagnosis.
- **Imputation, all features**: a logistic regression on the other features, the diagnosis and the hospital.

The adversary's sample of the population is the test rows (non-members). For the training rows it is all of them; for
the test rows, each is predicted from the other four fifths (five folds), so no row informs its own guess. UCI Heart
Disease is public, so this shows the mechanism, not a disclosure. Only aggregate figures are written.

    python scripts/audit_attribute_inference.py -o docs/measurements/attribute-inference-fedheart-<date>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SEX = 1           # the column of `sex` in heart_disease_sites.json's feature_names
FOLDS = 5


def _mia():
    spec = importlib.util.spec_from_file_location("mia", REPO / "scripts" / "audit_membership_inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidates(centre: dict) -> tuple[float, float]:
    """The standardised value of sex 0 and of sex 1 at this hospital, from its published mean and deviation."""
    m, s = centre["mean"][SEX], centre["std"][SEX]
    return (0.0 - m) / s, (1.0 - m) / s


def rows_of(sample: dict, part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, diagnosis, hospital, sex) for one split, sex read back from its standardised value."""
    X, y, h, sex = [], [], [], []
    for k, c in enumerate(sample["centres"]):
        c0, c1 = candidates(c)
        for x, yy in zip(c[part]["X"], c[part]["y"], strict=True):
            v = x[SEX]
            if min(abs(v - c0), abs(v - c1)) > 1e-5:
                raise ValueError(f"{c['key']}: sex value {v} is neither candidate")
            X.append(x)
            y.append(int(yy))
            h.append(k)
            sex.append(int(abs(v - c1) < abs(v - c0)))
    return np.asarray(X, dtype=float), np.asarray(y), np.asarray(h), np.asarray(sex)


def logit(theta: list[float], X: np.ndarray) -> np.ndarray:
    return X @ np.asarray(theta[:-1], dtype=float) + float(theta[-1])


def with_sex(X: np.ndarray, h: np.ndarray, sample: dict, value: int) -> np.ndarray:
    out = X.copy()
    out[:, SEX] = [candidates(sample["centres"][k])[value] for k in h]
    return out


def loss_attack(theta, X, y, h, sample) -> np.ndarray:
    """Yeom et al.: the value with the lower logistic loss on the known diagnosis (ties to male, the majority)."""
    z0, z1 = logit(theta, with_sex(X, h, sample, 0)), logit(theta, with_sex(X, h, sample, 1))
    sgn = np.where(y == 1, 1.0, -1.0)
    loss0, loss1 = np.logaddexp(0, -sgn * z0), np.logaddexp(0, -sgn * z1)
    return (loss1 <= loss0).astype(int)


def inversion_attack(theta, X, y, h, sample, prior_male: np.ndarray) -> np.ndarray:
    """Fredrikson et al.: argmax over the value of p(diagnosis | features, value) times the value's frequency."""
    sgn = np.where(y == 1, 1.0, -1.0)
    lp0 = -np.logaddexp(0, -sgn * logit(theta, with_sex(X, h, sample, 0))) + np.log(np.clip(1 - prior_male, 1e-12, 1))
    lp1 = -np.logaddexp(0, -sgn * logit(theta, with_sex(X, h, sample, 1))) + np.log(np.clip(prior_male, 1e-12, 1))
    return (lp1 >= lp0).astype(int)


def label_rule(y_pop, h_pop, sex_pop, y, h) -> np.ndarray:
    """The more frequent sex at the hospital among the sample's patients with that diagnosis (ties to male)."""
    out = np.ones(len(y), dtype=int)
    for i, (yi, hi) in enumerate(zip(y, h, strict=True)):
        sel = (y_pop == yi) & (h_pop == hi)
        out[i] = int(sex_pop[sel].mean() >= 0.5) if sel.any() else 1
    return out


def male_share(h_pop, sex_pop, h) -> np.ndarray:
    shares = {k: sex_pop[h_pop == k].mean() if (h_pop == k).any() else sex_pop.mean() for k in np.unique(h)}
    return np.asarray([shares[k] for k in h])


def features_imputation(X_pop, y_pop, h_pop, sex_pop, X, y, h, n_sites: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    def design(X_, y_, h_):
        others = np.delete(X_, SEX, axis=1)
        return np.column_stack([others, y_, np.eye(n_sites)[h_]])

    if len(np.unique(sex_pop)) < 2:
        return np.full(len(y), int(sex_pop[0]))
    model = LogisticRegression(max_iter=5000).fit(design(X_pop, y_pop, h_pop), sex_pop)
    return model.predict(design(X, y, h)).astype(int)


def score(pred: np.ndarray, sex: np.ndarray) -> dict:
    women = sex == 0
    called = pred == 0
    return {"accuracy": round(float((pred == sex).mean()), 4),
            "women": int(women.sum()), "women_called_women": int((called & women).sum()),
            "called_women": int(called.sum()),
            "recall_of_women": round(float((called & women).sum() / max(women.sum(), 1)), 4),
            "precision_on_women": round(float((called & women).sum() / called.sum()), 4) if called.any() else None}


def evaluate(theta, sample, members, population, n_sites: int) -> dict:
    """Each attack and imputation on the training rows (population: every test row) and on the test rows (cross-fitted)."""
    Xm, ym, hm, sm = members
    Xp, yp, hp, sp = population
    out = {"members": {}, "non_members": {}}
    preds = {"loss": loss_attack(theta, Xm, ym, hm, sample),
             "model_inversion": inversion_attack(theta, Xm, ym, hm, sample, male_share(hp, sp, hm)),
             "imputation_label_only": label_rule(yp, hp, sp, ym, hm),
             "imputation_all_features": features_imputation(Xp, yp, hp, sp, Xm, ym, hm, n_sites)}
    out["members"] = {k: score(v, sm) for k, v in preds.items()}
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(yp)) % FOLDS
    cross = {k: np.zeros(len(yp), dtype=int) for k in preds}
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        cross["loss"][te] = loss_attack(theta, Xp[te], yp[te], hp[te], sample)
        cross["model_inversion"][te] = inversion_attack(theta, Xp[te], yp[te], hp[te], sample,
                                                        male_share(hp[tr], sp[tr], hp[te]))
        cross["imputation_label_only"][te] = label_rule(yp[tr], hp[tr], sp[tr], yp[te], hp[te])
        cross["imputation_all_features"][te] = features_imputation(Xp[tr], yp[tr], hp[tr], sp[tr], Xp[te], yp[te],
                                                                   hp[te], n_sites)
    out["non_members"] = {k: score(v, sp) for k, v in cross.items()}
    out["always_male"] = {"members": score(np.ones(len(sm), dtype=int), sm),
                          "non_members": score(np.ones(len(sp), dtype=int), sp)}
    w = float(theta[SEX])
    # The identity in the docstring: for a logistic model the loss attack's guess is male exactly when the diagnosis
    # moves the logit the way the weight on sex does.
    rule = np.where(ym == 1, int(w > 0), int(w < 0))
    out["loss_attack_is_the_label_rule"] = bool(np.array_equal(preds["loss"], rule))
    out["weight_on_sex"] = round(w, 6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    from qsh.ai import fed_heart

    t0 = time.time()
    sample = json.loads(fed_heart.SAMPLE.read_text(encoding="utf-8"))
    mia = _mia()
    parts = [(c["train"]["X"], c["train"]["y"]) for c in sample["centres"]]
    members, population = rows_of(sample, "train"), rows_of(sample, "test")
    n_sites = len(sample["centres"])
    models = {"federated": mia.fedheart_federated(parts, 0), "pooled": mia.fedheart_pooled(parts, 0)}
    results = {name: evaluate(theta, sample, members, population, n_sites) for name, theta in models.items()}
    record = {
        "what": "attribute inference of a patient's sex from the published heart models, against imputation",
        "attacks": {"loss": "Yeom, Giacomelli, Fredrikson and Jha, IEEE CSF 2018; implemented here",
                    "model_inversion": "Fredrikson et al., USENIX Security 2014; implemented here",
                    "comparison": "Jayaraman and Evans, ACM CCS 2022 (arXiv:2209.01292)"},
        "data": f"{sample['corpus']} ({sample['licence']}): training rows {len(members[1])} (members), test rows "
                f"{len(population[1])} (the adversary's sample of the population; cross-fitted when they are scored)",
        "models": "seed 0, the page's default, of its federation through the Python twin: FedAvg and pooled",
        "attribute": "sex (UCI: 1 male, 0 female), read back from the standardised column with each hospital's "
                     "published mean and deviation",
        "results": results,
        "reading": "An attack shows disclosure from the model only where it beats the imputation with the same knowledge "
                   "and does better on members than on non-members. Accuracy is beside the always-male rate, because "
                   "most patients here are men; recall and precision on women are the minority value's.",
        "seconds": round(time.time() - t0, 1),
        "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                              text=True).stdout.strip(),
                        "python": platform.python_version(), "numpy": np.__version__, "date": time.strftime("%Y-%m-%d")},
    }
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
