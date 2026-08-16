#!/usr/bin/env python3
"""Build the two bundled federated-training samples the browser federates over.

WHY THIS EXISTS

``web/public/data/medical/breast_cancer_sample.json`` had **three readers and no writer**. It was a
committed artefact nobody could regenerate, which is the same defect the genomics payload comment
records ("a hand-built committed file with three readers and no writer, so the browser's view of this
use case could only be changed by editing JSON and hoping it stayed consistent"). The finance file
had a writer - ``scripts/prep_real_kaggle_data.py`` - but that script needs the Kaggle download, so
in practice it could not be re-run either.

WHAT WAS WRONG WITH WHAT THEY SHIPPED

Both were far too small for the metric computed on them. Finance shipped 16 rows and medical 12, and
``clientQFL`` holds out a fraction of that - so ``federated_accuracy`` was a **4-row** and a **3-row**
score. Measured with the real function before this change, the round-by-round accuracy was CONSTANT
across all six rounds on both, at three seeds each: finance ``[0.25 x6]`` every time, medical
``[0.67 x6]``, ``[0.0 x6]`` and ``[0.33, 0.33, 0.67 x4]``. A federated accuracy of exactly zero was
being computed, rendered and never read.

And the finance features were not on a common scale while the medical ones were. ``amount`` is a
log-amount spanning 0 to 17.69 and ``amount_zscore`` is already standardised, in the same eight
columns. That matters more than it looks: the browser's quantum feature map is
``tanh(mean_i sin(x_i + i/2))``, so an input spanning 17.7 radians wraps ``sin`` almost three times
and two transactions 2*pi apart in log-amount land on the same feature. Measured class separation in
that map, raw versus standardised: ``0.0233 -> 0.2180`` and ``0.0416 -> 0.3051`` on the two
informative components - an order of magnitude, and the same order as medical's.

WHAT THIS SCRIPT DOES NOT DO

It does not drop ``velocity_24h``, which is non-zero in 2 of the 4,000 bundled rows and is therefore
very nearly a constant. Dropping it would change the eight-column contract that ``loaders.load_paysim``,
the registry, the published QCNN weights and ``finance_aml``'s feature order all share, for a column
that costs nothing to carry. It is recorded in the payload's ``eda`` block instead, so the
degeneracy is visible in the artefact rather than discovered by whoever next wonders why a feature
never moves. ``StandardScaler`` leaves zero-variance columns alone (``scale_ = 1``) rather than
dividing by ~0, and this script asserts that it did.

Run:  .venv/bin/python scripts/data/build_federated_samples.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.models.registry import BUNDLED_DEMO_SAMPLES, DEMO_SAMPLE_FILES  # noqa: E402

#: The eight AML features, in the order `finance_aml` and the published weights expect.
FIN_NAMES = ["amount", "velocity_24h", "cross_border", "night_tx",
             "new_payee", "amount_zscore", "device_risk", "geo_mismatch"]

#: The eight WDBC columns the medical demo has always shipped, by scikit-learn feature name.
MED_NAMES = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
             "worst_radius", "worst_concavity", "worst_symmetry"]

#: Fraction of the shipped rows the standardiser is fitted on. NOT a train/test protocol: the browser
#: reshuffles with its own run seed, so this cannot be made to coincide with its split. What it buys
#: is that the constants applied to the redistributed rows were not computed from every one of them,
#: which is the same discipline - and the same fraction - as `loaders.load_breast_cancer`.
SCALER_FIT_FRACTION = 0.7


def _stratified(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """A balanced pick, capped by whichever class is scarcer.

    50/50 is the convention both files already shipped (8 fraud + 8 legit; 5 and 7). It is a further
    oversample on top of the TSV's 5 % - PaySim's own base rate is 0.13 % - and DATA_LICENSES says so.
    Balanced is what makes the accuracy readable at all: against a 0.13 % base rate, predicting
    "never fraud" scores 99.87 % and the number would say nothing about the model.
    """
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    half = min(len(pos), len(neg), n // 2)
    pick = np.concatenate([rng.choice(pos, half, replace=False),
                           rng.choice(neg, min(len(neg), n - half), replace=False)])
    rng.shuffle(pick)
    return pick


def _eda(X: np.ndarray, names: list[str]) -> dict:
    """Per-column statistics, shipped with the data so nobody has to rediscover them.

    ``degenerate`` flags a column carrying almost no information. It is reported rather than removed:
    see the module docstring.
    """
    cols = []
    for i, name in enumerate(names):
        c = X[:, i]
        cols.append({
            "name": name,
            "min": round(float(c.min()), 5), "max": round(float(c.max()), 5),
            "mean": round(float(c.mean()), 5), "sd": round(float(c.std()), 5),
            "n_unique": int(len(np.unique(c))),
        })
    # DUPLICATE PAIRS, computed rather than remembered. `amount` and `amount_zscore` shipped as
    # byte-identical columns for eleven rounds with nothing saying so: two of eight QCNN wires
    # carried one feature, and the exact-linear SHAP panel split that feature's attribution across
    # both as though they were independent evidence. `degenerate` already named the constant column
    # (`velocity_24h`); nothing named the duplicated one, because nobody had looked.
    dupes = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if float(np.max(np.abs(X[:, i] - X[:, j]))) <= 1e-4:
                dupes.append([names[i], names[j]])
    return {
        "columns": cols,
        "degenerate": [c["name"] for c in cols if c["n_unique"] < 2 or c["sd"] < 1e-6],
        "duplicates": dupes,
        "note": ("Per-column statistics of the rows shipped in this file. `degenerate` names columns "
                 "that carry effectively no signal and `duplicates` names pairs that are the same "
                 "column twice; both are kept rather than removed so the feature order stays the "
                 "one the backend, the registry and the published weights all use."),
    }


def build_finance(n: int, seed: int) -> dict:
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tsv = REPO / "data" / "finance" / "paysim_sample.tsv"
    rows = [ln for ln in tsv.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    data = np.asarray([[float(v) for v in ln.split("\t")] for ln in rows[1:]], dtype=np.float64)
    X_all, y_all = data[:, :8], data[:, 8].astype(int)
    assert rows[0].split("\t")[:8] == FIN_NAMES, f"TSV column order changed: {rows[0]}"

    rng = np.random.default_rng(seed)
    pick = _stratified(y_all, n, rng)
    X, y = X_all[pick], y_all[pick]

    # Standardise. The medical sample has always shipped standardised and this one never did, so the
    # same browser feature map saw one page's inputs in radians and the other's in log-currency.
    cut = max(1, int(len(X) * SCALER_FIT_FRACTION))
    scaler = StandardScaler().fit(X[:cut])
    zero_var = np.where(scaler.var_ <= 1e-12)[0]
    assert all(scaler.scale_[i] == 1.0 for i in zero_var), (
        "StandardScaler divided a zero-variance column by its own (near-zero) scale. That turns the "
        "two non-zero velocity_24h rows into a ~45-sigma spike and everything else into a constant."
    )
    Xs = scaler.transform(X)

    # The demo model is refit on the SHIPPED feature space. It has to be: `realInput` hands these rows
    # and this model to the CKKS private-inference path and to the exact-linear SHAP, so a model
    # fitted on unstandardised columns would be applied to standardised ones.
    teacher = xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.2, subsample=0.9,
                                eval_metric="logloss", n_jobs=2, random_state=seed).fit(Xs, y)
    student = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xs, teacher.predict(Xs))
    return {
        "provenance": BUNDLED_DEMO_SAMPLES["finance_aml"][0],
        "names": FIN_NAMES,
        "samples": [[round(float(v), 5) for v in r] for r in Xs],
        # THE SAME ROWS IN THE SPACE THE QCNN ARTIFACT DECLARES, and the reason both are needed.
        #
        # `samples` is StandardScaler output and must stay that way: the distilled CKKS-linear model
        # above is fitted on it, and the exact-linear SHAP panel attributes in the same space. But
        # `models/qcnn_finance/weights.json` declares `input_space: "raw"` - its moments were fitted
        # by the trainer on the whole raw TSV - so the browser standardises whatever it is handed a
        # SECOND time, with a DIFFERENT scaler (the trainer's, not this file's 70% fit).
        #
        # Measured on the shipped pair before this field existed: wire 0 (`amount`) encoded to a
        # tanh range of [-0.99999992, -0.99982701], sd 3.3e-05; 556 of 3200 values saturated; 69 of
        # 400 rows changed class, 25.25%; accuracy 0.8200 as deployed against 0.9125 correctly fed.
        # (An earlier revision of this comment said 17.25% / 0.9875. That pair compares `raw` against
        # `zscore` on the SAME array - one array through two encoders - and describes a different fix,
        # "stop standardising", which the trainer's raw-fitted moments rule out. The pair above is
        # the deployed feed against the correct feed, which is what a visitor actually saw.)
        # The model's most informative input carried nothing at all.
        #
        # The mismatch cannot be undone in the browser - this file ships no numeric scaler constants,
        # and the two scalers differ - so the raw columns have to come from here.
        "feature_space": "zscore",
        "samples_by_space": {"raw": [[round(float(v), 5) for v in r] for r in X]},
        "labels": [int(v) for v in y],
        "model": {"W": [[0.0] * 8, [float(w) for w in student.coef_[0]]],
                  "b": [0.0, float(student.intercept_[0])]},
        "real_model": ("XGBoost (n_estimators=80, depth=4) distilled to a CKKS-linear logistic "
                       "surrogate, fitted on the standardised features shipped here"),
        "scaling": (f"StandardScaler fitted on the first {SCALER_FIT_FRACTION:.0%} of these rows and "
                    "applied to all of them, so the constants were not computed from every row. The "
                    "browser reshuffles with its own run seed, so this is not - and cannot be - "
                    "aligned with any particular train/test split it makes."),
        "eda": _eda(Xs, FIN_NAMES),
    }


def build_medical(n: int, seed: int) -> dict:
    from sklearn.datasets import load_breast_cancer as _lbc
    from sklearn.linear_model import LogisticRegression

    from qsh.data.loaders import load_breast_cancer

    # Already standardised, and standardised the leakage-aware way: `load_breast_cancer` fits its
    # scaler on the first 70 % of rows. Reusing it keeps one definition of "the medical features".
    X_all, y_all = load_breast_cancer(scaler_fit_fraction=SCALER_FIT_FRACTION)
    feature_names = [name.replace(" ", "_") for name in _lbc().feature_names]
    idx = [feature_names.index(name) for name in MED_NAMES]

    rng = np.random.default_rng(seed)
    pick = _stratified(np.asarray(y_all), n, rng)
    X, y = np.asarray(X_all)[np.ix_(pick, idx)], np.asarray(y_all)[pick]
    student = LogisticRegression(max_iter=500, class_weight="balanced").fit(X, y)
    return {
        "provenance": BUNDLED_DEMO_SAMPLES["medical_fl"][0],
        "names": MED_NAMES,
        "samples": [[round(float(v), 5) for v in r] for r in X],
        # The space `samples` is in, named rather than implied. The browser keys its per-space row
        # map on this, so an artifact asks for the space it was trained on BY NAME.
        "feature_space": "zscore",
        "labels": [int(v) for v in y],
        "model": {"W": [[0.0] * 8, [float(w) for w in student.coef_[0]]],
                  "b": [0.0, float(student.intercept_[0])]},
        "real_model": ("logistic regression over the eight shipped WDBC columns, fitted on the rows "
                       "shipped here"),
        "scaling": (f"StandardScaler fitted on the first {SCALER_FIT_FRACTION:.0%} of the full 569-row "
                    "cohort by qsh.data.loaders.load_breast_cancer, which is where this file's "
                    "feature definition comes from."),
        "eda": _eda(X, MED_NAMES),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=400,
                    help="rows per file (default 400 -> a 20%% holdout is 80 rows)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for usecase, build in (("finance_aml", build_finance), ("medical_fl", build_medical)):
        payload = build(args.rows, args.seed)
        out = REPO / DEMO_SAMPLE_FILES[usecase]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload))
        pos = sum(payload["labels"])
        print(f"[{usecase}] wrote {out.relative_to(REPO)} "
              f"({out.stat().st_size:,} B, {len(payload['labels'])} rows, {pos} positive)"
              + (f"  degenerate: {payload['eda']['degenerate']}" if payload["eda"]["degenerate"] else ""))


if __name__ == "__main__":
    main()
