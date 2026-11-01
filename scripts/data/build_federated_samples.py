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

WHICH FINANCE COLUMNS THE MODEL READS

Four of the TSV's eight: ``amount``, ``cross_border``, ``night_tx`` and ``new_payee``
(``registry.PAYSIM_MODEL_FEATURES`` says what each is in PaySim's terms). The other four are left
out of ``samples`` and of the model, for the reasons ``registry.PAYSIM_EXCLUDED_FEATURES`` gives:
``device_risk`` and ``geo_mismatch`` are computed from PaySim's four balance columns, which the
dataset's own description says must not be used for fraud detection because fraudulent
transactions are cancelled, so a model reading them learns the label from the simulator's
bookkeeping; ``amount_zscore`` is an affine copy of ``amount``; ``velocity_24h`` is non-zero in 2
of the 4,000 TSV rows and in none of the 400 shipped here.

The procedure is unchanged: the same 400 class-balanced rows (seed 0), a scaler fitted on the first
70 percent of them, the XGBoost teacher and the logistic student both fitted on all 400. No held-out
split was carved out of them for the model before and none is now; the payload's ``evaluation``
block reports the fitting-row figures beside a 5-fold cross-validation of the same procedure and
the false-positive rate on the TSV's legitimate rows that were not drawn, so neither is mistaken
for the other.

``samples_by_space.raw`` holds the SAME four columns, in the same order, unstandardised. That block
feeds the finance QCNN artifact (``models/qcnn_finance``), which since its Round 84 retrain reads
these four in raw units, in this order (its ``feature_names``), and standardises them with its own
training moments; it was trained on all eight until then, and the eight-wide block, with the
``names_by_space`` and ``eda_by_space`` entries that named and described it, is gone.

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

from qsh.models.registry import (  # noqa: E402
    BUNDLED_DEMO_SAMPLES,
    DEMO_SAMPLE_FILES,
    DS_PAYSIM,
    DS_WDBC,
    PAYSIM_BALANCE_WARNING,
    PAYSIM_EXCLUDED_FEATURES,
    PAYSIM_MODEL_FEATURES,
    PAYSIM_SAMPLE_COLUMNS,
)

#: The TSV's eight columns, in file order: the row layout this builder parses.
FIN_TSV_NAMES = list(PAYSIM_SAMPLE_COLUMNS)
#: The four the bundled linear model reads, in this order (`samples`, `names`, `model.W`), and the
#: four the finance QCNN reads in the same order (`samples_by_space.raw`, its `feature_names`).
FIN_NAMES = list(PAYSIM_MODEL_FEATURES)

#: The eight WDBC columns the medical demo has always shipped, by scikit-learn feature name.
MED_NAMES = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
             "worst_radius", "worst_concavity", "worst_symmetry"]

#: Fraction of the shipped rows the standardiser is fitted on. NOT a train/test protocol: the browser
#: reshuffles with its own run seed, so this cannot be made to coincide with its split. What it buys
#: is that the constants applied to the redistributed rows were not computed from every one of them,
#: which is the same discipline - and the same fraction - as `loaders.load_breast_cancer`.
SCALER_FIT_FRACTION = 0.7

#: WHY THE MEDICAL FILE NOW SPLITS (Round 84). The private-inference figure named its 400 medical
#: rows "held-out" while the regression it plotted had been fitted on exactly those 400 (this file's
#: own `real_model` said "fitted on the rows shipped here"). Two evaluations of one model on its
#: fitting rows do measure the encryption, but the word invited reading the figure as generalisation.
#:
#: The split is the loader's own: `loaders.load_breast_cancer` fits its scaler on the first
#: SCALER_FIT_FRACTION of the 569 rows, int(569 * 0.7) = 398, and the regression is fitted on those
#: same 398. The other 171 touched neither, so the figure can call them held out and be right, and
#: the z-scores stay the loader's - which the medical QCNN artifact and the Python medical_fl
#: pipeline both assume. A stratified random split was tried first and rejected for that reason: it
#: needs its own scaler, and every consumer that maps these z-scores back to raw units with the
#: loader's constants would then decode the wrong cell measurements. The cost is balance: the held-out
#: tail is 22.8 percent malignant against 43.5 in the fitting rows (37.3 overall), stated in the file.

#: Licence text the CC BY 4.0 credit asks to accompany the data (section 3(a)(1)(C)): the licence's
#: URI. Written into the file, not only the page, because the file is what gets copied.
CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"
CC_BY_SA_4 = "https://creativecommons.org/licenses/by-sa/4.0/"


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

    ``degenerate`` flags a column carrying almost no information. It is reported, not acted on: which
    columns the finance model leaves out is decided by name in the registry
    (``PAYSIM_EXCLUDED_FEATURES``), so a flag computed on one draw cannot silently change the model.
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


def _fit_paysim_surrogate(X: np.ndarray, y: np.ndarray, seed: int):
    """The demo's procedure, written once so the shipped model and its cross-validation are the same
    thing: a scaler on the first SCALER_FIT_FRACTION of the rows, the XGBoost teacher on all of them,
    and the logistic student on the teacher's labels."""
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cut = max(1, int(len(X) * SCALER_FIT_FRACTION))
    scaler = StandardScaler().fit(X[:cut])
    # Every column the model reads must move in the scaler's rows: a zero-variance column is left at
    # scale 1 by StandardScaler, so a value it never saw there becomes a spike of arbitrary size.
    assert float(scaler.var_.min()) > 1e-12, (
        f"a model column is constant in the scaler's rows: {scaler.var_.tolist()}")
    Xs = scaler.transform(X)
    teacher = xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.2, subsample=0.9,
                                eval_metric="logloss", n_jobs=2, random_state=seed).fit(Xs, y)
    student = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xs, teacher.predict(Xs))
    return scaler, teacher, student


def _paysim_evaluation(X: np.ndarray, y: np.ndarray, X_not_drawn: np.ndarray,
                       y_not_drawn: np.ndarray, seed: int) -> dict:
    """What the shipped model scores, and on which rows, kept apart.

    `fitting_rows` is in-sample: the 400 rows the scaler, the teacher and the student were all
    fitted on. `cross_validated` refits the WHOLE procedure inside each of 5 stratified folds and
    scores the rows each fit did not see. `legit_rows_not_drawn` applies the shipped model to the
    TSV's rows outside the 400, which are all legitimate (the 400 already hold every fraud row), so
    only a false-positive rate can be measured there. None of the three is a PaySim-wide estimate:
    the rows are a 50/50 draw from a 5 percent fraud TSV, itself drawn from a corpus where fraud is
    rarer still.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    r4 = lambda v: round(float(v), 4)   # noqa: E731

    scaler, teacher, student = _fit_paysim_surrogate(X, y, seed)
    Xs = scaler.transform(X)
    margin = Xs @ student.coef_[0] + student.intercept_[0]
    pred, tpred = (margin > 0).astype(int), teacher.predict(Xs)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_margin, cv_teacher = np.zeros(len(y)), np.zeros(len(y), dtype=int)
    for tr, te in folds.split(X, y):
        sc, t, s = _fit_paysim_surrogate(X[tr], y[tr], seed)
        Z = sc.transform(X[te])
        cv_margin[te] = Z @ s.coef_[0] + s.intercept_[0]
        cv_teacher[te] = t.predict(Z)
    cv_pred = (cv_margin > 0).astype(int)

    assert int(y_not_drawn.sum()) == 0, "the rows outside the draw hold fraud rows; say so here"
    nd_margin = scaler.transform(X_not_drawn) @ student.coef_[0] + student.intercept_[0]
    flagged = int((nd_margin > 0).sum())
    return {
        "fitting_rows": {"rows": int(len(y)), "accuracy": r4((pred == y).mean()),
                         "fraud_f1": r4(f1_score(y, pred)), "roc_auc": r4(roc_auc_score(y, margin)),
                         "teacher_accuracy": r4((tpred == y).mean()),
                         "distillation_fidelity": r4((pred == tpred).mean())},
        "cross_validated": {"folds": 5, "accuracy": r4((cv_pred == y).mean()),
                            "fraud_f1": r4(f1_score(y, cv_pred)),
                            "roc_auc": r4(roc_auc_score(y, cv_margin)),
                            "teacher_accuracy": r4((cv_teacher == y).mean())},
        "legit_rows_not_drawn": {"rows": int(len(y_not_drawn)), "flagged": flagged,
                                 "false_positive_rate": r4(flagged / max(1, len(y_not_drawn)))},
        "note": ("fitting_rows is in-sample. cross_validated refits the scaler, the teacher and the "
                 f"student inside each of 5 stratified folds (seed {seed}) and scores the held-out fold. "
                 "legit_rows_not_drawn applies the shipped model to the bundled TSV's rows outside "
                 "the 400, all of them legitimate. The rows are a 50/50 draw, so none of these is "
                 "an estimate at PaySim's own fraud rate."),
    }


def build_finance(n: int, seed: int) -> dict:
    tsv = REPO / "data" / "finance" / "paysim_sample.tsv"
    rows = [ln for ln in tsv.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    data = np.asarray([[float(v) for v in ln.split("\t")] for ln in rows[1:]], dtype=np.float64)
    X_all, y_all = data[:, :8], data[:, 8].astype(int)
    assert rows[0].split("\t")[:8] == FIN_TSV_NAMES, f"TSV column order changed: {rows[0]}"
    keep = [FIN_TSV_NAMES.index(name) for name in FIN_NAMES]

    rng = np.random.default_rng(seed)
    pick = _stratified(y_all, n, rng)
    X8, y = X_all[pick], y_all[pick]
    # THE MODEL READS FOUR COLUMNS (module docstring). The draw above is over all eight, exactly as
    # before, so the 400 rows are the same transactions the eight-column model was fitted on.
    X = X8[:, keep]
    not_drawn = np.setdiff1d(np.arange(len(y_all)), pick)

    # Standardise. The medical sample has always shipped standardised and this one never did, so the
    # same browser feature map saw one page's inputs in radians and the other's in log-currency.
    # The demo model is refit on the SHIPPED feature space. It has to be: `realInput` hands these rows
    # and this model to the CKKS private-inference path and to the exact-linear SHAP, so a model
    # fitted on unstandardised columns would be applied to standardised ones.
    scaler, _teacher, student = _fit_paysim_surrogate(X, y, seed)
    Xs = scaler.transform(X)
    eda = _eda(Xs, FIN_NAMES)
    eda["note"] = ("Per-column statistics of `samples`, the four standardised columns the linear "
                   "model reads. `degenerate` names columns that carry effectively no signal and "
                   "`duplicates` names pairs that are the same column twice. The columns left out "
                   "of the model are in `excluded_features`; samples_by_space.raw holds the same "
                   "four columns unstandardised, the finance QCNN's input.")
    return {
        "provenance": BUNDLED_DEMO_SAMPLES["finance_aml"][0],
        # What CC BY-SA 4.0 asks to travel with Adapted Material (sections 3(a)(1) and 3(b)): the
        # creator, the licence's URI and a note of the changes. The file named the licence in prose
        # only, and a copy of the file is what a reuser actually holds.
        "licence": "CC-BY-SA-4.0",
        "licence_url": CC_BY_SA_4,
        "attribution": ("E. A. Lopez-Rojas, A. Elmir and S. Axelsson, 'PaySim: A financial mobile "
                        "money simulator for fraud detection', 28th European Modeling and "
                        "Simulation Symposium (EMSS), Larnaca, Cyprus, 2016. Data: Kaggle "
                        "ealaxi/paysim1."),
        "modifications": (f"{n} rows drawn class-balanced from the bundled data/finance/"
                          "paysim_sample.tsv (itself a fraud-enriched subset of the Kaggle file), "
                          "whose eight columns are quantities computed from PaySim fields under "
                          f"AML-style names; `samples` keeps {len(FIN_NAMES)} of them, standardised "
                          "and rounded to 5 decimals, and samples_by_space.raw the same "
                          f"{len(FIN_NAMES)}, unstandardised and rounded to 5 decimals. Adapted "
                          "Material, shared under CC BY-SA 4.0."),
        "names": FIN_NAMES,
        # What each name means in PaySim's terms, and what was left out and why. The names relabel
        # PaySim quantities (`cross_border` is a transaction-type flag), so a file that shipped the
        # names alone would let a reader take them at their word.
        "feature_definitions": {name: PAYSIM_MODEL_FEATURES[name] for name in FIN_NAMES},
        "excluded_features": dict(PAYSIM_EXCLUDED_FEATURES),
        "exclusion_source": {"quote": PAYSIM_BALANCE_WARNING, "url": DS_PAYSIM.url,
                             "applies_to": ["device_risk", "geo_mismatch"]},
        "samples": [[round(float(v), 5) for v in r] for r in Xs],
        # THE SAME ROWS IN THE SPACE THE QCNN ARTIFACT DECLARES, and the reason both are needed.
        #
        # `samples` is StandardScaler output and must stay that way: the distilled CKKS-linear model
        # above is fitted on it, and the exact-linear SHAP panel attributes in the same space. But
        # `models/qcnn_finance/weights.json` declares `input_space: "raw"` - its moments were fitted
        # by the trainer on its own training split of the raw TSV - so the browser standardises
        # whatever it is handed a SECOND time, with a DIFFERENT scaler (the trainer's, not this
        # file's 70% fit).
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
        #
        # THE MODEL'S FOUR, IN `names` ORDER (Round 84). The artifact was retrained on the four
        # columns `samples` carries, and its `feature_names` lists them in this order; the browser
        # hands it this row by position, so the block is exactly as wide as `samples` and is named
        # by `names`. It was the TSV's eight while the artifact read the eight.
        "feature_space": "zscore",
        "samples_by_space": {"raw": [[round(float(v), 5) for v in r] for r in X]},
        "labels": [int(v) for v in y],
        "model": {"W": [[0.0] * len(FIN_NAMES), [float(w) for w in student.coef_[0]]],
                  "b": [0.0, float(student.intercept_[0])]},
        "real_model": ("XGBoost (n_estimators=80, depth=4) distilled to a CKKS-linear logistic "
                       "surrogate, both fitted on the four standardised columns in `samples`"),
        "evaluation": _paysim_evaluation(X, y, X_all[not_drawn][:, keep], y_all[not_drawn], seed),
        "scaling": (f"StandardScaler fitted on the first {SCALER_FIT_FRACTION:.0%} of these rows and "
                    "applied to all of them, so the constants were not computed from every row. The "
                    "browser reshuffles with its own run seed, so this is not - and cannot be - "
                    "aligned with any particular train/test split it makes."),
        "eda": eda,
    }


def build_medical(n: int, seed: int) -> dict:
    """WDBC split ONCE into the rows the bundled regression is fitted on and the rows it is not.

    `samples` / `labels` are the 398 fitting rows: the federated demo trains on them (it holds out
    its own share, as before) and the SHAP reference point is their mean. `heldout` is the other
    171, which touched neither the regression nor the scaler, and they are what the browser scores
    under CKKS for the private-inference figure. `n` and `seed` are ignored here: the split is the
    loader's fixed prefix (see SCALER_FIT_FRACTION), over the whole cohort.

    THE CLASS IS RECODED TO 1 = MALIGNANT. scikit-learn ships 0 = malignant; `real_2d.json` and the
    medical QCNN both use 1 = malignant, so one page carried two encodings and an overlay drawing
    this file's rows beside the other would have swapped the colours of the two diagnoses.
    """
    from sklearn.datasets import load_breast_cancer as _lbc
    from sklearn.linear_model import LogisticRegression

    from qsh.data.loaders import load_breast_cancer

    del n, seed                                # see the docstring: a fixed split of the cohort
    ds = _lbc()
    # The loader's z-scores, one definition of "the medical features": its scaler is fitted on
    # rows [0, cut) only, which is the fitting side of this split by construction.
    X_all, _ = load_breast_cancer(scaler_fit_fraction=SCALER_FIT_FRACTION)
    cut = max(1, int(len(ds.data) * SCALER_FIT_FRACTION))
    feature_names = [name.replace(" ", "_") for name in ds.feature_names]
    idx = [feature_names.index(name) for name in MED_NAMES]
    Z = np.asarray(X_all, dtype=np.float64)[:, idx]
    y_all = (np.asarray(ds.target) == 0).astype(int)          # 1 = malignant
    fit_idx, held_idx = np.arange(cut), np.arange(cut, len(y_all))
    # The claim the figure prints, checked rather than assumed: refit the loader's scaler on the
    # fitting rows alone and it must reproduce every shipped z-score, held-out rows included.
    mu, sd = ds.data[:cut, idx].mean(axis=0), ds.data[:cut, idx].std(axis=0)
    assert float(np.abs((ds.data[:, idx] - mu) / sd - Z).max()) < 1e-4, (
        "load_breast_cancer no longer fits its scaler on the first rows only; the held-out rows "
        "would be scaled by constants they contributed to")

    student = LogisticRegression(max_iter=500, class_weight="balanced").fit(Z[fit_idx], y_all[fit_idx])
    # The overlay's own plane, in its own units, so the browser can draw the held-out rows where
    # `real_2d.json` draws them without a second copy of the scaler.
    names = list(ds.feature_names)
    xi, yi = names.index("mean radius"), names.index("mean texture")

    def plane(rows: np.ndarray) -> list[list[float]]:
        return [[float(f"{ds.data[i, xi]:.4g}"), float(f"{ds.data[i, yi]:.4g}")] for i in rows]

    def zs(rows: np.ndarray) -> list[list[float]]:
        return [[round(float(v), 5) for v in Z[i]] for i in rows]

    # THE SAME ROWS IN SOURCE UNITS (Round 84, for the QCNN owner's note C2-for-integrator.md 1). The
    # medical QCNN now declares `input_space: "raw"` and standardises with its own trainer's
    # constants, and this file's z-scores are made with a different scaler, so it must be fed raw
    # values by name. `.6g` is lossless for WDBC, whose values carry at most 4 significant figures.
    X_raw = np.asarray(ds.data, dtype=np.float64)[:, idx]

    def raw(rows: np.ndarray) -> list[list[float]]:
        return [[float(f"{v:.6g}") for v in X_raw[i]] for i in rows]

    held_pred = (Z[held_idx] @ student.coef_[0] + student.intercept_[0] > 0).astype(int)
    share = lambda rows: round(float(y_all[rows].mean()), 4)   # noqa: E731
    return {
        "provenance": BUNDLED_DEMO_SAMPLES["medical_fl"][0],
        # What CC BY 4.0 asks to travel with shared data (section 3(a)(1)): the creator as the
        # provider asks, the licence's URI and a note of the changes. The registry's credit, not a
        # second copy of it: one string, in the order and year UCI asks for.
        "licence": "CC-BY-4.0",
        "licence_url": CC_BY_4,
        "attribution": DS_WDBC.attribution,
        "modifications": (f"Eight of the 30 columns, standardised with constants fitted on the first "
                          f"{cut} rows (scikit-learn order) and rounded to 5 decimals; class recoded "
                          f"to 1 = malignant (the source codes 0 = malignant); split into those {cut} "
                          f"rows and the last {len(held_idx)}. samples_by_space.raw keeps the eight "
                          "columns in source units, and heldout.plane mean radius and mean texture, "
                          "rounded to 4 significant figures."),
        "names": MED_NAMES,
        "class_names": ["benign", "malignant"],
        "samples": zs(fit_idx),
        # The space `samples` is in, named rather than implied. The browser keys its per-space row
        # map on this, so an artifact asks for the space it was trained on BY NAME.
        "feature_space": "zscore",
        "samples_by_space": {"raw": raw(fit_idx)},
        "labels": [int(v) for v in y_all[fit_idx]],
        "model": {"W": [[0.0] * 8, [float(w) for w in student.coef_[0]]],
                  "b": [0.0, float(student.intercept_[0])]},
        "real_model": (f"logistic regression over the eight shipped WDBC columns, fitted on the "
                       f"{cut} rows in `samples`; the {len(held_idx)} rows in `heldout` were not "
                       "used to fit it or its scaler"),
        # Page-facing, read by the private-inference figure's caption: what the model WAS fitted on.
        "fit_note": (f"the regression and its scaler were fitted on rows 0 to {cut - 1} of the "
                     f"569-row cohort, and these are rows {cut} to {len(y_all) - 1}."),
        "scaling": (f"StandardScaler fitted on the first {SCALER_FIT_FRACTION:.0%} of the full 569-row "
                    f"cohort (rows 0 to {cut - 1}) by qsh.data.loaders.load_breast_cancer, which is "
                    "where this file's feature definition comes from; those are exactly the rows in "
                    "`samples`, so no held-out row contributed to the constants."),
        "heldout": {
            "n": int(len(held_idx)),
            "note": (f"Rows {cut} to {len(y_all) - 1} of the 569-row cohort in scikit-learn's order, "
                     f"which the regression in `model` was never fitted on and its scaler never saw. "
                     f"{share(held_idx):.1%} malignant, against {share(fit_idx):.1%} in the fitting "
                     "rows: WDBC's order is not perfectly exchangeable, and the split is the loader's "
                     "fixed prefix rather than a stratified draw. The browser scores these rows "
                     "under CKKS."),
            "split": (f"the last {len(held_idx)} of the 569 rows in scikit-learn's order, "
                      f"{share(held_idx):.1%} malignant against {share(fit_idx):.1%} in the fitting "
                      "rows (a fixed prefix split, not a stratified draw)"),
            "samples": zs(held_idx),
            "samples_by_space": {"raw": raw(held_idx)},
            "labels": [int(v) for v in y_all[held_idx]],
            "row_index": [int(i) for i in held_idx],
            "plane": plane(held_idx),
            "plaintext_accuracy": round(float((held_pred == y_all[held_idx]).mean()), 6),
        },
        "plane": {"x": {"key": "mean radius", "label": "mean nucleus radius"},
                  "y": {"key": "mean texture", "label": "mean nucleus texture"}},
        "eda": _eda(Z[fit_idx], MED_NAMES),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=400,
                    help="finance rows (default 400 -> a 20%% holdout is 80 rows); medical always "
                         "splits the whole 569-row cohort 398 / 171")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["finance_aml", "medical_fl"], default=None,
                    help="rebuild one file and leave the other untouched")
    args = ap.parse_args()

    for usecase, build in (("finance_aml", build_finance), ("medical_fl", build_medical)):
        if args.only and usecase != args.only:
            continue
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
