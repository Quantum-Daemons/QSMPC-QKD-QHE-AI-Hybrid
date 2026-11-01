"""The rows the private-inference figure encrypts were never used to fit the model it plots.

WHY THIS EXISTS (Round 84, checklist PI7 and PI8)

`breast_cancer_sample.json` said of its own regression "fitted on the rows shipped here", and the
figure encrypting that regression over those rows called them "held-out". Finance's figure did the
same with a PaySim surrogate. The payloads now carry a real split, and this checks it from the data
side, independently of the browser code (`test_r84_c3_encrypted_heldout_in_node.py` runs that):

  medical   the regression REFITS on `samples` alone to the shipped coefficients, its scaler refits on
            the same rows to every shipped z-score, and `heldout` shares no row with `samples`; one
            class encoding with the overlay (1 = malignant); the federated demo keeps enough rows.
  finance   `student_heldout.json` is the overlay's 8,192 test-window rows in order, through the
            published student's own preprocessing, with byte-identical weights.
  telecom   `student_heldout.json` and the overlay are the seed-42 split's held-out quarter, not the
            3,200-row corpus a QCNN trains on.

On the baseline every test here fails: there is no `heldout` block, no held-out payload, and the
telecom overlay draws 3,200 rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "web" / "public" / "data"
MED = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
       "worst radius", "worst concavity", "worst symmetry"]


def _load(rel: str) -> dict:
    path = PUBLIC / rel
    assert path.exists(), f"{rel} is not shipped"
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------------------- #
#  medical
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def wdbc():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    idx = [list(d.feature_names).index(n) for n in MED]
    return d.data[:, idx], (d.target == 0).astype(int), d


def test_medical_ships_a_split_the_model_was_fitted_on_one_side_of(wdbc) -> None:
    X, y, _ = wdbc
    ds = _load("medical/breast_cancer_sample.json")
    h = ds.get("heldout")
    assert h and h["n"] == len(h["samples"]) == len(h["labels"]) == 171, "no 171-row held-out block"
    assert len(ds["samples"]) == 398
    held = set(h["row_index"])
    fit = set(range(len(y))) - held
    assert len(fit) == 398 and not fit & held
    assert "fitted on the rows shipped here" not in ds["real_model"], ds["real_model"]


def test_medical_regression_and_scaler_refit_on_the_fitting_rows_alone(wdbc) -> None:
    """The claim the caption prints, re-derived: refit on `samples` only and get the shipped model."""
    from sklearn.linear_model import LogisticRegression

    X, y, _ = wdbc
    ds = _load("medical/breast_cancer_sample.json")
    h = ds["heldout"]
    fit = np.array(sorted(set(range(len(y))) - set(h["row_index"])))
    mu, sd = X[fit].mean(axis=0), X[fit].std(axis=0)
    z_held = (X[np.array(h["row_index"])] - mu) / sd
    assert np.abs(z_held - np.array(h["samples"])).max() < 1e-4, (
        "the held-out z-scores are not the fitting rows' scaler applied to them: the constants came "
        "from somewhere else, possibly including the held-out rows")
    assert np.abs((X[fit] - mu) / sd - np.array(ds["samples"])).max() < 1e-4
    lr = LogisticRegression(max_iter=500, class_weight="balanced").fit(np.array(ds["samples"]), ds["labels"])
    assert np.abs(lr.coef_[0] - np.array(ds["model"]["W"][1])).max() < 1e-3, (
        "refitting on `samples` does not reproduce the shipped coefficients, so the model was not "
        "fitted on exactly the fitting rows")
    assert ds["model"]["W"][0] == [0.0] * 8 and ds["model"]["b"][0] == 0.0


def test_medical_class_encoding_and_plane_match_the_overlay(wdbc) -> None:
    """One page, one encoding (1 = malignant), and the plane points are the overlay's own values."""
    X, y, d = wdbc
    ds = _load("medical/breast_cancer_sample.json")
    overlay = _load("medical/real_2d.json")
    h = ds["heldout"]
    assert ds["class_names"] == overlay["class_names"] == ["benign", "malignant"]
    assert h["labels"] == [int(y[i]) for i in h["row_index"]]
    assert ds["labels"] == [int(y[i]) for i in sorted(set(range(len(y))) - set(h["row_index"]))]
    for (px, py), i in zip(h["plane"], h["row_index"], strict=True):
        ox, oy, oc = overlay["points"][i]
        assert (px, py, h["labels"][h["row_index"].index(i)]) == (ox, oy, oc), (
            f"row {i}: held-out plane point {(px, py)} is not the overlay's {(ox, oy)}")


def test_medical_file_carries_the_licence_link_and_change_note() -> None:
    ds = _load("medical/breast_cancer_sample.json")
    assert ds["licence_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert "recoded" in ds["modifications"] and "398" in ds["modifications"]


# --------------------------------------------------------------------------------------------- #
#  finance
# --------------------------------------------------------------------------------------------- #

def test_finance_held_out_rows_are_the_overlays_rows_through_the_published_student() -> None:
    ho = _load("finance/student_heldout.json")
    overlay = _load("finance/real_2d.json")
    weights = PUBLIC / "finance" / "student_weights.json"
    assert weights.read_bytes() == (REPO / "models" / "finance_aml_student" / "weights.json").read_bytes(), (
        "the bundled student weights are not the published file")
    assert ho["model_key"] == "finance_aml.student" and ho["weights"] == "data/finance/student_weights.json"
    names = ho["feature_names"]
    xi, yi = names.index(overlay["x"]["key"]), names.index(overlay["y"]["key"])
    assert len(ho["samples"]) == len(overlay["points"]) == 8192
    for row, label, (ox, oy, oc) in zip(ho["samples"], ho["labels"], overlay["points"], strict=True):
        assert label == oc
        assert abs(float(f"{row[xi]:.4g}") - ox) <= 1e-9 * max(1, abs(ox)) and \
            abs(float(f"{row[yi]:.4g}") - oy) <= 1e-9 * max(1, abs(oy))
    # The published student's own preprocessing, on these rows, reproduces its published z-scores.
    w = json.loads(weights.read_text())
    assert w["input"]["preprocessing"] == ["subtract mean", "divide by scale"]
    z = (np.array(ho["samples"]) - np.array(w["input"]["mean"])) / np.array(w["input"]["scale"])
    pub = np.array(json.loads((REPO / "models" / "finance_aml_student" / "eval_sample.json").read_text())["samples"])
    assert np.abs(z - pub).max() < 1e-5
    logit = z @ np.array(w["W"][0]) + w["b"][0]
    assert int((logit > 0).sum()) == ho["expected"]["plaintext_flagged"]
    assert ho["licence_url"] == "https://cdla.dev/sharing-1-0/" and ho["modifications"]


# --------------------------------------------------------------------------------------------- #
#  telecom
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def seed42():
    from sklearn.model_selection import train_test_split

    from qsh.qml_ids.attacks import FEATURES, generate
    X, y, _ = generate(n_per_class=300, seed=42)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42, stratify=y)
    return X, y, tr, te, list(FEATURES)


def test_telecom_held_out_rows_are_the_students_own_test_split(seed42) -> None:
    X, y, tr, te, feats = seed42
    ho = _load("telecom/student_heldout.json")
    assert ho["model_key"] == "telecom_ids.student" and ho["feature_names"] == feats
    S = np.array(ho["samples"])
    assert S.shape == (600, 7) and np.abs(S - X[te]).max() <= 1e-6 * max(1.0, float(np.abs(X).max()))
    assert ho["labels"] == [int(v) for v in y[te]]
    # None of them is a training row of the student.
    train_rows = {tuple(np.round(r, 9)) for r in X[tr]}
    assert not train_rows & {tuple(np.round(r, 9)) for r in X[te]}


def test_telecom_overlay_draws_the_held_out_split_not_the_training_corpus(seed42) -> None:
    X, y, tr, te, feats = seed42
    overlay = _load("telecom/real_2d.json")
    assert len(overlay["points"]) == 600, (
        f"the telecom overlay draws {len(overlay['points'])} rows; 3,200 is the whole seed-0 corpus "
        "a QCNN trains on, 70 percent of it training rows")
    xi, yi = feats.index("qber_std"), feats.index("multiphoton_rate")
    for (ox, oy, oc), r, t in zip(overlay["points"], X[te], y[te], strict=True):
        assert (ox, oy) == (float(f"{r[xi]:.4g}"), float(f"{r[yi]:.4g}")) and oc == int(t != 0)
    assert overlay["models"] == ["telecom_ids.qml", "telecom_ids.student"]
    assert "held out" in overlay["split_note"]
