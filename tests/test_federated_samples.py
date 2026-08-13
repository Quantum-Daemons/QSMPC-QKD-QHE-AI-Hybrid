"""The bundled federated samples must be big enough, and on a common scale, to mean anything.

WHY THIS EXISTS

``federated_accuracy`` is computed in the browser on a held-out slice of these two files. They
shipped 16 rows (finance) and 12 (medical), so with the configured split that accuracy was a **4-row**
and a **3-row** score - and measured with the real ``clientQFL`` at five seeds each, it was worse
than doing nothing on every one of them:

    finance  untrained 0.75 -> trained 0.25, identical on all six rounds, at 4 of 5 seeds
    medical  untrained 1.00 -> trained 0.00, identical on all six rounds, at seed 1

Nothing caught it because nothing looked. The number rendered in a metrics table, the finance page
had no figure for the stage at all until Round 36, and no test asserted anything about the files
behind it.

The second half was scale. ``amount`` is a log-amount spanning 0 to 17.69 while ``amount_zscore`` in
the same eight columns is already standardised, and the medical file has always shipped standardised.
The browser's feature map is ``tanh(mean_i sin(x_i + i/2))``: an input spanning 17.7 radians wraps
``sin`` nearly three times, so two transactions 2*pi apart in log-amount produce the same feature.
Class separation in that map went from 0.0233 to 0.2180 on the leading component when the finance
rows were standardised - the same order as medical's.

WHAT THIS ASSERTS, AND WHY EACH ONE

Row count is tied to ``config.federated.test_split`` rather than to a number written here, so
changing the split cannot quietly shrink the evaluation set. Scale is asserted as a bound both files
must satisfy, which is what "on a common scale" means operationally and what the old finance file
failed. Both are properties of the artefact, so ``scripts/data/build_federated_samples.py --rows 16``
fails here rather than on the page.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qsh.config import get_config
from qsh.models.registry import DEMO_SAMPLE_FILES

ROOT = Path(__file__).resolve().parents[1]
FEDERATED = ["finance_aml", "medical_fl"]

#: Smallest held-out set worth calling an evaluation. At 50 rows a reported 0.90 carries a normal
#: approximate 95 % interval of about +/-8 points; at 4 rows the interval spans essentially the whole
#: scale and a single row moves the figure by 25.
MIN_EVAL_ROWS = 50

#: No shipped feature may exceed this in absolute value. Standardised columns sit near +/-3; the old
#: finance file reached 17.69, which is where the feature map's `sin` wraps. Deliberately loose - it
#: rejects "this column was never scaled", not "this column has an outlier".
MAX_ABS_FEATURE = 12.0


def _payload(usecase: str) -> dict:
    path = ROOT / DEMO_SAMPLE_FILES[usecase]
    assert path.exists(), f"{path} is registered but not shipped"
    return json.loads(path.read_text())


@pytest.mark.parametrize("usecase", FEDERATED)
def test_the_held_out_split_is_large_enough_to_be_an_evaluation(usecase: str) -> None:
    rows = len(_payload(usecase)["samples"])
    split = get_config().federated.test_split
    held_out = int(rows * split)
    assert held_out >= MIN_EVAL_ROWS, (
        f"{DEMO_SAMPLE_FILES[usecase]} ships {rows} rows, so a {split:.0%} holdout is {held_out} "
        f"rows and `federated_accuracy` is a {held_out}-row score. Regenerate with "
        f"`scripts/data/build_federated_samples.py --rows N` for N >= {int(MIN_EVAL_ROWS / split)}."
    )


@pytest.mark.parametrize("usecase", FEDERATED)
def test_the_features_are_on_a_common_scale(usecase: str) -> None:
    """Both files, one bound - the mismatch between them was the defect, not either file alone."""
    payload = _payload(usecase)
    worst, where = 0.0, ""
    for row in payload["samples"]:
        # strict: a row shorter than `names` would silently stop the scan early and pass this test
        # by not looking at the column that broke it.
        for name, value in zip(payload["names"], row, strict=True):
            if abs(value) > worst:
                worst, where = abs(value), name
    assert worst <= MAX_ABS_FEATURE, (
        f"{DEMO_SAMPLE_FILES[usecase]} carries |{where}| up to {worst:.2f}. The browser feature map "
        f"is tanh(mean sin(x + i/2)), so past a few radians the mapping stops being monotone in the "
        f"feature and two rows far apart score identically. Standardise in "
        f"`scripts/data/build_federated_samples.py`, do not raise this bound."
    )


@pytest.mark.parametrize("usecase", FEDERATED)
def test_both_classes_are_present_and_the_accuracy_is_readable(usecase: str) -> None:
    labels = _payload(usecase)["labels"]
    counts = {c: labels.count(c) for c in set(labels)}
    assert set(counts) == {0, 1}, f"{DEMO_SAMPLE_FILES[usecase]} does not ship both classes: {counts}"
    minority = min(counts.values()) / len(labels)
    assert minority >= 0.3, (
        f"{DEMO_SAMPLE_FILES[usecase]} is {counts}, minority share {minority:.2f}. Balanced is what "
        "makes a plain accuracy readable: against PaySim's own 0.13 % base rate, predicting 'never "
        "fraud' scores 99.87 % and the number would say nothing about the model."
    )


@pytest.mark.parametrize("usecase", FEDERATED)
def test_the_model_matches_the_feature_space_it_ships_with(usecase: str) -> None:
    """`realInput` hands these rows AND this model to private inference and to exact-linear SHAP."""
    payload = _payload(usecase)
    n_features = len(payload["names"])
    assert len(payload["samples"][0]) == n_features
    for row in payload["model"]["W"]:
        assert len(row) == n_features, (
            f"{DEMO_SAMPLE_FILES[usecase]}'s model has {len(row)} coefficients for {n_features} "
            "features. Regenerating the rows without refitting the model applies weights fitted on "
            "one feature space to another."
        )


@pytest.mark.parametrize("usecase", FEDERATED)
def test_the_payload_reports_its_own_eda(usecase: str) -> None:
    """A degenerate column is kept and DECLARED, not silently carried.

    ``velocity_24h`` is non-zero in 2 of the 4,000 bundled PaySim rows. Dropping it would change the
    eight-column order the backend, the registry and the published QCNN weights all share; carrying
    it unannounced is how the next person spends an afternoon on a feature that never moves.
    """
    eda = _payload(usecase).get("eda")
    assert eda and eda.get("columns"), f"{DEMO_SAMPLE_FILES[usecase]} carries no eda block"
    assert len(eda["columns"]) == len(_payload(usecase)["names"])
    assert "degenerate" in eda, "the eda block does not say which columns carry no signal"
    for column in eda["columns"]:
        assert {"name", "min", "max", "mean", "sd", "n_unique"} <= set(column)
