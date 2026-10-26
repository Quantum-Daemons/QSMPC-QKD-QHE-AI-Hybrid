"""`separability_auc_pr` must reproduce on the rows the chart actually plots.

WHY THIS EXISTS

The field is a hardcoded literal in `scripts/prep_real_data_2d.py` - three payloads carry one and
none of them states the ESTIMATOR that produced it. An audit reported the finance figure as not
reproducing, and a first check here agreed. Both were measuring the wrong thing: a single-feature
rank AUC-PR gives 0.0566 against a stated 0.2302, and the stated figure is a JOINT two-feature score.

Measured on the shipped rows, which is why the estimator has to be written down:

    finance    single-feature 0.0566 | RF 2D 5-fold CV 0.2016 | RF 2D in-sample 0.4378 | stated 0.2302
    satellite  single-feature 0.5888 | RF 2D 5-fold CV 0.8057 | RF 2D in-sample 0.9396 | stated 0.8492
    telecom    single-feature 0.9798 | RF 2D 5-fold CV 0.9925 | RF 2D in-sample 1.0000 | stated 0.9915

An unstated choice moves finance's answer by a factor of eight. The number was never wrong; it was
UNFALSIFIABLE, and a figure a reader cannot re-derive is one they have to take on trust - which is
the thing this project's whole verification posture exists to avoid.

ROUND 71: COMPUTED, AND THE BAND NARROWED. The finance points were redrawn from the held-out test
window, and on them the three readings are 0.0621 / 0.1269 / 0.5777. A band of 0.08 around 0.1269
then admitted the single-feature reading, so the gate could no longer tell which estimator produced
the figure. Satellite's literal, 0.8492, was never this estimator's: it is the pair-selection figure,
fitted on the train split, and 0.08 had admitted it as well. Both figures are now computed by
`_separability` in the generator with exactly the estimator below, so they reproduce to the
rounding, and the band is 0.01. Telecom keeps its own recorded estimator, a linear reader, which
lands within 0.001.

WHAT IS ASSERTED

That each stated figure reproduces, on the shipped points, under a cross-validated two-feature
classifier, within a tolerance wide enough for fold and seed variance and narrow enough to exclude
the in-sample and single-feature readings. Both of those are what a re-deriver would plausibly try,
so the band has to rule them out or it is not discriminating.

WHAT IS DELIBERATELY NOT ASSERTED

Not that the chosen feature pair is the best available. It is not, and the generator says so at
length: the finance AUC-PR winner scores 0.3195 and was rejected as unreadable. That is a recorded
decision, not a defect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "web" / "public" / "data"

#: Tolerance on the cross-validated reading. The estimator is seeded and its folds are fixed, so
#: what remains is the payload's 4-significant-figure rounding and telecom's different, recorded
#: estimator (0.9915 against 0.9925). Narrow enough that the in-sample and single-feature readings
#: fall outside it, which the anti-tautology test below checks rather than asserts. It was 0.08
#: until Round 71, which admitted both a mislabelled literal and, on the redrawn finance rows, the
#: single-feature reading.
TOL = 0.01


def _payloads():
    for p in sorted(DATA.glob("*/real_2d.json")):
        d = json.loads(p.read_text())
        if isinstance(d.get("separability_auc_pr"), (int, float)):
            yield p.parent.name, d


def _scores(d):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import cross_val_predict

    pts = np.asarray(d["points"], dtype=float)
    x, y = pts[:, :2], pts[:, 2].astype(int)
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    cv = average_precision_score(y, cross_val_predict(rf, x, y, cv=5, method="predict_proba")[:, 1])
    insample = average_precision_score(y, rf.fit(x, y).predict_proba(x)[:, 1])
    single = max(average_precision_score(y, s * f) for f in (x[:, 0], x[:, 1]) for s in (1.0, -1.0))
    return cv, insample, single


def test_there_are_payloads_to_check():
    """Anti-tautology: an empty glob would make every check below vacuous."""
    got = list(_payloads())
    assert len(got) >= 3, f"only {len(got)} payload(s) carry separability_auc_pr: {[g[0] for g in got]}"


@pytest.mark.parametrize("name", [n for n, _ in _payloads()])
def test_the_stated_figure_reproduces_under_the_stated_estimator(name):
    d = next(v for n, v in _payloads() if n == name)
    stated = d["separability_auc_pr"]
    cv, _insample, _single = _scores(d)
    assert abs(cv - stated) <= TOL, (
        f"{name}: payload states {stated}, a 5-fold cross-validated two-feature random forest on the "
        f"SHIPPED points gives {cv:.4f}. Either the points changed or the figure was produced by a "
        "different estimator - and if it was, say which in the payload, because a reader who cannot "
        "re-derive it has to take it on trust.")


@pytest.mark.parametrize("name", [n for n, _ in _payloads()])
def test_the_band_excludes_the_two_readings_a_re_deriver_would_try(name):
    """The tolerance must be narrower than the thing it is meant to distinguish.

    A band that also admits the in-sample or the single-feature reading would pass no matter which
    estimator produced the literal, which is the wider-tolerance-than-the-drift defect this project
    has already caught in a decoy-state band.
    """
    d = next(v for n, v in _payloads() if n == name)
    stated = d["separability_auc_pr"]
    _cv, insample, single = _scores(d)
    ambiguous = [f"{label} {val:.4f}" for label, val in
                 (("in-sample", insample), ("single-feature", single))
                 if abs(val - stated) <= TOL]
    if name == "telecom":
        pytest.skip(
            "telecom's readings are 0.9798 / 0.9915 / 0.9925 / 1.0000 - all four inside any usable "
            "band, because the corpus is near-perfectly separable in this projection. The estimator "
            "is genuinely undetermined here and no tolerance can fix that; it is recorded rather "
            "than asserted, and the other two payloads carry the discrimination.")
    assert not ambiguous, (
        f"{name}: the tolerance {TOL} also admits {', '.join(ambiguous)} against the stated {stated}, "
        "so this gate cannot tell which estimator produced the literal. Narrow the band or record "
        "the estimator explicitly in the payload.")
