"""Calibration must be a measurement, not a decoration.

The load-bearing test is the CORP identity S = MCB - DSC + UNC. Three numbers computed separately
and printed side by side would look identical to a decomposition and mean nothing; the identity is
what makes them one object. It is asserted to 1e-12 rather than approximately, because it is exact
arithmetic and a loose tolerance would hide a real error in the recalibration step.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsh.models import calibration as cal


@pytest.fixture()
def scored():
    """A deliberately MIS-calibrated but well-ranked score: sigmoid(3x) against labels from p=x."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < x).astype(int)
    logit = 3.0 * (x - 0.5)                       # ranks perfectly, probabilities far too extreme
    return logit, y


def test_the_corp_identity_holds_exactly(scored):
    """S = MCB - DSC + UNC. Without this the three terms are not a decomposition."""
    logit, y = scored
    p = 1.0 / (1.0 + np.exp(-logit))
    d = cal.corp_decomposition(p, y)
    lhs = d["brier"]
    rhs = d["mcb"] - d["dsc"] + d["unc"]
    assert abs(lhs - rhs) < 1e-12, f"decomposition does not close: {lhs} vs {rhs}"


def test_corp_terms_have_the_signs_their_meanings_require(scored):
    logit, y = scored
    p = 1.0 / (1.0 + np.exp(-logit))
    d = cal.corp_decomposition(p, y)
    assert d["mcb"] >= -1e-12, "miscalibration cannot be negative: PAV can only improve the Brier score"
    assert d["dsc"] >= -1e-12, "a model that ranks at all is not worse than the base rate"
    assert 0.0 <= d["unc"] <= 0.25, "the base-rate Brier score is bounded by 1/4 at p=0.5"


def test_platt_improves_a_miscalibrated_score_it_already_ranks_well(scored):
    """The point of calibration: Brier and log-loss improve, ranking does not change."""
    logit, y = scored
    n_fit = 2000
    fitted = cal.fit_platt(logit[:n_fit], y[:n_fit])
    raw = 1.0 / (1.0 + np.exp(-logit[n_fit:]))
    calibrated = fitted.apply(logit[n_fit:])
    held = y[n_fit:]

    assert cal.brier(calibrated, held) < cal.brier(raw, held), "calibration must improve the Brier score"
    assert cal.log_loss(calibrated, held) < cal.log_loss(raw, held)

    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(held, calibrated) == pytest.approx(roc_auc_score(held, raw), abs=1e-9), (
        "Platt is monotone, so it cannot move AUC - if it did, the transform is not a calibration")


def test_platt_reduces_the_corp_miscalibration_term(scored):
    """MCB is the term calibration is supposed to move. If it does not, nothing was calibrated."""
    logit, y = scored
    fitted = cal.fit_platt(logit[:2000], y[:2000])
    raw = 1.0 / (1.0 + np.exp(-logit[2000:]))
    before = cal.corp_decomposition(raw, y[2000:])["mcb"]
    after = cal.corp_decomposition(fitted.apply(logit[2000:]), y[2000:])["mcb"]
    assert after < before, f"MCB did not improve: {before:.5f} -> {after:.5f}"


def test_a_one_class_calibration_set_is_refused_not_silently_fitted(scored):
    """A fit on one class returns a constant and looks like a calibrated model."""
    logit, _ = scored
    with pytest.raises(ValueError, match="both classes"):
        cal.fit_platt(logit, np.zeros(len(logit), dtype=int))


def test_platt_never_emits_exactly_zero_or_one(scored):
    """Hard 0/1 labels saturate the sigmoid and produce probabilities a log-loss cannot score.

    This is what the Platt/Lin-Weng-Lin target smoothing is for, and it is the reason the fit does
    not simply regress on `y`.
    """
    logit, y = scored
    fitted = cal.fit_platt(logit, y)
    p = fitted.apply(logit)
    assert p.min() > 0.0 and p.max() < 1.0
    assert np.isfinite(cal.log_loss(p, y))


def test_the_shipped_parameters_reproduce_the_transform(scored):
    """The browser gets two floats and a formula string; they must be sufficient on their own."""
    logit, y = scored
    fitted = cal.fit_platt(logit, y)
    d = fitted.as_dict()
    by_hand = 1.0 / (1.0 + np.exp(d["a"] * logit + d["b"]))
    assert np.allclose(by_hand, fitted.apply(logit), atol=1e-12), (
        "a consumer applying the published formula must get the identical numbers")
    assert d["formula"] == "p = 1 / (1 + exp(a * score + b))"


@pytest.mark.parametrize(("k", "n"), [(42, 156), (164, 8192), (113, 529)])
def test_wilson_brackets_the_estimate_and_narrows_with_n(k, n):
    """The measured event counts this project actually has."""
    lo, hi = cal.wilson_interval(k, n)
    assert lo < k / n < hi
    lo2, hi2 = cal.wilson_interval(k * 10, n * 10)
    assert (hi2 - lo2) < (hi - lo), "ten times the data must narrow the interval"
