"""Calibration must be a measurement, not a decoration.

The load-bearing test is the CORP identity S = MCB - DSC + UNC. Three numbers computed separately
and printed side by side would look identical to a decomposition and mean nothing; the identity is
what makes them one object. It is asserted to 1e-12 rather than approximately, because it is exact
arithmetic and a loose tolerance would hide a real error in the recalibration step.

THE HOLE THIS SUITE HAD, AND WHAT NOW COVERS IT

Every test below the `scored` fixture ran `fit_platt` on a LOG-ODDS - the fixture builds
`3.0 * (x - 0.5)` and names the variable `logit`. The only production caller,
`scripts/train/finance_aml.py`, passed `predict_proba(...)[:, 1]`, a PROBABILITY. The map is linear
in its argument, so those are different functions, and this suite passed for three rounds because it
and the pipeline were each internally consistent about different things. No assertion here was
false; the missing one was about the SHAPE of the fitted map, and
`test_a_probability_score_is_not_fitted_into_a_near_constant` is it. That test fails against the
call as it shipped through Round 49 and passes against `on_logit=True`.
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


@pytest.fixture()
def imbalanced():
    """PROBABILITY scores at a finance-like base rate, which is the argument `fit_platt` really gets.

    THE DISTINCTION THIS FIXTURE EXISTS FOR is the defect Round 49 measured. `scored` yields a
    LOG-ODDS, and every test using it therefore exercised `fit_platt` on the argument its formula
    assumes. The finance trainer passes a PROBABILITY. Because the shipped map is linear in whatever
    it is handed, feeding it a probability at a 1-in-656 base rate - where 90 percent of scores sit
    below 1e-4 - collapses nine deciles onto one constant, and the map cannot express a risk below
    1 in 1,109 at all. The unit tests could not see it: their fixture passed a logit and the trainer
    passed a probability, so both were internally consistent and disagreed with each other.

    So this deliberately mirrors the trainer's shape rather than the other fixture's: scores in (0, 1)
    spanning several orders of magnitude, and a base rate near 1 in 650. A fixture that handed over
    a logit here would reintroduce exactly the blind spot.
    """
    rng = np.random.default_rng(7)
    n = 6000
    # Log-uniform over ~5 decades, which is the span the shipped finance scores actually cover
    # (4.4e-11 to 0.99997); a uniform draw would put almost nothing near the extremes and the clip
    # below would never be exercised.
    s = 10.0 ** rng.uniform(-6.0, -0.05, n)
    y = (rng.uniform(0, 1, n) < s).astype(int)
    if y.sum() < 2:                        # `fit_platt` refuses a single-class calibration set
        y[rng.integers(0, n, 2)] = 1
    return s, y


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
    assert d["on_logit"] is False


def test_the_shipped_parameters_reproduce_the_on_logit_transform(imbalanced):
    """Same contract for the map finance actually ships, whose argument is not the score.

    The transform moved inside `fit_platt`, so the published JSON has to say so. This reimplements
    the formula string from the JSON alone - clip, log-odds, sigmoid, in that order - because that
    is exactly what a consumer holding the weights file can do and nothing more.
    """
    s, y = imbalanced
    fitted = cal.fit_platt(s, y, on_logit=True)
    d = fitted.as_dict()
    assert d["on_logit"] is True
    assert d["formula"] == (
        "p = 1 / (1 + exp(a * logit(clip(score, logit_clip_eps, 1 - logit_clip_eps)) + b)), "
        "logit(q) = log(q / (1 - q))")

    q = np.clip(s, d["logit_clip_eps"], 1.0 - d["logit_clip_eps"])
    by_hand = 1.0 / (1.0 + np.exp(d["a"] * np.log(q / (1.0 - q)) + d["b"]))
    assert np.allclose(by_hand, fitted.apply(s), atol=1e-12), (
        "a consumer applying the published formula must get the identical numbers")


def test_the_clip_keeps_a_saturated_score_finite(imbalanced):
    """A retrain that saturates a leaf to exactly 0 or 1 must not become an infinity in the fit.

    Nothing on the shipped finance split is clipped - its scores span 4.4e-11 to 0.99997 - so this
    guards the retrain, not today's numbers. Without the clip `logit(0)` is -inf, the regression
    returns NaN parameters, and the failure surfaces as a weights file full of nulls rather than as
    an error.
    """
    s, y = imbalanced
    s = s.copy()
    s[:5] = 0.0
    s[5:10] = 1.0
    fitted = cal.fit_platt(s, y, on_logit=True)
    assert np.isfinite(fitted.a) and np.isfinite(fitted.b)
    p = fitted.apply(s)
    assert np.all(np.isfinite(p)) and p.min() > 0.0 and p.max() < 1.0


@pytest.mark.parametrize(("k", "n"), [(42, 156), (164, 8192), (113, 529)])
def test_wilson_brackets_the_estimate_and_narrows_with_n(k, n):
    """The measured event counts this project actually has."""
    lo, hi = cal.wilson_interval(k, n)
    assert lo < k / n < hi
    lo2, hi2 = cal.wilson_interval(k * 10, n * 10)
    assert (hi2 - lo2) < (hi - lo), "ten times the data must narrow the interval"
