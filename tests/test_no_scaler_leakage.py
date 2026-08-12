"""A scaler must not be fitted on rows it will later be evaluated against.

Round 34 found two of these in a project whose posture is that every number is attributable:

  * `src/qsh/data/loaders.py` ran `StandardScaler().fit_transform(ds.data)` over all 569 WDBC rows
    before any consumer split them. The demo and benchmark paths print those numbers.
  * `scripts/train/satellite_telemanom.py` computed the z-score over the whole series and split
    afterwards, so the test window contributed to the constants applied to itself - leakage and,
    on a temporal split, look-ahead.

Both are behavioural tests rather than greps for `fit_transform`, because the defect is not the call,
it is what the call was fitted on. A grep would pass on a correctly-written `fit_transform(X[:cut])`
and fail on nothing that matters.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_the_wdbc_scaler_is_not_fitted_on_every_row():
    """Fitting on all rows makes the FULL set have mean 0 exactly. That is the signature to catch."""
    from qsh.data.loaders import load_breast_cancer

    X, _ = load_breast_cancer()
    full_mean = np.abs(X.mean(axis=0)).max()
    assert full_mean > 1e-3, (
        "every column of the full corpus has mean ~0, which is what StandardScaler produces when it "
        "is fitted on the same rows it transforms. The scaler must be fitted on a training slice, so "
        "the held-out rows should be off-centre by however much they differ from it")


def test_the_wdbc_scaler_centres_the_slice_it_was_fitted_on():
    """The complement of the check above: the FIT window must be centred, or nothing was fitted."""
    from qsh.data.loaders import load_breast_cancer

    X, _ = load_breast_cancer(scaler_fit_fraction=0.7)
    cut = int(len(X) * 0.7)
    fit_mean = np.abs(X[:cut].mean(axis=0)).max()
    assert fit_mean < 1e-4, (
        f"the fitted window should be centred to numerical precision, got {fit_mean:.2e} - either "
        f"the fraction is not being honoured or the scaler was fitted on a different slice")


def test_the_fit_fraction_actually_changes_the_constants():
    """Guards against a parameter that is accepted and ignored - a whole defect class in this repo."""
    from qsh.data.loaders import load_breast_cancer

    a, _ = load_breast_cancer(scaler_fit_fraction=0.5)
    b, _ = load_breast_cancer(scaler_fit_fraction=0.9)
    assert not np.allclose(a, b), (
        "the same output for two different fit fractions means the argument is decorative")


@pytest.mark.parametrize("fraction", [0.5, 0.7, 0.9])
def test_no_row_is_dropped_whatever_the_fit_fraction(fraction):
    """Fitting on a slice must not silently truncate what is returned."""
    from qsh.data.loaders import load_breast_cancer

    X, y = load_breast_cancer(scaler_fit_fraction=fraction)
    assert len(X) == len(y) == 569, "all rows are still transformed and returned; only the FIT is a slice"


def test_the_satellite_trainer_fits_its_zscore_after_the_split():
    """A source-level check, because running the trainer needs the ESA corpus and a GPU.

    Asserted on ORDER rather than on the presence of a call: the defect was that `cut` was computed
    AFTER `mu, sigma`, so the whole series fed the constants. Both lines still exist either way.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "train" / "satellite_telemanom.py"
    text = src.read_text()
    cut_at = text.index("cut = int(len(values)")
    mu_at = text.index("mu, sigma = values")
    assert cut_at < mu_at, (
        "the temporal split must be computed BEFORE the z-score, or the scaler sees the test window")
    assert "values[:cut].mean(0)" in text, (
        "the z-score must be fitted on the training window, not on the whole series")
    assert ".ffill().bfill()" not in text, (
        "bfill carries a FUTURE value backwards across a gap, which is look-ahead on a time series")
