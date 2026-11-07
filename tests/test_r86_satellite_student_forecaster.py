"""The satellite CKKS student as a linear next-step forecaster (Round 86), its helpers on cases whose answer is known.

`scripts/train/satellite_student_forecaster.py` reads the published teacher and the ESA corpus, so these pin the parts
that decide what it publishes: the feature layout the page must reproduce, the ridge fit, and that the teacher's
trainer no longer overwrites the student's record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from train import satellite_student_forecaster as S  # noqa: E402


def test_features_are_the_k_rows_before_the_target_oldest_first_row_major() -> None:
    z = np.arange(20 * 6, dtype=float).reshape(20, 6)        # z[t, c] = 6 t + c
    x = S.features(z, np.array([10]), k=3)[0]
    # Rows 7, 8, 9 in that order; feature i * 6 + c is channel c at row 10 - 3 + i.
    assert x.tolist() == z[7:10].ravel().tolist()
    assert x[1 * 6 + 2] == z[8, 2]


def test_ridge_recovers_a_linear_map_and_its_intercept() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 12))
    W_true = rng.normal(size=(3, 12))
    b_true = np.array([0.5, -1.0, 2.0])
    Y = X @ W_true.T + b_true
    W, b = S.ridge(X, Y, 0.0)
    assert np.allclose(W, W_true, atol=1e-9) and np.allclose(b, b_true, atol=1e-9)
    assert np.allclose(S.forecast(W, b, X), Y, atol=1e-9)
    # A penalty shrinks the slopes and leaves the intercept to absorb the mean.
    W_r, _ = S.ridge(X, Y, 1e6)
    assert np.abs(W_r).max() < np.abs(W).max()


def test_the_teacher_trainer_no_longer_writes_the_students_record() -> None:
    src = (REPO / "scripts" / "train" / "satellite_telemanom.py").read_text()
    assert 'record_metrics("satellite_comms.student"' not in src, "a teacher run would overwrite the student again"
    assert "satellite_student_forecaster.py" in src
    student = (REPO / "scripts" / "train" / "satellite_student_forecaster.py").read_text()
    assert "record_metrics(STUDENT" in student and 'STUDENT = "satellite_comms.student"' in student
