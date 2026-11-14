"""The satellite overlay's decision boundary (Round 86, `scripts/measure_satellite_boundary.py`), without the data.

- the medoid is the row with the least summed distance;
- a row is near the slice only when every held feature is within one SD of the anchor's;
- the surrogate's terms are the two features, their squares and their product, and a surrogate fitted to a model's
  own quadratic probabilities agrees with it;
- the grid spans the drawn points widened by a quarter of their range, never below the data's minimum.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "measure_satellite_boundary.py"


@pytest.fixture(scope="module")
def mb():
    spec = importlib.util.spec_from_file_location("measure_satellite_boundary_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_medoid_and_near_slice(mb):
    z = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [10.0, 5.0, 5.0]])
    assert mb.medoid(z) == 0
    near = mb.near_slice(z, z[0], held=np.array([1, 2]))
    assert near.tolist() == [True, True, True, False]


def test_quadratic_terms_and_surrogate_fidelity(mb):
    x = np.array([[2.0, 3.0]])
    assert mb.quad(x).tolist() == [[2.0, 3.0, 4.0, 9.0, 6.0]]
    rng = np.random.default_rng(0)
    x2 = rng.normal(size=(4000, 2))
    p = 1 / (1 + np.exp(-(x2[:, 0] ** 2 - x2[:, 1] - 0.5)))
    sur = mb.fit_surrogate(x2, p)
    got = sur(x2)
    assert ((got >= 0.5) == (p >= 0.5)).mean() > 0.98


def test_grid_spans_the_points_with_a_margin_above_the_floor(mb):
    x2 = np.array([[1.0, 5.0], [3.0, 1.0], [2.0, 9.0]])
    xs, ys = mb.grid_over(x2, floor=np.array([1.0, 0.0]), n=4)
    assert (xs[0], xs[-1]) == (1.0, 3.5)                  # 1 - 0.5 clipped at the floor 1; 3 + 0.25 * 2
    assert (ys[0], ys[-1]) == (0.0, 11.0)                 # 1 - 2 clipped at 0; 9 + 0.25 * 8
    assert len(xs) == len(ys) == 4
