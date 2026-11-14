"""The satellite forecaster's VUS-PR (Round 86, `scripts/measure_satellite_vus_pr.py`), without the data.

- the autocorrelation equals statsmodels' definition summed directly: the mean removed, every lag divided by n;
- the window rule finds a sine's period, and falls back to 125 as TSB-AD's does where the period is out of range;
- the score is each channel's z-score against its validation errors, the largest of them;
- with TSB-AD installed, a score that marks the anomalies ranks above the same score permuted and above a random walk.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "measure_satellite_vus_pr.py"


@pytest.fixture(scope="module")
def vp():
    spec = importlib.util.spec_from_file_location("measure_satellite_vus_pr_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_acf_is_the_direct_sum(vp):
    rng = np.random.default_rng(0)
    x = rng.normal(size=777).cumsum()
    got = vp.acf_fft(x, 50)
    xc = x - x.mean()
    direct = np.array([np.dot(xc[: len(x) - k], xc[k:]) for k in range(51)]) / len(x)
    assert np.allclose(got, direct / direct[0], atol=1e-12)
    assert got[0] == pytest.approx(1.0)


def test_window_rule_finds_a_period_and_falls_back(vp):
    t = np.arange(20000)
    assert vp.find_length_rank1(np.sin(2 * np.pi * t / 50)) == 50
    assert vp.find_length_rank1(np.sin(2 * np.pi * t / 1000)) == 125      # beyond lag 300: TSB-AD's fall-back
    assert vp.find_length_rank1(np.arange(500.0)) == 125                 # no local maximum at all


def test_score_is_the_largest_channel_z(vp):
    val = np.array([[1.0, 10.0], [3.0, 30.0]])          # means 2 and 20, standard deviations 1 and 10
    test = np.array([[2.0, 40.0], [5.0, 20.0]])
    assert vp.channel_max_z(test, val).tolist() == [2.0, 3.0]


def test_a_marking_score_ranks_above_no_skill(vp):
    pytest.importorskip("TSB_AD")
    rng = np.random.default_rng(1)
    n = 20000
    labels = np.zeros(n, int)
    for s in (3000, 9000, 15000):
        labels[s:s + 60] = 1
    score = rng.normal(size=n) + 4 * labels
    got = vp.vus(labels, score, 50)["vus_pr"]
    base = vp.no_skill(labels, score, 50, draws=2)
    assert got > 0.8
    assert base["permuted"]["max"] < 0.2 and base["random_walk"]["max"] < 0.5
    assert base["permuted"]["draws"] == 2
