"""How far one AUROC on 156 images can be trusted, and that the card says so (Round 86, fact-check C6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qsh.models import metrics

REPO = Path(__file__).resolve().parents[1]
RECORD = json.loads((REPO / "models" / "metrics.json").read_text())
KEYS = ["medical_fl.resnet18", "medical_fl.resnet50", "medical_fl.student"]


def _brute(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """DeLong's structural components, written out pair by pair."""
    pos, neg = s[y == 1], s[y == 0]
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    return float(psi.mean()), float(np.var(v10, ddof=1) / len(pos) + np.var(v01, ddof=1) / len(neg))


@pytest.mark.parametrize("seed", range(4))
def test_the_fast_algorithm_is_delongs_variance(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 90)
    s = np.round(rng.normal(y * 0.7, 1.0), 1)         # rounded, so there are ties
    got = metrics.delong_auc_ci(y, s)
    auc, var = _brute(y, s)
    assert got["auc"] == pytest.approx(auc, abs=1e-12)
    assert got["se"] ** 2 == pytest.approx(var, abs=1e-12)


def test_it_refuses_a_class_too_small_for_a_variance() -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        metrics.delong_auc_ci([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3])


def test_a_bootstrap_interval_contains_its_point() -> None:
    rng = np.random.default_rng(5)
    y = np.asarray([1] * 40 + [0] * 110)
    s = rng.normal(y * 1.2, 1.0)
    ci = metrics.bootstrap_auc_ci(y, s, n_boot=800)
    assert ci["lo"] < metrics.delong_auc_ci(y, s)["auc"] < ci["hi"]


@pytest.mark.parametrize("key", KEYS)
def test_the_record_is_the_measurements(key: str) -> None:
    measured = json.loads((REPO / "docs" / "measurements" / "medical-auc-ci-2026-11-08.json").read_text())["results"][key]
    ci = RECORD[key]["auc_ci_95"]
    assert [ci["lo"], ci["hi"]] == measured["delong_95"]
    assert (ci["n_pos"], ci["n_neg"]) == (114, 42)
    assert ci["lo"] < measured["auc_onnx_graph"] < ci["hi"]


def test_the_published_baseline_lies_inside_the_interval() -> None:
    # Why 0.9147 against 0.901 is not a win: the interval on 156 images is wider than the gap.
    ci = RECORD["medical_fl.resnet18"]["auc_ci_95"]
    assert ci["lo"] < 0.901 < ci["hi"]


def test_the_card_prints_the_interval_and_the_seeds() -> None:
    panel = (REPO / "web" / "src" / "components" / "ModelPanel.tsx").read_text()
    for needle in ('"auc_ci_95"', '"seed_spread"', "95% CI {aucCi.lo.toFixed(3)} to {aucCi.hi.toFixed(3)}", "earlier runs differing only in the seed"):
        assert needle in panel, needle
