"""Round 84: the finance overlay says its rows are enriched, by how much, and what that does to its figure.

WHY THIS EXISTS

`web/public/data/finance/real_2d.json` draws 8,192 rows of AMLworld's held-out test window with
laundering enriched to 2.0 percent (164 rows), while the window itself holds it at 0.15 percent
(2,321 of 1,523,504). The caption printed "AUC-PR 0.1269 ... against a base rate of 0.0200 - a 6.34x
lift" with nothing saying the base rate was the enriched one, and AUC-PR's baseline IS the
prevalence, so a reader took a property of the draw for a property of the task.

WHAT IS ASSERTED, AND WHAT EACH FAILS ON AT THE BASE COMMIT (de1111c)

- The payload carries both prevalences as numbers, the window's taken from the trainer's own count
  in models/metrics.json. Base: no `enrichment` block.
- `split_note`, which the caption renders immediately before the AUC-PR sentence, states the two
  rates and says the AUC-PR, base rate and lift are measured on the enriched rows. Base: "enriched to
  2 percent" with no window rate and nothing about the figure.
- The figure at the window's rate reproduces from the shipped points: the separability estimator's
  out-of-fold scores, weighted by the inverse of each class's sampling fraction. Base: absent.
- The generator writes exactly the shipped file. A drift guard, not a behaviour test: it holds at
  the base commit too, for the base generator and the base file.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
OVERLAY = REPO / "web" / "public" / "data" / "finance" / "real_2d.json"
METRICS = REPO / "models" / "metrics.json"


def _overlay() -> dict:
    return json.loads(OVERLAY.read_text())


def _window() -> tuple[int, int]:
    corp = json.loads(METRICS.read_text())["finance_aml.gbm"]["calibration"]["corp_raw"]
    return int(corp["positives"]), int(corp["n"])


def test_the_payload_carries_both_prevalences() -> None:
    d = _overlay()
    labels = [int(p[2]) for p in d["points"]]
    e = d["enrichment"]
    assert (e["sample_positives"], e["sample_rows"]) == (sum(labels), len(labels)) == (164, 8192)
    assert (e["window_positives"], e["window_rows"]) == _window() == (2321, 1523504)
    assert e["sample_rate"] == d["base_rate"]
    assert abs(e["window_rate"] - 2321 / 1523504) < 5e-7
    assert e["factor"] == round((164 / 8192) / (2321 / 1523504), 2)
    assert "metrics.json" in e["source"]


def test_the_caption_text_states_the_enrichment_before_the_figure() -> None:
    d = _overlay()
    s = d["split_note"]
    for frag in ("enriched about 13x", "2.0 percent (164 of 8,192)", "0.15 percent in that window",
                 "(2,321 of 1,523,504)", "AUC-PR, base rate and lift below are measured on the enriched rows"):
        assert frag in s, (frag, s)
    # The note keeps the enriched-row lift and adds the window's, both computed from the payload.
    n = d["note"]
    assert f"lift of {d['separability_auc_pr'] / d['base_rate']:.1f}x on these enriched rows" in n
    win = d["separability_auc_pr_at_window_rate"] / d["enrichment"]["window_rate"]
    assert f"AUC-PR {d['separability_auc_pr_at_window_rate']}, a lift of {win:.1f}x" in n


def test_the_window_rate_figure_reproduces_from_the_shipped_points() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import cross_val_predict

    d = _overlay()
    pts = np.asarray(d["points"], dtype=float)
    x, y = pts[:, :2], pts[:, 2].astype(int)
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    scores = cross_val_predict(rf, x, y, cv=5, method="predict_proba")[:, 1]
    pos, n = _window()
    w = np.where(y == 1, pos / y.sum(), (n - pos) / (len(y) - y.sum()))
    got = average_precision_score(y, scores, sample_weight=w)
    stated = d["separability_auc_pr_at_window_rate"]
    assert abs(got - stated) <= 5e-5 * max(1.0, stated), (got, stated)
    # Unweighted, the same scores give the enriched figure the caption already prints.
    assert abs(average_precision_score(y, scores) - d["separability_auc_pr"]) <= 1e-4
    assert f"{pos:,}/{int(y.sum()):,}" in d["separability_at_window_rate_method"]


def test_the_generator_writes_exactly_the_shipped_overlay() -> None:
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("prep_real_data_2d_r84",
                                                  REPO / "scripts" / "prep_real_data_2d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    payload = mod.finance()
    assert json.dumps(payload, separators=(",", ":")) + "\n" == OVERLAY.read_text(), (
        "web/public/data/finance/real_2d.json is not what scripts/prep_real_data_2d.py writes")
