"""The published IBM GNN's calibration record agrees with the published record and with itself (Round 86, batch 4).

`docs/measurements/finance-gnn-calibration-2026-11-25.json` is computed by `scripts/measure_finance_gnn_calibration.py`
from the dumps the published record was scored from, which it checks against `models/metrics.json`; the dumps are
outside the repository, so this checks what can be checked without them: every seed's test AUPRC, rows and positives
are the published record's; each reliability table covers every scored row and every laundering row once; and the map
fitted on validation lowers both the ECE and the CORP miscalibration term on test, for every seed.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "finance-gnn-calibration-2026-11-25.json"


def test_the_record_is_the_published_models() -> None:
    d = json.loads(RECORD.read_text())
    pub = json.loads((REPO / "models" / "metrics.json").read_text())["finance_aml.gnn"]
    assert d["model"] == "finance_aml.gnn" and d["selected_seed"] == pub["selected_seed"]
    by_seed = {str(s["seed"]): s for s in pub["seeds"]}
    assert set(d["seeds"]) == set(by_seed)
    for k, s in d["seeds"].items():
        p = by_seed[k]
        assert (s["test_auprc"], s["test_rows_scored"], s["test_laundering"]) == (
            p["test_auprc"], p["n_test_scored"], p["test_positives_scored"]), k
        assert s["decision_threshold"] == p["decision_threshold"]
        assert len(d["sources"]) == 2 * len(by_seed)


def test_each_reliability_table_covers_every_row_once() -> None:
    d = json.loads(RECORD.read_text())
    for k, s in d["seeds"].items():
        for arm in ("raw", "calibrated"):
            table = s[arm]["reliability_10_equal_width"]
            assert len(table) == 10 and table[0]["from"] == 0.0 and table[-1]["to"] == 1.0
            assert sum(r["rows"] for r in table) == s["test_rows_scored"], (k, arm)
            assert sum(r["laundering"] for r in table) == s["test_laundering"], (k, arm)
            for r in table:
                if r["rows"]:
                    lo, hi = r["observed_ci95"]
                    assert lo <= r["observed_rate"] <= hi and r["from"] <= r["mean_prediction"] <= r["to"]


def test_the_map_fitted_on_validation_improves_calibration_on_test() -> None:
    d = json.loads(RECORD.read_text())
    for k, s in d["seeds"].items():
        raw, cal = s["raw"], s["calibrated"]
        for key in ("ece_10_equal_width", "ece_15_equal_width", "ece_15_equal_mass", "brier"):
            assert cal[key] < raw[key], (k, key)
        assert cal["corp"]["mcb"] < raw["corp"]["mcb"], k
        # The map is monotone, so it cannot change how the model ranks transfers: discrimination is the same.
        assert abs(cal["corp"]["dsc"] - raw["corp"]["dsc"]) < 1e-6, k
