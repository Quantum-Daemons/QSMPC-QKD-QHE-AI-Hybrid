"""5G-NIDD over eight Dirichlet-split clients with a 1D-CNN baseline (Round 86, batch 8; checklist MS31).

`references/telecom-nidd-dirichlet-preregistration.md` fixed the partition (Yurochkin et al.'s per-class Dir(0.3) over
eight clients, redrawn from seed 100 s + j while a client holds fewer than 100 fit rows), the three arms and two stated
comparisons, none deciding what ships. `scripts/train/telecom_nidd_dirichlet.py` wrote the record and the page's file.
The flows are not in CI, so what is held here is what the committed files say about each other:

  - the record ran under the committed pre-registration, against the committed two-station federation record;
  - each seed's partition follows the redraw rule, covers every fit and validation row once, and its clients' combined
    standardisation is the federation's within its tolerance;
  - each seed's linear penalty is the grid's best by pooled validation log loss;
  - the summaries, C1 and C2 are what the five seeds' figures give, C2's interval Student's t on the paired differences;
  - the page file is exactly the record's payload, and the telecom federation card mounts the section.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train" / "telecom_nidd_dirichlet.py"
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-dirichlet-2026-11-27.json"
WEB = REPO / "web" / "public" / "data" / "telecom_nidd" / "dirichlet.json"
ARMS = ("linear_fedavg_masked", "cnn_fedavg", "cnn_pooled")


@functools.cache
def _m():
    spec = importlib.util.spec_from_file_location("telecom_nidd_dirichlet", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _r() -> dict:
    return json.loads(RECORD.read_text())


def test_the_record_ran_under_the_committed_preregistration_and_federation() -> None:
    r, m = _r(), _m()
    assert hashlib.sha256((REPO / r["preregistration"]["file"]).read_bytes()).hexdigest() == r["preregistration"]["sha256"]
    assert r["preregistration"]["sha256"] == m.PREREG_SHA256
    assert r["of_federation"]["sha256"] == hashlib.sha256((REPO / r["of_federation"]["file"]).read_bytes()).hexdigest()
    assert (r["clients"], r["alpha"]) == (m.CLIENTS, m.ALPHA) == (8, 0.3)
    assert sorted(r["seeds"], key=int) == ["1", "2", "3", "4", "5"]


def test_each_partition_follows_the_rule_and_covers_every_row_once() -> None:
    r, m = _r(), _m()
    for s, v in r["seeds"].items():
        p = v["partition"]
        tried, used = p["seeds_tried"], p["partition_seed_used"]
        assert tried[0] == int(s) and tried[-1] == used
        assert tried[1:] == [100 * int(s) + j for j in range(1, len(tried))]
        fit, val = p["fit_rows_by_client"], p["validation_rows_by_client"]
        assert len(fit) == len(val) == m.CLIENTS and min(fit) >= m.MIN_FIT_ROWS, s
        assert [sum(c) for c in p["class_counts_by_client"]] == fit, s
        assert sum(fit) == r["rows"]["fit"] and sum(val) == r["rows"]["validation"], s
        assert p["standardisation_max_relative_difference"] <= m.F.STANDARDISATION_TOLERANCE, s


def test_each_linear_penalty_is_the_grid_best_by_pooled_validation_loss() -> None:
    m = _m()
    for s, v in _r()["seeds"].items():
        lin = v["linear_fedavg_masked"]
        table = lin["penalty_table"]
        assert [row["l2"] for row in table] == list(m.S.L2_GRID), s
        assert lin["l2"] == min(table, key=lambda row: (row["validation_log_loss"], row["l2"]))["l2"], s
        assert lin["temperature"] > 0 and lin["masked_messages"] > 0, s


def test_the_summaries_and_both_comparisons_are_the_seeds() -> None:
    from scipy import stats

    r = _r()
    fed = json.loads((REPO / r["of_federation"]["file"]).read_text())
    for arm in ARMS:
        for metric, got in r["summary"][arm].items():
            vals = [v[arm]["test"][metric] for v in r["seeds"].values()]
            assert got["mean"] == pytest.approx(np.mean(vals), abs=1e-6), (arm, metric)
            assert got["sd"] == pytest.approx(np.std(vals, ddof=1), abs=1e-6), (arm, metric)
    c1 = r["C1"]
    assert c1["pooled_flamby"] == fed["pooled"]["flamby_metric"]
    assert c1["linear_eight_clients_mean_flamby"] == r["summary"]["linear_fedavg_masked"]["flamby_metric"]["mean"]
    assert c1["gap"] == pytest.approx(c1["pooled_flamby"] - c1["linear_eight_clients_mean_flamby"], abs=1e-6)
    assert c1["within_0_01"] is (c1["pooled_flamby"] - c1["linear_eight_clients_mean_flamby"] <= 0.01)
    d = np.array([v["cnn_fedavg"]["test"]["flamby_metric"] - v["linear_fedavg_masked"]["test"]["flamby_metric"]
                  for v in r["seeds"].values()])
    half = stats.t.ppf(0.975, len(d) - 1) * np.std(d, ddof=1) / np.sqrt(len(d))
    c2 = r["C2"]
    assert c2["seeds"] == len(d) == 5
    assert c2["cnn_minus_linear_mean_flamby"] == pytest.approx(d.mean(), abs=1e-6)
    assert c2["ci95"] == pytest.approx([d.mean() - half, d.mean() + half], abs=1e-6)
    assert r["references"]["two_station_federated_flamby"] == fed["federated"]["flamby_metric"]
    assert r["references"]["alone_flamby"] == [a["flamby_metric"] for a in fed["alone"]]


def test_the_page_file_is_the_record_payload() -> None:
    r = _r()
    assert json.dumps(_m().web_payload(r, r["device_cnn"]), indent=1) + "\n" == WEB.read_text()


def test_the_telecom_federation_card_mounts_the_section() -> None:
    card = (REPO / "web" / "src" / "components" / "NiddFederatedCard.tsx").read_text()
    section = (REPO / "web" / "src" / "components" / "NiddDirichletSection.tsx").read_text()
    assert "<NiddDirichletSection" in card and "data/telecom_nidd/dirichlet.json" in section
