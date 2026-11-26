"""The finance QCNN's fair comparison is the pre-registered one, and the page states every arm (Round 86, batch 6).

WHY THIS EXISTS (the owner's choices of 2026-11-26; references/finance-qcnn-amlworld-comparison-preregistration.md)

The first AMLworld QCNN chose its checkpoint and seed by a validation F1 at a threshold the four-to-one fit sample sets.
The comparison re-ran it selected by average precision over twenty seeds, beside an improved circuit, quantum kernels,
classical models, a tensor network and a circuit without entanglement, and its criterion 2 has the page state every
arm's figure with its interval whatever ships. This file holds the record to the pre-registration and the page's
summary to the record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECORD_PATH = REPO / "docs" / "measurements" / "finance-qcnn-amlworld-comparison-2026-11-26.json"
SITE_PATH = REPO / "web" / "public" / "data" / "finance" / "qcnn_comparison.json"
pytestmark = pytest.mark.skipif(not RECORD_PATH.exists(), reason="the comparison's record is not written yet")


def _record() -> dict:
    return json.loads(RECORD_PATH.read_text())


def test_the_record_is_under_the_committed_preregistration() -> None:
    r = _record()
    assert hashlib.sha256((REPO / r["preregistration"]["file"]).read_bytes()).hexdigest() == r["preregistration"]["sha256"]
    assert [len(r["arms"][a]["seeds"]) for a in ("qcnn1", "qcnn2", "noent")] == [20, 20, 5]
    for arm in ("product", "zz", "projected", "logistic_regression", "gradient_boosting", "rbf_svm", "mps",
                "logistic_regression_whole_fit_window", "gradient_boosting_whole_fit_window"):
        assert arm in r["arms"], arm


def test_the_shipped_seed_is_the_one_best_on_validation_and_the_criterion_is_applied() -> None:
    r = _record()
    seeds = r["arms"]["qcnn1"]["seeds"]
    best = max(seeds, key=lambda s: (s["val_ap"], -s["seed"]))
    ship = r["arms"]["qcnn1"]["shipped_seed"]
    assert ship["seed"] == best["seed"] and ship["test_ap"] == best["test_ap"]
    assert r["criteria"]["1_shipped_seed_at_least_twice_prevalence"] == (ship["test_ap"] >= 2 * r["test_prevalence"])


def test_the_simulability_check_traced_in_double_precision() -> None:
    sch = _record()["schmidt"]
    assert sch and sch["precision"] == "complex128" and sch["bound"] == 16
    assert all(v["encode"] == 1 for v in sch["by_seed"].values()), "the encoding is a product state"
    assert sch["largest_rank_over_seeds_layers_cuts_and_rows"] <= 16


def test_the_page_summary_states_every_arm_with_its_interval() -> None:
    r, site = _record(), json.loads(SITE_PATH.read_text())
    keys = {row["key"] for row in site["rows"]}
    assert {"qcnn1_shipped", "qcnn1_ensemble", "qcnn2_ensemble", "noent_ensemble", "product", "zz", "projected",
            "logistic_regression", "gradient_boosting", "rbf_svm", "mps"} <= keys
    for row in site["rows"]:
        lo, hi = row["ap_ci95"]
        assert lo <= row["test_ap"] <= hi or row["key"].endswith("_ensemble"), row
        assert row["ap_ci95"] == r["intervals"][row["key"]]["ap_ci95"]
    assert site["prevalence"] == r["test_prevalence"] and "No quantum advantage is claimed" in site["statement"]
    page = (REPO / "web" / "src" / "pages" / "UseCasePage.tsx").read_text()
    assert 'usecase === "finance_aml" && <FinanceQcnnComparisonCard />' in page


def test_the_row_that_is_not_an_arm_is_labelled_and_is_the_weights_files_baseline() -> None:
    """The owner's choice of 2026-11-26, after the record was first written: logistic regression on the circuit's own
    tanh-encoded inputs, which leads the circuit's seed average, stands beside the arms, labelled as not one of them. It
    is the served weights file's own baseline, it is outside `arms`, so no pre-registered figure or criterion can move with
    it, and the card says what it is in its table and in its summary."""
    r, site = _record(), json.loads(SITE_PATH.read_text())
    extra = [row for row in site["rows"] if row.get("not_an_arm")]
    assert [row["key"] for row in extra] == ["logistic_regression_encoded"], extra
    row = extra[0]
    assert "tanh-encoded" in row["label"] and "not an arm" in row["kind"], row
    assert "logistic_regression_encoded" not in r["arms"] and "logistic_regression_encoded" in r["post_hoc"]
    assert "not pre-registered" in r["post_hoc"]["logistic_regression_encoded"]["not_an_arm"]
    served = json.loads((REPO / "models" / "qcnn_finance" / "weights.json").read_text())
    assert row["test_ap"] == round(served["classical_lr_auprc"], 6) == r["post_hoc"]["logistic_regression_encoded"]["test_ap"]
    assert row["ap_ci95"] == r["intervals"]["logistic_regression_encoded"]["ap_ci95"]
    assert "Logistic regression on the circuit's tanh-encoded inputs" not in r["criteria"]["comparators_above_the_shipped_seed"]
    card = (REPO / "web" / "src" / "components" / "FinanceQcnnComparisonCard.tsx").read_text()
    assert "r.not_an_arm" in card and "That row is not an arm" in card
    assert "arXiv:2403.07059" in card, "the card names the benchmark study whose checks it runs"
