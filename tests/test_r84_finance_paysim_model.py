"""Round 84: the finance page's bundled PaySim model no longer learns from PaySim's balance columns.

WHY THIS EXISTS

The model the finance page encrypts for its one-row CKKS step and explains with exact linear SHAP
was a logistic surrogate over eight columns of the bundled PaySim sample. Two of them, `device_risk`
and `geo_mismatch`, are computed from oldbalanceOrg, newbalanceOrig, oldbalanceDest and
newbalanceDest, and the dataset's own Kaggle description says those four "must not be used" for
fraud detection because fraudulent transactions are cancelled. `device_risk` carried the largest
weight (-3.06), so the page's explanation credited the simulator's bookkeeping. `amount_zscore` was
an affine copy of `amount` and `velocity_24h` was constant in the shipped rows. The same eight fed
the Python finance pipeline.

WHAT IS ASSERTED, AND WHAT EACH FAILS ON AT THE BASE COMMIT (de1111c)

- The payload's model reads exactly the registry's four columns, defines each, names the four left
  out with a reason, and quotes the description it rests on. Base: eight names, no definitions.
- The builder reproduces the shipped payload byte for byte. A drift guard, not a behaviour test:
  it holds at the base commit too, where the eight-column builder wrote the eight-column file.
- The evaluation block and the model reproduce from the shipped rows. Base: no evaluation block.
- The raw block the QCNN is fed is the artifact's own columns in its order: the same four as
  `samples`, unstandardised, since the artifact's Round 84 retrain on those four (it was the eight,
  named in `names_by_space` and described in `eda_by_space`, while the artifact read the eight).
  Base: an eight-wide block beside an eight-feature artifact.
- The TSV header carries the balance-column note, written by the generator. Base: neither.
- The Python use case reads the same four columns and hands the QCNN the same four, the artifact's
  own, in its order. Base: eight.
- Visitor text and the browser's fallback names. Base: the blurb names device_risk; FEATURE_NAMES
  lists the eight.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from qsh.models import registry

REPO = Path(__file__).resolve().parents[1]
PAYLOAD = REPO / "web" / "public" / "data" / "finance" / "paysim_sample.json"
QCNN = REPO / "web" / "public" / "data" / "finance" / "qcnn_weights.json"
TSV = REPO / "data" / "finance" / "paysim_sample.tsv"
BALANCE = {"oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
LEAKY = {"device_risk", "geo_mismatch"}
#: Written out so each check below reads the ARTIFACT first and the registry second: on a tree that
#: predates the registry constants a test then fails on what the file ships, not on a missing name.
FOUR = ["amount", "cross_border", "night_tx", "new_payee"]
EIGHT = ["amount", "velocity_24h", "cross_border", "night_tx",
         "new_payee", "amount_zscore", "device_risk", "geo_mismatch"]


def _payload() -> dict:
    return json.loads(PAYLOAD.read_text())


def _module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_model_reads_the_four_registry_columns_and_no_balance_derived_one() -> None:
    d = _payload()
    assert d["names"] == FOUR, d["names"]
    assert not LEAKY & set(d["names"]), "a balance-derived column is a model input again"
    assert {"amount_zscore", "velocity_24h"}.isdisjoint(d["names"])
    assert len(d["samples"][0]) == len(FOUR)
    assert [len(row) for row in d["model"]["W"]] == [len(FOUR), len(FOUR)]
    assert "TRANSFER or CASH_OUT" in d["feature_definitions"]["cross_border"]
    # The registry is the authority both the payload and the Python use case read.
    assert list(registry.PAYSIM_MODEL_FEATURES) == FOUR
    assert list(registry.PAYSIM_SAMPLE_COLUMNS) == EIGHT
    assert d["feature_definitions"] == {k: registry.PAYSIM_MODEL_FEATURES[k] for k in FOUR}


def test_the_columns_left_out_are_named_with_the_reason_and_its_source() -> None:
    d = _payload()
    left_out = d["excluded_features"]
    assert set(left_out) == set(EIGHT) - set(FOUR)
    for name in LEAKY:
        assert "must not be used for fraud detection" in left_out[name], left_out[name]
    src = d["exclusion_source"]
    assert set(src["applies_to"]) == LEAKY
    assert all(col in src["quote"] for col in BALANCE) and "must not be used" in src["quote"]
    assert src["url"] == registry.DS_PAYSIM.url
    # The quote is the dataset description's, so it names the columns `_paysim_features` reads.
    feat_src = (REPO / "scripts" / "prep_real_kaggle_data.py").read_text()
    body = feat_src[feat_src.index("def _paysim_features"):feat_src.index("def prep_finance")]
    for name in LEAKY:
        line = next(ln for ln in body.splitlines() if ln.strip().startswith(f"{name} ="))
        assert re.search(r"err_orig|dest_delta", line), line
    assert all(col in body for col in BALANCE)


def test_the_builder_reproduces_the_shipped_payload_byte_for_byte() -> None:
    pytest.importorskip("xgboost")
    build = _module("scripts/data/build_federated_samples.py", "build_federated_samples_r84")
    assert json.dumps(build.build_finance(400, 0)) == PAYLOAD.read_text(), (
        "web/public/data/finance/paysim_sample.json is not what build_federated_samples.py writes; "
        "regenerate it with --only finance_aml")


def test_the_stated_fitting_row_figures_reproduce_from_the_shipped_rows() -> None:
    from sklearn.metrics import f1_score, roc_auc_score

    d = _payload()
    X, y = np.asarray(d["samples"], dtype=float), np.asarray(d["labels"])
    W, b = np.asarray(d["model"]["W"]), np.asarray(d["model"]["b"])
    margin = X @ (W[1] - W[0]) + (b[1] - b[0])
    pred = (margin > 0).astype(int)
    fit = d["evaluation"]["fitting_rows"]
    # The shipped rows are rounded to 5 decimals, so the margins move by ~1e-5: no row sits that close.
    assert fit["rows"] == len(y) == 400
    assert fit["accuracy"] == round(float((pred == y).mean()), 4)
    assert fit["fraud_f1"] == round(float(f1_score(y, pred)), 4)
    assert fit["roc_auc"] == round(float(roc_auc_score(y, margin)), 4)
    cv = d["evaluation"]["cross_validated"]
    assert cv["folds"] == 5 and 0.5 < cv["accuracy"] < 1.0
    nd = d["evaluation"]["legit_rows_not_drawn"]
    assert nd["rows"] == 3600 and nd["false_positive_rate"] == round(nd["flagged"] / nd["rows"], 4)
    # The leak is gone, so the figures are no longer the near-perfect ones it produced.
    assert fit["accuracy"] < 0.95 and cv["accuracy"] < 0.95


def test_the_qcnn_feed_is_the_artifacts_own_four_columns_in_its_order() -> None:
    """Updated in Round 84 (qcnnfin): the artifact was retrained on the four model columns, so the
    block it is fed is those four, named by `names`, and the eight-column fields that named and
    described the old block (`names_by_space`, `eda_by_space`) are gone with it."""
    d, w = _payload(), json.loads(QCNN.read_text())
    raw = d["samples_by_space"]["raw"]
    assert d["names"] == FOUR == w["feature_names"], (d["names"], w["feature_names"])
    assert not LEAKY & set(w["feature_names"]), "the QCNN reads a balance-derived column again"
    assert len(raw) == len(d["samples"]) and {len(r) for r in raw} == {w["n_features"]} == {4}
    assert "names_by_space" not in d and "eda_by_space" not in d, (
        "the payload still names or describes an eight-column raw block it no longer ships")
    # The SAME transactions in both spaces, column for column: the raw block, z-scored with the
    # file's own scaler (first 70 percent), is the `samples` block.
    R = np.asarray(raw, dtype=float)
    cut = int(len(R) * 0.7)
    Z = (R - R[:cut].mean(axis=0)) / R[:cut].std(axis=0)
    assert float(np.abs(Z - np.asarray(d["samples"])).max()) < 1e-4
    # Nothing the wires carry is constant or a copy, in either space.
    assert d["eda"]["degenerate"] == [] and d["eda"]["duplicates"] == []
    assert float(R.std(axis=0).min()) > 0.1


def test_the_tsv_header_carries_the_balance_note_and_the_generator_writes_it() -> None:
    gen = _module("scripts/prep_real_kaggle_data.py", "prep_real_kaggle_data_r84")
    lines = TSV.read_text().splitlines()
    header = [ln for ln in lines if ln.startswith("#")]
    assert lines[:len(header)] == header, "comment lines are not all at the top of the TSV"
    text = " ".join(h.lstrip("# ") for h in header)
    assert all(col in text for col in BALANCE) and "must not be used for fraud detection" in text
    assert lines[len(header)].split("\t") == [*EIGHT, "isFraud"]
    assert tuple(header) == gen.FIN_TSV_HEADER
    assert gen.FIN_NAMES == EIGHT


def test_the_tsv_figures_the_registry_quotes_hold() -> None:
    rows = [ln for ln in TSV.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    data = np.asarray([[float(v) for v in ln.split("\t")] for ln in rows[1:]])
    col = {n: i for i, n in enumerate(EIGHT)}
    y = data[:, 8].astype(int)
    assert int((data[:, col["velocity_24h"]] != 0).sum()) == 2 and len(data) == 4000
    assert "non-zero in 2 of the 4,000 rows" in registry.PAYSIM_EXCLUDED_FEATURES["velocity_24h"]
    # cross_border is the type flag, and every fraud row of the sample carries it.
    assert set(np.unique(data[:, col["cross_border"]])) == {0.0, 1.0}
    assert int(data[y == 1, col["cross_border"]].sum()) == int(y.sum()) == 200
    # amount_zscore is an affine copy of amount.
    a, z = data[:, col["amount"]], data[:, col["amount_zscore"]]
    assert np.corrcoef(a, z)[0, 1] > 0.99999


def test_the_python_pipeline_reads_the_same_four_and_feeds_the_qcnn_the_same_four() -> None:
    """Updated in Round 84 (qcnnfin): the artifact reads the model's four columns, so the QCNN's row
    is the scored transaction's `features` themselves, in the artifact's order."""
    pytest.importorskip("xgboost")
    from qsh.qhe.qcnn_trained import load_weights
    from qsh.usecases.finance_aml import build

    _, ctx = build({"seed": 0, "inference": "qcnn"})
    assert ctx.data["feature_names"] == FOUR
    assert np.asarray(ctx.data["features"]).shape == (4,)
    assert np.asarray(ctx.data["linear_model"].weight).shape == (2, 4)
    assert np.asarray(ctx.data["explain_baseline"]).shape == (4,)
    assert ctx.data["_finance_model"]["features"] == FOUR
    w = load_weights("finance_aml")
    assert w["feature_names"] == FOUR
    q = np.asarray(ctx.data["qcnn_features"])
    assert q.shape == (len(w["feature_names"]),) == (4,)
    # The QCNN's row is the scored transaction itself, column for column, in raw units.
    np.testing.assert_allclose(q, ctx.data["features"], rtol=0, atol=0)


def test_the_visitor_text_and_the_browser_fallback_names_follow_the_model() -> None:
    uc = (REPO / "web" / "src" / "lib" / "useCases.ts").read_text()
    m = re.search(r'id: "finance_aml".*?blurb: "([^"]*)"', uc, re.S)
    assert m, "finance blurb not found"
    blurb = m.group(1)
    assert "device_risk" not in blurb and "origin-balance" not in blurb
    assert "TRANSFER or CASH_OUT" in blurb and "balance columns" in blurb
    cc = (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text()
    fn = re.search(r"finance_aml: \[([^\]]*)\],", cc[cc.index("const FEATURE_NAMES"):])
    assert fn and re.findall(r'"([^"]+)"', fn.group(1)) == FOUR
    rd = (REPO / "web" / "src" / "lib" / "realData.ts").read_text()
    body = rd[rd.index('if (usecase === "finance_aml") {'):rd.index("} catch (e) {")]
    assert "row.slice(0, 8)" not in body and "columnMeans(rows, 8)" not in body
