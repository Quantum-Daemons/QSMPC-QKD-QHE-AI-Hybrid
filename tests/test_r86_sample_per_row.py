"""The per-row record's Python twin follows the page's rule (Round 86).

`qsh.models.sample_per_row` writes each sample's `expected.per_row` by the rule `evaluatePublishedModel`
(web/src/lib/onnxInference.ts) applies, with the probabilities of `outputKind` and `toProbabilities`
(web/src/lib/calibration.ts) and the preprocessing of `applyPreprocessing`. A twin that drifts from its original
writes records the page then fails against, or worse, passes against a rule it does not apply. Held here:

1. the same output names, preprocessing steps, decision rule and margin rule appear in both languages;
2. on small graphs built here, the record's decisions and margins are the rule's, for a threshold on probabilities,
   an argmax on logits and a single sigmoid column, and an unknown preprocessing step is refused;
3. the metrics a record implies are the page's: accuracy, rare-class F1, macro F1 over the classes seen, forecast MAE.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from qsh.models import sample_per_row as spr

REPO = Path(__file__).resolve().parents[1]
ONNX_TS = (REPO / "web/src/lib/onnxInference.ts").read_text()
CALIB_TS = (REPO / "web/src/lib/calibration.ts").read_text()


def test_the_twin_names_what_the_page_names() -> None:
    assert 'if (name === "probabilities") return "probabilities";' in CALIB_TS
    assert 'if (name === "logits") return "logits";' in CALIB_TS
    assert spr.output_kind("probabilities") == "probabilities" and spr.output_kind("logits") == "logits"
    assert spr.output_kind("label") is None
    steps = set(re.findall(r'step === "([^"]+)"', ONNX_TS))
    assert steps == set(spr._STEPS), (steps, set(spr._STEPS))


def test_the_twin_decides_and_measures_as_the_page_does() -> None:
    # The decision rule, verbatim from the page.
    assert "const pred = thr != null && width === 2 ? (prob![i * 2 + 1] >= thr ? 1 : 0)" in ONNX_TS
    assert ": width > 1 ? best : ((prob ? prob[i] : data[i]) > (thr ?? 0.5) ? 1 : 0);" in ONNX_TS
    # The margin rule, the same three branches.
    assert "if (thr != null && width === 2) return prob![i * 2 + 1] - thr;" in ONNX_TS
    assert "return top - second;" in ONNX_TS
    assert "return (prob ? prob[i] : data[i]) - (thr ?? 0.5);" in ONNX_TS
    # The output the page reads: the first float32 output of two or more dimensions.
    assert 'outputs.findIndex((o) => o && o.type === "float32" && (o.dims?.length ?? 0) >= 2)' in ONNX_TS


def _graph(tmp_path: Path, width: int, out_name: str) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["n", width])
    y = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ["n", width])
    graph = helper.make_graph([helper.make_node("Identity", ["input"], [out_name])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / f"{out_name}_{width}.onnx"
    onnx.save(model, path)
    return path


def _sample(rows, threshold=None, steps=(), task="classification") -> dict:
    exp = {"graph": "g.onnx", "note": ""}
    if threshold is not None:
        exp["decision_threshold"] = threshold
    return {"task": task, "input": {"shape": [1, len(rows[0])], "preprocessing": list(steps)},
            "samples": rows, "expected": exp}


def test_a_threshold_on_probabilities(tmp_path) -> None:
    g = _graph(tmp_path, 2, "probabilities")
    # 0.75 and 0.875 are exact in float32; 0.7 is not (it reads 0.69999999 against a float64 threshold of 0.7, and
    # the page, reading a Float32Array, calls it class 0 too).
    rec = spr.per_row_record(_sample([[0.25, 0.75], [0.6, 0.4], [0.125, 0.875], [0.3, 0.7]], threshold=0.75), g)
    assert rec["predicted"] == [1, 0, 1, 0]               # 0.75 >= 0.75 is class 1
    assert rec["margin"] == pytest.approx([0.0, -0.35, 0.125, -0.05], abs=1e-7)
    assert rec["read_from"] == "probabilities" and rec["tolerance"] == spr.TOLERANCE


def test_an_argmax_on_logits_and_its_tie(tmp_path) -> None:
    g = _graph(tmp_path, 3, "logits")
    rec = spr.per_row_record(_sample([[1.0, 3.0, 2.0], [5.0, 5.0, 1.0]]), g)
    assert rec["predicted"] == [1, 0]                     # a tie goes to the first maximum, as on the page
    assert rec["margin"] == pytest.approx([1.0, 0.0])


def test_a_single_sigmoid_column(tmp_path) -> None:
    g = _graph(tmp_path, 1, "logits")
    rec = spr.per_row_record(_sample([[0.0], [2.0]]), g)
    assert rec["predicted"] == [0, 1]                     # sigmoid(0) = 0.5 is not above 0.5
    assert rec["margin"] == pytest.approx([0.0, 1 / (1 + np.exp(-2.0)) - 0.5], abs=1e-7)


def test_the_page_preprocessing_and_nothing_else(tmp_path) -> None:
    g = _graph(tmp_path, 2, "logits")
    rec = spr.per_row_record(_sample([[255, 0]], steps=["divide by 255", "subtract 0.5", "divide by 0.5"]), g)
    assert rec["margin"] == pytest.approx([2.0])
    with pytest.raises(ValueError, match="unknown preprocessing step"):
        spr.per_row_record(_sample([[1.0, 2.0]], steps=["multiply by 2"]), g)


def test_the_metrics_a_record_implies() -> None:
    sample = {"labels": [1, 0, 1, 2], "expected": {"accuracy": 0, "minority_class_f1": 0, "macro_f1": 0}}
    got = spr.implied_metrics(sample, {"predicted": [1, 1, 0, 2]})
    assert got["accuracy"] == 0.5
    assert got["minority_class_f1"] == pytest.approx(2 * 1 / (2 * 1 + 1 + 1))
    # Classes seen in either list: 0 (F1 0), 1 (F1 0.5), 2 (F1 1).
    assert got["macro_f1"] == pytest.approx((0 + 0.5 + 1) / 3)
    fc = {"targets": [[1.0, 2.0]], "expected": {"mae": 0}}
    assert spr.implied_metrics(fc, {"values": [[1.5, 2.0, 9.0]]})["mae"] == pytest.approx(0.25)
