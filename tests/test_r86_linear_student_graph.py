"""The telecom student's ONNX graph is its weights.json, operation for operation (Round 86).

`qsh.models.linear_onnx` builds the graph `telecom_ids.student` is published with from its weights file rather than
refitting it. Held here: the graph's logits equal the file's arithmetic, (x - mean) / scale @ W^T + b, on the 600
held-out rows the page scores, to float32 precision; the class each row gets is the same; the bundled copy the page's
CKKS tier reads is the same bytes as the model directory's file; and a file the graph cannot express is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "models" / "telecom_ids_student" / "weights.json"
HELDOUT = REPO / "web" / "public" / "data" / "telecom" / "student_heldout.json"

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def _run(model, x: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": x.astype(np.float32)})[0]


def test_the_graph_is_the_weights_file() -> None:
    from qsh.models.linear_onnx import linear_student_graph

    w = json.loads(WEIGHTS.read_text())
    rows = np.asarray(json.loads(HELDOUT.read_text())["samples"], dtype=np.float64)
    want = ((rows - np.asarray(w["input"]["mean"])) / np.asarray(w["input"]["scale"])) @ np.asarray(w["W"]).T \
        + np.asarray(w["b"])
    got = _run(linear_student_graph(w), rows)
    assert got.shape == (600, 8)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    assert (got.argmax(1) == want.argmax(1)).all()


def test_the_page_reads_the_same_weights_bytes() -> None:
    bundled = REPO / "web" / "public" / "data" / "telecom" / "student_weights.json"
    assert bundled.read_bytes() == WEIGHTS.read_bytes(), "the CKKS tier's copy is not the published weights file"


def test_a_file_the_graph_cannot_express_is_refused() -> None:
    from qsh.models.linear_onnx import linear_student_graph

    w = json.loads(WEIGHTS.read_text())
    with pytest.raises(ValueError, match="standardisation only"):
        linear_student_graph({**w, "input": {**w["input"], "preprocessing": ["quadratic expansion"]}})
    with pytest.raises(ValueError, match="not a linear multiclass student"):
        linear_student_graph({**w, "kind": "something else"})
