"""An ONNX graph for a linear student, built from its own weights.json (Round 86).

WHY THIS EXISTS

`telecom_ids.student` was bundled with the page and never published, so it was the one student the page did not
re-score from a pinned graph: its encrypted tier read 64 floats nobody checked against anything. Publishing it the
way the other students are published needs a graph the page can run in plaintext, and this student was fitted in
numpy and saved as `weights.json` only. The graph is built from that file, not refitted, so it is the same model:
input [n, F] -> (x - mean) / scale -> x W^T + b -> "logits" [n, C], the operations the CKKS tier performs on the
same weights. `tests/test_r86_linear_student_graph.py` holds the graph to the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: ONNX opset of the graph: Sub, Div, MatMul and Add are all in ONNX Runtime Web's WASM build at 17.
OPSET = 17


def linear_student_graph(weights: dict):
    """The ONNX model for a `logistic-linear-multiclass` weights file: standardise, then 8 dot products."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if weights.get("kind") != "logistic-linear-multiclass":
        raise ValueError(f"not a linear multiclass student: kind {weights.get('kind')!r}")
    steps = weights["input"]["preprocessing"]
    if steps != ["subtract mean", "divide by scale"]:
        raise ValueError(f"the graph builds standardisation only, and this file declares {steps}")
    W = np.asarray(weights["W"], dtype=np.float32)              # C x F
    b = np.asarray(weights["b"], dtype=np.float32)
    mean = np.asarray(weights["input"]["mean"], dtype=np.float32)
    scale = np.asarray(weights["input"]["scale"], dtype=np.float32)
    n_classes, n_features = W.shape
    if mean.shape != (n_features,) or scale.shape != (n_features,) or b.shape != (n_classes,):
        raise ValueError("mean, scale and b do not match W's shape")
    inits = [numpy_helper.from_array(mean, "mean"), numpy_helper.from_array(scale, "scale"),
             numpy_helper.from_array(np.ascontiguousarray(W.T), "W_t"), numpy_helper.from_array(b, "b")]
    nodes = [helper.make_node("Sub", ["input", "mean"], ["centred"]),
             helper.make_node("Div", ["centred", "scale"], ["z"]),
             helper.make_node("MatMul", ["z", "W_t"], ["scores"]),
             helper.make_node("Add", ["scores", "b"], ["logits"])]
    graph = helper.make_graph(
        nodes, str(weights.get("student_key", "linear_student")),
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["n", n_features])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", n_classes])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="qsh")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def write_linear_student_graph(weights: dict, out: Path) -> Path:
    """Write the graph to `out` (`onnx/model.onnx` beside the weights file)."""
    import onnx

    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(linear_student_graph(weights), out)
    return out
