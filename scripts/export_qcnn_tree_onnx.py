#!/usr/bin/env python3
"""A binary tree QCNN's classical contraction as an ONNX graph (references/satellite-qcnn-tn-preregistration.md, C).

The graph takes the artifact's raw features (rows, n), standardises them with the artifact's own moments, encodes each
as the Bloch vector of RY(c pi tanh z)|0>, applies each pooling stage's bilinear map (`qsh.qhe.qcnn_tree`) to the even
and odd wires of the stage with one Einsum, and returns <Z0> and the logit (scale times <Z0> plus bias). It is the
circuit's own function, not a model fitted to it, so the criterion is equality: on every test row the ONNX readout is
within 1e-5 of the state-vector simulation's (`qsh.qhe.qcnn_trained`), and no decision differs; otherwise nothing is
written.

    python scripts/export_qcnn_tree_onnx.py WEIGHTS_JSON OUT_ONNX --usecase satellite_opssat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

OPSET = 17
TOLERANCE = 1e-5


def build(weights: dict):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from qsh.qhe import qcnn_tree
    from qsh.qhe.qcnn_trained import validate

    layout = validate(weights)
    nq, nf, c = layout["nq"], layout["nf"], float(layout["c"])
    if nf != nq:
        raise SystemExit(f"{nf} features on {nq} wires: this export encodes every wire")
    net = qcnn_tree.network(weights)
    mu = np.asarray(weights["feature_mean"], dtype=np.float32)[:nf]
    sd = np.asarray(weights["feature_std"], dtype=np.float32)[:nf]
    sd = np.where(sd == 0, 1.0, sd).astype(np.float32)
    inits = [numpy_helper.from_array(mu, "mean"), numpy_helper.from_array(sd, "std"),
             numpy_helper.from_array(np.array(c * np.pi, dtype=np.float32), "angle_scale"),
             numpy_helper.from_array(np.array([1], dtype=np.int64), "axis_last"),
             numpy_helper.from_array(net["readout"].astype(np.float32), "readout_vector"),
             numpy_helper.from_array(np.array(float(weights["scale"]), dtype=np.float32), "head_scale"),
             numpy_helper.from_array(np.array(float(weights["bias"]), dtype=np.float32), "head_bias")]
    nodes = [helper.make_node("Sub", ["x", "mean"], ["centred"]), helper.make_node("Div", ["centred", "std"], ["z"]),
             helper.make_node("Tanh", ["z"], ["t"]), helper.make_node("Mul", ["t", "angle_scale"], ["theta"]),
             helper.make_node("Sin", ["theta"], ["s"]), helper.make_node("Cos", ["theta"], ["co"]),
             helper.make_node("Mul", ["s", "zero"], ["y0"]), helper.make_node("Add", ["y0", "one"], ["ones"]),
             helper.make_node("Unsqueeze", ["ones", "axis_neg"], ["e0"]), helper.make_node("Unsqueeze", ["s", "axis_neg"], ["e1"]),
             helper.make_node("Unsqueeze", ["y0", "axis_neg"], ["e2"]), helper.make_node("Unsqueeze", ["co", "axis_neg"], ["e3"]),
             helper.make_node("Concat", ["e0", "e1", "e2", "e3"], ["r0"], axis=-1)]
    inits += [numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero"),
              numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one"),
              numpy_helper.from_array(np.array([-1], dtype=np.int64), "axis_neg")]
    cur, width = "r0", nq
    for s, m in enumerate(net["maps"]):
        inits.append(numpy_helper.from_array(m.astype(np.float32), f"map{s}"))
        inits.append(numpy_helper.from_array(np.arange(0, width, 2, dtype=np.int64), f"even{s}"))
        inits.append(numpy_helper.from_array(np.arange(1, width, 2, dtype=np.int64), f"odd{s}"))
        nodes += [helper.make_node("Gather", [cur, f"even{s}"], [f"a{s}"], axis=1),
                  helper.make_node("Gather", [cur, f"odd{s}"], [f"b{s}"], axis=1),
                  helper.make_node("Einsum", [f"map{s}", f"a{s}", f"b{s}"], [f"r{s + 1}"], equation="kij,nwi,nwj->nwk")]
        cur, width = f"r{s + 1}", width // 2
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), "first"))
    nodes += [helper.make_node("Gather", [cur, "first"], ["last"], axis=1),
              helper.make_node("Squeeze", ["last", "axis_last"], ["survivor"]),
              helper.make_node("MatMul", ["survivor", "readout_vector"], ["readout"]),
              helper.make_node("Mul", ["readout", "head_scale"], ["scaled"]),
              helper.make_node("Add", ["scaled", "head_bias"], ["logit"])]
    graph = helper.make_graph(
        nodes, "qcnn_tree_contraction",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["rows", nf])],
        [helper.make_tensor_value_info("readout", TensorProto.FLOAT, ["rows"]),
         helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["rows"])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="qsh")
    # IR version 8 is opset 17's: the onnx package writes its own newest IR, which ONNX Runtime may not read yet.
    model.ir_version = 8
    model.doc_string = ("The binary tree QCNN's classical contraction (qsh.qhe.qcnn_tree): the circuit's own function, "
                        "not a model fitted to it. Not quantum hardware.")
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> int:
    import onnxruntime as ort

    from qcnn_variational import load, split_for
    from qsh.qhe.qcnn_trained import _run_circuit

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("weights", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--usecase", required=True)
    args = ap.parse_args()
    weights = json.loads(args.weights.read_text())
    model = build(weights)
    X, y, _, _ = load(args.usecase)
    order, _, vcut = split_for(args.usecase, y, int(weights["training"]["seed"]))
    Xte = X[order[vcut:]]
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    got, logit = sess.run(["readout", "logit"], {"x": Xte.astype(np.float32)})
    ref = np.array([_run_circuit(x, weights)[0] for x in Xte])
    ref_logit = ref * float(weights["scale"]) + float(weights["bias"])
    worst = float(np.max(np.abs(got.astype(np.float64) - ref)))
    flipped = int(((logit > 0) != (ref_logit > 0)).sum())
    print(f"{len(Xte)} test rows: largest |ONNX - state vector| {worst:.3e}; decisions differing {flipped}")
    if worst > TOLERANCE or flipped:
        raise SystemExit(f"not written: the criterion is within {TOLERANCE:g} and no decision differing")
    raw = model.SerializeToString()
    args.out.write_bytes(raw)
    record = {"onnx": args.out.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "opset": OPSET,
              "test_rows": int(len(Xte)), "max_abs_readout_difference": worst, "decisions_differing": flipped,
              "against": "qsh.qhe.qcnn_trained._run_circuit (state vector, float64)", "onnxruntime": ort.__version__}
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
