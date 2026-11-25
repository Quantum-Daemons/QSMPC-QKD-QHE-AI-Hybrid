"""A binary tree QCNN's classical contraction is the circuit, and its ONNX graph is too (Round 86).

references/satellite-qcnn-tn-preregistration.md, sections C and D, addendum 3. `qsh.qhe.qcnn_tree` contracts the
circuit stage by stage, one bilinear map on single-qubit Bloch vectors a pooling stage; these tests hold it to the
state-vector simulation (`qsh.qhe.qcnn_trained`) on the shipped medical QCNN, and hold the ONNX graph
`scripts/export_qcnn_tree_onnx.py` writes to the same.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "train"))
sys.path.insert(0, str(REPO / "scripts"))

from qsh.qhe import qcnn_tree  # noqa: E402
from qsh.qhe.qcnn_trained import _run_circuit  # noqa: E402

WEIGHTS = json.loads((REPO / "models" / "qcnn_medical" / "weights.json").read_text(encoding="utf-8"))


def _rows() -> np.ndarray:
    from qcnn_variational import load

    X, _, _, _ = load("medical")
    return X


def test_the_contraction_is_the_circuit_on_every_row() -> None:
    X = _rows()
    tree = qcnn_tree.readout(WEIGHTS, X)
    ref = np.array([_run_circuit(x, WEIGHTS)[0] for x in X])
    assert np.max(np.abs(tree - ref)) < 1e-12


def test_each_stage_map_keeps_a_state_a_state() -> None:
    """The survivor's first coordinate is the trace, 1 for any pair of states: M[0] is 1 at (0, 0) and 0 elsewhere."""
    for m in qcnn_tree.network(WEIGHTS)["maps"]:
        want = np.zeros((4, 4))
        want[0, 0] = 1.0
        assert np.allclose(m[0], want, atol=1e-12)
        assert np.isrealobj(m)


def test_a_pure_input_never_leaves_the_bloch_ball() -> None:
    net = qcnn_tree.network(WEIGHTS)
    r = qcnn_tree.encoded_bloch(WEIGHTS, _rows()[:64])
    for m in net["maps"]:
        r = np.einsum("kij,nwi,nwj->nwk", m, r[:, 0::2], r[:, 1::2])
        assert np.all(np.linalg.norm(r[..., 1:], axis=-1) <= 1 + 1e-12)


def test_the_onnx_graph_is_the_circuit() -> None:
    ort = pytest.importorskip("onnxruntime")
    from export_qcnn_tree_onnx import TOLERANCE, build

    model = build(WEIGHTS)
    assert model.ir_version == 8
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    X = _rows()
    got, logit = sess.run(["readout", "logit"], {"x": X.astype(np.float32)})
    ref = np.array([_run_circuit(x, WEIGHTS)[:3] for x in X])
    assert np.max(np.abs(got - ref[:, 0])) < TOLERANCE
    assert np.array_equal(logit > 0, ref[:, 2] > 0)
