"""An exported graph declares the batch axis it computes with, on its outputs as on its inputs.

WHY THIS EXISTS

Round 71. The published Telemanom LSTM declared input ``[batch, 250, 6]`` and output ``[1, 6]``. The
graph computed every row correctly - its export gate compared 256 rows at once and passed - but
onnxruntime checks the declared shape after each run, and the browser scores 128 windows at once, so
every satellite run logged "Expected shape from model of {1,6} does not match actual shape of
{128,6}". The cause was the example the exporter traced on: one row. torch>=2.9 exports through
torch.export, which specialises a dimension whose example size is 0 or 1. The ResNets went through
the same call with the same one-row example and kept ``[batch, 2]``, which is why one row looked
safe.

WHAT IS ASSERTED

- Behaviourally, through the export path the trainers use: a small two-layer LSTM with a linear head,
  the Telemanom shape, exported from a ONE-row example, declares a symbolic output batch and runs at
  batch 5 with the same numbers as torch.
- The detector the exporter now refuses on names a fixed output batch, and stays quiet both for a
  symbolic one and for a graph whose input batch is fixed too.
- Every ONNX graph present locally declares its batch consistently. CI has no artefacts, so there the
  first two carry the check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from qsh.models import export  # noqa: E402


def _lstm():
    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(3, 8, num_layers=2, batch_first=True)
            self.head = torch.nn.Linear(8, 3)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    torch.manual_seed(0)
    return Net().eval()


def _dims(vi) -> list:
    return [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]


def test_a_one_row_example_exports_a_symbolic_output_batch(tmp_path: Path) -> None:
    model = _lstm()
    x = np.random.default_rng(0).standard_normal((5, 12, 3)).astype(np.float32)
    path = export.export_torch(model, x[:1], tmp_path / "model.onnx")

    graph = onnx.load(str(path)).graph
    assert isinstance(_dims(graph.input[0])[0], str), "the input lost its batch axis"
    out = _dims(graph.output[0])
    assert isinstance(out[0], str), (
        f"output declares {out}: a fixed batch, so onnxruntime warns on every batched call")
    assert export.fixed_batch_outputs(path) == []

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        ref = model(torch.tensor(x)).numpy()
    assert got.shape == (5, 3)
    assert np.abs(got - ref).max() < 1e-5, "tracing on a repeated row changed what the graph computes"


def _identity(tmp_path: Path, name: str, inp: list, out: list) -> Path:
    from onnx import TensorProto, helper

    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], name,
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, inp)],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, out)])
    p = tmp_path / f"{name}.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), str(p))
    return p


def test_the_detector_names_a_fixed_output_batch_and_nothing_else(tmp_path: Path) -> None:
    assert export.fixed_batch_outputs(_identity(tmp_path, "fixed", ["batch", 3], [1, 3])) == ["y [1, 3]"]
    assert export.fixed_batch_outputs(_identity(tmp_path, "symbolic", ["batch", 3], ["batch", 3])) == []
    # A graph built for one row throughout is consistent, not defective.
    assert export.fixed_batch_outputs(_identity(tmp_path, "single", [1, 3], [1, 3])) == []


def test_every_local_graph_declares_its_batch_consistently() -> None:
    graphs = sorted((REPO / "models").glob("*/onnx/*.onnx"))
    if not graphs:
        pytest.skip("no ONNX artefacts in this checkout (they are not committed); the two tests "
                    "above carry the check here")
    bad = {str(p.relative_to(REPO)): export.fixed_batch_outputs(p) for p in graphs}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"graphs whose outputs declare a fixed batch: {bad}"
