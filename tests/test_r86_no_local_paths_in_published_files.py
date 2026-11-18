"""Round 86 (2026-11-18): no graph this project exports or publishes carries the exporting machine's file paths.

torch>=2.9 exports through torch.export, which records on every node of the ONNX graph its FX node, its module path
and its Python stack trace, and the stack trace holds absolute file paths: seven graphs on the Hub carried this
machine's home directory. The exporter now drops node metadata (`qsh.models.export.strip_node_metadata`, also applied
to the int8 graph the quantiser derives), and the publish step refuses any file that still holds the home directory
(`scripts/publish_hf.py::local_paths_in`), dry runs included.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

onnx = pytest.importorskip("onnx")

from qsh.models import export  # noqa: E402

spec_ = importlib.util.spec_from_file_location("publish_hf", REPO / "scripts" / "publish_hf.py")
publish_hf = importlib.util.module_from_spec(spec_)
sys.modules.setdefault("publish_hf", publish_hf)
spec_.loader.exec_module(publish_hf)


def _graph_with_a_stack_trace(path: Path, where: str) -> Path:
    node = onnx.helper.make_node("Relu", ["x"], ["y"], doc_string="a doc string")
    node.metadata_props.add(key="pkg.torch.onnx.stack_trace", value=f'File "{where}/train.py", line 1, in main')
    node.metadata_props.add(key="namespace", value="Net/relu")
    graph = onnx.helper.make_graph([node], "g", [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
                                   [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])])
    onnx.save(onnx.helper.make_model(graph), str(path))
    return path


def test_strip_node_metadata_drops_what_torch_export_writes(tmp_path):
    path = _graph_with_a_stack_trace(tmp_path / "m.onnx", "/home/user/project")
    assert export.strip_node_metadata(path) == 3          # two metadata entries and the doc string
    assert export.strip_node_metadata(path) == 0          # nothing left, and the file is not rewritten
    model = onnx.load(str(path))
    assert [n.op_type for n in model.graph.node] == ["Relu"]
    assert not model.graph.node[0].metadata_props and not model.graph.node[0].doc_string
    assert b"/home/user" not in path.read_bytes()


def test_the_publish_step_names_a_file_holding_the_home_directory(tmp_path):
    held = _graph_with_a_stack_trace(tmp_path / "held.onnx", "/home/user/project")
    clean = tmp_path / "eval_sample.json"
    clean.write_text('{"kind": "graph"}\n')
    assert publish_hf.local_paths_in([held, clean], home="/home/user") == [str(held)]
    assert publish_hf.local_paths_in([held, clean], home="/home/other") == []
    assert publish_hf.local_paths_in([held], home="/") == []    # no home directory to look for


def test_a_torch_export_keeps_no_node_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)).eval()
    x = np.arange(8, dtype=np.float32).reshape(2, 4) / 8
    path = export.export_torch(model, x[:1], tmp_path / "model.onnx")
    graph = onnx.load(str(path))
    assert all(not n.metadata_props and not n.doc_string for n in graph.graph.node)
    assert str(Path.home()).encode() not in path.read_bytes()
    with torch.no_grad():
        want = model(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    np.testing.assert_allclose(got, want, atol=1e-5)
