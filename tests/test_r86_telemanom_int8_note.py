"""The Telemanom card says its int8 graph quantises only the output layer, and the record holds it to that (Round 86).

The card printed `export.int8_bytes` beside `fp32_bytes` as though the forecaster had been quantised. It had not:
onnxruntime's dynamic quantiser converts an LSTM only when both of its weight tensors are stored constants, and the
exported graph computes them (through Unsqueeze), so the int8 graph keeps both LSTM layers in fp32 and quantises the
dense output layer alone. The int8 file is 96% of the fp32 one.

The graphs are not tracked, so CI holds the note to the export report: an int8 file about the size of the fp32 one is
what a graph whose LSTM stayed fp32 looks like, and an export that did quantise the LSTM would shrink it to about a
quarter and fail here until the note changes. On the machine that trained it, where the graph is present, its
operators are read as well.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qsh.models import registry

REPO = Path(__file__).resolve().parents[1]
KEY = "satellite_comms.telemanom"
NOTE = "Its int8 graph quantises only the output layer."


def _model_dir() -> Path:
    return REPO / registry.ARTIFACT_DIRS[KEY]


def test_the_note_matches_the_export_report() -> None:
    report = json.loads((_model_dir() / "export_report.json").read_text())
    assert report["int8_bytes"] > 0 and not report["int8_skipped_reason"], "no int8 graph is shipped any more"
    ratio = report["int8_bytes"] / report["fp32_bytes"]
    notes = registry.get(KEY).notes
    if ratio > 0.5:
        assert NOTE in notes, (f"the int8 graph is {ratio:.0%} of the fp32 one, so its LSTM layers stayed fp32, "
                               "and the card has to say so")
    else:
        assert NOTE not in notes, (f"the int8 graph is {ratio:.0%} of the fp32 one: the LSTM is quantised now, "
                                   "so the note is wrong")


def test_the_shipped_graph_keeps_both_lstm_layers_in_fp32() -> None:
    graph = _model_dir() / "onnx" / "model_int8.onnx"
    if not graph.exists():
        pytest.skip("the int8 graph is not tracked; test_the_note_matches_the_export_report holds the note in CI")
    import onnx

    ops = [n.op_type for n in onnx.load(str(graph)).graph.node]
    assert ops.count("LSTM") == 2 and "DynamicQuantizeLSTM" not in ops, ops
    assert ops.count("MatMulInteger") == 1, ops
