"""The satellite QCNNs as shipped: artifacts, the card's data, the ONNX graphs and the tree network under CKKS (Round 86).

references/satellite-qcnn-tn-preregistration.md fixed the design before any training; the owner chose on 2026-11-24 to
show both circuits on the satellite page, OPS-SAT first, and to run the tree network under CKKS there. These tests hold
the page's files to the artifacts and the records, the ONNX graphs to their criterion, and the encrypted path to its
criterion on shipped rows, on the real node-seal.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "public" / "data" / "satellite"
CORPORA = ("opssat", "esa")


def _web(short: str) -> dict:
    return json.loads((WEB / f"satellite_qcnn_{short}.json").read_text(encoding="utf-8"))


def test_the_card_files_are_what_the_exporter_writes() -> None:
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "export_satellite_qcnn.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr


def test_without_the_corpora_the_check_still_checks_everything_but_the_rows(tmp_path: Path, monkeypatch) -> None:
    """CI has no corpus, so there the check takes the rows, labels and feature names from the shipped files; it must
    still fail when any other field differs from what the artifacts and the records give."""
    sys.path.insert(0, str(REPO / "scripts"))
    import export_satellite_qcnn as ex

    monkeypatch.setenv("QSH_DATA_ROOT", str(tmp_path / "no-corpus"))
    out = tmp_path / "out"
    out.mkdir()
    for short in CORPORA:
        (out / f"satellite_qcnn_{short}.json").write_bytes((WEB / f"satellite_qcnn_{short}.json").read_bytes())
    monkeypatch.setattr(ex, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["export_satellite_qcnn.py", "--check"])
    assert ex.main() == 0
    for short, field in (("opssat", "test_rows"), ("esa", "test_positive")):
        f = out / f"satellite_qcnn_{short}.json"
        kept = f.read_bytes()
        d = json.loads(kept)
        d["figures"][field] += 1
        f.write_text(json.dumps(d, separators=(",", ":")) + "\n", encoding="utf-8")
        assert ex.main() == 1, f"{short}: a changed {field} passed the check without the corpus"
        f.write_bytes(kept)


@pytest.mark.parametrize("short", CORPORA)
def test_the_page_runs_the_shipped_weights_and_their_tree(short: str) -> None:
    from qsh.qhe import qcnn_tree

    w = json.loads((REPO / "models" / f"qcnn_satellite_{short}" / "weights.json").read_text(encoding="utf-8"))
    web = _web(short)
    assert web["weights"] == w
    net = qcnn_tree.network(w)
    for got, want in zip(web["tree"]["maps"], net["maps"], strict=True):
        assert np.allclose(np.asarray(got), want, atol=0, rtol=0)
    assert np.allclose(web["tree"]["readout"], net["readout"], atol=0, rtol=0)
    assert len(web["rows"]) == len(web["labels"]) and set(web["labels"]) <= {0, 1}


@pytest.mark.parametrize("short", CORPORA)
def test_each_onnx_graph_met_its_criterion_and_rebuilds_from_the_weights(short: str) -> None:
    """The repository commits no .onnx (`*.onnx` is ignored), so the graph is rebuilt from the committed weights and must
    be, byte for byte, the one the export checked and recorded in tree.json."""
    pytest.importorskip("onnx")
    sys.path.insert(0, str(REPO / "scripts"))
    from export_qcnn_tree_onnx import build

    d = REPO / "models" / f"qcnn_satellite_{short}"
    rec = json.loads((d / "tree.json").read_text(encoding="utf-8"))
    w = json.loads((d / "weights.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(build(w).SerializeToString()).hexdigest() == rec["sha256"]
    assert rec["max_abs_readout_difference"] <= 1e-5 and rec["decisions_differing"] == 0
    scores = json.loads((REPO / "docs" / "measurements" / "satellite-qcnn-2026-11-24.json").read_text(encoding="utf-8"))
    assert rec["test_rows"] == scores[short]["rows"]["test"]


def test_the_encrypted_record_met_its_criterion_on_every_test_row() -> None:
    rec = json.loads((REPO / "docs" / "measurements" / "satellite-qcnn-tree-ckks-2026-11-24.json").read_text(encoding="utf-8"))
    scores = json.loads((REPO / "docs" / "measurements" / "satellite-qcnn-2026-11-24.json").read_text(encoding="utf-8"))
    assert rec["params"]["N"] == 16384 and sum(rec["params"]["coeff_mod_bits"]) == 360
    for case in rec["cases"]:
        assert case["rows"] == scores[case["name"]]["rows"]["test"]
        assert case["criterion_met"] and case["tree"]["decisions_differing"] == 0
        assert case["tree"]["max_abs_readout_error"] <= 1e-3


@pytest.mark.parametrize("short", CORPORA)
def test_each_card_states_its_seed_and_that_the_surrogate_leads(short: str) -> None:
    card = (REPO / "models" / f"qcnn_satellite_{short}" / "README.md").read_text(encoding="utf-8")
    scores = json.loads((REPO / "docs" / "measurements" / "satellite-qcnn-2026-11-24.json").read_text(encoding="utf-8"))[short]
    spread = scores["qcnn_spread"]["average_precision"]
    assert f"{spread['mean']:.3f} +- {spread['sd']:.3f}" in card
    assert f"{scores['classical_surrogate']['average_precision']:.3f}" in card
    assert "**The classical surrogate leads" in card
    assert scores["classical_surrogate"]["average_precision"] > spread["max"]


def test_the_tree_network_under_ckks_on_shipped_rows_in_node() -> None:
    """The page's module on the real node-seal, on the first 256 shipped rows of each corpus."""
    if shutil.which("node") is None or not (REPO / "web" / "node_modules" / "node-seal").exists():
        pytest.skip("node or web/node_modules is not installed")
    from qsh.qhe import qcnn_tree

    cases = []
    for short in CORPORA:
        web = _web(short)
        w = web["weights"]
        rows = np.asarray(web["rows"][:256], dtype=float)
        cases.append({"name": short, "weights_path": str(REPO / "models" / f"qcnn_satellite_{short}" / "weights.json"),
                      "pauli_rows": 1, "rows": rows.tolist(), "expected": qcnn_tree.readout(w, rows).tolist(),
                      "network": {"maps": web["tree"]["maps"], "readout": web["tree"]["readout"], "scale": w["scale"],
                                  "bias": w["bias"], "feature_mean": w["feature_mean"], "feature_std": w["feature_std"],
                                  "encoding_scale": w.get("encoding_scale", 1.0), "n_qubits": w["n_qubits"]}})
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps({"cases": cases}))
        r = subprocess.run(["node", str(REPO / "web" / "scripts" / "check_qcnn_tree_ckks.mjs"), str(src), str(dst)],
                           capture_output=True, text=True, cwd=REPO, timeout=600)
        assert r.returncode == 0, r.stdout + r.stderr
        out = json.loads(dst.read_text())
    for case in out["cases"]:
        assert case["tree"]["max_abs_readout_error"] < 1e-3 and case["tree"]["decisions_differing"] == 0, case
        assert case["clear_twin_max_difference"] < 1e-12, case


def test_the_satellite_card_draws_its_circuit_from_the_shipped_weights() -> None:
    """The owner's choice of 2026-11-25: the satellite QCNNs' circuit drawn as the other pages draw theirs, by
    `recordQcnn` on the weights the card ships, so a retrain that ships new weights redraws it and nothing is typed in."""
    card = (REPO / "web" / "src" / "components" / "SatelliteQcnnCard.tsx").read_text()
    shared = (REPO / "web" / "src" / "components" / "QcnnCircuitCard.tsx").read_text()
    assert 'import { BlockDiagram } from "./QcnnCircuitCard";' in card
    assert "recordQcnn(file.weights)" in card and "<BlockDiagram r={recorded} weights={file.weights} />" in card
    assert "export function BlockDiagram(" in shared
    assert not list((REPO / "web" / "public").rglob("*circuit*.svg")), "a circuit drawn as a fixed SVG file"
