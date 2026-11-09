"""The trained QCNN's prediction with its input under CKKS (Round 86, the owner's choice).

WHY THIS EXISTS. A trained QCNN ran on plaintext features and the page said so; the owner asked for the private form. Every quantum route needs quantum resources a browser does not have (Broadbent and Jeffery 2015;
Dulek, Schaffner and Speelman 2016; Ortega and Martin-Delgado, arXiv:2602.12712), but the circuit here is simulated classically, so the private form is a classical HE problem. The encoding is a product of real
states, so the readout is exactly a linear form in 3^n products of sines and cosines of the encoded angles (`qcnnPauli.ts`, `qsh.qhe.qcnn_pauli`), and one multiplicative level evaluates it.

WHAT IS ASSERTED. The form is the circuit's readout in Python and is the browser's form to rounding. The class under CKKS is the clear circuit's. The selector offers it only where a trained circuit ships, both QCNN
sites route through one function, the page states what this is not, the classical surrogate is on the circuit's own split, and the node check is registered where the two CI paths run it.
"""

from __future__ import annotations

import json
import re
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from qsh.fhe.backends import get_backend
from qsh.qhe.qcnn_pauli import pauli_features, pauli_form, under_ckks
from qsh.qhe.qcnn_trained import infer

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
CLIENT = (WEB / "src/lib/clientCompute.ts").read_text()
BINARY = {"finance": "models/qcnn_finance/weights.json", "medical": "models/qcnn_medical/weights.json"}


def _weights(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def _rows(w: dict, n: int) -> np.ndarray:
    nf = w["n_features"]
    rng = np.random.default_rng(0)
    return np.asarray(w["feature_mean"])[:nf] + rng.normal(size=(n, nf)) * np.asarray(w["feature_std"])[:nf]


@pytest.mark.parametrize("name", BINARY)
def test_the_form_is_the_circuits_readout(name: str) -> None:
    w = _weights(BINARY[name])
    c = pauli_form(w)
    assert c.size == 3 ** w["n_features"]
    for x in _rows(w, 32):
        assert float(c @ pauli_features(w, x)) == pytest.approx(infer(x, w)[0], abs=1e-12)


@pytest.mark.parametrize("name", BINARY)
def test_the_browser_builds_the_same_form(name: str, tmp_path: Path) -> None:
    w = _weights(BINARY[name])
    (tmp_path / "w.json").write_text(json.dumps(w))
    script = textwrap.dedent(f"""
        import {{ build }} from "esbuild";
        import {{ readFileSync, writeFileSync }} from "node:fs";
        await build({{ stdin: {{ contents: 'export {{ pauliForm }} from "./src/lib/qcnnPauli";', resolveDir: {json.dumps(str(WEB))}, loader: "ts" }}, bundle: true, format: "esm", platform: "node", outfile: {json.dumps(str(tmp_path / "b.mjs"))}, logLevel: "error" }});
        const m = await import({json.dumps((tmp_path / "b.mjs").as_uri())});
        const form = m.pauliForm(JSON.parse(readFileSync({json.dumps(str(tmp_path / "w.json"))}, "utf8")));
        writeFileSync({json.dumps(str(tmp_path / "out.json"))}, JSON.stringify(Array.from(form.coefficients[0])));
    """)
    (WEB / ".tmp-qcnn-form.mjs").write_text(script)
    try:
        subprocess.run(["node", ".tmp-qcnn-form.mjs"], cwd=WEB, check=True, capture_output=True, text=True, timeout=120)
    finally:
        (WEB / ".tmp-qcnn-form.mjs").unlink()
    browser = np.asarray(json.loads((tmp_path / "out.json").read_text()))
    assert np.abs(browser - pauli_form(w)).max() < 1e-12


@pytest.mark.parametrize("name", BINARY)
def test_the_class_under_ckks_is_the_clear_circuits(name: str) -> None:
    w = _weights(BINARY[name])
    backend = get_backend("tenseal")
    for x in _rows(w, 2):
        out = under_ckks(w, x, backend)
        assert out["class"] == out["clear_class"], out
        assert out["max_error"] < 1e-3, out


def test_the_selector_offers_it_where_a_trained_circuit_ships() -> None:
    panel = (WEB / "src/components/ParameterPanel.tsx").read_text()
    assert re.search(r'hasTrainedQcnn\(usecase\) && <option value="qcnn-ckks">QCNN \(trained, under CKKS\)</option>', panel)
    assert '<option value="qcnn">QCNN (trained, not encrypted)</option>' in panel, "the plaintext run stays as the comparison"


def test_both_qcnn_sites_route_through_one_function() -> None:
    assert len(re.findall(r'await surfaceQcnnUnderCkks\(', CLIENT)) == 2, "the qml_ids site and the private_inference site"
    assert 'if (!(qcnnAsked === "qcnn-ckks" && await surfaceQcnnUnderCkks(' in CLIENT and 'if (!(inferBackend === "qcnn-ckks" && await surfaceQcnnUnderCkks(' in CLIENT, "the plaintext backend is reported only if CKKS did not run"
    helper = CLIENT[CLIENT.index("async function surfaceQcnnUnderCkks"):]
    helper = helper[:helper.index("\n}\n")]
    assert 'metric(stage, "primary_inference_backend", "qcnn-ckks")' in helper
    assert "is not encrypted" in helper, "a failed run must say the class is the plaintext circuit's"
    flows = (WEB / "src/lib/useCaseFlows.ts").read_text()
    assert 'ran === "qcnn-ckks"' in flows


def test_the_panel_says_what_this_is_not() -> None:
    panel = (WEB / "src/components/QcnnCkksPanel.tsx").read_text()
    for needle in ("not quantum homomorphic encryption", "notional", "This is classical computation under RLWE", "chosen scores"):
        assert needle in panel, needle
    page = (WEB / "src/pages/UseCasePage.tsx").read_text()
    assert "artifacts?.qcnn_ckks ? <QcnnCkksPanel" in page


@pytest.mark.parametrize("model,web", [("models/qcnn_medical/weights.json", "web/public/data/medical/qcnn_weights.json"),
                                       ("models/qcnn_finance/weights.json", "web/public/data/finance/qcnn_weights.json"),
                                       ("models/qcnn_telecom/weights.json", "web/public/data/telecom/qcnn_weights.json"),
                                       ("models/qcnn_telecom_q7/weights.json", "web/public/data/telecom/qcnn_weights_q7.json")])
def test_the_surrogate_is_recorded_on_the_circuits_own_features(model: str, web: str) -> None:
    m, w = _weights(model), _weights(web)
    s = m["classical_surrogate"]
    assert s == w["classical_surrogate"], "the page reads the bundled copy"
    assert s["n_features"] == 3 ** m["n_features"]
    assert "same split" in s["model"] and s["test_accuracy"] > 0.5


def test_the_trainer_writes_the_surrogate() -> None:
    trainer = (REPO / "scripts/train/qcnn_variational.py").read_text()
    assert '"classical_surrogate": classical_surrogate(' in trainer


def test_both_ci_paths_run_the_check() -> None:
    assert '"check:qcnn-ckks"' in (WEB / "package.json").read_text()
    assert "check:qcnn-ckks" in (REPO / "scripts/ci_local.sh").read_text()
    assert "check:qcnn-ckks" in (REPO / ".github/workflows/ci.yml").read_text()


def test_the_diagram_is_drawn_from_the_recording() -> None:
    card = (WEB / "src/components/QcnnCircuitCard.tsx").read_text()
    assert "recordQcnn(w)" in card and "layoutBlocks(" in card
    assert "gateCounts(recorded)" in card, "the summary's counts are the recording's"
