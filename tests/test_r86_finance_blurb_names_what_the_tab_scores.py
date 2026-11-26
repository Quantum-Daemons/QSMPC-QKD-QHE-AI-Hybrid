"""Round 86 (batch 6): the finance blurb names what the browser scores since F7b, and what the full deployment scores.

#94 put F7b's square network on the finance page in place of the linear student and moved the pipeline's transaction to
an AMLworld test-window row, but left the blurb saying "The scored rows are PaySim simulator output, and the model
reads four quantities ...", which the public demo showed until this batch. The blurb now says both: in the browser an
AMLworld row through the square network under CKKS, the QCNN on eight of that row's inputs (on a PaySim row of its own
until Round 86's batch 6); in the full deployment's Python pipeline, PaySim rows and the four-column model. Each statement is held to the file it describes, so a retrain
that changes one of them fails here instead of on the site.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"


def _blurb() -> str:
    uc = (WEB / "src" / "lib" / "useCases.ts").read_text()
    m = re.search(r'id: "finance_aml".*?blurb: "([^"]*)"', uc, re.S)
    assert m, "finance blurb not found"
    return m.group(1)


def test_the_blurb_no_longer_says_the_tab_scores_paysim_rows() -> None:
    assert "The scored rows are PaySim simulator output" not in _blurb()


def test_the_browser_half_matches_the_square_network_the_page_runs() -> None:
    blurb = _blurb()
    net = json.loads((WEB / "public" / "data" / "finance" / "square_student_weights.json").read_text())
    assert net["kind"] == "dense-bn-square-dense"
    assert f"square network over its {net['n_in']} inputs" in blurb
    assert "test-window row of IBM's AMLworld HI-Small" in blurb
    assert "under CKKS end to end in a Web Worker" in blurb
    data = (WEB / "src" / "lib" / "financeSquareData.ts").read_text()
    assert 'SQUARE_KEY = "finance_aml.square_network"' in data


def test_the_qcnn_clause_matches_the_circuit_the_page_loads() -> None:
    """Since Round 86's batch 6 (the owner's choice of 2026-11-26) the QCNN is the AMLworld circuit: the page feeds it eight
    of the scored transaction's own inputs, and the Python pipeline, which scores PaySim rows, an AMLworld row of the page's
    sample. Until then the clause said it encoded four columns of a PaySim row of its own, which was then true."""
    w = json.loads((WEB / "public" / "data" / "finance" / "qcnn_weights.json").read_text())
    assert len(w["feature_names"]) == w["n_qubits"] == 8 and "AMLworld" in w["provenance"] and "PaySim" not in w["provenance"]
    blurb = _blurb()
    assert "the QCNN beside it encodes eight of the same transaction's inputs" in blurb
    assert "its QCNN reads an AMLworld transaction of its own, from the page's sample" in blurb
    assert "PaySim row of its own" not in blurb
    src = (REPO / "src" / "qsh" / "usecases" / "finance_aml" / "__init__.py").read_text()
    assert "QCNN_HELDOUT" in src and "qcnn_heldout.json" in src


def test_the_full_deployment_half_matches_the_python_pipeline() -> None:
    blurb = _blurb()
    src = (REPO / "src" / "qsh" / "usecases" / "finance_aml" / "__init__.py").read_text()
    assert "registry.PAYSIM_MODEL_FEATURES" in src and "PaySim simulator sample" in src
    assert "The full deployment's Python pipeline scores PaySim simulator rows instead" in blurb
    for name in ("amount", "cross_border", "night_tx", "new_payee"):
        assert name in blurb, name
