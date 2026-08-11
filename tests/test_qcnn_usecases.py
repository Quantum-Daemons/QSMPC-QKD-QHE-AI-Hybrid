"""The QCNN backend option is offered exactly where trained weights ship.

WHY THIS EXISTS. Round 28 found that selecting ``Inference = QCNN`` on the genomics page produced a
class from CKKS while two panels announced that the quantum circuit had decided it. Three things had
to line up for that: genomics reaches the ``private_inference`` stage, so the selector legitimately
renders; the QCNN option was listed unconditionally; and no ``qcnn_weights.json`` ships for genomics,
so the loader returned ``None`` and the code fell back - correctly - to CKKS.

The fix hides the option where no weights exist, which introduces a list that can go stale in both
directions: train a third circuit and the option stays hidden, delete one and the option lies. A
hand-maintained list of what exists on disk is exactly the kind of claim this project checks rather
than trusts, so this test reads both sides and compares them.

It deliberately parses the TypeScript rather than importing it: ``tests/test_api_config.py`` sets the
precedent that a Python test may read web sources, and adding a JS test runner for one assertion is a
worse trade than a narrow regex over a line whose shape is asserted here too.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "web" / "src" / "lib" / "usePipelineRun.ts"
PUBLIC = REPO / "web" / "public" / "data"

#: ``clientCompute.dataDirFor``: use case -> directory under ``public/data``. Only ``medical_fl``
#: fails the ``split("_")[0]`` rule, and that exception lives in one place in the TS too.
def _data_dir(usecase: str) -> str:
    return "medical" if usecase == "medical_fl" else usecase.split("_")[0]


def _declared() -> set[str]:
    """The use cases ``QCNN_USECASES`` claims ship trained weights."""
    text = SRC.read_text()
    m = re.search(r"export const QCNN_USECASES = new Set\(\[(.*?)\]\)", text, re.S)
    assert m, f"QCNN_USECASES not found in {SRC.relative_to(REPO)} - did the declaration move?"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def _on_disk() -> set[str]:
    """Use cases whose weights file actually exists, mapped back from directory to use case."""
    dirs = {p.parent.name for p in PUBLIC.glob("*/qcnn_weights.json")}
    known = ["finance_aml", "medical_fl", "genomics", "satellite_comms", "telecom_ids"]
    return {u for u in known if _data_dir(u) in dirs}


def test_declared_set_matches_the_files_on_disk() -> None:
    declared, actual = _declared(), _on_disk()
    assert declared == actual, (
        "QCNN_USECASES disagrees with web/public/data/*/qcnn_weights.json.\n"
        f"  declared but absent: {sorted(declared - actual) or 'none'}\n"
        f"  present but undeclared: {sorted(actual - declared) or 'none'}\n"
        "Declaring a use case whose weights are missing puts an inert option back on the page; "
        "omitting one that has weights hides a working backend."
    )


@pytest.mark.parametrize("usecase", sorted(_on_disk()))
def test_shipped_weights_are_loadable_and_carry_a_headline_metric(usecase: str) -> None:
    """A shipped file must be usable, not merely present.

    ``headline_metric`` is load-bearing: ``clientCompute`` chooses between accuracy and minority-class
    F1 from it, and the live deploy once served a weights file predating the field, which silently
    made a 5 percent-prevalence use case report accuracy.
    """
    import json

    w = json.loads((PUBLIC / _data_dir(usecase) / "qcnn_weights.json").read_text())
    assert w["headline_metric"] in {"accuracy", "minority_class_f1"}
    assert len(w["theta"]) == w["layers"], "theta rows must equal the layer count"
    assert all(len(row) == w["n_qubits"] for row in w["theta"]), "each layer needs one angle per qubit"
    if w.get("input_space") == "raw":
        # Raw-space models standardise with shipped moments; without them the circuit sees mixed
        # scales and pins its features, which is how the first trained model became meaningless.
        assert len(w["feature_mean"]) == w["n_qubits"]
        assert len(w["feature_std"]) == w["n_qubits"]
