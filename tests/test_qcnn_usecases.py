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

    ``headline_metric`` is load-bearing: ``clientCompute`` chooses between accuracy, minority-class
    F1 and macro F1 from it, and the live deploy once served a weights file predating the field, which
    silently made a 5 percent-prevalence use case report accuracy.

    The accepted set gained ``macro_f1`` when the 8-class telecom circuit shipped. The two-way version
    asked only whether the BINARY problem was imbalanced, so a balanced multiclass run fell through to
    accuracy while the trainer printed "MACRO F1 leads" - the artifact carried floors for macro F1 and
    told its consumers to headline something else. Widened deliberately rather than by deleting the
    assertion; ``tests/test_qcnn_reachable.py`` now also requires that whichever value is set names a
    field the file actually populates.
    """
    import json

    w = json.loads((PUBLIC / _data_dir(usecase) / "qcnn_weights.json").read_text())
    assert w["headline_metric"] in {"accuracy", "minority_class_f1", "macro_f1"}
    assert len(w["theta"]) == w["layers"], "theta rows must equal the layer count"
    assert all(len(row) == w["n_qubits"] for row in w["theta"]), "each layer needs one angle per qubit"
    if w.get("input_space") == "raw":
        # Raw-space models standardise with shipped moments; without them the circuit sees mixed
        # scales and pins its features, which is how the first trained model became meaningless.
        # One moment per ENCODED FEATURE, and features may be fewer than wires. This asserted
        # `== n_qubits` until Round 34, which was true only while every wire carried data; telecom
        # puts 7 features on 8 wires and leaves the last UNENCODED, so the strict equality failed on
        # a model that is correct. The browser feeds `features[q] ?? 0` with a missing mean of 0 and
        # a missing std of 1, giving RY(0) - the identity - which is exactly what the trainer's
        # zero-pad produces, so the two agree by construction and this checks that they can.
        n_feat = w.get("n_features", w["n_qubits"])
        assert len(w["feature_mean"]) == n_feat, "one mean per encoded feature"
        assert len(w["feature_std"]) == n_feat, "one std per encoded feature"
        assert n_feat <= w["n_qubits"], (
            f"{n_feat} features cannot be encoded on {w['n_qubits']} wires")


#: The training artifact for a use case, whose directory is named for the data dir rather than the
#: use case: ``scripts/train/qcnn_variational.py --usecase medical`` writes ``models/qcnn_medical/``.
def _trained_weights(usecase: str) -> Path:
    return REPO / "models" / f"qcnn_{_data_dir(usecase)}" / "weights.json"


@pytest.mark.parametrize("usecase", sorted(_on_disk()))
def test_the_shipped_copy_agrees_with_the_trained_artifact(usecase: str) -> None:
    """The browser's copy of the weights must not drift from the model that was trained.

    WHY. ``qcnn_variational.py`` writes only ``models/qcnn_<uc>/weights.json``. **Nothing copies it
    to** ``web/public/data/<dir>/qcnn_weights.json`` - not the Makefile, not a script, not a
    documented step. The shipped copies were made by hand as trimmed subsets, dropping
    ``feature_names``, ``n_test``, ``n_train`` and ``usecase``.

    So a retrain updates one file and leaves the other, and the browser goes on serving the previous
    circuit's angles while every local check passes and the metrics table shows the NEW numbers - the
    displayed accuracy would belong to weights the page is not running. This project has already shipped
    that shape once, when a retrain silently unpublished two models because the pinned revision no
    longer matched the metrics beside it.

    Asserts agreement on the SHARED keys only, because the subsetting is deliberate. A key present in
    both and different in each is the drift; a key present in one is the trim.
    """
    import json

    trained_path = _trained_weights(usecase)
    shipped_path = PUBLIC / _data_dir(usecase) / "qcnn_weights.json"
    assert trained_path.exists(), (
        f"{shipped_path.relative_to(REPO)} ships weights for {usecase} but "
        f"{trained_path.relative_to(REPO)} does not exist. The browser is serving a circuit with no "
        "training artifact behind it, so nothing can reproduce or re-derive it."
    )
    trained = json.loads(trained_path.read_text())
    shipped = json.loads(shipped_path.read_text())

    missing = sorted(set(shipped) - set(trained))
    assert not missing, (
        f"the shipped copy has keys the trained artifact does not: {missing}. The shipped file is "
        "meant to be a SUBSET of the trained one; an extra key was invented by hand."
    )
    drifted = sorted(k for k in set(shipped) & set(trained) if shipped[k] != trained[k])
    assert not drifted, (
        f"{usecase}: these keys differ between the trained artifact and the copy the browser "
        f"serves: {drifted}.\n"
        f"  trained: {trained_path.relative_to(REPO)}\n"
        f"  shipped: {shipped_path.relative_to(REPO)}\n"
        "Nothing copies one to the other, so a retrain updates only the first. Re-copy the shared "
        "keys. If a difference is deliberate, it needs a reason here rather than a silent divergence "
        "between the numbers shown and the circuit run."
    )
