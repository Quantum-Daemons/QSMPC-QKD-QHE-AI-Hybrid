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

import json
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


# --------------------------------------------------------------------------------------------- #
#  Every shipped QCNN card says where its seed sits, or says no spread exists
# --------------------------------------------------------------------------------------------- #

#: Checklist row `QC20`, rewritten in Round 42 because the previous wording could not be satisfied by
#: anything. It asked for "the five-seed spread reported beside any shipped QCNN figure", and the
#: spread does not exist for two of the four shipped circuits (one run is 6 layers x 400 epochs) and
#: is rendered on no page in the browser at all. A row nothing can pass is not a standard; it is a
#: permanent FAIL, and a column of permanent FAILs is a column readers learn to skip.
#:
#: What replaced it is the property the spread was FOR: a reader must be able to tell whether the
#: number in front of them is typical. Two ways to satisfy that, and both are honest - report the
#: seed's position in a measured spread, or state that no spread was measured. What is forbidden is
#: the third option, which is what a card silently does by default: print a figure and say nothing,
#: so a single draw reads as a settled result.
#:
#: The medical card is why this matters and not a formality. Its seed 0 is the WORST of five on
#: minority-class F1 - 0.622951 against a median of 0.8710 - so a reader comparing it against the
#: classical baseline was comparing against the low end of a wide spread and had no way to know.
#:
#: A MARKER, NOT A PROSE MATCH - AND THE PROSE VERSION FAILED ITS OWN NEGATIVE CONTROL. The first
#: version searched for the phrasings the cards happened to use: "worst of five", "lands exactly on
#: its median", "No five-seed spread is published". Deleting the disclosure from `qcnn_telecom_q7`
#: and re-running produced GREEN, because that card cross-references the MEDICAL card's spread - "its
#: seed 0 turned out to be the worst of five" - and a regex cannot tell a card's statement about
#: itself from its statement about a sibling. A card could have satisfied a check about its own
#: numbers by mentioning somebody else's.
#:
#: That is this project's tautology class in another costume - a matcher satisfied by prose ABOUT the
#: subject rather than by the subject - and the answer has been the same every time it appeared: an
#: explicit marker nobody writes by accident. `BLOCKED UNTIL`, `DEADLINE EXECUTED`, and now
#: `**Seed position:**` at the start of a line. Cross-references stay in the prose where they are
#: useful, and the line that discharges the row is the line that speaks about itself.
_SEED_POSITION = re.compile(r"^\*\*Seed position:\*\*\s+\S", re.M)


def test_every_qcnn_artifact_has_a_card() -> None:
    """The set is the ARTIFACTS, not the cards that happen to exist.

    Found by training two new widths in Round 43. `test_every_shipped_qcnn_card_places_its_seed`
    globs `qcnn_*/README.md`, so `models/qcnn_finance_q9/weights.json` with **no card at all** went
    straight through it - the check enumerated its own subject from the thing it was checking, and an
    artifact that documents nothing is the strongest possible way to satisfy "every card documents
    its seed".

    Enumerating from `weights.json` fixes the direction: an artifact must produce a card, not a card
    must produce a card.
    """
    artifacts = sorted((REPO / "models").glob("qcnn_*/weights.json"))
    assert len(artifacts) >= 4, (
        f"only {len(artifacts)} QCNN artifacts found; the glob has stopped finding them"
    )
    undocumented = [
        str(a.parent.relative_to(REPO)) for a in artifacts if not (a.parent / "README.md").exists()
    ]
    assert not undocumented, (
        f"these QCNN artifacts ship weights with no model card: {undocumented}.\n"
        "A card is where the seed position, the floors and the classical comparator live. Without "
        "one the numbers reach the browser and nothing states what they mean."
    )


def test_every_shipped_qcnn_card_places_its_seed() -> None:
    """`QC20`. Enumerated from disk, because a hand-listed set is a second place to forget one."""
    cards = sorted((REPO / "models").glob("qcnn_*/README.md"))
    assert len(cards) >= 4, (
        f"only {len(cards)} QCNN model cards found; this check enumerates the shipped circuits from "
        "disk and an empty or shrunken glob passes for the same reason an empty check passes"
    )
    silent = [
        str(c.relative_to(REPO)) for c in cards if not _SEED_POSITION.search(c.read_text())
    ]
    assert not silent, (
        "these QCNN model cards print figures without saying whether the seed behind them is "
        f"typical: {silent}.\n"
        "State the seed's position in the measured spread, or state that no spread was measured for "
        "that circuit. Do NOT re-pick the seed to close the gap - selecting on test performance is "
        "test-set selection, refused in references/qcnn-entanglement-ablation.md - and do NOT close "
        "it by deleting the disclosure, which is the only move that makes the numbers look better "
        "while making them mean less."
    )


# --------------------------------------------------------------------------------------------- #
#  The offered widths are the widths that ship - checked in both directions, like the set above
# --------------------------------------------------------------------------------------------- #

def _declared_widths() -> dict[str, list[int]]:
    """`QCNN_WIDTHS` as the browser declares it."""
    text = SRC.read_text()
    m = re.search(r"export const QCNN_WIDTHS: Record<string, number\[\]> = \{(.*?)\};", text, re.S)
    assert m, f"QCNN_WIDTHS not found in {SRC.relative_to(REPO)} - did the declaration move?"
    return {
        k: [int(n) for n in re.findall(r"\d+", v)]
        for k, v in re.findall(r"(\w+):\s*\[([^\]]*)\]", m.group(1))
    }


def _widths_on_disk() -> dict[str, list[int]]:
    """Widths with a shipped artifact, read from `web/public/data/<dir>/qcnn_weights*.json`.

    The unsuffixed file is the default width; `_q<n>` names its own. Both are read from the FILE's
    `n_qubits` rather than from its name, so a mislabelled file is a failure and not a silent pass.
    """
    known = ["finance_aml", "medical_fl", "genomics", "satellite_comms", "telecom_ids"]
    out: dict[str, list[int]] = {}
    for uc in known:
        found = []
        for p in sorted((PUBLIC / _data_dir(uc)).glob("qcnn_weights*.json")):
            found.append(int(json.loads(p.read_text())["n_qubits"]))
        if found:
            out[uc] = sorted(found)
    return out


def test_offered_qcnn_widths_are_exactly_the_widths_that_ship() -> None:
    """Both directions, because a one-way check rots the other way.

    The control offered 2..14 on every page with trained weights while two artifacts existed for
    telecom and one each for finance and medical. Every other value silently loaded the 8-wire file,
    and the run then reported `qcnn_qubits` as the width that actually ran - so the control and the
    metric disagreed with nothing on the page to say which was which.

    Declaring a width that does not ship recreates that. Shipping one that is not declared makes it
    unreachable, which is the same waste as the artifact nobody could select.
    """
    declared, disk = _declared_widths(), _widths_on_disk()
    assert declared, "no widths declared; the parse has stopped working"
    assert set(declared) == set(disk), (
        f"the use cases with declared widths {sorted(declared)} differ from those with shipped "
        f"artifacts {sorted(disk)}"
    )
    wrong = {
        uc: {"declared": sorted(declared[uc]), "on disk": disk[uc]}
        for uc in declared
        if sorted(declared[uc]) != disk[uc]
    }
    assert not wrong, (
        f"declared QCNN widths do not match the artifacts on disk: {wrong}.\n"
        "A width with no artifact loads a different circuit in silence; an artifact with no declared "
        "width cannot be selected at all."
    )


def test_the_width_floor_is_the_feature_count() -> None:
    """No shipped width may be narrower than the features it must encode.

    This is why the sets differ per use case and not by preference: a wire carries a feature or is
    padded, so the floor is the feature count. `qcnn_variational.py` refuses a narrower run outright
    ("8 features exceeds 7 wires"), which is how Round 43 learned that finance and medical cannot
    have a 7-wire twin the way telecom does - the plan had assumed they could.
    """
    thin = []
    for uc in _declared_widths():
        for p in sorted((PUBLIC / _data_dir(uc)).glob("qcnn_weights*.json")):
            w = json.loads(p.read_text())
            if int(w["n_qubits"]) < int(w["n_features"]):
                thin.append(f"{p.relative_to(REPO)}: {w['n_qubits']} wires for {w['n_features']} features")
    assert not thin, "these shipped circuits have fewer wires than features:\n  " + "\n  ".join(thin)
