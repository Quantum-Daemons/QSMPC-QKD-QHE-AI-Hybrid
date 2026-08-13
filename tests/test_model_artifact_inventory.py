"""Every artefact that influences a page is registered or declared, and every one has a card.

WHY THIS EXISTS

Two things were found by reading rather than by any check, in Round 37:

    ``models/metrics.json`` carried ``finance_aml.scorecard`` with no matching ``ModelSpec``. It is a
    real record - BFV-exact, with its own test - but nothing tied it to the registry, so a reader
    comparing the two lists would find a record with no owner and could not tell whether it was a
    leftover or a deliberate non-registry artefact.

    ``models/qcnn_telecom/`` had no README while ``qcnn_finance`` and ``qcnn_medical`` both did - and
    telecom is the most complicated of the three: the only multiclass circuit, the only one shipping
    a readout convention as data, and the only one the Python reimplementation cannot consume. Round
    28 wrote the other two cards precisely because "an artefact with real influence and no
    documentation" is a defect. The check that produced them was a person looking at two directories.

Both are inventory properties: a set on disk against a set in code. That is the shape this repository
already guards for QCNN use cases (``tests/test_qcnn_usecases.py``) and for federated scope
(``tests/test_federated_scope.py``); it simply had no guard for the model artefacts themselves.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from qsh.models.registry import _SPECS

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "models" / "metrics.json"

#: Records in ``metrics.json`` that deliberately have no ``ModelSpec``, each with the reason.
#:
#: An allow-list rather than a silent exclusion: adding a key here is a decision someone writes down,
#: and the test prints this docstring when an unexpected key appears, so the next person knows the
#: two options are "register it" or "declare it here with why".
NON_REGISTRY_METRICS: dict[str, str] = {
    "finance_aml.scorecard": (
        "A points-based scorecard evaluated under BFV, not CKKS, and never published to the Hub. It "
        "exists to demonstrate EXACT integer evaluation where CKKS is approximate - "
        "tests/test_scorecard_bfv_exact.py asserts `got == plain` for BFV and `approx != plain` for "
        "CKKS on the same model. It has no hf_repo because there is nothing to publish: the weights "
        "are a coordinate search over a bounded integer grid, reproducible from the recorded "
        "fit_method in less time than a download."
    ),
}


def test_every_metrics_record_is_registered_or_declared() -> None:
    metrics = json.loads(METRICS.read_text())
    registered = {spec.key for spec in _SPECS}
    orphans = sorted(set(metrics) - registered - set(NON_REGISTRY_METRICS))
    assert not orphans, (
        f"{orphans} appear in models/metrics.json with no ModelSpec and no entry in "
        "NON_REGISTRY_METRICS. A record with no owner cannot be read as either a leftover or a "
        "deliberate non-registry artefact. Register it, or declare it there with the reason."
    )


def test_the_declared_exceptions_still_exist() -> None:
    """An allow-list that outlives the thing it allows is how a stale exemption hides a real orphan."""
    metrics = json.loads(METRICS.read_text())
    stale = sorted(set(NON_REGISTRY_METRICS) - set(metrics))
    assert not stale, (
        f"{stale} are declared as non-registry metrics records but are no longer in metrics.json. "
        "Delete the declaration - an exemption for something absent will silently cover a future "
        "record that happens to reuse the key."
    )


def _qcnn_dirs() -> list[Path]:
    return sorted(p for p in (REPO / "models").glob("qcnn_*") if p.is_dir())


def test_every_shipped_qcnn_has_a_model_card() -> None:
    dirs = _qcnn_dirs()
    assert dirs, "no models/qcnn_* directories found - did they move?"
    missing = [p.name for p in dirs if not (p / "README.md").is_file()]
    assert not missing, (
        f"{missing} ship weights with no README.md. These weights decide the class a use-case page "
        "surfaces, and a card is the only place their measured performance, their classical baseline "
        "and their scope are written down. models/qcnn_telecom went several rounds without one."
    )


@pytest.mark.parametrize("directory", [p.name for p in _qcnn_dirs()])
def test_each_card_states_it_is_not_published(directory: str) -> None:
    """The claim that most needs to survive a copy-paste, since these sit beside published models."""
    card = (REPO / "models" / directory / "README.md").read_text()
    assert "NOT PUBLISHED TO HUGGING FACE" in card, (
        f"models/{directory}/README.md does not state that it is not published. These directories sit "
        "next to the tiered published models and carry the same weights.json filename; a card that "
        "omits it invites a reader to look for a Hub repo that does not exist."
    )


@pytest.mark.parametrize("directory", [p.name for p in _qcnn_dirs()])
def test_each_card_reports_the_classical_baseline(directory: str) -> None:
    """The number this project prints beside every quantum figure, including when it loses.

    `QC6` requires it on the page. It has to be on the card too: the card is what a reader finds
    first when the weights turn up in a repository listing, and every one of these circuits currently
    loses to logistic regression on its own split.
    """
    card = (REPO / "models" / directory / "README.md").read_text().lower()
    assert "logistic regression" in card, (
        f"models/{directory}/README.md does not name the classical baseline. Every one of these "
        "circuits is behind logistic regression on its own split, and a card that omits it reports a "
        "quantum number with nothing to read it against."
    )


# --------------------------------------------------------------------------------------------- #
#  Documentation hierarchy: an investigation nobody links to is an investigation nobody reads
# --------------------------------------------------------------------------------------------- #

#: The checklist is GITIGNORED. A link from it is not a link for anyone who clones this repository,
#: which is what "reachable" has to mean here.
UNPUBLISHED = {"docs/browser-verification-checklist.md"}


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.split() if f not in UNPUBLISHED]


def test_every_reference_investigation_is_linked_from_somewhere() -> None:
    """`references/*.md` are the project's dated fact-checks. An orphan is a decision that is lost.

    The recurring per-round request asks for a documentation-hierarchy sweep, and the checklist had
    **zero** numbered rows covering it - only prose. "Re-read everything" is not a pass condition.
    This is: every investigation must be reachable from a file that someone who clones this
    repository can open.

    THE FIRST VERSION OF THIS TEST WAS WRONG IN BOTH DIRECTIONS, and CI caught it where local runs
    did not. It searched `docs/*.md`, which includes the gitignored checklist, so a link that no repo
    reader can follow counted as a link - and it passed locally for that reason while failing in CI,
    where the checklist does not exist. It also searched only markdown and TeX, so
    `qcnn-inference-investigation.md` (linked from `quantumSim.ts` and the trainer) and
    `medical-student-collapse-investigation.md` (linked from two model cards) were reported as
    orphans when a reader can reach both.

    Searching every tracked file fixes both: a link from source, from a model card, from
    `metrics.json` or from `REFERENCES.md` all count, because all of them ship.
    """
    refs = sorted((REPO / "references").glob("*.md"))
    assert refs, "references/ has no markdown - did it move?"
    tracked = _tracked_files()
    orphans = []
    for ref in refs:
        rel = f"references/{ref.name}"
        if not any(
            ref.name in (REPO / f).read_text(errors="ignore")
            for f in tracked
            if f != rel and (REPO / f).is_file()
        ):
            orphans.append(ref.name)
    assert not orphans, (
        f"{orphans} are in references/ and no TRACKED file points at them. An investigation nobody "
        "can reach gets re-run: the question reappears, someone answers it again, and the dated "
        "answer already in the repository is never found. A link from the gitignored checklist does "
        "not count - it does not ship."
    )
