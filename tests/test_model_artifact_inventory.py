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

DOC_ROOTS = ["README.md", "DATA_LICENSES.md"]


def test_every_reference_investigation_is_linked_from_somewhere() -> None:
    """`references/*.md` are the project's dated fact-checks. An orphan is a decision that is lost.

    The recurring per-round request asks for a documentation-hierarchy sweep, and the checklist had
    **zero** numbered rows covering it - only prose. "Re-read everything" is not a pass condition.
    This is: every investigation must be reachable from a document someone opens.

    Measured when written: all 13 were linked. The check exists so the fourteenth cannot be written,
    merged, and then answered again from scratch two rounds later - which is exactly what happened to
    `model-choice-vs-2026-practice.md`, an answered question that kept being re-asked because nothing
    pointed at it from where the question arises.
    """
    refs = sorted((REPO / "references").glob("*.md"))
    assert refs, "references/ has no markdown - did it move?"
    haystack = ""
    for pattern in ("docs/*.md", "references/*.md", "report/*.tex"):
        for path in REPO.glob(pattern):
            haystack += path.read_text(errors="ignore")
    for root in DOC_ROOTS:
        p = REPO / root
        if p.is_file():
            haystack += p.read_text(errors="ignore")

    orphans = []
    for ref in refs:
        if ref.name in ("REFERENCES.md",):        # the index itself; nothing needs to link to it
            continue
        # Its own text does not count as a link to itself.
        others = haystack.replace(ref.read_text(errors="ignore"), "")
        if ref.name not in others:
            orphans.append(ref.name)
    assert not orphans, (
        f"{orphans} are in references/ and linked from nowhere. An investigation that no document "
        "points at gets re-run: the question reappears, someone answers it again, and the dated "
        "answer already in the repository is never found. Link it from the checklist section where "
        "the question arises, or from the document whose claim it supports."
    )
