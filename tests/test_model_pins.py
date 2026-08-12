"""A published model's pin must survive everything except the one event that invalidates it.

WHY THIS TEST EXISTS

Round 20 retrained the finance gradient-boosting model and the OPS-SAT classifier so their
StandardScaler could be published alongside their weights. Both trainings ended with a call to
``record_metrics``, which wrote ``current[key] = payload`` -- a wholesale replace under a docstring
that said "Merge". ``hf_revision`` is written by ``scripts/publish_hf.py`` into that same entry, so
both models silently lost their pins.

Nothing failed. No test broke, the training logs looked perfect, and CI was green. The damage was
only visible three layers away, in a browser: ``browserModels()`` skips any spec without a revision,
so both models dropped out of the browser tier, and the finance use-case page went from one
verifiable model to none while still printing the metrics of a model a visitor could no longer run.
The Models panel said "published but not pinned to a commit" -- which is true, and reads exactly like
a model that was never pushed, so the message pointed at the wrong fix.

The correct behaviour is neither of the two obvious ones. Keeping the pin across a retrain is wrong:
the artefact at that sha did not produce the new metrics, so the browser would fetch weights that
disagree with the numbers printed beside them. Dropping it silently is wrong for the reason above.
The pin is therefore demoted to ``hf_revision_invalidated_by_retrain`` and surfaced as its own
distinct reason, so the state says which action fixes it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qsh.models import registry, train  # noqa: E402

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture
def metrics_file(tmp_path, monkeypatch):
    """Point both modules at a scratch metrics.json and clear the registry's cache around it."""
    path = tmp_path / "metrics.json"
    monkeypatch.setattr(train, "METRICS_PATH", path)
    monkeypatch.setattr(registry, "METRICS_PATH", path)
    registry._achieved.cache_clear()
    yield path
    registry._achieved.cache_clear()


def entry(path: Path, key: str = "m") -> dict:
    return json.loads(path.read_text())[key]


def test_publish_records_the_pin(metrics_file):
    train.record_metrics("m", {"accuracy": 0.9, "hf_revision": SHA_A})
    assert entry(metrics_file)["hf_revision"] == SHA_A


def test_retrain_demotes_the_pin_rather_than_keeping_it(metrics_file):
    """The exact Round 20 regression: a retrain must not leave the browser on stale weights."""
    train.record_metrics("m", {"accuracy": 0.9, "hf_revision": SHA_A})
    train.record_metrics("m", {"accuracy": 0.95})

    e = entry(metrics_file)
    assert "hf_revision" not in e, "a retrain kept a pin that no longer matches its metrics"
    assert e["hf_revision_invalidated_by_retrain"] == SHA_A, "the pin vanished without a trace"
    assert e["accuracy"] == 0.95


def test_a_metric_absent_from_a_fresh_run_cannot_read_as_fresh(metrics_file):
    """An un-remeasured metric survives, but LABELLED - the invariant changed in Round 34.

    This test used to assert `"auc_pr" not in entry(...)`: the entry was replaced, not merged, so a
    metric absent from a fresh run could not survive. That rule was deliberate and its reasoning was
    sound - a stale number must never read as a fresh measurement.

    What it did not anticipate is a run that legitimately measures a strict SUBSET. Round 33 re-ran
    `telecom_ids.py` without `--quantum` to add a CKKS student, and the rule deleted all eight
    `quantum_*` fields, including `quantum_beats_classical`, which `references/REFERENCES.md` cites
    by name as being recorded in `models/metrics.json`. Three published claims spent a round pointing
    at fields that no longer existed, and no gate could see it.

    So the invariant is now narrower and does the same job: the value survives, and it is marked with
    the sha of the run that actually measured it, so it cannot be read as this run's output. Deleting
    the value was never what protected the reader - being able to tell the two apart is.
    """
    train.record_metrics("m", {"accuracy": 0.9, "auc_pr": 0.7, "git_sha": SHA_A})
    train.record_metrics("m", {"accuracy": 0.95, "git_sha": SHA_B})

    e = entry(metrics_file)
    assert e["auc_pr"] == 0.7, "an un-remeasured metric must not be deleted"
    assert e["git_sha"] == SHA_B, "the record's own sha is this run's"
    carried = e["carried_forward_not_remeasured"]
    assert carried["fields"] == ["auc_pr"]
    assert carried["measured_at_git_sha"] == SHA_A, (
        "without this a reader would attribute auc_pr to the run named in the record's git_sha, "
        "which is the confusion the old delete-it rule was preventing")


def test_republishing_clears_the_demotion(metrics_file):
    train.record_metrics("m", {"accuracy": 0.9, "hf_revision": SHA_A})
    train.record_metrics("m", {"accuracy": 0.95})
    train.record_metrics("m", {"accuracy": 0.95, "hf_revision": SHA_B})

    e = entry(metrics_file)
    assert e["hf_revision"] == SHA_B
    assert "hf_revision_invalidated_by_retrain" not in e, "a republished model still reads as stale"


def test_a_model_that_was_never_published_gains_no_marker(metrics_file):
    """Absence of a pin is not the same state as a retired one, and must not be dressed up as one."""
    train.record_metrics("m", {"accuracy": 0.9})
    assert "hf_revision_invalidated_by_retrain" not in entry(metrics_file)


def test_registry_surfaces_the_retired_pin_distinctly(metrics_file, monkeypatch):
    """The UI branches on this: 'needs republishing' and 'never pinned' need different actions."""
    spec = replace(registry._SPECS[0], key="m", hf_repo="scratch-repo", hf_revision=None)
    monkeypatch.setattr(registry, "_SPECS", (spec,))

    train.record_metrics("m", {"accuracy": 0.9, "hf_revision": SHA_A})
    registry._achieved.cache_clear()
    assert registry.get("m").hf_revision == SHA_A
    assert registry.get("m").hf_revision_stale is None

    train.record_metrics("m", {"accuracy": 0.95})
    registry._achieved.cache_clear()
    retrained = registry.get("m")
    assert retrained.hf_revision is None, "the browser would still fetch the superseded artefact"
    assert retrained.hf_revision_stale == SHA_A, "the reason the UI prints would be the wrong one"
