"""A partial re-run must not delete measurements it did not take.

Round 33 re-ran `scripts/train/telecom_ids.py` WITHOUT `--quantum` in order to add a CKKS student.
`record_metrics` replaced the record outright, so all eight `quantum_*` fields disappeared from
`models/metrics.json` - including `quantum_beats_classical`, which `references/REFERENCES.md` cites
by name as being recorded there. Three published claims spent a round pointing at fields that no
longer existed.

Nothing caught it. The config-parity gate compares two DERIVED artifacts and both stayed consistent
with each other; the prose gate compares prose literals and no prose changed. What was lost was
numeric, machine-written and cited, and that combination had no gate at all.

These tests pin the two halves of the fix, because the fix has a tension in it and either half alone
is wrong. Merging without marking would let a stale number read as a fresh measurement - which is
exactly what the original replace-don't-merge docstring was defending against, and that argument was
right as far as it went. Marking without merging would not have prevented the deletion at all.
"""

from __future__ import annotations

import json

import pytest

from qsh.models import train


@pytest.fixture()
def metrics_file(tmp_path, monkeypatch):
    """Point `record_metrics` at a scratch file. It writes a real path by module constant."""
    path = tmp_path / "metrics.json"
    monkeypatch.setattr(train, "METRICS_PATH", path)
    return path


def _read(path):
    return json.loads(path.read_text())


def test_a_partial_rerun_keeps_the_measurements_it_did_not_take(metrics_file):
    """The exact Round 33 sequence: a full run, then a run that skips the optional stage."""
    train.record_metrics("telecom_ids.qml", {
        "classical_rf_macro_f1": 0.986629,
        "quantum_beats_classical": False,
        "quantum_svc_macro_f1": 0.831611,
        "git_sha": "aaaaaaa",
        "wall_clock_s": 267.1,
    })
    # the same trainer, invoked without --quantum
    train.record_metrics("telecom_ids.qml", {
        "classical_rf_macro_f1": 0.986629,
        "git_sha": "bbbbbbb",
        "wall_clock_s": 4.2,
    })

    rec = _read(metrics_file)["telecom_ids.qml"]
    assert rec["quantum_beats_classical"] is False, (
        "the field REFERENCES.md cites by name was deleted by a partial re-run - the Round 33 defect")
    assert rec["quantum_svc_macro_f1"] == 0.831611


def test_carried_fields_are_labelled_so_they_cannot_read_as_fresh(metrics_file):
    """Merging alone would recreate the defect the replace-don't-merge rule was guarding against."""
    train.record_metrics("m", {"measured": 1.0, "optional": 2.0, "git_sha": "aaaaaaa"})
    train.record_metrics("m", {"measured": 1.5, "git_sha": "bbbbbbb"})

    rec = _read(metrics_file)["m"]
    carried = rec["carried_forward_not_remeasured"]
    assert carried["fields"] == ["optional"], "only the un-remeasured field is marked"
    assert carried["measured_at_git_sha"] == "aaaaaaa", (
        "a reader must be able to attribute the inherited value to the run that took it, not to "
        "this record's own git_sha")
    assert rec["measured"] == 1.5, "a field the new run DID measure is overwritten, not inherited"


def test_run_scoped_fields_are_never_inherited(metrics_file):
    """A wall clock or a device copied from another run is not stale, it is false."""
    train.record_metrics("m", {
        "measured": 1.0, "git_sha": "aaaaaaa", "wall_clock_s": 267.1, "seed": 42,
        "device": "cuda", "checkpoint": "best.pt", "torch": "2.13.0",
    })
    train.record_metrics("m", {"measured": 1.5, "git_sha": "bbbbbbb", "wall_clock_s": 4.2})

    rec = _read(metrics_file)["m"]
    assert rec["git_sha"] == "bbbbbbb" and rec["wall_clock_s"] == 4.2
    for field in ("seed", "device", "checkpoint", "torch"):
        assert field not in rec, (
            f"{field} describes the RUN, so inheriting it would attach another run's context to "
            f"these numbers")
    assert "carried_forward_not_remeasured" not in rec, (
        "nothing was carried, so no marker - an empty marker would read as a warning about nothing")


def test_a_full_rerun_clears_a_previous_carry_marker(metrics_file):
    """Once every field is measured again the record must stop advertising inherited values."""
    train.record_metrics("m", {"measured": 1.0, "optional": 2.0, "git_sha": "aaaaaaa"})
    train.record_metrics("m", {"measured": 1.5, "git_sha": "bbbbbbb"})
    assert "carried_forward_not_remeasured" in _read(metrics_file)["m"]

    train.record_metrics("m", {"measured": 2.0, "optional": 3.0, "git_sha": "ccccccc"})
    rec = _read(metrics_file)["m"]
    assert "carried_forward_not_remeasured" not in rec, "the marker is stale once nothing is inherited"
    assert rec["optional"] == 3.0


def test_hf_revision_is_still_demoted_rather_than_carried(metrics_file):
    """The pre-existing rule this change must not weaken: a retrain invalidates the published pin."""
    train.record_metrics("m", {"measured": 1.0, "hf_revision": "deadbee"})
    train.record_metrics("m", {"measured": 1.5})

    rec = _read(metrics_file)["m"]
    assert "hf_revision" not in rec, (
        "carrying the pin forward would point the browser at weights that did not produce these "
        "numbers")
    assert rec["hf_revision_invalidated_by_retrain"] == "deadbee"
