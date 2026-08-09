"""A published baseline must never be rendered beside a number measured a different way.

`ModelPanel` shows `spec.metric.baseline` next to whichever recorded key matches
`spec.metric.name`. That is correct when the two describe the same quantity and silently wrong when
they do not - the reader sees "achieved X, baseline Y" for an X and a Y that are different metrics.

Two distilled students declare their TEACHER's metric while recording their own:

    medical_fl.student              declares auc     records metric_name = accuracy
    satellite_comms.opssat_student  declares auc_pr  records metric_name = minority_class_f1

Today neither renders a false comparison, because the declared name does not appear among their
recorded numeric keys, so nothing matches and no baseline is drawn. That is luck rather than design:
adding an `auc` key to the medical student's metrics would immediately put a BreastMNIST AUC baseline
of 0.901 beside a number that is not an AUC.

These tests pin the invariant rather than the accident.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from qsh.models.registry import _SPECS

METRICS = json.loads((pathlib.Path(__file__).resolve().parents[1]
                      / "models" / "metrics.json").read_text())


def _numeric_keys(record: dict) -> set[str]:
    """Numeric metric keys, excluding booleans - `bool` is a subclass of `int` in Python, so a
    naive isinstance check treats every False flag as a metric."""
    return {k for k, v in record.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


@pytest.mark.parametrize("spec", [s for s in _SPECS if s.key in METRICS], ids=lambda s: s.key)
def test_baseline_is_never_shown_against_a_different_metric(spec):
    """If a spec declares metric A while the record says it measured B, then A must not appear as a
    recorded number - otherwise the panel prints A's baseline beside a value that is not an A."""
    record = METRICS[spec.key]
    recorded_name = record.get("metric_name")
    declared = spec.metric.name if spec.metric else None
    if not recorded_name or not declared or recorded_name == declared:
        return                                   # no mismatch: the comparison is honest
    assert declared not in _numeric_keys(record), (
        f"{spec.key} declares metric '{declared}' (baseline {spec.metric.baseline}) but records "
        f"metric_name '{recorded_name}', AND has a numeric '{declared}' key. ModelPanel will print "
        f"the '{declared}' baseline beside it, comparing two different quantities. Either drop that "
        f"key, or give the spec a metric that matches what was measured."
    )


@pytest.mark.parametrize("spec", [s for s in _SPECS if s.key in METRICS], ids=lambda s: s.key)
def test_a_declared_baseline_has_a_source(spec):
    """A baseline without a citation is a number this project invented. Round 22 retired a whole
    metric for that, so the rule is worth enforcing on the ones that remain."""
    if spec.metric is None or spec.metric.baseline is None:
        return
    assert spec.metric.baseline_source, (
        f"{spec.key} declares baseline {spec.metric.baseline} with no baseline_source"
    )
