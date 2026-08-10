"""A published baseline must never be rendered beside a number measured a different way.

`ModelPanel` shows `spec.metric.baseline` next to whichever recorded key matches
`spec.metric.name`. That is correct when the two describe the same quantity and silently wrong when
they do not - the reader sees "achieved X, baseline Y" for an X and a Y that are different metrics.

Two distilled students declare their TEACHER's metric while recording their own:

    medical_fl.student              declares auc     records metric_name = accuracy
    satellite_comms.opssat_student  declares auc_pr  records metric_name = minority_class_f1

When this was written neither rendered a false comparison, because the declared name did not appear
among their recorded numeric keys, so nothing matched and no baseline was drawn. That was luck
rather than design, and this docstring said so: "adding an `auc` key to the medical student's
metrics would immediately put a BreastMNIST AUC baseline of 0.901 beside a number that is not
an AUC."

ROUND 24 resolved it the other way, and this test now pins that. Both trainers were changed to
MEASURE the declared metric and record it under its own name: the medical student's scores were
being thrown away by an `.argmax(1)` before AUC could be computed, and the OPS-SAT student is a
logistic model whose `predict_proba` was simply never called. `auc` and `auc_pr` are therefore now
present AND genuine, so the panel's comparison is honest rather than merely absent.

The silence had been expensive. Recording `accuracy` where the registry declared `auc` is what hid
the medical student being a CONSTANT PREDICTOR - it emitted one class for all 156 test images,
scoring accuracy 0.7308, which is exactly the majority-class rate, while its AUC was 0.354, below
random. The missing metric was not an untidy label; it was the reason nobody could see that the
model had learned nothing.

So the invariant is no longer "the declared key must be absent". It is: whatever is rendered beside
a baseline must BE that metric, and a record that measures two different things must say which is
which. A record carrying both therefore has to carry a `metric_note` distinguishing them.
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
    if declared not in _numeric_keys(record):
        return          # nothing matches the declared name, so no baseline is drawn at all

    # The declared metric IS recorded alongside a differently-named fidelity metric. That is the
    # good case - the panel compares like with like - but only if the record distinguishes them,
    # because two numbers in one record with no statement of which is which is how the medical
    # student shipped an accuracy under an AUC's baseline for several rounds.
    assert record.get("metric_note"), (
        f"{spec.key} records both '{declared}' (the declared metric, baseline {spec.metric.baseline}) "
        f"and a fidelity metric named '{recorded_name}', but has no `metric_note` saying which is "
        f"which. ModelPanel renders the baseline beside '{declared}', so that key must genuinely be "
        f"a '{declared}' and the record must say so."
    )
    for name in (declared, recorded_name):
        assert name in record["metric_note"], (
            f"{spec.key}'s metric_note does not mention '{name}'. It carries two different "
            f"quantities and the note has to name both, or it does not do its job."
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
