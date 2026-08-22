"""A published baseline may only be drawn on a chart whose axis is the metric that baseline measures.

WHY THIS EXISTS

The standing request asks for prior-research accuracy to appear on the private-inference plots. The
obvious implementation is a `ReferenceLine` at `MetricSpec.baseline`. Measured across every published
model, that would be WRONG ON EVERY ONE OF THEM and right only on the single model that has no
baseline to draw:

    medical_fl.resnet18       baseline is AUC 0.901        chart is per-class F1      no
    medical_fl.resnet50       baseline is AUC 0.857        chart is per-class F1      no
    finance_aml.gbm           baseline is min-class F1     chart is precision/recall  no
    satellite_comms.telemanom baseline is event-wise F0.5  chart is precision/recall  no
    satellite_comms.opssat    baseline is AUC-PR 0.979     chart is precision/recall  no
    telecom_ids.nidd          declared macro F1, chart IS per-class F1                YES - baseline None

An AUC of 0.901 drawn as a horizontal line on an F1 axis is a number in the right numeric range and
the wrong quantity, which is the most convincing way to be wrong. The project already has the
vocabulary for this: `test_metric_baseline_consistency.py` exists because a baseline was once
rendered beside a differently-named metric in a TABLE. This is the same defect one step along, in a
CHART, where the reader has no column header to check it against.

THE SECOND TRAP, and it is a name collision rather than a judgement call. There are two fields called
`baseline` and they are different quantities:

  * `MetricSpec.baseline` (registry, reaches `web/src/lib/config.ts`) is PRIOR RESEARCH - what a
    published paper achieved.
  * `ModelEvalResult.baseline` (`web/src/lib/onnxInference.ts`) is the TRIVIAL PREDICTOR - majority
    class, all-positive F1, persistence MAE, or chance.

`VerificationChart` receives the second. Passing the first into the same slot would render a
published 0.901 as a chance line, and nothing about the call site would look wrong.

WHAT IS ASSERTED

1. The two fields stay distinguishable: the chart component does not read a `metric.baseline`.
2. The declared-metric-to-chart-axis mapping is pinned. If a model's declared metric changes, or its
   recorded keys change which chart it gets, this fails - and whoever changed it has to decide again
   whether a baseline is drawable, rather than inheriting a decision made when the pairing was
   different.

WHAT IS DELIBERATELY NOT ASSERTED

Not that a baseline must be drawn. The honest state today is that none is drawable, and
`ModelPanel` already renders each baseline beside the metric row it belongs to, which is the place
where the metric is named and the comparison is checkable.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "web" / "src" / "lib" / "config.defaults.json"
CHART = REPO / "web" / "src" / "components" / "VerificationChart.tsx"

#: Which chart `evaluatePublishedModel` produces, decided by which `expected.*` key the sample has.
#: Mirrors the branch order in `onnxInference.ts`; the axis is what a y-value on that chart MEANS.
_AXES = {
    "per-class-f1": "f1",
    "forecast": "value",
    "pr-curve": "precision",
}

#: Metric families. A baseline is drawable only when its family matches the chart's axis family.
_FAMILY = {
    "macro_f1": "f1",
    "minority_class_f1": "f1",
    "event_wise_f0_5": "f1",
    "auc": "auc",
    "auc_pr": "auc",
    "accuracy": "accuracy",
    "mae": "value",
}


def _published() -> list[dict]:
    cfg = json.loads(CONFIG.read_text())
    return [s for specs in cfg.get("models", {}).values() for s in specs if s.get("hf_revision")]


def _chart_kind(achieved: dict) -> str:
    if achieved.get("macro_f1") is not None:
        return "per-class-f1"
    if achieved.get("mae") is not None or achieved.get("persistence_mae") is not None:
        return "forecast"
    return "pr-curve"


def test_the_scan_sees_the_published_models() -> None:
    """Non-vacuity: an empty set would make every assertion below pass."""
    pub = _published()
    assert len(pub) >= 8, f"only {len(pub)} published models resolved; the config scan is not working"
    assert any(s.get("metric", {}).get("baseline") is not None for s in pub), (
        "no published model carries a MetricSpec.baseline, so this gate has nothing to protect")


def test_the_chart_does_not_read_the_registry_baseline() -> None:
    """The name collision, asserted on the code rather than on intent.

    `VerificationChart` is handed `ModelEvalResult`, whose `baseline` is the trivial predictor. If it
    ever reads a `metric.baseline` it is reading prior research and drawing it as a chance line.
    """
    src = CHART.read_text()
    offenders = [m.group(0) for m in re.finditer(r"metric\s*\.\s*baseline|metric\.baseline_source", src)]
    assert not offenders, (
        f"{CHART.relative_to(REPO)} reads the registry's prior-research baseline: {offenders}. "
        "That field is a published result; `res.baseline` in this component is the TRIVIAL predictor "
        "(majority class / all-positive F1 / persistence MAE / chance). Drawing one as the other "
        "renders a paper's score as a chance line and nothing at the call site looks wrong.")


def test_no_baseline_is_drawable_on_its_own_chart_and_that_is_recorded() -> None:
    """The mapping, pinned. Red the moment a metric or a chart pairing changes.

    This is not asserting that the current state is desirable - it is asserting that the current
    state is what the code was written against, so a change forces the drawability question to be
    re-answered instead of inherited.
    """
    drawable, blocked = [], []
    for s in _published():
        metric = s.get("metric", {})
        name, base = metric.get("name"), metric.get("baseline")
        kind = _chart_kind(s.get("achieved") or {})
        axis = _AXES[kind]
        if base is None:
            continue
        (drawable if _FAMILY.get(name) == axis else blocked).append(
            f"{s['key']}: baseline is {name}={base} but its chart plots {axis} ({kind})")

    assert not drawable, (
        "a published baseline now shares an axis with its chart, so it IS drawable:\n  "
        + "\n  ".join(drawable)
        + "\n\nThat is a good state, not a failure - but it changes the answer this gate records. "
        "Draw it with `MetricSpec.baseline` explicitly (never `ModelEvalResult.baseline`), label the "
        "line with the metric name, and move the entry here.")
    assert len(blocked) >= 5, (
        f"only {len(blocked)} baseline/chart pairs are mismatched; this file was written when every "
        "one of them was, and a drop means the pairings moved:\n  " + "\n  ".join(blocked))
