"""Two fields are called `baseline` and they are different quantities.

WHY THIS EXISTS

`MetricSpec.baseline` in the registry is PRIOR PUBLISHED RESEARCH - finance 0.6323 from GFP+XGBoost,
medical 0.901 from the BreastMNIST leaderboard, satellite 0.968 from ESA-ADB.

`ModelEvalResult.baseline` in `web/src/lib/onnxInference.ts` is the TRIVIAL PREDICTOR - majority-class
accuracy, all-positive F1, persistence MAE, or `2/(n+1)/n` chance.

They share a name, they are both numbers between 0 and 1, and they sit one import apart. Round 54
wired the published figure onto the verification chart as a reference line, and drawing the wrong one
would have rendered a published 0.901 as a chance line - a plot that looks entirely reasonable and
says the opposite of the truth.

That is this project's signature defect in a new place: not a fix reaching one language, but one WORD
covering two quantities, which is how it reached several languages in the first place.

WHAT IS ASSERTED

That the two are numerically distinguishable on the shipped configuration, so a swap changes what a
reader sees rather than passing unnoticed; and that the chart takes the registry field by name.

WHAT IS DELIBERATELY NOT ASSERTED

Not that either value is correct - the registry decides one and the evaluator computes the other.
Only that they are not interchangeable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHART = REPO / "web" / "src" / "components" / "VerificationChart.tsx"


def _config() -> dict:
    return json.loads((REPO / "web" / "src" / "lib" / "config.defaults.json").read_text())


def test_the_two_baselines_are_numerically_distinguishable():
    """A swap must be visible. If they coincided, the gate below could not detect a mix-up."""
    cfg = _config()
    published = []
    for specs in cfg.get("models", {}).values():
        for s in specs:
            b = (s.get("metric") or {}).get("baseline")
            if isinstance(b, (int, float)):
                published.append((s["key"], b))
    assert len(published) >= 4, (
        f"only {len(published)} models carry a published baseline; this gate has too little subject "
        "to be meaningful")
    # The trivial-predictor figures the evaluator computes live in the achieved record under names
    # that say what they are. None of them should equal the published figure for the same model.
    TRIVIAL = ("majority_baseline", "all_positive_f1", "majority_class_rate", "persistence_mae")
    collisions = []
    for specs in cfg.get("models", {}).values():
        for s in specs:
            b = (s.get("metric") or {}).get("baseline")
            if not isinstance(b, (int, float)):
                continue
            a = s.get("achieved") or {}
            for t in TRIVIAL:
                v = a.get(t)
                if isinstance(v, (int, float)) and abs(v - b) < 1e-9:
                    collisions.append(f"{s['key']}: published {b} == {t} {v}")
    assert not collisions, (
        "these models have a published baseline numerically equal to their trivial predictor:\n  "
        + "\n  ".join(collisions)
        + "\n\nThat is not necessarily wrong, but it means a mix-up between the two would be "
        "invisible on this model, so the distinction has to be checked another way here."
    )


def test_the_chart_reads_the_registry_field_not_the_evaluator_field():
    """Asserted on the source, because the rendered output cannot carry which field it came from.

    A chart drawing `res.baseline` and a chart drawing `spec.metric.baseline` produce the same SHAPE
    of plot with a different line in it. Nothing downstream distinguishes them, so the distinction
    has to be made where the value is chosen.
    """
    src = CHART.read_text()
    assert "ReferenceLine" in src, "the chart no longer draws reference lines; re-derive this test"

    # The published line must come from the registry lookup.
    assert re.search(r"spec\?\.metric\?\.baseline", src), (
        "VerificationChart no longer reads `spec.metric.baseline`. If the published reference line "
        "now comes from somewhere else, check it is not `res.baseline` - that field is the TRIVIAL "
        "predictor, and drawing it would render a chance line labelled as prior research.")

    # And it must not be drawn from the evaluator's own baseline field.
    bad = re.search(r"ReferenceLine[^/]{0,200}?y=\{\s*res\.baseline\s*\}", src, re.S)
    assert not bad, (
        "a ReferenceLine is being drawn at `res.baseline`, which is the trivial predictor - "
        "majority-class accuracy, all-positive F1 or persistence MAE - not prior published research.")


def test_at_least_one_shipped_model_can_actually_render_the_comparison():
    """The gate Round 54 did not have, and its absence let a dead feature ship.

    Round 54 drew the published baseline as a reference line on the per-class-F1 chart and asserted
    that the value came from `spec.metric.baseline` rather than `res.baseline`. Both assertions were
    true and the line rendered NOWHERE: every model carrying a baseline plots as a PR curve, and every
    model plotting per-class F1 is a telecom multiclass model whose baseline is deliberately null. The
    intersection was empty and nothing checked it, because the gate tested the MECHANISM and not the
    OUTCOME.

    So this asserts reachability: at least one shipped model must have both a published baseline and a
    measured value for the SAME declared metric, or the comparison chart is unreachable and the
    feature is decoration.
    """
    cfg = _config()
    drawable = []
    for specs in cfg.get("models", {}).values():
        for spec in specs:
            metric = (spec.get("metric") or {})
            name, base = metric.get("name"), metric.get("baseline")
            measured = (spec.get("achieved") or {}).get(name)
            if isinstance(base, (int, float)) and isinstance(measured, (int, float)):
                drawable.append(f"{spec['key']}: {name} measured {measured} vs published {base}")
    assert drawable, (
        "no shipped model has BOTH a published baseline and a measured value under the same declared "
        "metric, so `MeasuredAgainstPublished` can never draw. A comparison nobody can see is not a "
        "comparison - check `metric.name` against the keys in `achieved` before adding the chart back."
    )
    assert len(drawable) >= 3, (
        f"only {len(drawable)} model(s) can render the comparison: {drawable}. That is thin enough "
        "that a single registry edit would make the feature unreachable again.")


def test_the_published_line_is_labelled_with_its_metric_name():
    """A bare number on an axis is not attributable, and these two look identical when bare."""
    src = CHART.read_text()
    assert "MeasuredAgainstPublished" in src, (
        "the measured-against-published chart is gone. If the comparison moved, check it still puts "
        "both quantities on ONE axis - a published aggregate drawn over a precision axis or over "
        "per-class F1 bars is the same category error as drawing the trivial predictor as prior "
        "research.")
    assert "<code>{metric}</code>" in src, (
        "the comparison chart no longer names its metric. Two bars labelled only 'this project' and "
        "'published' do not say what is being compared, and this project has twice shipped a plot "
        "where the two quantities were not the same.")
