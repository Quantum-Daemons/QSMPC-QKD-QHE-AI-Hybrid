"""Every explanatory note the registry exports to the browser must be rendered by something.

WHY THIS EXISTS

`metric_note` was written by the trainers, exported into `config.defaults.json`, shipped in the
bundle, and displayed on no page. Round 50 found it and rendered it, and `ModelPanel` still carries
the comment describing that.

That fix reached ONE field. Two siblings were in exactly the same state five rounds later:

    sample_size_note              satellite_comms.telemanom, telecom_ids.nidd
    classical_rf_attribution_note telecom_ids.qml

which is this project's most-repeated defect - a fix reaching one of several twins - at field
granularity rather than at file or language granularity.

`sample_size_note` is the load-bearing one. On `satellite_comms.telemanom` it is the sentence saying
the measured 0.4545 "is NOT comparable to a published benchmark computed over the full event set",
and while it rendered nowhere the use-case page drew that number in a bar chart beside the published
0.968 under a caption asserting the two were the same measurement.

WHAT IS ASSERTED

That every `*_note` string a model carries in the exported registry appears in at least one web
source file. Enumerated FROM THE CONFIG, not from a list written here - a hard-coded list of field
names would need updating by the same person who forgot to render the field, so it would go stale in
exactly the case it exists to catch.

WHAT IS DELIBERATELY NOT ASSERTED

Not that the note renders on any particular page, and not that a visitor can reach it - `RS`/`RG` rows
in the browser checklist do that, on the deployed site. A source-level reference is the weakest claim
that still distinguishes "wired to a component" from "exported and forgotten", which was the defect.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "web" / "src" / "lib" / "config.defaults.json"
WEB_SRC = REPO / "web" / "src"

#: Fields that are data for a chart rather than prose for a reader. `baseline_source` and
#: `comparison_caveat` live on `metric`, not on `achieved`, so they are outside the scan anyway; this
#: set exists so that a future numeric field ending in `_note` can be excused explicitly rather than
#: by weakening the scan.
NOT_PROSE: set[str] = set()


def _note_fields() -> dict[str, list[str]]:
    cfg = json.loads(CONFIG.read_text())
    found: dict[str, list[str]] = {}
    for specs in cfg.get("models", {}).values():
        for spec in specs:
            for key, value in (spec.get("achieved") or {}).items():
                if key.endswith("_note") and isinstance(value, str) and key not in NOT_PROSE:
                    found.setdefault(key, []).append(spec["key"])
    return found


#: `//` to end of line, and `/* ... */` including the `{/* ... */}` form JSX uses. Order matters:
#: block comments first, because a `//` inside one would otherwise truncate the block's opener.
_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_LINE = re.compile(r"//[^\n]*")


def _web_sources() -> str:
    """Source with COMMENTS STRIPPED, and that is the whole subtlety of this gate.

    The first version matched the raw text. It passed a seeded revert - because the comment in
    `ModelPanel` explaining why `sample_size_note` must be rendered NAMES the field, so removing every
    render site left the identifier in the file and the gate green.

    That is the same failure the QRNG health gate had, where `"raw=raw" in src` matched the sentence
    above the call rather than the call. A gate that its own documentation can satisfy tests nothing,
    and prose about a field is what an unrendered field tends to accumulate.
    """
    parts = []
    for path in sorted(WEB_SRC.rglob("*")):
        if path.suffix in (".ts", ".tsx") and path.is_file():
            parts.append(_LINE.sub(" ", _BLOCK.sub(" ", path.read_text())))
    return "\n".join(parts)


def test_the_scan_has_something_to_scan():
    """Anti-tautology. An empty enumeration would make the gate below vacuously green."""
    fields = _note_fields()
    assert len(fields) >= 3, (
        f"only {len(fields)} note field(s) found in the exported registry: {sorted(fields)}. This "
        "gate enumerates from the config on purpose; if the fields moved, re-derive it rather than "
        "letting it pass on an empty set.")


def test_every_exported_note_field_is_read_by_some_component():
    """The property: exported prose has a render site. Enumerated from the config, not from a list."""
    sources = _web_sources()
    unrendered = []
    for field, models in sorted(_note_fields().items()):
        # Word-boundary match so `metric_note` cannot be satisfied by `metric_notes` or a substring
        # of a longer identifier.
        if not re.search(rf"\b{re.escape(field)}\b", sources):
            unrendered.append(f"{field} (carried by {', '.join(sorted(models))})")
    assert not unrendered, (
        "these registry note fields are exported to the browser and read by no web source:\n  "
        + "\n  ".join(unrendered)
        + "\n\nThey ship in the bundle and are shown to nobody, which is the state `metric_note` was "
        "in before Round 50 and `sample_size_note` was in for five rounds afterwards. Render it in "
        "`ModelPanel`, or - if it is genuinely not for a reader - add it to NOT_PROSE with the "
        "reason, so the silence is a decision."
    )


def test_the_satellite_comparison_is_like_for_like_and_its_replication_reaches_the_page():
    """The specific case the general gate above would not have caught on its own, as it stands since Round 86.

    Until Round 86 `MeasuredAgainstPublished` drew the satellite model's event-wise F0.5 beside ESA-ADB's 0.968
    while the two were not one measurement: three scored events here against ESA-ADB's 29, another split, and a
    time-level TNR factor applied there only. The registry's `comparison_caveat` carried that to the chart. Since
    Round 86 the model is trained and scored by ESA-ADB's own recipe and rules over the same 29 anomalies, so the
    caveat is withdrawn and the chart's like-for-like caption is true. The gap between the bars is the two choices
    the baseline source names, and the record carries the arm that applies them and reproduces the figure, which
    the model panel renders.
    """
    cfg = json.loads(CONFIG.read_text())
    spec = next((s for specs in cfg.get("models", {}).values() for s in specs
                 if s["key"] == "satellite_comms.telemanom"), None)
    assert spec is not None, "satellite_comms.telemanom is gone; re-derive this test"

    metric, achieved = spec.get("metric") or {}, spec.get("achieved") or {}
    assert not metric.get("comparison_caveat"), "the satellite pair is flagged as not like-for-like again"
    assert metric.get("name") == "corrected_event_wise_f0_5", "the headline is no longer ESA-ADB's own metric"
    assert achieved.get("n_events_scored") == 29, "the scored events are no longer ESA-ADB's 29 anomalies"
    replication = (((achieved.get("esa_scores") or {}).get("restored_annotated_samples") or {})
                   .get("paper_floor") or {}).get("corrected_event_wise_f0_5")
    assert isinstance(replication, float), "the record lost the replication arm the page reads"
    assert "two choices" in metric.get("baseline_source", "").lower(), "the baseline source no longer names the gap"
    panel = (WEB_SRC / "components" / "ModelPanel.tsx").read_text()
    assert "function EsaScoresRows" in panel and "<EsaScoresRows spec={spec} />" in panel, (
        "the model panel no longer renders the arms table")

    chart = (WEB_SRC / "components" / "VerificationChart.tsx").read_text()
    assert "caveat" in chart, "MeasuredAgainstPublished no longer takes a caveat; re-derive"
    assert re.search(r"caveat\s*\?", chart), (
        "the chart no longer branches on the caveat, so its like-for-like caption is unconditional "
        "again - which is the assertion that was false for this model.")

    page = (WEB_SRC / "pages" / "UseCasePage.tsx").read_text()
    assert "comparison_caveat" in page, (
        "the use-case page no longer passes `comparison_caveat` to the chart. The field would be in "
        "the config, the chart would be able to render it, and the one model that needs it would "
        "still show the unqualified caption - a fix present everywhere except the wiring.")
