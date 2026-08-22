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


def test_the_incomparable_baseline_carries_its_caveat_to_the_chart():
    """The specific case the general gate above would not have caught on its own.

    `sample_size_note` being unrendered was one half. The other half is that
    `MeasuredAgainstPublished` asserted "both bars are the same metric on the same axis" for a model
    whose registry says three times that they are not. Rendering the note in the model panel does not
    fix a caption on a different component, so the caveat is a separate field and this pins it to the
    chart that makes the claim.
    """
    cfg = json.loads(CONFIG.read_text())
    spec = next((s for specs in cfg.get("models", {}).values() for s in specs
                 if s["key"] == "satellite_comms.telemanom"), None)
    assert spec is not None, "satellite_comms.telemanom is gone; re-derive this test"

    caveat = (spec.get("metric") or {}).get("comparison_caveat") or ""
    assert caveat, (
        "satellite_comms.telemanom carries no `comparison_caveat`, so the use-case page draws its "
        "measured event-wise F0.5 beside ESA-ADB's 0.968 under a caption claiming the two bars are "
        "the same measurement. The registry's own `sample_size_note` says they are not: three scored "
        "events here against 29 in ESA-ADB's test partition, a different split, and a time-level TNR "
        "factor applied there and not here.")
    assert "three" in caveat.lower() or "3" in caveat, (
        "the caveat no longer states the event count, which is the number that makes the gap "
        "unreadable - at three events one false alarm is worth 0.26.")

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
