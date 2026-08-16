"""A metric published from more than one place must be rounded the same way in all of them.

WHY THIS EXISTS

`skr_per_pulse` is emitted from four sites across two languages. Round 47 shipped it at five decimals
from Python and six from the browser, so one panel showed 0.00198 and 0.001977 for one quantity.
Round 48 fixed that - and reached three of the four sites, leaving
`src/qsh/orchestration/stages.py` at five while the satellite twin NINE LINES ABOVE IT in the same
emit block already used six. The commit line was "one panel, two roundings"; the fix itself was
spelled two ways.

So this is the third instance of one defect class in three rounds, and the second where the fix
carried the defect. What makes it recur is that nothing relates the emitting sites to each other:
each one is locally reasonable, and the disagreement only exists across files.

WHAT IS ASSERTED

For every metric NAME emitted from more than one site, all sites round it to the same number of
decimals. Sites are found by pattern across tracked Python and TypeScript, so a fifth emitter added
tomorrow is in scope the day it lands.

WHAT IS DELIBERATELY NOT ASSERTED

Not that any particular precision is correct - six against five is a presentation choice, and this
gate has no opinion about which. Only that the choice is made once.

Not that every metric must be rounded. A raw float is a decision too, and `ctx.metrics.update(...)`
deliberately stores unrounded values for downstream computation; only the DISPLAY emitters are
compared.

DECLARED EXCEPTIONS carry their reason inline, because an exception with no reason is how a gate
turns into a list of things people got tired of fixing.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Metric names whose emitters may legitimately disagree, each with the reason it is allowed.
DECLARED = {
    # 5 dp in `orchestration/stages.py`, 4 dp in `api/inference_service.py`. Narrowing the first to
    # match the second would LOSE a digit on a quantity whose own sampling noise exceeds the
    # difference; widening the second changes a published API payload for no gain. Recorded as a
    # decision rather than silently equalised.
    "qber": "5 dp in the orchestration path, 4 dp in the demo API; narrowing would lose a digit",
}

#: `bus.metric("name", round(expr, N)` / `"name": round(expr, N)` - the two Python shapes.
_PY = re.compile(r'["\'](?P<name>[a-z0-9_]+)["\']\s*(?:,|:)\s*round\((?P<expr>[^,]+),\s*(?P<dp>\d+)\)')

#: A name only qualifies for the cross-site comparison if at least one of its sites publishes it
#: through a METRIC CHANNEL - `bus.metric(...)` in Python, `metric(stage, ...)` in the browser - or
#: through the demo API payload.
#:
#: Without this the scan compared `mean` and `sd`, which are per-column keys inside two unrelated
#: EDA blocks (`build_federated_samples.py` and `genomics_prs.py`) rounded to 5 and 6 decimals. Those
#: are not one metric spelled two ways; they are two artifacts' private statistics that happen to
#: share an English word. Adding them to DECLARED would have been the wrong fix - it would have made
#: the exception list absorb a scoping error, which is how an exception list becomes a list of things
#: people got tired of fixing.
_CHANNEL = re.compile(r'bus\.metric\(|(?<![A-Za-z0-9_])metric\(\s*stage|def qkd_demo')

#: `metric(stage, "name", Math.round(expr * 1e6) / 1e6)` / `name: Math.round(expr * 1e6) / 1e6`.
_TS = re.compile(
    r'(?:["\'](?P<qname>[a-z0-9_]+)["\']\s*,|(?P<bname>[a-z0-9_]+)\s*:)\s*'
    r'Math\.round\([^)]*?\*\s*1e(?P<dp>\d+)\s*\)\s*/\s*1e(?P=dp)')


def _tracked(*globs: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--", *globs], cwd=REPO,
                         check=True, capture_output=True, text=True).stdout.split()
    return [REPO / f for f in out if not f.startswith("tests/")]


def _emitters() -> dict[str, set[tuple[str, int]]]:
    """metric name -> {(file:line, decimals)} across both languages."""
    found: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for path in _tracked("*.py"):
        text = path.read_text(errors="ignore")
        for m in _PY.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            found[m.group("name")].add(
                (f"{path.relative_to(REPO)}:{line}", int(m.group("dp"))))
    for path in _tracked("*.ts", "*.tsx"):
        text = path.read_text(errors="ignore")
        for m in _TS.finditer(text):
            name = m.group("qname") or m.group("bname")
            line = text.count("\n", 0, m.start()) + 1
            found[name].add((f"{path.relative_to(REPO)}:{line}", int(m.group("dp"))))
    return found


def _channel_names() -> set[str]:
    """Names published through a metric channel somewhere - the ones a panel can show."""
    names: set[str] = set()
    for path in _tracked("*.py", "*.ts", "*.tsx"):
        text = path.read_text(errors="ignore")
        for m in _CHANNEL.finditer(text):
            window = text[m.start():m.start() + 220]
            names.update(re.findall(r'["\']([a-z0-9_]+)["\']', window))
    return names


def test_the_scan_finds_the_metrics_it_is_about() -> None:
    """Non-vacuity: a pattern that matches nothing would make the comparison below pass."""
    found = _emitters()
    assert len(found) >= 8, (
        f"found rounded emissions for only {len(found)} metric names; the patterns are not matching "
        "the code they were written against")
    assert "skr_per_pulse" in found and len(found["skr_per_pulse"]) >= 3, (
        f"`skr_per_pulse` resolves to {sorted(found.get('skr_per_pulse', ()))}. It is emitted from "
        "at least three sites across two languages and is the metric this gate was written for; if "
        "the scan cannot see them, it cannot see a disagreement either.")
    unknown = set(DECLARED) - set(found)
    assert not unknown, f"declared exceptions naming metrics that no longer exist: {sorted(unknown)}"


def test_every_metric_is_rounded_the_same_way_everywhere() -> None:
    """The property. Red on the unmodified tree at `skr_per_pulse`: 5 dp in one of four sites."""
    split = []
    channels = _channel_names()
    for name, sites in sorted(_emitters().items()):
        precisions = {dp for _, dp in sites}
        if len(precisions) <= 1 or name in DECLARED or name not in channels:
            continue
        where = "\n      ".join(f"{loc}  ->  {dp} dp" for loc, dp in sorted(sites))
        split.append(f"{name}: {sorted(precisions)}\n      {where}")
    assert not split, (
        "these metric names are published with different precisions by different sites, so one "
        "panel can show two numbers for one quantity:\n  " + "\n  ".join(split)
        + "\n\nPick one precision and use it everywhere, or add the name to DECLARED with the "
        "reason the disagreement is intended. `skr_per_pulse` reached this state twice: once as a "
        "Python/browser split, and once when the fix for that split missed the fourth site."
    )
