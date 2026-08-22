"""`published_lightweight_f0_5` and `published_full_channel_f0_5` must come from ONE table.

WHY THIS EXISTS

The pair exists to show CHANNEL-SCOPE sensitivity: the same method, the same event set, scored on six
channels and then on all of them. That argument only works if the only difference is the scope.

ESA-ADB publishes two tables that both give event-wise F0.5 for Telemanom-ESA-Pruned on Mission1:

    Table 2               ALL events excluding communication gaps      0.786 lightweight / 0.061 full
    Supplementary Table 9 ANOMALIES ALONE, x time-level TNR            0.968 lightweight / 0.039 full

Round 24 moved `MetricSpec.baseline` from 0.786 to 0.968, correctly - this project scores anomalies
alone, so Supplementary Table 9 is the like-for-like comparator.

The trainer then read the lightweight half from `spec.metric.baseline` and kept the full-channel half
as a hardcoded `0.061`. So it paired **0.968 from Supplementary Table 9 with 0.061 from Table 2**, and
labelled the difference "the same method on all Mission1 channels". Two tables, two event sets, under
a label whose entire meaning is that only the channel scope changed.

THE SHAPE IS WORTH MORE THAN THE INSTANCE. Replacing a literal with a registry read looks like
removing duplication. Here it silently swapped which table one half of a matched pair came from, and
it did so at the moment the registry value was corrected - the fix to one number broke a different
number that had been reading it. It was latent when found: the shipped `models/metrics.json` still
holds the consistent Table 2 pair, because the trainer had not run since Round 24.

WHAT IS ASSERTED

That the two halves are a known same-table pair, and that the lightweight half is NOT sourced from
`MetricSpec.baseline` - which is the field that legitimately holds the other table's number.

WHAT IS DELIBERATELY NOT ASSERTED

Not that either table is the right comparator. The registry decides that and Round 24 already did.
Only that one pair does not straddle both.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "scripts" / "train" / "satellite_telemanom.py"
METRICS = REPO / "models" / "metrics.json"

#: The two published pairs, read off arXiv:2406.17826v2. A value from one row may only be paired with
#: the other value from the SAME row.
TABLE_2 = (0.786, 0.061)          # all events excluding communication gaps
SUPP_TABLE_9 = (0.968, 0.039)     # anomalies alone, event precision x time-level TNR


def _emitted_pair() -> tuple[object, object]:
    """The literal pair the trainer writes, parsed rather than executed."""
    tree = ast.parse(TRAINER.read_text())
    found: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value in (
                "published_lightweight_f0_5", "published_full_channel_f0_5"
            ):
                found[k.value] = v.value if isinstance(v, ast.Constant) else ast.unparse(v)
    return found.get("published_lightweight_f0_5"), found.get("published_full_channel_f0_5")


def test_the_trainer_writes_a_same_table_pair():
    """The property: both halves from one row of the table above."""
    light, full = _emitted_pair()
    assert light is not None and full is not None, (
        "the trainer no longer writes both halves of the channel-scope pair; re-derive this test")
    assert (light, full) in (TABLE_2, SUPP_TABLE_9), (
        f"the trainer writes lightweight={light!r} with full_channel={full!r}, which is not a "
        f"published pair. Table 2 is {TABLE_2} and Supplementary Table 9 is {SUPP_TABLE_9}. Mixing "
        "them compares two event sets under a label that says only the channel scope differs - "
        "which is the Round 24 category error, reintroduced.")


def test_the_lightweight_half_is_not_read_from_the_registry_baseline():
    """The mechanism, because the values could coincide and still be wrongly sourced.

    `MetricSpec.baseline` legitimately holds Supplementary Table 9's 0.968. Reading it into a key
    paired with a Table 2 literal is the defect, and it is invisible in the output the moment the two
    tables happen to agree - which they do not today, but a future correction to either could make
    them. Asserted on the source so the provenance is checked, not the coincidence.
    """
    light, _ = _emitted_pair()
    assert not (isinstance(light, str) and "baseline" in light), (
        f"`published_lightweight_f0_5` is sourced from `{light}`. That field carries Supplementary "
        "Table 9's figure, and it is being paired with a Table 2 literal. Pin the Table 2 value "
        "instead - the pair is a within-table comparison and the registry field is not part of it.")


def test_the_shipped_metrics_agree_with_the_trainer():
    """End to end: what is committed must be what the trainer would write.

    This is what made the defect latent rather than live - `models/metrics.json` still held the
    consistent pair because the trainer had not run since the baseline moved. A drift here means the
    two have diverged again.
    """
    rec = json.loads(METRICS.read_text()).get("satellite_comms.telemanom", {})
    achieved = rec.get("achieved", rec)
    shipped = (achieved.get("published_lightweight_f0_5"), achieved.get("published_full_channel_f0_5"))
    if shipped == (None, None):
        return  # the record does not carry the pair; nothing to compare
    assert shipped in (TABLE_2, SUPP_TABLE_9), (
        f"models/metrics.json ships lightweight/full = {shipped}, which straddles two ESA-ADB "
        "tables. This is the SHIPPED artefact, so unlike the trainer check above this one is live.")
    assert shipped == _emitted_pair(), (
        f"the committed pair {shipped} differs from what the trainer would now write "
        f"{_emitted_pair()}. Re-run the trainer or reconcile deliberately - a silent divergence "
        "here is how the mixed pair stayed invisible.")


def test_the_two_tables_are_distinguishable():
    """Anti-tautology: if the pairs coincided, neither check above could tell them apart."""
    assert TABLE_2 != SUPP_TABLE_9
    assert TABLE_2[0] != SUPP_TABLE_9[0] and TABLE_2[1] != SUPP_TABLE_9[1], (
        "the two published pairs now share a value, so a mix-up would be partly invisible; "
        "re-derive these constants against arXiv:2406.17826 before trusting the gate.")


def test_the_registry_comment_still_names_the_table():
    """The comment that explains the companion sat directly under the other table's number."""
    src = (REPO / "src" / "qsh" / "models" / "registry.py").read_text()
    m = re.search(r"the same method scores 0\.061[^\n]*\n(?:[^\n]*\n){0,4}", src)
    assert m, "the companion comment moved; re-derive this test"
    assert "Table 2" in m.group(0), (
        "the companion comment no longer names which table 0.061 comes from. It sits a few lines "
        "under `baseline=0.968`, which is a DIFFERENT table, so a reader pairs them by proximity.")
