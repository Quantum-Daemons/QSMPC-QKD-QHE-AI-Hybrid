"""The finance test window's tail is disclosed with the numbers its measurement recorded (Round 86).

IBM's HI-Small holds ten days of data, and this project's test window - the last 30 percent of the transactions in
time order - runs eight days past them, through rows the dataset's creator says exist so that laundering patterns
can complete. `scripts/measure_finance_test_window_tail.py` counts them and scores both published models with and
without them. Two surfaces state the result: the dataset's notes, which every finance model card prints, and the
finance chart's split note, which the page shows. This holds both to the measurement, so a re-measurement that
moves a number fails here until the prose moves with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "measurements"


def _latest() -> dict:
    found = sorted(MEASUREMENTS.glob("finance-test-window-tail-*.json"))
    assert found, "no finance-test-window-tail measurement is committed"
    return json.loads(found[-1].read_text())


def test_the_measurement_adds_up() -> None:
    t = _latest()
    fw, wt, tl = t["full_window"], t["without_tail"], t["tail_only"]
    assert fw["rows"] == wt["rows"] + tl["rows"] and fw["positives"] == wt["positives"] + tl["positives"]
    assert abs(t["tail_share_of_window_positives"] - tl["positives"] / fw["positives"]) < 1e-6
    for block in (fw, wt, tl):
        assert abs(block["base_rate"] - block["positives"] / block["rows"]) < 1e-8
        for model in ("teacher", "student"):
            m = block[model]
            assert m["true_positives"] + m["false_negatives"] == block["positives"]
            assert m["true_positives"] + m["false_positives"] == m["flagged"]
    # The published graph reproduced the record exactly before the measurement was written.
    rec = json.loads((REPO / "models" / "metrics.json").read_text())
    assert t["reproduced"]["teacher_minority_class_f1"] == rec["finance_aml.gbm"]["minority_class_f1"]
    assert fw["rows"] == rec["finance_aml.gbm"]["n_test"]
    assert t["tail_starts"] == "2022-09-11" and t["stated_period"]["last"] == "2022-09-10"


def test_the_dataset_notes_state_the_measured_tail() -> None:
    sys.path.insert(0, str(REPO / "src"))
    from qsh.models import registry

    t, notes = _latest(), registry.DS_AMLWORLD.notes
    fw, wt, tl = t["full_window"], t["without_tail"], t["tail_only"]
    for needle in (
        f"The {tl['rows']:,} rows dated September 11-18",
        f"{tl['positives']:,} of them are laundering, {t['tail_share_of_window_positives']:.1%} of the test "
        f"window's {fw['positives']:,} positives",
        f"the window's rate is {100 * wt['base_rate']:.3f} percent rather than {100 * fw['base_rate']:.3f}",
        f"the teacher's minority-class F1 {wt['teacher']['minority_class_f1']:.4f} rather than "
        f"{fw['teacher']['minority_class_f1']:.4f}",
        f"its AUPRC {wt['teacher']['auprc']:.4f} rather than {fw['teacher']['auprc']:.4f}",
        f"the CKKS student's F1 {wt['student']['minority_class_f1']:.4f} rather than "
        f"{fw['student']['minority_class_f1']:.4f}",
        f"carries the same {t['published_602020_test_split']['tail_rows']:,} rows",
        f"finance-test-window-tail-{t['measured']}.json",
    ):
        assert needle in notes, needle


def test_the_page_states_it_from_the_measurement() -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from prep_real_data_2d import finance_tail_sentence

    payload = json.loads((REPO / "web" / "public" / "data" / "finance" / "real_2d.json").read_text())
    assert finance_tail_sentence(_latest()) in payload["split_note"], "regenerate with scripts/prep_real_data_2d.py"
