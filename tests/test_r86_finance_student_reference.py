"""What the finance student's agreement is worth, and that its card says so (Round 86).

The student agrees with its teacher on 0.9829 of 1,523,504 window rows, and the teacher flags 759 of them, so a student
that never flags would agree 0.9995. Printed alone, the agreement reads as fidelity. The card now prints that reference
and both flag counts, from the record, and the calibration table prints the Brier skill against the base rate, which for
this student is negative.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
METRICS = json.loads((REPO / "models" / "metrics.json").read_text())


def test_the_counts_are_the_measurement_scripts():
    rec = METRICS["finance_aml.student"]
    flags = rec["flag_counts"]
    measured = json.loads((REPO / "docs" / "measurements" / "finance-test-window-tail-2026-11-06.json").read_text())["full_window"]
    assert measured["rows"] == rec["n_eval"]
    assert flags["teacher_flagged"] == measured["teacher"]["flagged"] == 759
    assert flags["student_flagged"] == measured["student"]["flagged"] == 26151


def test_a_student_that_never_flags_would_beat_this_agreement():
    rec = METRICS["finance_aml.student"]
    never = 1 - rec["flag_counts"]["teacher_flagged"] / rec["n_eval"]
    assert round(never, 4) == 0.9995
    assert never > rec["agreement"]


def test_the_calibrated_student_is_worse_than_the_base_rate_and_the_teacher_is_not():
    # The card computes this from the same two fields, so the sign it prints is the sign here.
    def skill(key: str) -> float:
        c = METRICS[key]["calibration"]
        return 1 - c["brier_calibrated"] / c["corp_calibrated"]["unc"]

    assert round(skill("finance_aml.student"), 4) == -0.0124
    assert skill("finance_aml.gbm") > 0


def test_the_card_prints_them():
    panel = (REPO / "web" / "src" / "components" / "ModelPanel.tsx").read_text()
    assert '"flag_counts"' in panel, "the counts would be walked into the metric rows"
    assert "A student that never flags would agree {fmt(neverFlagsAgrees)}" in panel
    assert "Brier skill against the base rate" in panel
    assert "1 - c.brier_calibrated / unc" in panel


def test_the_curve_says_what_below_zero_means():
    chart = (REPO / "web" / "src" / "components" / "VerificationChart.tsx").read_text()
    assert "below = data.filter((d) => d.model < 0)" in chart
    assert "laundering rate is under t" in chart
