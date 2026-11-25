"""The satellite page's threshold-comparison card is its measurement record, and keeps the page's rule (Round 86, batch 6).

The owner chose on 2026-11-26 to show the pre-registered comparison on the page. The card file is copied from the record
by scripts/export_satellite_thresholds_card.py; this holds the two together and holds the card to what it says: NDT
first, the page's rule, the calibrated rules below it, and ESA-ADB's credit.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CARD = json.loads((REPO / "web" / "public" / "data" / "satellite" / "thresholds.json").read_text())
RECORD = json.loads((REPO / "docs" / "measurements" / "satellite-thresholds-2026-11-25.json").read_text())


def test_every_figure_is_the_records() -> None:
    for arm in CARD["arms"]:
        a = RECORD["arms"][arm["key"]]
        assert arm["f0_5"] == round(a["anomalies"]["f_0.50"], 4)
        assert (arm["tp"], arm["fp"], arm["fn"]) == (a["anomalies"]["tp"], a["anomalies"]["fp"], a["anomalies"]["fn"])
        assert arm["share_flagged"] == a["share_of_test_steps_flagged"]
    assert CARD["vus_pr_of_the_shared_score"] == round(RECORD["vus"]["vus_pr"], 4)
    assert CARD["preregistration"] == RECORD["preregistration"]


def test_the_pages_rule_leads_and_wins_and_matches_its_record() -> None:
    assert CARD["arms"][0]["key"] == "ndt" and "the page's rule" in CARD["arms"][0]["label"]
    assert all(CARD["arms"][0]["f0_5"] > a["f0_5"] for a in CARD["arms"][1:])
    assert CARD["ndt_matches_the_students_record"] is True


def test_the_card_carries_esa_adbs_credit() -> None:
    a = CARD["attribution"]
    assert a["licence"] == "CC BY 3.0 IGO" and a["licence_url"].endswith("/by/3.0/igo/") and "12528696" in a["download"]
