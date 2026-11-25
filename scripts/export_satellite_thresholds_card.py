#!/usr/bin/env python3
"""The satellite page's threshold-comparison card, from its measurement record (Round 86, batch 6).

On the owner's choice of 2026-11-26 the page shows the pre-registered comparison of its forecaster's threshold with two
calibrated ones (references/satellite-threshold-calibration-preregistration.md), without changing the page's rule. This
copies the figures from docs/measurements/satellite-thresholds-2026-11-25.json, which
scripts/measure_satellite_thresholds.py wrote, into web/public/data/satellite/thresholds.json with ESA-ADB's credit, so
the card reads a file and computes nothing. tests/test_r86_satellite_thresholds_card.py holds the two together.

Usage: python -I scripts/export_satellite_thresholds_card.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "satellite-thresholds-2026-11-25.json"
CARD = REPO / "web" / "public" / "data" / "satellite" / "thresholds.json"
VUS = REPO / "web" / "public" / "data" / "satellite" / "vus_pr.json"

#: The arms in the order the card lists them, each with its label.
ARMS = (
    ("ndt", "Telemanom's NDT with ESA-ADB's pruning (the page's rule)"),
    ("pot_alpha_0.0001", "extreme value (POT), alpha 1e-4, primary"),
    ("conformal_alpha_0.0001", "weighted conformal, alpha 1e-4, primary"),
    ("pot_alpha_0.001", "extreme value (POT), alpha 1e-3"),
    ("conformal_alpha_0.001", "weighted conformal, alpha 1e-3"),
    ("pot_alpha_1e-05", "extreme value (POT), alpha 1e-5"),
    ("conformal_alpha_1e-05", "weighted conformal, alpha 1e-5"),
)


def main() -> int:
    rec = json.loads(RECORD.read_text())
    arms = []
    for key, label in ARMS:
        a = rec["arms"][key]["anomalies"]
        arms.append({"key": key, "label": label, "f0_5": round(a["f_0.50"], 4), "tp": a["tp"], "fp": a["fp"], "fn": a["fn"],
                     "share_flagged": rec["arms"][key]["share_of_test_steps_flagged"]})
    card = {
        "_generated_by": "scripts/export_satellite_thresholds_card.py from " + str(RECORD.relative_to(REPO)) + "; do not edit by hand",
        "model": rec["model"], "preregistration": rec["preregistration"], "record": str(RECORD.relative_to(REPO)),
        "events": "ESA-ADB's anomalies of Mission1 channels 41-46 in the test period, 29 events",
        "metric": "ESA-ADB's corrected event-wise F0.5",
        "arms": arms,
        "vus_pr_of_the_shared_score": round(rec["vus"]["vus_pr"], 4),
        "ndt_matches_the_students_record": rec["arms"]["ndt"]["matches_the_students_record"],
        # The corpus's credit, as the VUS-PR card carries it: an aggregate of ESA-ADB, CC BY 3.0 IGO.
        "attribution": json.loads(VUS.read_text())["attribution"],
    }
    CARD.write_text(json.dumps(card, indent=1) + "\n")
    print(f"wrote {CARD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
