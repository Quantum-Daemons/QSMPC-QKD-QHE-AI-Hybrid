#!/usr/bin/env python3
"""The 3D nodule student's report at each step, for the page (Round 86, batch 8).

Reads `docs/measurements/medical-nodule3d-reporting-2026-11-27.json`, which `scripts/train/medical_nodule3d_report.py`
writes under `references/medical-nodule3d-reporting-preregistration.md`, and writes its figures, and nothing else, to
`web/public/data/medical/nodule3d_reporting.json`: aggregates over the test part, no crop and no patient.

Usage: python -I scripts/export_nodule3d_reporting.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "measurements" / "medical-nodule3d-reporting-2026-11-27.json"
SITE = REPO / "web" / "public" / "data" / "medical" / "nodule3d_reporting.json"
FIGS = ("auc", "auc_ci95", "auprc", "auprc_ci95")


def main() -> int:
    r = json.loads(RECORD.read_text())
    pick = lambda d: {k: d[k] for k in FIGS}  # noqa: E731
    payload = {
        "_generated_by": "scripts/export_nodule3d_reporting.py; do not edit by hand",
        "preregistration": r["preregistration"]["file"], "record": str(RECORD.relative_to(REPO)),
        "test_part": r["test_part"], "device": r["device"], "resamples": r["teacher"]["resamples"],
        "teacher": {**pick(r["teacher"]), "what": r["teacher"]["what"]},
        "students": {arm: r["students"][arm]["mean"] for arm in ("square", "control")},
        "seeds": len([k for k in r["students"]["square"] if k != "mean"]),
        "shipped_seed": {"seed": r["shipped_seed"]["seed"], **pick(r["shipped_seed"]["folded_float64"])},
        "encrypted": {name: {**pick(e["decrypted"]), "plaintext_same_run": pick(e["plaintext_same_run"]),
                             "max_abs_logit_error": e["max_abs_logit_error"], "crops": e["crops"]}
                      for name, e in r["encrypted"].items()},
    }
    student = json.loads((REPO / "web" / "public" / "data" / "medical" / "nodule3d_square_student.json").read_text())
    payload["licence"], payload["licence_url"] = student["licence"], student["licence_url"]
    payload["attribution"] = {"dataset": "LIDC-IDRI", "download": "https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX",
                              "citation": student["attribution"], "licence": student["licence"]}
    payload["modifications"] = "figures over the test part from the record; no crop, scan or patient is shipped"
    SITE.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {SITE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
