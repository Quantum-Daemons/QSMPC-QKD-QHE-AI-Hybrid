"""The telecom page's file for the 5G-NIDD re-run under leakage-aware protocols (Round 86, batch 9; checklist MS33).

Reads `docs/measurements/telecom-nidd-clean-protocols-2026-11-27.json` and writes
`web/public/data/telecom_nidd/clean_protocols.json`: each arm's figure for each input set and split (the mean and
standard deviation over seeds where an arm has seeds), the single-feature check's highest values, the CKKS re-check's
figures, and the corpora's credit. No flow, threshold or weight is shipped. `--check` compares the page's file with what this would write.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models.registry import DS_5G_NIDD  # noqa: E402

RECORD = REPO / "docs" / "measurements" / "telecom-nidd-clean-protocols-2026-11-27.json"
SITE = REPO / "web" / "public" / "data" / "telecom_nidd" / "clean_protocols.json"
SPLITS = {"P0": "the shipped random split", "P1_train_1_test_2": "station 1 to station 2",
          "P1_train_2_test_1": "station 2 to station 1", "P2": "a time split in each capture file", "P3": "LEAF's Seq rule"}
INPUTS = {"T46": "all 46 inputs", "T40": "without LEAF's six", "T38": "without the six and TTL",
          "F36": "all 36 inputs", "F30": "without LEAF's six", "F28": "without the six and TTL"}


def metric(t: dict) -> float:
    """FLamby's metric where the test part holds both stations; the macro-F1 where it holds one."""
    return t["flamby_metric"] if len(t.get("station_macro_f1", {})) > 1 else t["macro_f1"]


def cell(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 4),
            "sd": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else None, "runs": len(values)}


def build() -> dict:
    r = json.loads(RECORD.read_text())
    arms = {}
    if "teacher" in r:
        arms["teacher"] = {s: {p: cell([metric(r["teacher"][f"{s}/{p}/seed{k}"]["test"]) for k in range(5)]) for p in SPLITS}
                           for s in ("T46", "T40", "T38")}
    if "student" in r:
        arms["student"] = {s: {p: cell([metric(r["student"][f"{s}/{p}"]["test"])]) for p in SPLITS} for s in ("F36", "F30", "F28")}
    if "mlp" in r:
        arms["mlp"] = {"F30": {p: cell([metric(r["mlp"][f"F30/{p}/seed{k}"]["test"]) for k in range(5)])
                               for p in ("P1_train_1_test_2", "P1_train_2_test_1", "P2")}}
    if "federation" in r:
        arms["federation"] = {s: {"P2": cell([metric(v["test"])])} for s, v in r["federation"].items()}
    top = {}
    for p, v in r["single_feature_check"].items():
        pairs = sorted(((f, n, int(c)) for n, d in v["validation_f1_at_least_0_5"].items() for c, f in d.items()), reverse=True)
        top[p] = [{"input": n, "class": r["classes"][c], "f1": f} for f, n, c in pairs[:3]]
    out = {"_generated_by": "scripts/export_nidd_clean_protocols.py; do not edit by hand",
           "preregistration": r["preregistration"], "record": str(RECORD.relative_to(REPO)),
           "splits": SPLITS, "inputs": INPUTS, "arms": arms, "single_feature_check": {
               "flagged": {p: v["flagged"] for p, v in r["single_feature_check"].items()}, "highest": top},
           "licence": DS_5G_NIDD.licence, "licence_url": "https://creativecommons.org/licenses/by/4.0/",
           "attribution": {"dataset": "5G-NIDD", "licence": DS_5G_NIDD.licence,
                           "licence_url": "https://creativecommons.org/licenses/by/4.0/", "citation": DS_5G_NIDD.attribution,
                           "download": "https://doi.org/10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361"},
           "modifications": "figures computed from 5G-NIDD by the pre-registered re-run; no flow, threshold or weight is shipped"}
    if "ckks_recheck" in r:
        out["ckks_recheck"] = {k: r["ckks_recheck"][k] for k in ("rows", "ciphertexts", "decisions_differ",
                                                                 "max_abs_logit_error", "encrypted_accuracy",
                                                                 "plaintext_accuracy")}
    if "crosstestbed" in r:
        ct = r["crosstestbed"]
        out["crosstestbed"] = {"teacher": {"macro_f1": cell([ct[f"teacher/seed{k}"]["macro_f1"] for k in range(5)]),
                                           "binary_attack_f1": cell([ct[f"teacher/seed{k}"]["binary_attack_f1"] for k in range(5)])},
                               "student": {"macro_f1": cell([ct["student"]["macro_f1"]]),
                                           "binary_attack_f1": cell([ct["student"]["binary_attack_f1"]])},
                               "shared_rows": ct["student"]["shared_rows"],
                               "dataset": "NetsLab-5GORAN-IDD", "licence": "CC-BY-4.0",
                               "source": "https://zenodo.org/records/18923275",
                               "citation": ("F. Abed Zadeh, A. Civciss, V. R. Wickrama Arachchilage, M. Liyanage and "
                                            "C. Sandeepa, '5G Open Radio Access Network Multi-Modal Intrusion Detection "
                                            "Dataset (NetsLab-5GORAN-IDD)', Zenodo 18923275, 2025, CC BY 4.0; figures "
                                            "computed from its flow CSV, no row shipped")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the page's file is not what this writes")
    args = ap.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if args.check:
        same = SITE.exists() and SITE.read_text() == text
        print("the page's file is the record's" if same else "the page's file differs from the record")
        return 0 if same else 1
    SITE.write_text(text)
    print(f"wrote {SITE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
