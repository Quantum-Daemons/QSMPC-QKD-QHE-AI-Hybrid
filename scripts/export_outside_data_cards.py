#!/usr/bin/env python3
"""The page files for three checks against outside data (Round 86, batch 13; the owner's choice of 2026-11-28 to show
them): the BreastMNIST models on BUS-BRA (medical page), SatQuMA's key lengths recomputed (satellite page), and a PFCP
attack detector on attack days it never saw (telecom page).

Each file is built from its committed record only, keeps aggregate figures, and carries its dataset's credit; no image,
flow, row or per-patient figure is shipped. `--check` compares each page file with what this would write.

    python scripts/export_outside_data_cards.py [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEAS = REPO / "docs" / "measurements"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"

BUSBRA = MEAS / "medical-busbra-external-2026-11-28.json"
SATQKD = MEAS / "satellite-satqkd-key-length-2026-11-28.json"
SATQKD_B_PRIME = MEAS / "satellite-satqkd-key-length-b-prime-2026-11-28.json"
SATQKD_B_DOUBLE_PRIME = MEAS / "satellite-satqkd-key-length-b-double-prime-2026-11-28.json"
PFCP = MEAS / "telecom-pfcp-held-out-day-2026-11-28.json"
PFCP_AUDIT = MEAS / "telecom-pfcp-audit-app-layer-normal-only-2026-11-28.json"

OUT = {
    "busbra": REPO / "web" / "public" / "data" / "medical" / "busbra_external.json",
    "satqkd": REPO / "web" / "public" / "data" / "satellite" / "satqkd_check.json",
    "pfcp": REPO / "web" / "public" / "data" / "telecom" / "pfcp_held_out_day.json",
}

LABELS = {
    "medical_fl.resnet18 (int8)": "ResNet-18, its int8 graph (the page's build)",
    "medical_fl.resnet18 (fp32)": "ResNet-18, its fp32 graph",
    "medical_fl.resnet50": "ResNet-50",
    "medical_fl.student": "CKKS student (14 x 14)",
    "medical_fl.square_student": "Square student, fully encrypted (14 x 14)",
}
#: Each PFCP run by its folder's name: the attack and its date (the deletion run's folder writes its date day first).
RUNS = {
    "20221005_PFCP_Sess_Establishment_DoS_Test_05_4h": ("session establishment", "2022-10-05"),
    "20221310_PFCP_PFCP_Sess_Deletion_DoS_Test_07_4h": ("session deletion", "2022-10-13"),
    "20221101_PFCP_PFCP_Sess_Modification_DoS_DROP_Test_07_4h": ("session modification, DROP flag", "2022-11-01"),
    "20221122_PFCP_PFCP_Sess_Modification_DoS_DUPL_Test_08_4h": ("session modification, DUPL flag", "2022-11-22"),
}


def r4(v: float | None) -> float | None:
    return None if v is None else round(float(v), 4)


def busbra() -> dict:
    r = json.loads(BUSBRA.read_text())
    models = []
    for key, m in r["models"].items():
        b, t = m["busbra"], m["breastmnist_test_same_graph"]
        models.append({"key": key, "label": LABELS[key], "breastmnist_auc": r4(t["auc"]), "busbra_auc": r4(b["auc"]),
                       "busbra_auc_ci95": [r4(x) for x in b["auc_ci95"]], "busbra_auprc": r4(b["auprc"]),
                       "busbra_accuracy": r4(b["accuracy"]),
                       "malignant_detected": b["confusion_rows_true_cols_predicted_malignant_benign"][0][0],
                       "per_device_auc": {d: r4(v["auc"]) for d, v in m["per_device"].items() if not v.get("counts_only")}})
    ckks = {"student": r["models"]["medical_fl.student"]["ckks"]["page_context"],
            "square_student": r["models"]["medical_fl.square_student"]["ckks"]}
    return {"_generated_by": "scripts/export_outside_data_cards.py; do not edit by hand",
            "preregistration": r["preregistration"]["file"], "record": str(BUSBRA.relative_to(REPO)),
            "images": r["images"], "patients": r["patients"], "malignant": r["malignant"], "devices": r["devices"],
            "models": models,
            "ckks": {"student_largest_logit_error": ckks["student"]["largest_abs_logit_difference"],
                     "student_decisions_changed": ckks["student"]["decisions_changed"],
                     "square_student": {s: {"largest_logit_error": v["largest_abs_logit_difference"],
                                            "decisions_changed": v["decisions_changed"]} for s, v in ckks["square_student"].items()}},
            "licence": "CC-BY-4.0", "licence_url": CC_BY,
            "attribution": ("W. Gomez-Flores, M. J. Gregorio-Calas and W. C. de Albuquerque Pereira, 'BUS-BRA: A breast "
                            "ultrasound dataset for assessing computer-aided diagnosis systems', Medical Physics 51(4):3110-3123 "
                            "(2024), doi:10.1002/mp.16812; dataset Zenodo 8231412, CC BY 4.0 on the record, whose LICENSE.txt "
                            "asks that work using it cite that article"),
            "modifications": "figures computed from BUS-BRA by the pre-registered check; no image or per-patient figure is shipped"}


def satqkd() -> dict:
    a, b = json.loads(SATQKD.read_text()), json.loads(SATQKD_B_PRIME.read_text())
    c = json.loads(SATQKD_B_DOUBLE_PRIME.read_text())
    if not c["check_b_prime_reproduced"]:
        raise SystemExit(f"{SATQKD_B_DOUBLE_PRIME.name} did not reproduce Check B': export nothing")
    files = {k: v for k, v in a["check_a"].items() if k != "meets"}
    variants = {(x["file"], x["dt_s"]): x["variants"] for x in c["check_b_double_prime"]["rows"]}
    rows = []
    for x in b["check_b_prime"]["rows"]:
        s, o = x["satquma"], x["this_project"]
        v = variants[(x["file"], x["dt_s"])]
        rows.append({"file": x["file"], "half_window_s": x["dt_s"], "satquma_key": s["key"], "this_project_key": round(o["key"]),
                     "key_ratio": r4(x["key_ratio"]), "key_basis_count_ratio": r4(x["key_basis_count_ratio"]),
                     "satquma_phase_error": r4(s["phase_error"]), "this_project_phase_error": r4(o["phase_error"]),
                     # Check B'' (the second addendum): SatQuMA's differences added one at a time, each a ratio to its key.
                     "with_cap_key_ratio": r4(v["a"]["key_ratio"]),
                     "with_cap_and_satellite_statistics_key_ratio": r4(v["b"]["key_ratio"]),
                     "with_cap_statistics_and_slots_key_ratio": r4(v["c"]["key_ratio"]),
                     "with_cap_statistics_and_slots_key_basis_count_ratio": r4(v["c"]["key_basis_count_ratio"]),
                     "with_cap_statistics_and_slots_phase_error": r4(v["c"]["phase_error"])})
    return {"_generated_by": "scripts/export_outside_data_cards.py; do not edit by hand",
            "preregistration": a["preregistration"]["file"], "addenda": [b["addendum"]["file"], c["addendum"]["file"]],
            "records": [str(SATQKD.relative_to(REPO)), str(SATQKD_B_PRIME.relative_to(REPO)),
                        str(SATQKD_B_DOUBLE_PRIME.relative_to(REPO))],
            "satellite_statistics": c["check_b_double_prime"]["statistics"]["satellite_page"],
            "check_a": {"rows": sum(v["rows"] for v in files.values()),
                        "rows_with_a_key": sum(v["rows_with_a_key"] for v in files.values()),
                        "keyed_equal": sum(v["keyed_equal"] for v in files.values()),
                        "rows_without_a_key": sum(v["rows_without_a_key"] for v in files.values()),
                        "meets": a["check_a"]["meets"]},
            "check_b_found_a_key": any(r.get("computed") and (r.get("this_project") or {}).get("key", float("-inf")) > 0
                                       for r in a["check_b"]["rows"]),
            "check_b_prime": rows,
            "licence": "CC-BY-4.0", "licence_url": CC_BY,
            "attribution": ("J. S. Sidhu, 'SatQKD_sim_data', Zenodo 8101679 (2023), CC BY 4.0: the data behind J. S. Sidhu et "
                            "al., 'Finite key performance of satellite quantum key distribution under practical constraints', "
                            "arXiv:2301.13209"),
            "modifications": "figures recomputed from the deposit's terms by the pre-registered check and its two addenda; no row is shipped"}


def pfcp() -> dict:
    r = json.loads(PFCP.read_text())
    audit = json.loads(PFCP_AUDIT.read_text())["arm_a_audit"]
    days = []
    for run, (attack, day) in RUNS.items():
        v = r["held_out_days"][run]
        # The audit (pre-registered, run after this check): does each label follow the address pairs, and how often
        # do each label's application-layer flows carry the day's attack message?
        a = audit[run]
        share = a["app"]["attack_counter"]["share_non_zero_per_label"]
        labels = {"follow_address_pairs": a["cic"]["share_whose_label_is_their_address_pairs_majority"] == 1.0,
                  "attack_message_in_malicious": r4(share.get("Malicious")), "attack_message_in_normal": r4(share.get("Normal")),
                  "attack_message": a["app"]["attack_counter"]["column"]}
        labels["consistent_with_the_attack"] = bool(labels["attack_message_in_malicious"] > labels["attack_message_in_normal"])
        if not v.get("scorable"):
            days.append({"attack": attack, "date": day, "scorable": False, "labels": labels})
            continue
        s, t = v["student"], v["teacher"]
        days.append({"attack": attack, "date": day, "scorable": True, "labels": labels, "flows": v["held_out_rows"],
                     "malicious": v["held_out_malicious"],
                     "teacher_auc": {k: r4(t["auc"][k]) for k in ("mean", "min", "max")},
                     "teacher_detection_rate": r4(t["detection_rate"]["mean"]),
                     "teacher_false_positive_rate": r4(t["false_positive_rate"]["mean"]),
                     "student_auc": r4(s["auc"]), "student_auc_ci95": [r4(x) for x in s["auc_ci95"]],
                     "student_detection_rate": r4(s["detection_rate"]), "student_false_positive_rate": r4(s["false_positive_rate"])})
    return {"_generated_by": "scripts/export_outside_data_cards.py; do not edit by hand",
            "preregistration": r["preregistration"]["file"], "record": str(PFCP.relative_to(REPO)), "days": days,
            "audit": {"preregistration": json.loads(PFCP_AUDIT.read_text())["preregistration"]["file"],
                      "record": str(PFCP_AUDIT.relative_to(REPO))},
            "licence": "CC-BY-4.0", "licence_url": CC_BY,
            "attribution": ("G. Amponis, P. Radoglou-Grammatikis, G. Nakas, M. Zevgara, S. Giannakidou, S. Ouzounidis, "
                            "G. Kakamoukas and A. Sarigiannidis, '5GC PFCP Intrusion Detection Dataset', IEEE DataPort / Zenodo "
                            "7888347 (2023), doi:10.21227/e0gj-ev03, CC BY 4.0"),
            "modifications": "figures computed from the dataset's UPF flows by the pre-registered check; no flow is shipped"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a page file is not what this writes")
    args = ap.parse_args()
    bad = 0
    for name, build in (("busbra", busbra), ("satqkd", satqkd), ("pfcp", pfcp)):
        text = json.dumps(build(), indent=1) + "\n"
        path = OUT[name]
        if args.check:
            same = path.exists() and path.read_text() == text
            print(f"{path.relative_to(REPO)} {'is' if same else 'is NOT'} the record's")
            bad += not same
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(REPO)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
