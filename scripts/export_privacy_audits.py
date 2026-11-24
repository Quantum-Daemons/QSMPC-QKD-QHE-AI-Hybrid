#!/usr/bin/env python3
"""The privacy audits' figures, from their records in docs/measurements, as the Validations page reads them.

On the owner's choice of 2026-11-24, the third layer of the PETs evaluation (references/pets-evaluation-2026-11.md:
what a model's outputs leak) is shown on the Validations page as recorded results, not run in the tab. This copies,
from each committed record, only the figures the card draws, with the record's file name and SHA-256, into
`web/src/lib/privacyAudits.json`. Nothing is recomputed, and no reconstructed record or image is copied: the records
hold none (references/pets-evaluation-2026-11.md, "Metrics are published, never a reconstructed record or image").

    python scripts/export_privacy_audits.py            # writes web/src/lib/privacyAudits.json
    python scripts/export_privacy_audits.py --check    # exits 1 if the file is not what this writes

`tests/test_r86_privacy_audits_card.py` runs the check and holds the card to the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "measurements"
OUT = REPO / "web" / "src" / "lib" / "privacyAudits.json"
FPRS = (0.001, 0.01, 0.1)

#: Each membership-inference record, the use case it belongs to, what one record is, and a short name for each model it
#: audits, in the order the card lists them. A record or a model missing here is an error, so none is dropped silently.
MEMBERSHIP = (
    ("mia-breastmnist-pipeline-2026-11-23.json", "Medical", "BreastMNIST, the whole pipeline",
     "an ultrasound image", {"teacher": "ResNet-18 teacher", "student": "CNN student", "square_student": "square-activation student"}),
    ("mia-breastmnist-students-2026-11-23.json", "Medical", "BreastMNIST students, the teacher held fixed",
     "an ultrasound image", {"student": "CNN student", "square_student": "square-activation student"}),
    ("mia-fedheart-2026-11-23.json", "Medical", "The heart federation (four hospitals)",
     "a patient's record", {"federated": "FedAvg", "pooled": "pooled, one model on all rows", "dp_clip_only": "record-level DP, clipping only",
                            "dp_epsilon_8": "record-level DP, epsilon 8", "dp_epsilon_1": "record-level DP, epsilon 1"}),
    ("mia-opssat-2026-11-23.json", "Satellite", "OPS-SAT-AD anomaly detection",
     "a telemetry segment", {"teacher": "gradient-boosted teacher", "student": "CKKS student"}),
    ("mia-telecom-qml-2026-11-23.json", "Telecom", "The QKD-attack classifier (generated rows)",
     "a generated row", {"forest": "random forest"}),
)
DP_AUDIT = "dp-one-run-audit-2026-11-23.json"
DP_STUDIES = {"fedheart": "Heart federation, record-level DP", "murmur": "Murmur study, DP-SGD"}
#: Each seeded bug, the study it belongs to and a short name; each is listed after its study's honest arms.
DP_SEEDED = {"fedheart": ("fedheart_a_sixteenth_of_the_variance", "bug: a sixteenth of the variance"),
             "murmur": ("murmur_every_person_every_step", "bug: every person, every step")}
RECONSTRUCTION = (
    ("reconstruction-opssat-student-2026-11-23.json", "Satellite", "OPS-SAT-AD CKKS student",
     {"A_published_preprocessing_mean": "from the published preprocessing mean",
      "B_weights_at_the_optimum": "from the weights, as the trainer's solver left them",
      "C_weights_solved_to_the_optimum": "from the weights, solved to the optimum"}),
    ("reconstruction-telecom-student-2026-11-24.json", "Telecom", "the QKD-attack CKKS student (generated rows)",
     {"A_published_preprocessing_mean": "from the published preprocessing mean",
      "B_weights_at_the_optimum": "from the weights, as the trainer's solver left them",
      "C_weights_solved_to_the_optimum": "from the weights, solved to the optimum"}),
    ("reconstruction-fedheart-2026-11-24.json", "Medical", "the heart federation's models",
     {"A_published_preprocessing_mean": "from each hospital's published mean",
      "B_pooled_as_the_page_trains_it": "from the pooled model, as the page trains it",
      "B_fedavg_as_the_page_trains_it": "from the FedAvg model, as the page trains it",
      "C_pooled_solved_to_the_optimum": "from the pooled model, solved to the optimum"}),
)
ATTRIBUTE = "attribute-inference-fedheart-2026-11-23.json"
ATTRIBUTE_ATTACKS = {"model_inversion": "model inversion (Fredrikson et al.)", "loss": "loss attack (Yeom et al.)",
                     "imputation_label_only": "imputation from the diagnosis alone",
                     "imputation_all_features": "imputation from every other feature, no model"}
INVERSION = "gradient-inversion-cxr-2026-11-23.json"


def _read(name: str, sources: dict[str, str]) -> dict:
    raw = (MEASUREMENTS / name).read_bytes()
    sources[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def _attack(a: dict) -> dict:
    return {"auc": a["auc"], "tpr": [a.get(f"tpr_at_fpr_{f:g}") for f in FPRS], "not_resolvable": a.get("not_resolvable", [])}


def build() -> dict:
    sources: dict[str, str] = {}
    membership = []
    for name, use_case, title, unit, labels in MEMBERSHIP:
        d = _read(name, sources)
        if set(d["results"]) != set(labels):
            raise SystemExit(f"{name}: the record audits {sorted(d['results'])}, the export names {sorted(labels)}")
        models = []
        for key, label in labels.items():
            r = d["results"][key]
            models.append({"key": key, "label": label, "description": d["models"][key],
                           "lira": _attack(r["lira_online"]), "rmia": _attack(r["rmia_offline"]),
                           "exposed": r["records_exposed_in_most_targets_at_fpr_1e-3"], "audit_records": r["audit_records"]})
        membership.append({"file": name, "use_case": use_case, "title": title, "unit": unit, "data": d["data"],
                           "shadow_models": d["design"]["shadow_models"], "audit_records": d["design"]["audit_records"],
                           "membership": d["design"]["membership"], "models": models})

    d = _read(DP_AUDIT, sources)
    arms = []
    if set(DP_SEEDED) != set(DP_STUDIES) or {k for k, _ in DP_SEEDED.values()} != set(d["seeded_checks"]):
        raise SystemExit(f"{DP_AUDIT}: seeded checks {sorted(d['seeded_checks'])}, the export names {sorted(DP_SEEDED.values())}")
    for study, label in DP_STUDIES.items():
        for arm, a in d[study].items():
            arms.append({"study": label, "arm": arm.replace("noise_", "noise multiplier ").replace("_", " "), "accounted": a["epsilon_accounted"],
                         "lower_bound": a["epsilon_lower_bound"], "canaries": a["canaries"], "seeded_bug": False})
        key, what = DP_SEEDED[study]
        a = d["seeded_checks"][key]
        arms.append({"study": label, "arm": what, "accounted": a["epsilon_accounted"],
                     "lower_bound": a["epsilon_lower_bound"], "canaries": a["canaries"], "seeded_bug": True})
    dp = {"file": DP_AUDIT, "method": d["method"], "confidence": d["design"]["confidence"], "arms": arms}

    reconstruction = []
    for name, use_case, title, routes in RECONSTRUCTION:
        r = _read(name, sources)
        if set(r["routes"]) != set(routes):
            raise SystemExit(f"{name}: routes {sorted(r['routes'])}, the export names {sorted(routes)}")
        reconstruction.append({"file": name, "use_case": use_case, "title": title, "records": r["training_records"],
                               "routes": [{"label": label, "median_relative_error": r["routes"][k]["relative_error_median"],
                                           "share_within": [r["routes"][k][f"share_below_{t:g}"] for t in (0.001, 0.01, 0.1)]}
                                          for k, label in routes.items()]})

    a = _read(ATTRIBUTE, sources)
    attribute = {"file": ATTRIBUTE, "attribute": a["attribute"], "rows": [
        {"model": model, "attack": label, "members": a["results"][model]["members"][k]["accuracy"],
         "non_members": a["results"][model]["non_members"][k]["accuracy"]}
        for model in ("federated", "pooled") for k, label in ATTRIBUTE_ATTACKS.items()]}

    g = _read(INVERSION, sources)
    inversion = {"file": INVERSION, "rows": [
        {"setting": f"{v['films']} films, {v['gradients']} gradient{'s' if v['gradients'] != 1 else ''}",
         "ssim_median": v["ssim_median"], "flat_grey_ssim_median": v["baseline_flat_mean_ssim_median"],
         "nearest_other_film_ssim_median": v["baseline_best_other_film_ssim_median"],
         "films_closer_to_themselves": v["films_above_best_other_film_ssim"]} for v in g["results"].values()]}

    return {"_generated_by": "scripts/export_privacy_audits.py from docs/measurements; do not edit by hand",
            "fprs": list(FPRS), "sources": sources, "membership": membership, "dp_one_run": dp,
            "reconstruction": reconstruction, "attribute_inference": attribute, "gradient_inversion": inversion}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.relative_to(REPO)} is not what scripts/export_privacy_audits.py writes", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
