#!/usr/bin/env python3
"""The 3D nodule student's shipped files: its weights, and the twelve test crops the medical page's card runs.

Run after the pre-registered stages (`scripts/train/medical_nodule3d_student.py students`, then `ckks`), and only if
the candidate is adopted: the pre-registration's criteria 1 and 2 hold. Otherwise the page does not change and this
refuses to write. It writes:

- `models/medical_lidc_nodule3d_square_student/weights.json`: the exported seed's folded float64 network, its Platt
  scaling and validation threshold, and only the CKKS parameter sets that met criterion 3 (a set that did not is not
  offered, the pre-registration's rule), in the form the CKKS stage ran (`weights_json`);
- `web/public/data/medical/nodule3d_square_student.json`: those weights; the twelve test crops the addendum's section 4
  fixed before any model was trained (the first six labelled malignant and the first six benign, in the cluster
  records' order), each as base64 of its uint8 voxels (axes z, y, x), with its label, its readers' count, their median
  rating, its SHA-256 and the folded network's plaintext logits; the figures the card quotes, copied from the record;
  the sources' SHA-256; and the licence, credit and change note. No patient or series identifier is written.

    python scripts/export_nodule3d_student.py --crops DIR --work W     # writes both files
    python scripts/export_nodule3d_student.py --check                  # exits 1 if the web file is not these files'

`--check` needs no crop: it rebuilds the web file from the committed weights and record, takes the crops from the web
file itself, and checks each against its SHA-256 and its logits against the weights.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

import medical_nodule3d_student as s  # noqa: E402

MODEL_DIR = REPO / "models" / "medical_lidc_nodule3d_square_student"
WEIGHTS = MODEL_DIR / "weights.json"
WEB = REPO / "web" / "public" / "data" / "medical" / "nodule3d_square_student.json"
PER_CLASS = 6
ATTRIBUTION = (
    "Armato III, S. G., et al. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. "
    "doi:10.7937/K9/TCIA.2015.LO9QL9SX; Armato et al., The Lung Image Database Consortium (LIDC) and Image Database "
    "Resource Initiative (IDRI), Medical Physics 38(2), 915-931 (2011), doi:10.1118/1.3528204; nodule clusters by "
    "pylidc, Hancock and Magnan, Journal of Medical Imaging 3(4), 044504 (2016), doi:10.1117/1.JMI.3.4.044504. "
    "The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, "
    "and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.")
MODIFICATIONS = (
    "twelve test nodules only, no whole scan: each a 28 mm cube at 1 mm centred on the mean of its readers' centroids, "
    "sampled by trilinear interpolation, clipped to [-1000, 400] HU and scaled to uint8 "
    "(scripts/data/lidc_make_crops.py); no patient or series identifier")
POLICY_URL = "https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def adopted(record: dict) -> None:
    c = record.get("criteria", {})
    if not (c.get("1_accuracy") and c.get("2_no_collapse")):
        raise SystemExit("the candidate is not adopted (criteria 1 and 2): the page does not change, nothing is written")
    if "ckks" not in record:
        raise SystemExit("the record has no CKKS stage yet: run `medical_nodule3d_student.py ckks` first")


def offered_sets(record: dict) -> list[str]:
    sets = [k for k, v in record["ckks"].items() if v["criterion_3"]]
    if not sets:
        raise SystemExit("no parameter set met criterion 3: nothing is offered under CKKS")
    return sets


def page_rows(cubes: np.ndarray, records: list[dict], split: dict) -> list[int]:
    """The addendum's section 4: labelled test crops in record order, the first six of each label."""
    keep = set(split["parts"]["test"])
    idx = [i for i, r in enumerate(records) if r["patient"] in keep and s.label_of(r["malignancy"]) is not None]
    pick = lambda cls: [i for i in idx if s.label_of(records[i]["malignancy"]) == cls][:PER_CLASS]  # noqa: E731
    return sorted(pick(1) + pick(0))


def crop_entry(cube: np.ndarray, rec: dict, p: dict) -> dict:
    raw = np.ascontiguousarray(cube, dtype=np.uint8).tobytes()
    logits = s.plain_logits((cube.astype(np.float64) / 255.0 - 0.5) / 0.5, p)
    return {"label": int(s.label_of(rec["malignancy"])), "readers": len(rec["malignancy"]),
            "median_rating": float(np.median(rec["malignancy"])), "sha256": _sha(raw),
            "plain_logits": [float(v) for v in logits], "u8_base64": base64.b64encode(raw).decode("ascii")}


def params_of(weights: dict) -> dict:
    return {"conv_w": np.array(weights["conv"]["weight"], dtype=np.float64),
            "conv_b": np.array(weights["conv"]["bias"], dtype=np.float64),
            "fc_w": np.array(weights["dense"][0]["weight"], dtype=np.float64),
            "fc_b": np.array(weights["dense"][0]["bias"], dtype=np.float64)}


def artifact_metrics(record: dict) -> dict:
    """The record's figures for the exported seed, kept in weights.json so its card's measured table is checked against
    its own artifact (tests/test_a_model_card_states_what_its_weights_state.py)."""
    seed = record["criteria"]["export_seed"]
    arm = record["arms"]["square"][str(seed)]
    boot = record["export_bootstrap"]
    return {"test_auc": boot["auc"], "test_auc_ci95_low": boot["ci95"][0], "test_auc_ci95_high": boot["ci95"][1],
            "bootstrap_resamples": boot["resamples"], "val_auc": arm["val_auc"],
            "test_balanced_accuracy": arm["test_balanced_accuracy"],
            "mean_test_auc_square": record["criteria"]["mean_test_auc"]["square"],
            "mean_test_auc_control": record["criteria"]["mean_test_auc"]["control"],
            "majority_class_rate_test": record["majority_class_rate_test"],
            "test_nodules": record["split"]["counts"]["test"]["nodules"]}


def figures(record: dict) -> dict:
    seed = record["criteria"]["export_seed"]
    arm = record["arms"]["square"][str(seed)]
    return {
        "export_seed": seed, "teacher": record["teacher"],
        "teachers_oof_auc": {k: v["oof_auc"] for k, v in record["teachers"].items()},
        "mean_test_auc": record["criteria"]["mean_test_auc"], "sd_test_auc": record["criteria"]["sd_test_auc"],
        "square_test_aucs": [record["arms"]["square"][k]["test_auc"] for k in sorted(record["arms"]["square"], key=int)],
        "export": {"test_auc": arm["test_auc"], "val_auc": arm["val_auc"],
                   "test_balanced_accuracy": arm["test_balanced_accuracy"], "bootstrap": record["export_bootstrap"]},
        "criteria": {k: record["criteria"][k] for k in ("1_accuracy", "2_no_collapse", "3_fidelity")},
        "ckks": {k: {kk: v[kk] for kk in ("N", "coeff_mod_bits", "scale_bits", "modulus_bits", "crops",
                                          "max_abs_logit_error", "decisions_differ", "median_ms", "max_ms",
                                          "ciphertexts", "bytes", "criterion_3")}
                 for k, v in record["ckks"].items()},
        "split_counts": record["split"]["counts"], "malignant_share": record["malignant_share"],
        "majority_class_rate_test": record["majority_class_rate_test"],
        "context_not_compared": "NoduleMNIST3D's official split (test AUC: ResNet-18 3D 0.863, ResNet-50 3D 0.875, "
                                "auto-sklearn 0.914) shares scans across its parts and labels differently",
    }


def build(weights_text: str, crops: list[dict]) -> dict:
    record_text = s.RECORD.read_text()
    record = json.loads(record_text)
    adopted(record)
    w = json.loads(weights_text)
    return {
        "_generated_by": "scripts/export_nodule3d_student.py; do not edit by hand",
        "weights": w, "crops_about": "twelve test crops fixed by the 3D addendum's section 4 before any model was "
                                     "trained: the first six labelled malignant and the first six benign, in the "
                                     "cluster records' order; uint8, axes z, y, x, read as (u / 255 - 0.5) / 0.5",
        "crops": crops, "figures": figures(record),
        "sources": {str(s.RECORD.relative_to(REPO)): _sha(record_text.encode()),
                    str(WEIGHTS.relative_to(REPO)): _sha(weights_text.encode()),
                    str(s.SPLIT.relative_to(REPO)): _sha(s.SPLIT.read_bytes())},
        "licence": "CC-BY-3.0", "licence_url": "https://creativecommons.org/licenses/by/3.0/",
        "attribution": ATTRIBUTION, "modifications": MODIFICATIONS, "data_usage_policy": POLICY_URL,
    }


def write(crops_dir: Path, work: Path) -> int:
    record = json.loads(s.RECORD.read_text())
    adopted(record)
    seed = record["criteria"]["export_seed"]
    arm = record["arms"]["square"][str(seed)]
    p = dict(np.load(work / f"square_seed{seed}_folded.npz"))
    weights_text = json.dumps({**s.weights_json(p, offered_sets(record), int(seed), arm["platt"], arm["threshold"]),
                               "metrics": artifact_metrics(record)}) + "\n"
    cubes, records = s.load_crops(crops_dir)
    rows = page_rows(cubes, records, s.read_split())
    crops = [crop_entry(cubes[i], records[i], p) for i in rows]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS.write_text(weights_text)
    WEB.write_text(json.dumps(build(weights_text, crops), separators=(",", ":")) + "\n")
    print(f"wrote {WEIGHTS.relative_to(REPO)} and {WEB.relative_to(REPO)} ({len(WEB.read_text()) / 1e3:.0f} kB): "
          f"{sum(c['label'] for c in crops)} malignant and {sum(1 - c['label'] for c in crops)} benign crops")
    return 0


SAMPLE = MODEL_DIR / "eval_sample.json"
#: The registry's `eval_split` for this model, which the sample states as its own `split`.
SAMPLE_SPLIT = ("twelve of the 347 test nodules of the 148 held-out patients: the first six labelled malignant and the "
                "first six benign in the records' order, fixed by the 3D addendum's section 4 before any model was trained")


def eval_sample(shipped: dict) -> dict:
    """The Hub repo's evaluation sample: the twelve crops the site ships, with their labels, SHA-256 and the folded
    network's plaintext logits, under the corpus's licence and credit, so the repo can be checked without the site."""
    crops = shipped["crops"]
    return {
        "_generated_by": "scripts/export_nodule3d_student.py --metrics; do not edit by hand",
        "task": "classification",
        "provenance": ("LIDC-IDRI chest CT (The Cancer Imaging Archive): " + SAMPLE_SPLIT + ". " + shipped["attribution"]),
        "split": SAMPLE_SPLIT,
        "licence": shipped["licence"], "licence_url": shipped["licence_url"], "attribution": shipped["attribution"],
        "modifications": shipped["modifications"], "data_usage_policy": shipped["data_usage_policy"],
        "input": {"shape": [1, 1, s.SIDE, s.SIDE, s.SIDE], "dtype": "uint8",
                  "preprocessing": ["x / 255", "(x - 0.5) / 0.5"],
                  "layout": "samples[i] is one crop's 28^3 voxels, base64 of uint8, axes z, y, x"},
        "classes": {"0": "benign", "1": "malignant"},
        "samples": [c["u8_base64"] for c in crops],
        "labels": [int(c["label"]) for c in crops],
        "expected": {"sha256": [c["sha256"] for c in crops], "plain_logits": [c["plain_logits"] for c in crops],
                     "note": ("The folded float64 network's logits on each crop, as the medical page's card shows them "
                              "beside its CKKS run; the decision is the weights' Platt map of the logit difference "
                              "against the validation threshold.")},
    }


def registry_record() -> dict:
    """What `models/metrics.json` records for `medical_fl.nodule3d_square_student` (the owner's choice of 2026-11-25 to
    stage it on the Hub): the exported seed's figures as `weights.json` carries them, the pre-registered comparison and
    the CKKS record, all copied from `docs/measurements/medical-nodule3d-square-student.json`, so the card and the
    registry state the record's numbers and no other."""
    record = json.loads(s.RECORD.read_text())
    w = json.loads(WEIGHTS.read_text())
    m, c = w["metrics"], record["criteria"]
    square = record["arms"]["square"]
    return {
        "auc": m["test_auc"], "auc_ci95_low": m["test_auc_ci95_low"], "auc_ci95_high": m["test_auc_ci95_high"],
        "bootstrap_resamples": m["bootstrap_resamples"], "val_auc": m["val_auc"],
        "balanced_accuracy": m["test_balanced_accuracy"], "majority_class_rate": m["majority_class_rate_test"],
        "n_eval": m["test_nodules"], "scoring_unit": "nodules", "export_seed": int(c["export_seed"]),
        "student_params": int(params_count(w)), "metric_name": "auc",
        "metric_note": ("`auc` is the exported seed's test AUC on all 347 test nodules, chosen by validation AUC as "
                        "pre-registered; `five_seeds_test_auc` is the pre-registered comparison with the same network "
                        "with a ReLU, and `ckks` the record's run of the page's module on every test crop."),
        "five_seeds_test_auc": {"mean": c["mean_test_auc"]["square"], "sd": c["sd_test_auc"]["square"],
                                "control_mean": c["mean_test_auc"]["control"], "control_sd": c["sd_test_auc"]["control"],
                                "seeds": {k: v["test_auc"] for k, v in square.items()}},
        "teacher": (f"{record['teacher']}, the best of three pre-registered candidates scored out of fold over the training "
                    "patients; not published"),
        "ckks": {k: {kk: v[kk] for kk in ("N", "coeff_mod_bits", "scale_bits", "modulus_bits", "crops",
                                          "max_abs_logit_error", "decisions_differ", "median_ms")}
                 for k, v in record["ckks"].items()},
        "preregistration": record["preregistration"], "split_counts": record["split"]["counts"],
    }


def params_count(w: dict) -> int:
    conv = w["conv"]
    n = int(np.asarray(conv["weight"]).size + np.asarray(conv["bias"]).size)
    for layer in w["dense"]:
        n += sum(int(np.asarray(v).size) for v in layer.values() if isinstance(v, list))
    return n


def write_metrics() -> int:
    from qsh.models import train

    payload = registry_record()
    prev = json.loads((REPO / "models" / "metrics.json").read_text()).get("medical_fl.nodule3d_square_student") or {}
    # A pin written by a publish is kept: the record's numbers do not change the weights the pin names.
    for k in ("hf_revision",):
        if k in prev:
            payload[k] = prev[k]
    train.record_metrics("medical_fl.nodule3d_square_student", payload)
    SAMPLE.write_text(json.dumps(eval_sample(json.loads(WEB.read_text())), separators=(",", ":")) + "\n")
    print(f"models/metrics.json: medical_fl.nodule3d_square_student, test AUC {payload['auc']} on {payload['n_eval']} "
          f"nodules; {SAMPLE.relative_to(REPO)}")
    return 0


def check() -> int:
    shipped = json.loads(WEB.read_text())
    weights_text = WEIGHTS.read_text()
    p = params_of(json.loads(weights_text))
    problems = []
    for k, c in enumerate(shipped["crops"]):
        raw = base64.b64decode(c["u8_base64"])
        if _sha(raw) != c["sha256"] or len(raw) != s.SIDE ** 3:
            problems.append(f"crop {k}: its bytes are not the ones recorded")
            continue
        cube = np.frombuffer(raw, dtype=np.uint8).reshape(s.SIDE, s.SIDE, s.SIDE)
        if np.max(np.abs(s.plain_logits((cube.astype(np.float64) / 255.0 - 0.5) / 0.5, p) - c["plain_logits"])) > 1e-9:
            problems.append(f"crop {k}: its logits are not the weights'")
    text = json.dumps(build(weights_text, shipped["crops"]), separators=(",", ":")) + "\n"
    if WEB.read_text() != text:
        problems.append(f"{WEB.relative_to(REPO)} is not what this script writes from the weights and the record")
    sample = json.dumps(eval_sample(shipped), separators=(",", ":")) + "\n"
    if not SAMPLE.exists() or SAMPLE.read_text() != sample:
        problems.append(f"{SAMPLE.relative_to(REPO)} is not what this script writes from the site's crops")
    for msg in problems:
        print(msg, file=sys.stderr)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--metrics", action="store_true",
                    help="write the registry's record (models/metrics.json) from the record and the shipped weights")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.metrics:
        return write_metrics()
    if not (args.crops and args.work):
        raise SystemExit("--crops and --work are needed to write")
    s.check_preregistration()
    return write(args.crops, args.work)


if __name__ == "__main__":
    raise SystemExit(main())
