#!/usr/bin/env python3
"""The 3D nodule student reported at each of its steps, as pre-registered
(`references/medical-nodule3d-reporting-preregistration.md`; Round 86, batch 8).

On the record's test part, on the CPU: the teacher (the mean of its five fold networks), the plaintext students (five
seeds of the square arm and of the ReLU control, and the shipped seed's folded float64 network), and the encrypted
student (the shipped seed through the page's module under A2 and B, keeping each crop's decrypted logits), each with AUC
and AUPRC and a 95 percent percentile interval over 2,000 resamples of the test patients. Writes
`docs/measurements/medical-nodule3d-reporting-2026-11-27.json`; per-crop scores stay in the work directory.

Usage: CUDA_VISIBLE_DEVICES= python scripts/train/medical_nodule3d_report.py --crops DIR --work W
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

import medical_nodule3d_student as S  # noqa: E402

PREREG = REPO / "references" / "medical-nodule3d-reporting-preregistration.md"
PREREG_SHA256 = "90bdaac561b78722626c50e7e5a414d1187d30b684753a10543ea9179432d268"  # with its addendum, 5fef91e
OUT = REPO / "docs" / "measurements" / "medical-nodule3d-reporting-2026-11-27.json"
RESAMPLES, AUC_TOL = 2000, 1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_logits(model, x: np.ndarray) -> np.ndarray:
    """The network's two logits on each crop, on the CPU, in float64 (the pre-registration's device)."""
    import torch

    model.eval().to("cpu")
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            out.append(model(torch.tensor(x[i:i + 64])).double().numpy())
    return np.concatenate(out)


def scores(y: np.ndarray, s: np.ndarray, patients: list[str]) -> dict:
    """AUC and AUPRC of malignant against benign, each with its percentile interval over resamples of the test patients,
    numpy's default generator seeded 0 for every figure (the record's `export_bootstrap`)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    groups = sorted(set(patients))
    rows = {q: [i for i, r in enumerate(patients) if r == q] for q in groups}
    rng = np.random.default_rng(0)
    aucs, aps = [], []
    for _ in range(RESAMPLES):
        idx = [i for q in rng.choice(groups, len(groups)) for i in rows[q]]
        if len(set(y[idx].tolist())) == 2:
            aucs.append(roc_auc_score(y[idx], s[idx]))
            aps.append(average_precision_score(y[idx], s[idx]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 6), round(float(np.percentile(v, 97.5)), 6)]  # noqa: E731
    return {"auc": round(float(roc_auc_score(y, s)), 6), "auc_ci95": ci(aucs),
            "auprc": round(float(average_precision_score(y, s)), 6), "auprc_ci95": ci(aps), "resamples": len(aucs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    got = sha256(PREREG)
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} has SHA-256 {got}, not the committed {PREREG_SHA256}: nothing runs")
    import torch
    from safetensors.torch import load_file

    from qsh.models import metrics

    torch.set_num_threads(min(6, os.cpu_count() or 1))
    started = time.time()
    record = json.loads(S.RECORD.read_text())
    split = S.read_split()
    cubes, records = S.load_crops(args.crops)
    xte, yte, pte = S.labelled(cubes, records, split["parts"], "test")
    c = record["split"]["counts"]["test"]
    if (len(yte), len(set(pte)), int(yte.sum())) != (c["nodules"], c["patients"], c["malignant"]):
        raise SystemExit(f"the test part is not the record's: {len(yte)} crops, {len(set(pte))} patients")
    out: dict = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
                 "of_record": {"file": str(S.RECORD.relative_to(REPO)), "sha256": sha256(S.RECORD)},
                 "test_part": {"crops": int(len(yte)), "patients": int(len(set(pte))), "malignant": int(yte.sum()),
                               "malignant_share": round(float(yte.mean()), 6)},
                 "device": "cpu", "environment": S.environment()}

    # 1. The teacher: the mean of its five fold networks' probabilities of malignancy.
    probs = []
    for k in range(S.FOLDS):
        model = S.resnet3d18()
        model.load_state_dict(load_file(args.work / "teachers" / "resnet3d18" / f"fold{k}" / "model.safetensors"))
        probs.append(S.softmax(cpu_logits(model, xte))[:, 1])
        print(f"  teacher fold {k} scored", flush=True)
    teacher = np.mean(probs, axis=0)
    out["teacher"] = {"what": "the mean of the 3D ResNet-18 candidate's five fold networks' probabilities; each fold "
                              "network trained on three fifths of the training patients and stopped on a fourth; none "
                              "fitted on every training patient",
                      **scores(yte, teacher, pte),
                      "per_fold_auc": [round(float(metrics.auroc(yte, q)), 6) for q in probs]}

    # 2. The plaintext students: each seed's checkpoint, its identity and its recorded AUC checked first.
    out["students"] = {}
    for arm in ("square", "control"):
        out["students"][arm] = {}
        for seed in S.SEEDS:
            ck = args.work / f"{arm}_seed{seed}" / "model.safetensors"
            rec = record["arms"][arm][str(seed)]
            if sha256(ck) != rec["checkpoint_sha256"]:
                raise SystemExit(f"{arm} seed {seed}: the checkpoint is not the recorded one")
            model = S.student(arm)
            model.load_state_dict(load_file(ck))
            lt = cpu_logits(model, xte)
            auc = metrics.auroc(yte, S.softmax(lt))
            # The addendum: two of the test part's malignant-benign pairs, since the record's AUCs came from GPU logits.
            pair = 1.0 / (int(yte.sum()) * int(len(yte) - yte.sum()))
            if abs(auc - rec["test_auc"]) > 2 * pair + 1e-9:
                raise SystemExit(f"{arm} seed {seed}: test AUC {auc:.6f} against the recorded {rec['test_auc']:.6f}")
            out["students"][arm][str(seed)] = {**scores(yte, lt[:, 1] - lt[:, 0], pte),
                                               "recorded_test_auc_gpu": rec["test_auc"]}
            print(f"  {arm} seed {seed}: AUC {auc:.6f}, recorded {rec['test_auc']:.6f}", flush=True)
        aps = [v["auprc"] for v in out["students"][arm].values()]
        aucs = [v["auc"] for v in out["students"][arm].values()]
        out["students"][arm]["mean"] = {"auc": round(float(np.mean(aucs)), 6), "auc_sd": round(float(np.std(aucs, ddof=1)), 6),
                                        "auprc": round(float(np.mean(aps)), 6), "auprc_sd": round(float(np.std(aps, ddof=1)), 6)}
    seed = int(record["criteria"]["export_seed"])
    p = dict(np.load(args.work / f"square_seed{seed}_folded.npz"))
    folded = np.array([S.plain_logits(x[0], p) for x in xte.astype(np.float64)])
    fs = scores(yte, folded[:, 1] - folded[:, 0], pte)
    if abs(metrics.auroc(yte, S.softmax(folded)) - record["export_bootstrap"]["auc"]) > AUC_TOL:
        raise SystemExit("the shipped seed's folded network does not give the recorded AUC")
    out["shipped_seed"] = {"seed": seed, "folded_float64": fs}

    # 3. The encrypted student: the page's module under each set, every test crop, decrypted logits kept here only.
    arm = record["arms"]["square"][str(seed)]
    a, b = arm["platt"]
    keep = set(split["parts"]["test"])
    idx = [i for i, r in enumerate(records) if r["patient"] in keep and S.label_of(r["malignancy"]) is not None]
    out["encrypted"] = {}
    for name in S.PARAMS:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "weights.json").write_text(json.dumps(S.weights_json(p, list(S.PARAMS), seed, [a, b], arm["threshold"])))
            (d / "crops.u8").write_bytes(np.ascontiguousarray(cubes[idx]).tobytes())
            (d / "harness.mjs").write_text(S.HARNESS)
            subprocess.run(["node", str(d / "harness.mjs"), str(REPO / "web"), str(d / "bundle.mjs"), str(d), name],
                           check=True, cwd=REPO / "web", timeout=7200)
            runs = json.loads((d / "runs.json").read_text())
        dec = np.array([r["logits"] for r in runs])
        plain = np.array([r["plain"] for r in runs])
        np.save(args.work / f"report_decrypted_logits_{name}.npy", dec)
        out["encrypted"][name] = {"decrypted": scores(yte, dec[:, 1] - dec[:, 0], pte),
                                  "plaintext_same_run": scores(yte, plain[:, 1] - plain[:, 0], pte),
                                  "max_abs_logit_error": float(max(r["err"] for r in runs)), "crops": len(runs)}
        print(f"  CKKS {name}: decrypted AUC {out['encrypted'][name]['decrypted']['auc']:.6f}, "
              f"AUPRC {out['encrypted'][name]['decrypted']['auprc']:.6f}", flush=True)
    out["wall_clock_s"] = round(time.time() - started, 1)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
