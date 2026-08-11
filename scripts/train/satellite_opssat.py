#!/usr/bin/env python
"""Detect anomalous OPS-SAT telemetry segments, and distil the detector to run under CKKS.

WHAT THIS IS
    A second, independent satellite model, trained on the OPS-SAT anomaly benchmark (Ruszczak et
    al., Scientific Data 12, 710, 2025; CC BY 4.0). It exists because that benchmark ships **30
    published baselines** on its own train/test split, so a result here has thirty comparators
    instead of one.

WHAT IT IS NOT
    It is NOT comparable with `satellite_comms.telemanom`, and the two must never be presented as
    a comparison. Telemanom is a FORECASTER scored on next-step error against a persistence
    baseline over continuous ESA-ADB series. This is a segment CLASSIFIER over 2,123 pre-segmented
    fragments with precomputed statistical features. Different task, different split, different
    metric. Putting the two numbers side by side would be the same category error this project
    corrected when it stopped scoring a 28px model against a 224px baseline.

WHY AUC PR
    The positive class is 20.4 percent of the data, so accuracy reads about 0.80 for a detector
    that never fires at all. The benchmark reports seven metrics; area under the precision-recall
    curve is the one that cannot be satisfied by abstaining, and F1 is carried beside it.

Usage:
    export QSH_DATA_ROOT=$PWD/data-full
    python scripts/train/satellite_opssat.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qsh.models import datasets, distil, export, metrics, train  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402

TEACHER = "satellite_comms.opssat"
STUDENT = "satellite_comms.opssat_student"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=400)
    args = ap.parse_args()

    started = time.time()
    train.seed_everything(args.seed)

    from sklearn.ensemble import HistGradientBoostingClassifier

    print("loading OPS-SAT-AD ...")
    xtr, ytr = datasets.opssat_ad("train")
    xte, yte = datasets.opssat_ad("test")
    print(f"  train {xtr.shape[0]} segments ({ytr.mean()*100:.1f}% anomalous) | "
          f"test {xte.shape[0]} ({yte.mean()*100:.1f}% anomalous) | {xtr.shape[1]} features")

    # ---- T2 teacher -----------------------------------------------------------
    teacher = HistGradientBoostingClassifier(
        max_iter=args.n_estimators, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=args.seed,
    )
    teacher.fit(xtr, ytr)
    proba = teacher.predict_proba(xte)
    pred = proba[:, 1] >= 0.5

    teacher_metrics = metrics.with_companion(yte, pred.astype(int), proba, headline="auc_pr")
    spec = get_spec(TEACHER)
    print("\n  teacher (gradient-boosted trees, test split)")
    for k, v in teacher_metrics.items():
        print(f"    {k:22s} {v:.6f}" if isinstance(v, float) else f"    {k:22s} {v}")
    print(f"    published best         {spec.metric.baseline}  ({spec.metric.baseline_source})")

    # ---- T1 student: what actually runs under CKKS ----------------------------
    # 19 features fit in a single packed dot product, and the encrypted tier has room to spare: a
    # degree-8192 ring has 4096 slots and the QUADRATIC expansion of 19 features is 209. The
    # expansion is a function of the plaintext input, so the client computes it before encrypting
    # and the encrypted step stays one dot product at unchanged multiplicative depth - capacity
    # bought without spending ciphertext.
    #
    # Measured, which is why this is not merely tidier: minority-class F1 0.8485 -> 0.9391 and
    # AUC-PR 0.9235 -> 0.9810, passing both the teacher (0.9189) and the published 0.979. It was
    # worth asking whether the encrypted tier had to be the weakest model in the project. Here it
    # did not, and the constraint that looked binding was the feature map, not the cipher.
    print("\n  distilling to the CKKS student (quadratic expansion, computed client-side) ...")
    student = distil.distil_linear(teacher.predict_proba(xtr), xtr, seed=args.seed, degree=2)
    student_pred = student.predict(xte)

    student_export = distil.save_student(
        student, Path("models/satellite_opssat_student"),
        n_features=xtr.shape[1], teacher_key=TEACHER, student_key=STUDENT,
        # Real held-out rows, not standard normals: the graph now scales internally, so normals
        # would exercise it far outside the range it was fitted on and the agreement check would
        # be measuring nothing.
        fixtures=xte[:64])
    print(f"  student persisted: {student_export['n_parameters']} params, "
          f"browser_runnable={student_export['browser_runnable']}")

    fidelity = distil.measure_fidelity(
        teacher_key=TEACHER, student_key=STUDENT,
        teacher_pred=pred.astype(int), student_pred=student_pred, y_true=yte,
        # Fidelity is measured on hard labels, so it uses F1 on the positive class: the student's
        # DECISION is what the encrypted path produces and what agreement compares. AUC PR is
        # recorded separately below from predict_proba - the student is a logistic model, so the
        # scores do exist, and the registry declares auc_pr as this model's headline.
        metric_fn=metrics.minority_class_f1, metric_name="minority_class_f1",
        # A pipeline student keeps its coefficients on the final estimator, and they describe the
        # EXPANDED space - 209 terms from 19 raw features - which is the honest parameter count.
        student_params=int(student[-1].coef_.size + student[-1].intercept_.size),
        teacher_params=int(sum(len(p) for p in teacher._predictors)),
    )
    print(f"    agreement with teacher      {fidelity.agreement:.4f}")
    print(f"    F1 (positive) teacher/stu   {fidelity.teacher_metric:.4f} / {fidelity.student_metric:.4f}"
          f"  (delta {fidelity.metric_delta_pp:+.2f} pp)")

    # ---- artefacts ------------------------------------------------------------
    out = Path("models/satellite_opssat")
    out.mkdir(parents=True, exist_ok=True)
    report = export.verify_and_quantize(teacher, xte.astype(np.float32), out,
                                        kind="sklearn", n_features=xtr.shape[1])
    print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB, domains={report.onnx_domains}, "
          f"browser_runnable={report.browser_runnable}")

    import json
    (out / "feature_spec.json").write_text(json.dumps({
        "input": {"shape": [1, xtr.shape[1]], "dtype": "float32",
                  "preprocessing": "none for either: the teacher takes raw features, and the\n                   student's ONNX graph now bakes in its own quadratic expansion and scaler"},
        "feature_names": list(datasets.OPSSAT_FEATURES),
        "classes": ["nominal", "anomalous"],
    }, indent=2) + "\n")

    train.record_metrics(TEACHER, {
        **teacher_metrics,
        "dataset": spec.dataset.name,
        "n_train_segments": int(xtr.shape[0]),
        "n_test_segments": int(xte.shape[0]),
        "n_features": int(xtr.shape[1]),
        "anomaly_rate_test": round(float(yte.mean()), 6),
        "task": ("segment classification - NOT comparable with satellite_comms.telemanom, which "
                 "forecasts continuous ESA-ADB series against a persistence baseline"),
        "git_sha": train.git_sha(),
        "export": {"fp32_bytes": report.fp32_bytes, "int8_bytes": report.int8_bytes,
                   "int8_agreement": report.int8_agreement, "opset": report.opset},
        "wall_clock_s": round(time.time() - started, 1),
    })
    # Record the metric the REGISTRY declares, beside the decision metric fidelity measures.
    # Without this the panel printed an auc_pr baseline of 0.979 under a 0.8649 minority-class F1 -
    # two different quantities, rendered as if one could be read against the other.
    student_auprc = metrics.auprc(yte, student.predict_proba(xte))   # pipeline scales internally
    print(f"    auc_pr student              {student_auprc:.4f}"
          f"   <- the metric the registry declares, so this is the one the baseline compares to")
    train.record_metrics(STUDENT, {
        **fidelity.as_metrics(),
        "auc_pr": round(float(student_auprc), 6),
        "metric_note": (
            "Two metrics are recorded and they answer different questions. `auc_pr` is the "
            "registry's declared headline and is what the published baseline compares against. "
            "`student_metric`/`teacher_metric` are `minority_class_f1`, because fidelity compares "
            "DECISIONS and a decision has no score to rank - so agreement and metric_delta are "
            "necessarily computed on a decision metric. Do not read the F1 against the auc_pr "
            "baseline."
        ),
    })
    print(f"\n  wrote {out} and models/metrics.json in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
