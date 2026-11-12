"""How much of the finance test window lies after HI-Small's ten days of data, and what it does to the scores (Round 86).

IBM's dataset card gives HI-Small's date range as September 1-10, 2022, its "primary" period, and the dataset's
creator explains on Kaggle (discussion 427517) that the small datasets hold only those ten days of "real" data and
that the transactions after September 10 are there to let laundering patterns that began earlier finish. This
project's test window is the last 30 percent of the transactions in time order, so it ends on September 18 and
carries that tail; the published 60/20/20 split's test set does too.

This script counts the tail and scores BOTH published finance models with and without it:

  - the teacher through its published ONNX graph, the file the Hub serves at the record's pinned revision, checked by
    digest, and required to reproduce the record's minority-class F1 exactly before anything is written;
  - the CKKS student through its published weights, checked the same way.

It writes `docs/measurements/finance-test-window-tail-<date>.json` and nothing else. Needs the AMLworld corpus
(`qsh.models.datasets`, `$QSH_DATA_ROOT`) and onnxruntime.

    .venv/bin/python scripts/measure_finance_test_window_tail.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

#: The first day after HI-Small's stated date range (IBM's card: "Sep 1-10" for the small datasets).
TAIL_STARTS = "2022-09-11"
#: `onnx/model.onnx` as the Hub serves it at the teacher's revision 6532ffa3 (LFS sha256 read 2026-11-11), the retrain on the
#: 114 inputs; `models/hub_file_hashes.json` pins the same digest.
TEACHER_ONNX_SHA256 = "d69b65320c55496001d7b6b77ed2b9c4ee6d88495ab4e1e12f967903947db8a5"
#: `weights.json` of the student, as in scripts/finance_student_calibration.py: revision f172f6b8 (2026-11-11), the retrain on the 114 inputs.
STUDENT_WEIGHTS_SHA256 = "dd58191c8279e7f6d36522db5545e46e1a6dda135abffb741eaba545d6a66528"


def _digest(path: Path, want: str) -> bytes:
    blob = path.read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    if got != want:
        raise SystemExit(f"{path.relative_to(REPO)} is not the published artefact: sha256 {got}")
    return blob


def _scores(y: np.ndarray, decision: np.ndarray, score: np.ndarray) -> dict:
    from qsh.models import metrics

    tp = int(((decision == 1) & (y == 1)).sum())
    fp = int(((decision == 1) & (y == 0)).sum())
    fn = int(((decision == 0) & (y == 1)).sum())
    both = bool(y.sum()) and bool((y == 0).sum())
    return {
        "minority_class_f1": round(float(metrics.minority_class_f1(y, decision)), 6),
        "auprc": round(float(metrics.auprc(y, score)), 6) if both else None,
        "flagged": int(decision.sum()), "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "recall": round(tp / max(int(y.sum()), 1), 6),
    }


def _block(mask: np.ndarray, y: np.ndarray, teacher: tuple, student: tuple) -> dict:
    ys = y[mask]
    return {
        "rows": int(mask.sum()), "positives": int(ys.sum()), "base_rate": round(float(ys.mean()), 8),
        "teacher": _scores(ys, teacher[0][mask], teacher[1][mask]),
        "student": _scores(ys, student[0][mask], student[1][mask]),
    }


def main() -> int:
    import onnxruntime as ort
    from train.finance_aml import (
        causal_account_features,
        load_sorted,
        model_inputs,
        temporal_windows,
    )

    from qsh.models import metrics, registry

    onnx_path = REPO / registry.ARTIFACT_DIRS["finance_aml.gbm"] / "onnx" / "model.onnx"
    weights_path = REPO / registry.ARTIFACT_DIRS["finance_aml.student"] / "weights.json"
    _digest(onnx_path, TEACHER_ONNX_SHA256)
    w = json.loads(_digest(weights_path, STUDENT_WEIGHTS_SHA256))
    rec_t, rec_s = registry.achieved("finance_aml.gbm"), registry.achieved("finance_aml.student")

    df = load_sorted()
    fit, val, te, tr = temporal_windows(df)
    (_, _, x_te), _, _ = model_inputs(df, fit, val, te, tr, causal_account_features(df))
    y = te["label"].to_numpy().astype(int)
    if len(y) != rec_t["n_test"]:
        raise SystemExit(f"{len(y)} test rows; the teacher's record says {rec_t['n_test']}")
    tail = (te["timestamp"] >= TAIL_STARTS).to_numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = dict(zip([o.name for o in sess.get_outputs()],
                   sess.run(None, {sess.get_inputs()[0].name: x_te.astype(np.float32)}), strict=True))
    prob = np.asarray(out["probabilities"], dtype=np.float64)[:, 1]
    t_dec = (prob >= rec_t["decision_threshold"]).astype(int)
    f1_t = metrics.minority_class_f1(y, t_dec)
    if f1_t != rec_t["minority_class_f1"]:
        raise SystemExit(f"the published graph gives minority-class F1 {f1_t!r}; the record says "
                         f"{rec_t['minority_class_f1']!r}")
    auprc_t = metrics.auprc(y, prob)

    mean, scale = np.asarray(w["input"]["mean"]), np.asarray(w["input"]["scale"])
    z = ((np.asarray(x_te, dtype=np.float64) - mean) / scale) @ np.asarray(w["W"][0]) + float(w["b"][0])
    s_dec = (z > 0).astype(int)
    f1_s = metrics.minority_class_f1(y, s_dec)
    if abs(f1_s - rec_s["student_metric"]) > 5e-7:
        raise SystemExit(f"the published student gives minority-class F1 {f1_s}; the record says "
                         f"{rec_s['student_metric']}")

    n = len(df)
    split_602020 = df.iloc[int(n * 0.8):]
    tail_602020 = (split_602020["timestamp"] >= TAIL_STARTS).to_numpy()
    by_day = te.groupby(te["timestamp"].dt.normalize())["label"].agg(["size", "sum"])

    teacher, student = (t_dec, prob), (s_dec, z)
    record = {
        "measured": dt.date.today().isoformat(),
        "script": "scripts/measure_finance_test_window_tail.py",
        "corpus": "IBM AMLworld HI-Small (CDLA-Sharing-1.0)",
        "stated_period": {"first": "2022-09-01", "last": "2022-09-10",
                          "source": "IBM's Kaggle dataset card, 'Date Range HI + LI (2022)', Small: Sep 1-10, the "
                                    "'primary' period; the dataset's creator in Kaggle discussion 427517: the small "
                                    "datasets hold ten days of 'real' data, and the later transactions let laundering "
                                    "patterns that began earlier complete"},
        "tail_starts": TAIL_STARTS,
        "test_window": {"first": str(te["timestamp"].min()), "last": str(te["timestamp"].max()),
                        "rule": "the last 30 percent of the transactions in time order (TEST_FRACTION)"},
        "test_window_by_day": {str(k.date()): {"rows": int(v["size"]), "positives": int(v["sum"])}
                               for k, v in by_day.iterrows()},
        "full_window": _block(np.ones_like(tail), y, teacher, student),
        "without_tail": _block(~tail, y, teacher, student),
        "tail_only": _block(tail, y, teacher, student),
        "tail_share_of_window_positives": round(float(y[tail].sum() / y.sum()), 6),
        "published_602020_test_split": {
            "rows": len(split_602020), "positives": int(split_602020["label"].sum()),
            "tail_rows": int(tail_602020.sum()),
            "tail_positives": int(split_602020["label"].to_numpy()[tail_602020].sum())},
        "reproduced": {
            "teacher_onnx_sha256": TEACHER_ONNX_SHA256, "teacher_hf_revision": rec_t["hf_revision"],
            "teacher_minority_class_f1": f1_t, "teacher_auprc_onnx": round(float(auprc_t), 8),
            "teacher_auprc_record": rec_t["auprc"],
            "teacher_auprc_note": ("the record's AUPRC is the native booster's, in float64; the published ONNX graph "
                                   "scores in float32, which moves the ranking's AUPRC in the sixth decimal and no "
                                   "decision at the threshold"),
            "student_weights_sha256": STUDENT_WEIGHTS_SHA256, "student_minority_class_f1": round(float(f1_s), 6),
        },
    }
    out_path = REPO / "docs" / "measurements" / f"finance-test-window-tail-{record['measured']}.json"
    out_path.write_text(json.dumps(record, indent=1) + "\n")
    # The student's record carries the two flag counts its card prints beside the agreement (Round 86), so they come from
    # this measurement and are rewritten with it, never carried from a previous student.
    from qsh.models import train

    payload = dict(registry.achieved("finance_aml.student"))
    # Measured here, so no longer carried forward: the previous record's marker named exactly this field.
    marker = payload.pop("carried_forward_not_remeasured", None)
    if marker and set(marker.get("fields", [])) - {"flag_counts"}:
        payload["carried_forward_not_remeasured"] = {**marker, "fields": sorted(set(marker["fields"]) - {"flag_counts"})}
    payload["flag_counts"] = {"source": f"{out_path.relative_to(REPO)}, full_window",
                              "teacher_flagged": record["full_window"]["teacher"]["flagged"],
                              "student_flagged": record["full_window"]["student"]["flagged"]}
    train.record_metrics("finance_aml.student", payload)
    fw, wt = record["full_window"], record["without_tail"]
    print(f"tail from {TAIL_STARTS}: {record['tail_only']['rows']:,} rows, {record['tail_only']['positives']:,} "
          f"positive = {record['tail_share_of_window_positives']:.1%} of the window's {fw['positives']:,}")
    print(f"base rate {fw['base_rate']:.6f} -> {wt['base_rate']:.6f} without the tail")
    for k in ("teacher", "student"):
        print(f"{k}: F1 {fw[k]['minority_class_f1']} -> {wt[k]['minority_class_f1']}, "
              f"AUPRC {fw[k]['auprc']} -> {wt[k]['auprc']}")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
