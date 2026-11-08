"""Fit the published finance student's calibration map without retraining it, and record it (Round 86).

The finance page draws a decision curve from the CKKS student's decrypted logits, which needs a map from logit to risk.
The trainer fits that map in the run that fits the student (`student_calibration` in `scripts/train/finance_aml.py`),
but the student on the Hub was fitted before the map existed, and retraining it to get one would publish different
weights. So this applies the trainer's own function to the PUBLISHED weights: the same windows, the same features,
and the weights file the Hub pins, checked by digest before anything is fitted.

It writes nothing unless the record's own numbers come back from these weights and windows: the test window's size
and the student's minority-class F1 at its folded threshold.

Needs the AMLworld corpus (`qsh.models.datasets`, `$QSH_DATA_ROOT`).

    .venv/bin/python scripts/finance_student_calibration.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

KEY = "finance_aml.student"

#: `models/finance_aml_student/weights.json` as the Hub serves it at the record's revision 641566e (read 2026-11-06:
#: git blob 0b0dcab90e88a38cb12387c6f69b1a5005f9af94, which is this file's).
#: Re-published 2026-11-08 with a rewritten `note` only: the weights, bias and scaler are those of sha256 c47afeadcc8816ce, which the 2026-11-06 measurements used.
WEIGHTS_SHA256 = "d8823c8961bbc98699223bfa446cc3c5763701d2f363f8be23ee7851d1a420b2"


def main() -> int:
    from qsh.models import metrics, registry, train
    from train.finance_aml import (  # type: ignore
        causal_account_features, graph_features, load_sorted, student_calibration, temporal_windows)

    path = REPO / registry.ARTIFACT_DIRS[KEY] / "weights.json"
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise SystemExit(f"{path.relative_to(REPO)} is not the published student: sha256 {digest}")
    w = json.loads(blob)
    if w["input"]["preprocessing"] != ["subtract mean", "divide by scale"] or len(w["W"]) != 1:
        raise SystemExit(f"unexpected student form: {w['input']['preprocessing']}, {len(w['W'])} logit rows")
    mean, scale = np.asarray(w["input"]["mean"]), np.asarray(w["input"]["scale"])
    coef, bias = np.asarray(w["W"][0]), float(w["b"][0])
    rec = registry.achieved(KEY)

    df = load_sorted()
    fit, val, te, tr = temporal_windows(df)
    cz = causal_account_features(df)
    x_val = graph_features(val, fit_on=fit, causal=cz)
    x_te = graph_features(te, fit_on=tr, causal=cz)
    y_val = val["label"].to_numpy().astype(int)
    y_te = te["label"].to_numpy().astype(int)

    def logit(x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - mean) / scale) @ coef + bias

    z_val, z_te = logit(x_val), logit(x_te)
    f1 = metrics.minority_class_f1(y_te, (z_te > 0).astype(int))
    if len(y_te) != rec["n_eval"] or abs(f1 - rec["student_metric"]) > 5e-7:
        raise SystemExit(f"these weights and windows give {len(y_te)} rows and minority-class F1 {f1:.6f}; the record "
                         f"says {rec['n_eval']} and {rec['student_metric']}")
    print(f"reproduced the record: {len(y_te):,} test rows, minority-class F1 {f1:.6f}")

    calib = student_calibration(
        z_val, y_val, z_te, y_te,
        fitted_by=(f"scripts/finance_student_calibration.py at git {train.git_sha()}, on the published weights "
                   f"(sha256 {digest[:16]}) without retraining"))
    payload = dict(rec)
    payload["calibration"] = calib
    train.record_metrics(KEY, payload)
    print(f"Platt a {calib['a']:.6f}, b {calib['b']:.6f} on {calib['n_calibration']:,} validation rows "
          f"({calib['positives_calibration']} positive)")
    print(f"test window: Brier {calib['brier_raw']:.8f} -> {calib['brier_calibrated']:.8f}, log-loss "
          f"{calib['log_loss_raw']:.8f} -> {calib['log_loss_calibrated']:.8f}, CORP MCB "
          f"{calib['corp_raw']['mcb']:.8f} -> {calib['corp_calibrated']['mcb']:.8f}")
    print(f"AUPRC raw {calib['auprc_raw']:.8f}, calibrated {calib['auprc_calibrated']:.8f}; isotonic "
          f"{calib['isotonic_measured']}")
    print(f"operating point (logit 0) is a calibrated risk of {1 / (1 + np.exp(calib['b'])):.6f}")
    print(f"recorded in models/metrics.json under {KEY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
