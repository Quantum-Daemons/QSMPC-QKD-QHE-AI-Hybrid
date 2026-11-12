"""Rewrite the published finance teacher's calibration record from its graph, without retraining it (Round 86).

The teacher the Hub pins was trained with a computed Platt map, and the record beside it still carried three notes
measured in Round 49 on an earlier teacher: the isotonic comparison, which its Hub card printed beside the current map's
own Brier score, and two notes on the ranking and the clip. `teacher_calibration` in `scripts/train/finance_aml.py`
measures them in the training run now. This applies it to the PUBLISHED graph, checked by digest, with the PUBLISHED map:
it refits nothing the page applies, and reports how far a refit on the graph's own probabilities would move the map.

It writes nothing unless the record's own numbers come back from this graph and these windows: the test window's size,
the minority-class F1 at the published threshold, and the Brier score of both arms. The log-loss is not held to the
record, and the new record says why: the record's came from the booster in the training run, and the graph's float32
output is exactly 0.0 wherever the probability is below its resolution, which the log-loss's clip at 1e-15 prices at
34.5 nats on a laundering row. Every score is now the graph's, the output the page reads, with the booster's beside it.

Needs the AMLworld corpus (`qsh.models.datasets`, `$QSH_DATA_ROOT`) and onnxruntime.

    .venv/bin/python scripts/finance_teacher_calibration.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

KEY = "finance_aml.gbm"

#: `models/finance_aml_gbm/onnx/model.onnx` as the Hub serves it at the record's revision 6532ffa3 (read 2026-11-11), the
#: teacher on the 114 inputs; `scripts/measure_finance_test_window_tail.py` pins the same digest.
GRAPH_SHA256 = "d69b65320c55496001d7b6b77ed2b9c4ee6d88495ab4e1e12f967903947db8a5"

#: The record rounds every score to 8 decimals, so a reproduced figure may differ from it by half a unit there.
TOLERANCE = 5e-9 + 1e-12


def main() -> int:
    import onnxruntime as ort
    from train.finance_aml import (  # type: ignore
        causal_account_features,
        load_sorted,
        model_inputs,
        teacher_calibration,
        temporal_windows,
    )

    from qsh.models import calibration as cal
    from qsh.models import metrics, registry, train

    path = REPO / registry.ARTIFACT_DIRS[KEY] / "onnx" / "model.onnx"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != GRAPH_SHA256:
        raise SystemExit(f"{path.relative_to(REPO)} is not the published teacher: sha256 {digest}")
    rec = registry.achieved(KEY)
    old = rec["calibration"]
    if old.get("method") != "platt" or old.get("on_logit") is not True:
        raise SystemExit(f"unexpected map in the record: {old.get('method')}, on_logit {old.get('on_logit')}")
    published = cal.Platt(a=float(old["a"]), b=float(old["b"]), n_calibration=int(old["n_calibration"]),
                          positives_calibration=int(old["positives_calibration"]), on_logit=True,
                          logit_clip_eps=float(old["logit_clip_eps"]))

    df = load_sorted()
    fit, val, te, tr = temporal_windows(df)
    (_, x_val, x_te), _, _ = model_inputs(df, fit, val, te, tr, causal_account_features(df))
    y_val = val["label"].to_numpy().astype(int)
    y_te = te["label"].to_numpy().astype(int)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]

    def prob(x: np.ndarray) -> np.ndarray:
        out = dict(zip(names, sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)}), strict=True))
        return np.asarray(out["probabilities"], dtype=np.float64)[:, 1]

    p_val, p_te = prob(x_val), prob(x_te)
    f1 = metrics.minority_class_f1(y_te, (p_te >= rec["decision_threshold"]).astype(int))
    if len(y_te) != rec["n_test"] or f1 != rec["minority_class_f1"] or len(y_val) != published.n_calibration \
            or int(y_val.sum()) != published.positives_calibration:
        raise SystemExit(f"this graph and these windows give {len(y_te)} test rows, minority-class F1 {f1!r}, and "
                         f"{len(y_val)} validation rows ({int(y_val.sum())} positive); the record says {rec['n_test']}, "
                         f"{rec['minority_class_f1']!r}, {published.n_calibration} and {published.positives_calibration}")
    print(f"reproduced the record: {len(y_te):,} test rows, minority-class F1 {f1:.6f}, {len(y_val):,} validation rows")

    calib = teacher_calibration(
        p_val, y_val, p_te, y_te, platt=published,
        fitted_by=(f"scripts/train/finance_aml.py, in the run that fitted the teacher; every figure beside the map "
                   f"rewritten by scripts/finance_teacher_calibration.py at git {train.git_sha()} from the published "
                   f"graph (sha256 {digest[:16]}) and the published map, without retraining"))
    for k in ("brier_raw", "brier_calibrated"):
        if abs(calib[k] - old[k]) > TOLERANCE:
            raise SystemExit(f"{k}: {calib[k]} from the published graph and map, {old[k]} in the record")
    zero, zero_events = int((p_te == 0.0).sum()), int(((p_te == 0.0) & (y_te == 1)).sum())
    calib["scored_from"] = "the published ONNX graph's float32 probability, the output the page reads"
    calib["booster_scores_in_training_run"] = {k: old[k] for k in ("brier_raw", "brier_calibrated", "log_loss_raw",
                                                                    "log_loss_calibrated")}
    calib["raw_is"] = (
        f"the published graph's float32 P(laundering). It is exactly 0 on {zero:,} of the {len(y_te):,} test rows, "
        f"{zero_events} of them laundering, where the probability is below float32's resolution; the log-loss clips at "
        f"1e-15, so each of those {zero_events} costs {-np.log(1e-15):.1f} nats. The training run's booster, which keeps "
        f"them above zero, scored the raw log-loss {old['log_loss_raw']:.8f} where the graph scores "
        f"{calib['log_loss_raw']:.8f}, and the calibrated {old['log_loss_calibrated']:.8f} where the graph scores "
        f"{calib['log_loss_calibrated']:.8f}; the Brier scores agree to 8 decimals.")
    refit = cal.fit_platt(p_val, y_val, on_logit=True)
    print(f"published map a {published.a:.12f}, b {published.b:.12f}; a refit on the graph's own validation "
          f"probabilities gives a {refit.a:.12f}, b {refit.b:.12f} (not used)")
    payload = dict(rec)
    payload["calibration"] = {**calib, **{k: old[k] for k in ("page_reads_this_map",) if k in old}}
    train.record_metrics(KEY, payload)
    iso = calib["isotonic_measured"]
    print(f"test window: Brier {calib['brier_raw']:.8f} -> {calib['brier_calibrated']:.8f} (isotonic {iso['brier']:.8f}), "
          f"log-loss {calib['log_loss_raw']:.8f} -> {calib['log_loss_calibrated']:.8f} (isotonic {iso['log_loss']:.8f})")
    print(f"AUROC raw {calib['auroc_raw']:.8f}, calibrated {calib['auroc_calibrated']:.8f}, isotonic {iso['auroc']:.8f}; "
          f"AUPRC raw {calib['auprc_raw']:.8f}, calibrated {calib['auprc_calibrated']:.8f}, isotonic {iso['auprc']:.8f}")
    print(f"isotonic: {iso['distinct_outputs']} distinct outputs, {iso['rows_at_zero']:,} rows at zero "
          f"({iso['events_at_zero']} laundering); {calib['test_rows_clipped_by_logit']} test rows at the logit clip")
    print(f"recorded in models/metrics.json under {KEY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
