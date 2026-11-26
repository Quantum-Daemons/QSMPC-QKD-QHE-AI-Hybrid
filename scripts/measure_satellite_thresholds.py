#!/usr/bin/env python3
"""The pre-registered threshold comparison for the satellite page's CKKS forecaster (Round 86, batch 6).

references/satellite-threshold-calibration-preregistration.md fixed everything below before any threshold was fitted
or any test window scored: the shipped CKKS student's forecasts (`satellite_comms.student`) on ESA-ADB Mission1's
lightweight channels, the errors smoothed as Telemanom smooths them, and three rules that turn them into flags:

  NDT        Telemanom's non-parametric dynamic threshold with ESA-ADB's pruning, the page's present rule;
  POT        per channel, a generalised Pareto fit to the validation window's excesses over its 0.98 quantile, the
             threshold at exceedance probability alpha;
  CONFORMAL  per channel, the weighted 1 - alpha quantile of the validation window's smoothed errors, weights
             2^(-age / 7 days).

alpha = 1e-4 is primary, 1e-3 and 1e-5 beside it. The script refuses to run unless the pre-registration is the
committed text whose hash it records. Writes docs/measurements/satellite-thresholds-2026-11-25.json.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/measure_satellite_thresholds.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from train import satellite_telemanom as T  # noqa: E402
from train.satellite_student_forecaster import (  # noqa: E402
    STUDENT,
    TEACHER,
    _padded,
    features,
    forecast,
)

from qsh.models import registry, telemanom_threshold  # noqa: E402
from qsh.models.esa_scores import esa_event_wise  # noqa: E402

PREREG = REPO / "references" / "satellite-threshold-calibration-preregistration.md"
PREREG_SHA256 = "a2436cde71d939c32b7d67f895f892fd0012262f1c7a1fb3a39505d85a153d65"
RECORD = REPO / "docs" / "measurements" / "satellite-thresholds-2026-11-25.json"
ALPHAS = (1e-4, 1e-3, 1e-5)          # the first is primary
Q0, MIN_EXCESSES, HALF_LIFE_S = 0.98, 30, 7 * 86_400


def pot_threshold(e: np.ndarray, alpha: float) -> tuple[float, str]:
    """The level a GPD over the excesses above the Q0 quantile exceeds with probability alpha (or the empirical one)."""
    from scipy.stats import genpareto

    u = float(np.quantile(e, Q0))
    exc = e[e > u] - u
    p_u = exc.size / e.size
    if exc.size < MIN_EXCESSES or alpha >= p_u:
        return float(np.quantile(e, 1 - alpha)), "empirical"
    c, _loc, scale = genpareto.fit(exc, floc=0.0)
    return float(u + genpareto.ppf(1 - alpha / p_u, c, loc=0.0, scale=scale)), f"gpd (shape {c:.4f}, scale {scale:.6g})"


def weighted_quantile(e: np.ndarray, w: np.ndarray, q: float) -> float:
    """The smallest value whose weighted share of the mass at or below it is at least q."""
    order = np.argsort(e, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    return float(e[order][min(np.searchsorted(cum, q, side="left"), e.size - 1)])


def scores(times, flagged, labels, full) -> dict:
    a = esa_event_wise(times, flagged, labels, select=T.ANOMALIES, full_range=full)
    e = esa_event_wise(times, flagged, labels, select=T.ALL_EVENTS, full_range=full)
    return {"anomalies": {k: a[k] for k in ("f_0.50", "precision", "recall", "tp", "fp", "fn") if k in a},
            "all_events": {k: e[k] for k in ("f_0.50", "precision", "recall", "tp", "fp", "fn") if k in e},
            "share_of_test_steps_flagged": round(float(np.mean(flagged)), 8)}


def main() -> int:
    started = time.time()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    rec = registry.achieved(TEACHER) or {}
    split = rec["split"]
    floor = rec["esa_scores"]["plain_zero_order_hold"]["floor_chosen_on_validation"]
    prep = T.prepare(rec["resample"], split["test_from"], split["validation_from"], hold="last")
    z = prep.zscore(prep.raw_plain).astype(np.float64)
    cut, vstart, sigma = prep.cut, prep.vstart, prep.sigma
    w = json.loads((REPO / registry.ARTIFACT_DIRS[STUDENT] / "weights.json").read_text())
    W, b = np.asarray(w["W"], dtype=np.float64), np.asarray(w["b"], dtype=np.float64)
    cfg = telemanom_threshold.TelemanomConfig(min_error_value=floor)

    # The validation months, as one contiguous stretch: forecasts from SEQ_LEN on, padded as the test period's are.
    v_rows = np.arange(vstart + T.SEQ_LEN, cut)
    err_v = np.abs(_padded(forecast(W, b, features(z, v_rows))) - z[vstart:cut]) * sigma
    es_v = telemanom_threshold.smooth_errors(err_v, cfg)
    v_times = prep.df.index[vstart:cut]
    age = (v_times[-1] - v_times).total_seconds().to_numpy()
    weights = np.power(2.0, -age / HALF_LIFE_S)

    # The test period, exactly as the student's trainer scores it.
    times = prep.df.index[cut:]
    st = prep.labels_all["StartTime"].dt.tz_localize(None)
    en = prep.labels_all["EndTime"].dt.tz_localize(None)
    labels = prep.labels_all[(st >= times[0]) & (en <= times[-1])]
    full = (times[0], times[-1])
    rows = np.arange(cut + T.SEQ_LEN, len(z))
    err = np.abs(_padded(forecast(W, b, features(z, rows))) - z[cut:]) * sigma
    es = telemanom_threshold.smooth_errors(err, cfg)
    vals = prep.raw_plain[cut:]

    _, ndt = telemanom_threshold.flag(err, vals, cfg)
    out = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
           "model": STUDENT, "floor": floor, "test_steps": int(len(times)), "validation_steps": int(len(v_times)),
           "arms": {"ndt": scores(times, ndt.astype(int), labels, full)}}
    # The student's own record of the same pipeline (models/metrics.json, esa_scores.student): the NDT arm must reproduce it.
    taught = ((registry.achieved(STUDENT) or {}).get("esa_scores") or {}).get("student", {}).get("corrected_event_wise_f0_5")
    out["arms"]["ndt"]["matches_the_students_record"] = (
        taught is not None and abs(out["arms"]["ndt"]["anomalies"]["f_0.50"] - taught) < 1e-5)
    if not out["arms"]["ndt"]["matches_the_students_record"]:
        raise SystemExit(f"the NDT arm gives {out['arms']['ndt']['anomalies']['f_0.50']}; the student's record says {taught}")
    for alpha in ALPHAS:
        pot = [pot_threshold(es_v[:, c], alpha) for c in range(es_v.shape[1])]
        con = [weighted_quantile(es_v[:, c], weights, 1 - alpha) for c in range(es_v.shape[1])]
        for name, thr in (("pot", [t for t, _ in pot]), ("conformal", con)):
            flagged = (es > np.asarray(thr)[None, :]).any(axis=1).astype(int)
            arm = scores(times, flagged, labels, full)
            arm["thresholds"] = [round(t, 9) for t in thr]
            if name == "pot":
                arm["fits"] = [how for _, how in pot]
            out["arms"][f"{name}_alpha_{alpha:g}"] = arm
    try:
        import measure_satellite_vus_pr as V
        # As scripts/measure_satellite_vus_pr.py builds them: the score, the Anomaly events inside the test half, the window.
        score = V.channel_max_z(es, es_v)
        sel = labels[labels["Category"] == "Anomaly"]
        steps = T.label_mask(times, sel, "30s").astype(int)
        window = V.find_length_rank1(prep.raw_plain[:, 0])
        out["vus"] = {"window": int(window), "anomalous_steps": int(steps.sum()), **V.vus(steps, score, window)}
    except Exception as exc:  # recorded, not hidden: VUS-PR is the one figure here that does not depend on a threshold
        out["vus"] = {"error": f"{type(exc).__name__}: {exc}"}
    out["wall_clock_s"] = round(time.time() - started, 1)
    RECORD.write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["arms"].items():
        print(f"  {k:24s} F0.5 {v['anomalies']['f_0.50']:.4f} (tp {v['anomalies'].get('tp')} fp {v['anomalies'].get('fp')} "
              f"fn {v['anomalies'].get('fn')}) all events {v['all_events']['f_0.50']:.4f} flagged {v['share_of_test_steps_flagged']}")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
