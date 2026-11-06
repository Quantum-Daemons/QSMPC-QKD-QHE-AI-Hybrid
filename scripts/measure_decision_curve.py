"""Finance's decision curve on the page's 8,192 rows, set beside the same curve over the whole test window.

The finance page draws net benefit across threshold probabilities from the published evaluation sample, which is
enriched to 2 percent laundering by drawing each class of the test window on its own (`prep_model_inputs.py`). Drawn
that way the true- and false-positive rates are unbiased and the rate is not, so the page puts the window's own rate
back (Pfeiffer and Gail, Biometrical Journal 62(3):764-776, 2020, eq. 11):

    NB(t) = mu * TPR(t) - (1 - mu) * FPR(t) * t / (1 - t)

This measures whether that holds here. The published graph and its record's Platt map are run over all 1,523,504 rows
of the window, which gives the exact curve, and the sample's estimate and pointwise interval are set beside it at every
threshold the page draws. It also records what weighting by the sample's own rate would have read, which is the error
the estimator exists to avoid, and the same window curve on the graph's uncalibrated output.

Needs the AMLworld corpus (`qsh.models.datasets`, `$QSH_DATA_ROOT`) and the graph under `models/finance_aml_gbm/onnx/`,
which it checks against the Hub's pinned revision by SHA-256 before anything is measured.

    .venv/bin/python scripts/measure_decision_curve.py --out docs/measurements/finance-decision-curve-2026-11-06.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

#: The graph the page downloads, pinned by the Hub's LFS digest at the record's revision (read 2026-11-06).
GRAPH_SHA256 = "54511ef70652acb02be201cfed309ff1f95e03c55e9f2aa2b24e2b35a9c8996f"


def _curve(y: np.ndarray, risk: np.ndarray, grid: list[float], mu: float, z: float) -> list[dict]:
    from qsh.models import metrics

    rows = []
    for t in grid:
        nb, se = metrics.net_benefit_at_prevalence(y, risk, t, mu)
        rows.append({"t": t, "net_benefit": nb, "se": se, "lo": nb - z * se, "hi": nb + z * se})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import onnxruntime as ort

    from qsh.config import get_config
    from qsh.models import metrics, registry
    from qsh.models.calibration import Platt
    from train.finance_aml import causal_account_features, graph_features, load_sorted, temporal_windows  # type: ignore

    cfg = get_config().decision_curve
    entry = cfg.curves["finance_aml"]
    key = entry["model_key"]
    grid = metrics.decision_grid(entry["thresholds"], cfg.grid_points)
    rec = registry.achieved(key)
    cal = rec["calibration"]
    platt = Platt(a=cal["a"], b=cal["b"], n_calibration=cal["n_calibration"],
                  positives_calibration=cal["positives_calibration"], on_logit=cal["on_logit"],
                  logit_clip_eps=cal["logit_clip_eps"])

    graph = REPO / registry.ARTIFACT_DIRS[key] / "onnx" / "model.onnx"
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    if digest != GRAPH_SHA256:
        raise SystemExit(f"{graph} is not the graph the page downloads: sha256 {digest}")
    sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def raw_p(x: np.ndarray) -> np.ndarray:
        out = sess.run(None, {name: x.astype(np.float32)})
        proba = next((o for o in out if getattr(o, "ndim", 0) == 2 and o.shape[1] > 1), out[0])
        return np.asarray(proba, dtype=np.float64)[:, 1]

    # ---- the whole window, exactly as prep_model_inputs.py and the trainer compute it ------------------------------
    t0 = time.time()
    df = load_sorted()
    _, _, te, tr = temporal_windows(df)
    x_te = graph_features(te, fit_on=tr, causal=causal_account_features(df))
    y_te = te["label"].to_numpy().astype(int)
    fixtures = np.load(REPO / registry.ARTIFACT_DIRS[key] / "fixtures.npy")
    drawn = np.random.default_rng(42).choice(len(x_te), size=len(fixtures), replace=False)
    if not np.array_equal(x_te[drawn], fixtures):
        raise SystemExit("the window's features differ from the 512 the trainer's export gate saved")
    feature_s = time.time() - t0
    p_window = raw_p(x_te)
    auprc = metrics.auprc(y_te, p_window)
    # prep_model_inputs.py's tolerance for the same check. The record is the trainer's booster; the graph scores
    # float32 inputs, and measured here they differ by 2.9e-6.
    if abs(auprc - rec["auprc"]) > 1e-3:
        raise SystemExit(f"the graph scores AUPRC {auprc:.8f} over the window against the recorded {rec['auprc']}")
    events_w, rows_w = int(y_te.sum()), int(len(y_te))
    corp = cal["corp_calibrated"]
    if (events_w, rows_w) != (corp["positives"], corp["n"]):
        raise SystemExit(f"the window holds {events_w} of {rows_w}, the record {corp['positives']} of {corp['n']}")
    mu = events_w / rows_w
    risk_window = platt.apply(p_window)

    # ---- the page's sample: the published file, read as the browser reads it ---------------------------------------
    sample_path = REPO / registry.ARTIFACT_DIRS[key] / "eval_sample.json"
    sample = json.loads(sample_path.read_text())
    blob = sample_path.read_bytes()
    x_s = np.asarray(sample["samples"], dtype=np.float32)
    y_s = np.asarray(sample["labels"], dtype=int)
    risk_s = platt.apply(raw_p(x_s))
    rate_s = float(y_s.mean())

    z = cfg.interval_z
    window = [{"t": t, "net_benefit": metrics.net_benefit(y_te, risk_window, t),
               "net_benefit_uncalibrated": metrics.net_benefit(y_te, p_window, t),
               "treat_all": mu - (1 - mu) * t / (1 - t)} for t in grid]
    page = _curve(y_s, risk_s, grid, mu, z)
    naive = [metrics.net_benefit(y_s, risk_s, t) for t in grid]
    inside = [w["t"] for w, s in zip(window, page) if s["lo"] <= w["net_benefit"] <= s["hi"]]
    worst = max(abs(s["net_benefit"] - w["net_benefit"]) / s["se"] for w, s in zip(window, page) if s["se"] > 0)
    t_med = float(np.median(entry["thresholds"]))
    k_med = grid.index(t_med)
    thr = float(sample["expected"]["decision_threshold"])

    record = {
        "machine": {"cpu": platform.processor() or platform.machine(), "date": time.strftime("%Y-%m-%d"),
                    "feature_seconds": round(feature_s, 1)},
        "versions": {"onnxruntime": ort.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "inputs": {
            "model_key": key, "hf_revision": rec["hf_revision"], "graph_sha256": digest,
            "eval_sample_git_blob": hashlib.sha1(b"blob %d\0" % len(blob) + blob).hexdigest(),
            "calibration": {k: cal[k] for k in ("method", "a", "b", "on_logit", "logit_clip_eps")},
            "thresholds": entry["thresholds"], "grid_points": cfg.grid_points, "interval_z": z,
            "window": {"rows": rows_w, "events": events_w, "prevalence": mu, "auprc_graph": auprc,
                       "auprc_recorded": rec["auprc"]},
            "sample": {"rows": int(len(y_s)), "events": int(y_s.sum()), "rate": rate_s},
        },
        "results": {
            "window": window,
            "sample": [{**s, "net_benefit_at_sample_rate": nv} for s, nv in zip(page, naive)],
            "window_inside_sample_interval": {"count": len(inside), "of": len(grid)},
            "largest_gap_in_standard_errors": worst,
            "at_median_threshold": {"t": t_med, "window": window[k_med]["net_benefit"],
                                    "sample": page[k_med]["net_benefit"], "lo": page[k_med]["lo"],
                                    "hi": page[k_med]["hi"], "at_sample_rate": naive[k_med]},
            "operating_threshold": {"raw": thr, "risk": float(platt.apply(np.array([thr]))[0])},
        },
    }
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"window: {events_w} of {rows_w} ({mu:.6%}), AUPRC {auprc:.6f}; sample {int(y_s.sum())} of {len(y_s)}")
    print(f"window inside the sample's {cfg.interval_coverage:.0%} interval at {len(inside)} of {len(grid)} thresholds;"
          f" largest gap {worst:.2f} standard errors")
    m = record["results"]["at_median_threshold"]
    print(f"t = {t_med}: window {m['window']:+.6f}, sample {m['sample']:+.6f} [{m['lo']:+.6f}, {m['hi']:+.6f}], "
          f"at the sample's rate {m['at_sample_rate']:+.6f}")
    print(f"operating point {thr} raw is a calibrated risk of {record['results']['operating_threshold']['risk']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
