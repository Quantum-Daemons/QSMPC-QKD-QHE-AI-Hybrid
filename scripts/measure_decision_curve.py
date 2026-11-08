"""Finance's decision curves on the page's 8,192 rows, set beside the same curves over the whole test window.

The finance page draws net benefit across threshold probabilities for two published models: the teacher
(`finance_aml.gbm`) from its evaluation sample, and the CKKS student (`finance_aml.student`) from its decrypted logits
on the same 8,192 rows. Both samples are enriched to 2 percent laundering by drawing each class of the test window on
its own (`prep_model_inputs.py`). Drawn that way the true- and false-positive rates are unbiased and the rate is not, so
the page puts the window's own rate back (Pfeiffer and Gail, Biometrical Journal 62(3):764-776, 2020, eq. 11):

    NB(t) = mu * TPR(t) - (1 - mu) * FPR(t) * t / (1 - t)

This measures whether that holds here. The model and its record's Platt map are run over all 1,523,504 rows of the
window, which gives the exact curve, and the sample's estimate and pointwise interval are set beside it at every
threshold the page draws. It also records what weighting by the sample's own rate would have read, which is the error
the estimator exists to avoid, and the window curve on the model's uncalibrated output. For the student, whose record
measures isotonic regression beside its Platt map, the window curve under an isotonic map fitted on the validation
window as well, so the choice of map can be read off the curve rather than off the scores.

Needs the AMLworld corpus (`qsh.models.datasets`, `$QSH_DATA_ROOT`) and the published files, which it checks against the
Hub's pinned revision by SHA-256 before anything is measured.

    .venv/bin/python scripts/measure_decision_curve.py --key finance_aml.gbm \
        --out docs/measurements/finance-decision-curve-2026-11-06.json
    .venv/bin/python scripts/measure_decision_curve.py --key finance_aml.student \
        --out docs/measurements/finance-student-decision-curve-2026-11-06.json
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
#: The student's weights as the Hub serves them at its record's revision (read 2026-11-06). The page's copy,
#: `web/public/data/finance/student_weights.json`, is the same bytes (`build_heldout_payloads.py`).
#: Re-published 2026-11-08 with a rewritten `note` only: the weights, bias and scaler are those of sha256 c47afeadcc8816ce, which the 2026-11-06 measurements used.
STUDENT_WEIGHTS_SHA256 = "d8823c8961bbc98699223bfa446cc3c5763701d2f363f8be23ee7851d1a420b2"

TEACHER, STUDENT = "finance_aml.gbm", "finance_aml.student"


def _curve(y: np.ndarray, risk: np.ndarray, grid: list[float], mu: float, z: float) -> list[dict]:
    from qsh.models import metrics

    rows = []
    for t in grid:
        nb, se = metrics.net_benefit_at_prevalence(y, risk, t, mu)
        rows.append({"t": t, "net_benefit": nb, "se": se, "lo": nb - z * se, "hi": nb + z * se})
    return rows


def _git_blob(blob: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(blob) + blob).hexdigest()


def _teacher(key: str, rec: dict, x_te: np.ndarray, y_te: np.ndarray):
    """The published graph's P(laundering), checked against the trainer's export fixtures and the record's AUPRC."""
    import onnxruntime as ort

    from qsh.models import metrics, registry

    graph = REPO / registry.ARTIFACT_DIRS[key] / "onnx" / "model.onnx"
    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    if digest != GRAPH_SHA256:
        raise SystemExit(f"{graph} is not the graph the page downloads: sha256 {digest}")
    sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def score(x: np.ndarray) -> np.ndarray:
        out = sess.run(None, {name: x.astype(np.float32)})
        proba = next((o for o in out if getattr(o, "ndim", 0) == 2 and o.shape[1] > 1), out[0])
        return np.asarray(proba, dtype=np.float64)[:, 1]

    fixtures = np.load(REPO / registry.ARTIFACT_DIRS[key] / "fixtures.npy")
    drawn = np.random.default_rng(42).choice(len(x_te), size=len(fixtures), replace=False)
    if not np.array_equal(x_te[drawn], fixtures):
        raise SystemExit("the window's features differ from the 512 the trainer's export gate saved")
    s_window = score(x_te)
    auprc = metrics.auprc(y_te, s_window)
    # prep_model_inputs.py's tolerance for the same check. The record is the trainer's booster; the graph scores
    # float32 inputs, and measured here they differ by 2.9e-6.
    if abs(auprc - rec["auprc"]) > 1e-3:
        raise SystemExit(f"the graph scores AUPRC {auprc:.8f} over the window against the recorded {rec['auprc']}")

    sample_path = REPO / registry.ARTIFACT_DIRS[key] / "eval_sample.json"
    sample = json.loads(sample_path.read_text())
    x_s = np.asarray(sample["samples"], dtype=np.float32)
    return {
        "s_window": s_window, "score": score, "x_s": x_s, "y_s": np.asarray(sample["labels"], dtype=int),
        "sample_path": sample_path, "operating_raw": float(sample["expected"]["decision_threshold"]),
        "inputs": {"graph_sha256": digest},
        "window": {"auprc_graph": auprc, "auprc_recorded": rec["auprc"]},
    }


def _student(key: str, rec: dict, x_te: np.ndarray, y_te: np.ndarray):
    """The published student's P(laundering), sigmoid of its decision logit, checked against the record's minority-class
    F1 at its operating point. Its record's map is fitted on this probability (`student_calibration`)."""
    from qsh.models import metrics, registry

    path = REPO / registry.ARTIFACT_DIRS[key] / "weights.json"
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != STUDENT_WEIGHTS_SHA256:
        raise SystemExit(f"{path} is not the published student: sha256 {digest}")
    w = json.loads(blob)
    mean, scale = np.asarray(w["input"]["mean"]), np.asarray(w["input"]["scale"])
    coef, bias = np.asarray(w["W"][0]), float(w["b"][0])

    def score(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(((np.asarray(x, dtype=np.float64) - mean) / scale) @ coef + bias)))

    s_window = score(x_te)
    f1 = metrics.minority_class_f1(y_te, (s_window > 0.5).astype(int))
    if abs(f1 - rec["student_metric"]) > 5e-7:
        raise SystemExit(f"the weights score minority-class F1 {f1:.6f} over the window against the recorded "
                         f"{rec['student_metric']}")
    # The rows the page encrypts: raw graph features, which it z-scores with the weights' own mean and scale.
    sample_path = REPO / "web" / "public" / "data" / "finance" / "student_heldout.json"
    sample = json.loads(sample_path.read_text())
    if hashlib.sha256((REPO / "web/public/data/finance/student_weights.json").read_bytes()).hexdigest() != digest:
        raise SystemExit("the page's copy of the student's weights is not the published file")
    return {
        "s_window": s_window, "score": score, "x_s": np.asarray(sample["samples"], dtype=np.float64),
        "y_s": np.asarray(sample["labels"], dtype=int), "sample_path": sample_path,
        # The encrypted path decides by argmax over [0, logit], so it classifies at logit 0, a probability of 0.5: the
        # trainer folded the tuned threshold into the bias.
        "operating_raw": 0.5,
        "inputs": {"weights_sha256": digest},
        "window": {"minority_class_f1_at_operating_point": f1, "minority_class_f1_recorded": rec["student_metric"]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--key", choices=[TEACHER, STUDENT], default=TEACHER)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from qsh.config import get_config
    from qsh.models import metrics, registry
    from qsh.models.calibration import Platt
    from train.finance_aml import causal_account_features, graph_features, load_sorted, temporal_windows  # type: ignore

    cfg = get_config().decision_curve
    entry = cfg.curves["finance_aml"]
    key = args.key
    if key not in entry["model_keys"]:
        raise SystemExit(f"the config draws no finance curve for {key}: {entry['model_keys']}")
    grid = metrics.decision_grid(entry["thresholds"], cfg.grid_points)
    rec = registry.achieved(key)
    cal = rec["calibration"]
    platt = Platt(a=cal["a"], b=cal["b"], n_calibration=cal["n_calibration"],
                  positives_calibration=cal["positives_calibration"], on_logit=cal["on_logit"],
                  logit_clip_eps=cal["logit_clip_eps"])

    # ---- the whole window, exactly as prep_model_inputs.py and the trainer compute it ------------------------------
    t0 = time.time()
    df = load_sorted()
    fit, val, te, tr = temporal_windows(df)
    cz = causal_account_features(df)
    x_te = graph_features(te, fit_on=tr, causal=cz)
    y_te = te["label"].to_numpy().astype(int)
    feature_s = time.time() - t0
    m = (_teacher if key == TEACHER else _student)(key, rec, x_te, y_te)
    events_w, rows_w = int(y_te.sum()), int(len(y_te))
    corp = cal["corp_calibrated"]
    if (events_w, rows_w) != (corp["positives"], corp["n"]):
        raise SystemExit(f"the window holds {events_w} of {rows_w}, the record {corp['positives']} of {corp['n']}")
    mu = events_w / rows_w
    s_window = m["s_window"]
    risk_window = platt.apply(s_window)
    # The model's own probability read as a risk, without the map.
    uncal_window = s_window

    iso_window = None
    if key == STUDENT:
        from sklearn.isotonic import IsotonicRegression

        x_val = graph_features(val, fit_on=fit, causal=cz)
        y_val = val["label"].to_numpy().astype(int)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(m["score"](x_val), y_val)
        iso_window = iso.predict(s_window)

    # ---- the page's sample: the published file, read as the browser reads it ---------------------------------------
    blob = m["sample_path"].read_bytes()
    y_s = m["y_s"]
    risk_s = platt.apply(m["score"](m["x_s"]))
    rate_s = float(y_s.mean())

    z = cfg.interval_z
    window = []
    for k, t in enumerate(grid):
        row = {"t": t, "net_benefit": metrics.net_benefit(y_te, risk_window, t),
               "net_benefit_uncalibrated": metrics.net_benefit(y_te, uncal_window, t),
               "treat_all": mu - (1 - mu) * t / (1 - t)}
        if iso_window is not None:
            row["net_benefit_isotonic"] = metrics.net_benefit(y_te, iso_window, t)
        window.append(row)
    page = _curve(y_s, risk_s, grid, mu, z)
    naive = [metrics.net_benefit(y_s, risk_s, t) for t in grid]
    inside = [w["t"] for w, s in zip(window, page) if s["lo"] <= w["net_benefit"] <= s["hi"]]
    worst = max(abs(s["net_benefit"] - w["net_benefit"]) / s["se"] for w, s in zip(window, page) if s["se"] > 0)
    t_med = float(np.median(entry["thresholds"]))
    k_med = grid.index(t_med)
    thr = m["operating_raw"]

    import onnxruntime as ort

    record = {
        "machine": {"cpu": platform.processor() or platform.machine(), "date": time.strftime("%Y-%m-%d"),
                    "feature_seconds": round(feature_s, 1)},
        "versions": {"onnxruntime": ort.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "inputs": {
            "model_key": key, "hf_revision": rec["hf_revision"], **m["inputs"],
            ("eval_sample_git_blob" if key == TEACHER else "page_rows_git_blob"): _git_blob(blob),
            "calibration": {k: cal[k] for k in ("method", "a", "b", "on_logit", "logit_clip_eps")},
            "thresholds": entry["thresholds"], "grid_points": cfg.grid_points, "interval_z": z,
            "window": {"rows": rows_w, "events": events_w, "prevalence": mu, **m["window"]},
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
    print(f"{key}: window {events_w} of {rows_w} ({mu:.6%}); sample {int(y_s.sum())} of {len(y_s)}")
    print(f"window inside the sample's {cfg.interval_coverage:.0%} interval at {len(inside)} of {len(grid)} thresholds;"
          f" largest gap {worst:.2f} standard errors")
    med = record["results"]["at_median_threshold"]
    print(f"t = {t_med}: window {med['window']:+.6f}, sample {med['sample']:+.6f} [{med['lo']:+.6f}, {med['hi']:+.6f}], "
          f"at the sample's rate {med['at_sample_rate']:+.6f}")
    if iso_window is not None:
        d = [w["net_benefit_isotonic"] - w["net_benefit"] for w in window]
        print(f"isotonic minus Platt over the window: {min(d):+.6f} to {max(d):+.6f}")
    print(f"operating point {thr} raw is a calibrated risk of {record['results']['operating_threshold']['risk']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
