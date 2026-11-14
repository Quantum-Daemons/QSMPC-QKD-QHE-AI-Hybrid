#!/usr/bin/env python3
"""VUS-PR beside the satellite forecaster's event-wise F0.5 (Round 86, the owner's choice of 2026-11-14).

The page scores the published Telemanom forecaster (`satellite_comms.telemanom`) on ESA-ADB Mission1's lightweight
channels 41-46 with ESA-ADB's corrected event-wise F0.5 at Telemanom's thresholds: a figure for the operating point.
TSB-AD (Liu and Paparrizos, NeurIPS 2024 Datasets and Benchmarks) recommends VUS-PR, a threshold-free figure for how
well the scores rank the anomalous steps, and this computes it with TSB-AD 1.5's own code
(`TSB_AD.evaluation.basic_metrics.generate_curve`, version 'opt', 250 thresholds spaced by rank). Everything below was
fixed before any VUS-PR figure was seen:

  THE SCORE   the forecaster's one-step absolute errors in raw units, as the F0.5's thresholding reads them, over the
              test half (2007-2013 on the 30 s zero-order hold), its first 250 rows taking the period's first forecast
              as ESA-ADB pads them; each channel's errors smoothed as Telemanom smooths them (EWMA, span 105,
              `qsh.models.telemanom_threshold.smooth_errors`); each channel z-scored by its smoothed errors' mean and
              standard deviation over the validation months (2006-10 to 2006-12, smoothed the same way); the step's
              score the largest of the six, as a step is flagged when any channel flags it.
  THE LABELS  the steps the test half's Anomaly events of channels 41-46 overlap, those that start and end inside it,
              as ESA-ADB's scoring selects them: the F0.5's own events.
  THE WINDOW  TSB-AD's rule for multivariate series (`benchmark_exp/Run_Detector_M.py`): `find_length_rank` on the
              first channel, its first 20,000 rows. That function needs statsmodels, which is not installed here; it is
              reimplemented below with the same autocorrelation, computed by FFT as statsmodels' `acf(fft=True)` does,
              and `tests/test_r86_satellite_vus_pr.py` checks it against the direct sum.
  SENSITIVITY W = 0 (no buffer), 100 (`get_metrics`' default) and twice the window, and the window with 1,000
              thresholds instead of 250.
  NO SKILL    the same score permuted at random, and a Gaussian random walk, five draws each at the window, median and
              range.
  THE GRAPH   the forecasts come from the PyTorch weights the published ONNX graph was exported from; the local graph
              must hash to the pinned Hub file (`models/hub_file_hashes.json`), and it is run on 2,048 test windows to
              show it gives the same forecasts.

Writes docs/measurements/satellite-vus-pr-2026-11-14.json and web/public/data/satellite/vus_pr.json (aggregate figures).

Usage: QSH_DATA_ROOT=<data root> python -I scripts/measure_satellite_vus_pr.py --artifacts <models/satellite_telemanom>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
RECORD = REPO / "docs" / "measurements" / "satellite-vus-pr-2026-11-14.json"
CARD = REPO / "web" / "public" / "data" / "satellite" / "vus_pr.json"
KEY = "satellite_comms.telemanom"
THRESHOLDS = 250
FINE_THRESHOLDS = 1000
DRAWS = 5
AGREEMENT_WINDOWS = 2048


def acf_fft(x: np.ndarray, nlags: int) -> np.ndarray:
    """The autocorrelation statsmodels' `acf(x, nlags, fft=True)` returns with its defaults: the mean removed, the
    autocovariance summed by FFT over a zero-padded length and divided by n at every lag (not n - k), then normalised by
    lag 0."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    n = len(x)
    nfft = 1 << int(math.ceil(math.log2(2 * n - 1)))
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[: nlags + 1] / n
    return acov / acov[0]


def find_length_rank1(series: np.ndarray) -> int:
    """TSB-AD 1.5's `find_length_rank(data, rank=1)` (TSB_AD/utils/slidingWindows.py) on one series: the lag of the
    highest local maximum of the autocorrelation over lags 3 to 403 of the first 20,000 values, or 125 when there is none
    or it lies below 3 or above 300 (the function's own fall-back, which it also takes on any error)."""
    from scipy.signal import argrelextrema

    data = np.asarray(series, dtype=np.float64).squeeze()
    data = data[: min(20000, len(data))]
    base = 3
    auto_corr = acf_fft(data, 400)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    if len(local_max) == 0:
        return 125
    best = local_max[np.argsort([auto_corr[i] for i in local_max])[::-1][0]]
    if best < 3 or best > 300:
        return 125
    return int(best + base)


def channel_max_z(es_test: np.ndarray, es_val: np.ndarray) -> np.ndarray:
    """Each channel's smoothed test errors z-scored by its validation mean and standard deviation, then the largest."""
    mu, sd = es_val.mean(axis=0), es_val.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return ((es_test - mu) / sd).max(axis=1)


def vus(labels: np.ndarray, score: np.ndarray, window: int, thresholds: int = THRESHOLDS) -> dict:
    """VUS-PR and VUS-ROC from TSB-AD 1.5's `generate_curve` (version 'opt'), the call `get_metrics` makes."""
    from TSB_AD.evaluation.basic_metrics import generate_curve

    out = generate_curve(labels.astype(int), np.asarray(score, dtype=np.float64), int(window), "opt", int(thresholds))
    return {"vus_roc": float(out[-2]), "vus_pr": float(out[-1])}


def no_skill(labels: np.ndarray, score: np.ndarray, window: int, draws: int, seed: int = 2026) -> dict:
    """VUS-PR of the score permuted at random and of a Gaussian random walk, `draws` each, at the same settings."""
    rng = np.random.default_rng(seed)
    perm = [vus(labels, rng.permutation(score), window)["vus_pr"] for _ in range(draws)]
    walk = [vus(labels, np.cumsum(rng.normal(size=len(score))), window)["vus_pr"] for _ in range(draws)]

    def summ(v):
        return {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)), "draws": len(v)}
    return {"permuted": summ(perm), "random_walk": summ(walk)}


def _load_training_module():
    spec = importlib.util.spec_from_file_location("satellite_telemanom", REPO / "scripts" / "train" / "satellite_telemanom.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("satellite_telemanom", mod)
    spec.loader.exec_module(mod)
    return mod


def rounded(x, digits: int = 6):
    if isinstance(x, float):
        return float(f"{x:.{digits}g}")
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v, digits) for v in x]
    return x


def main() -> int:
    import onnxruntime as ort
    import pandas as pd
    from safetensors.torch import load_file
    from sklearn.metrics import average_precision_score

    from qsh.models import registry, telemanom_threshold

    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("--artifacts", type=Path, required=True, help="the forecaster's folder: model.safetensors, manifest.json, onnx/model.onnx")
    ap.add_argument("--draws", type=int, default=DRAWS)
    a = ap.parse_args()
    t0 = time.time()
    tm = _load_training_module()

    # ---- the published graph, and the weights it was exported from
    spec = registry.get(KEY)
    pinned = json.loads((REPO / "models" / "hub_file_hashes.json").read_text())[KEY]
    graph = a.artifacts / "onnx" / "model.onnx"
    graph_sha = hashlib.sha256(graph.read_bytes()).hexdigest()
    if graph_sha != pinned["files"]["onnx/model.onnx"] or pinned["revision"] != spec.hf_revision:
        raise SystemExit(f"{graph} is not the pinned published graph of {KEY}")
    manifest = json.loads((a.artifacts / "manifest.json").read_text())

    prep = tm.prepare("30s", tm.ESA_ADB_SPLIT, tm.ESA_ADB_VALIDATION, hold="last")
    z = prep.zscore(prep.raw_plain)
    model = tm.TelemanomLSTM.build(int(z.shape[1]), horizon=int(manifest["horizon"]))
    model.load_state_dict(load_file(str(a.artifacts / "model.safetensors")))
    cut, vstart = prep.cut, prep.vstart

    # The graph against the weights on test windows drawn once.
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(np.arange(cut, len(z) - tm.SEQ_LEN), AGREEMENT_WINDOWS, replace=False))
    windows = np.stack([z[s:s + tm.SEQ_LEN] for s in starts]).astype(np.float32)
    sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: windows})[0]
    import torch

    model.eval().cpu()
    with torch.no_grad():
        torch_out = model(torch.from_numpy(windows))[:, 0, :].numpy()
    graph_vs_weights = float(np.abs(onnx_out - torch_out).max())

    # ---- the score
    v_err = np.abs(tm._padded_forecast(model, z, vstart, cut) - z[vstart:cut]) * prep.sigma
    t_err = np.abs(tm._padded_forecast(model, z, cut, len(z)) - z[cut:]) * prep.sigma
    es_v = telemanom_threshold.smooth_errors(v_err)
    es_t = telemanom_threshold.smooth_errors(t_err)
    score = channel_max_z(es_t, es_v)

    # ---- the labels: the test half's Anomaly events that start and end inside it
    t_index = prep.df.index[cut:]
    la = prep.labels_all
    st = la["StartTime"].dt.tz_localize(None) if la["StartTime"].dt.tz is not None else la["StartTime"]
    en = la["EndTime"].dt.tz_localize(None) if la["EndTime"].dt.tz is not None else la["EndTime"]
    lo, hi = pd.Timestamp(t_index[0]).tz_localize(None), pd.Timestamp(t_index[-1]).tz_localize(None)
    sel = la[(st >= lo) & (en <= hi) & (la["Category"] == "Anomaly")]
    labels = tm.label_mask(t_index, sel, "30s").astype(int)
    events = int(sel["ID"].nunique())

    # ---- the window, then VUS-PR
    window = find_length_rank1(prep.raw_plain[:, 0])
    main_fig = vus(labels, score, window)
    sensitivity = {"window_0": vus(labels, score, 0)["vus_pr"], "window_100": vus(labels, score, 100)["vus_pr"],
                   f"window_{2 * window}": vus(labels, score, 2 * window)["vus_pr"],
                   f"window_{window}_thresholds_{FINE_THRESHOLDS}": vus(labels, score, window, FINE_THRESHOLDS)["vus_pr"]}
    baselines = no_skill(labels, score, window, a.draws)
    point_ap = float(average_precision_score(labels, score))
    f05 = (registry.achieved(KEY) or {}).get("corrected_event_wise_f0_5")

    from importlib.metadata import version as _v

    import scipy
    import sklearn

    env = {"tsb_ad": _v("TSB-AD"), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
           "onnxruntime": ort.__version__, "torch": torch.__version__}
    record = {
        "_about": ("VUS-PR of the published satellite forecaster on ESA-ADB Mission1 lightweight channels 41-46, test half, "
                   "beside its event-wise F0.5 (Round 86). Written by scripts/measure_satellite_vus_pr.py; aggregate figures."),
        "model": KEY, "hf_revision": spec.hf_revision, "graph_sha256": graph_sha, "graph_vs_weights_max_abs": graph_vs_weights,
        "agreement_windows": AGREEMENT_WINDOWS,
        "series": {"grid": "30 s, zero-order hold", "test_range": [str(t_index[0]), str(t_index[-1])],
                   "steps": int(len(labels)), "anomalous_steps": int(labels.sum()), "anomalous_share": float(labels.mean()),
                   "events": events, "warm_up_rows": tm.SEQ_LEN,
                   "warm_up_rule": "the first 250 rows of each period take the period's first forecast, as ESA-ADB pads them"},
        "score": ("max over channels 41-46 of the Telemanom-smoothed one-step error (EWMA span 105), z-scored by the "
                  "channel's smoothed validation errors (2006-10-01 to 2006-12-31)"),
        "window": {"samples": window, "minutes_each_side": window / 2 * 0.5,
                   "rule": "TSB-AD find_length_rank(data[:, 0], rank=1) on channel 41's first 20,000 rows, numpy's FFT for statsmodels' acf"},
        "thresholds": THRESHOLDS, "version": "opt",
        "vus_pr": main_fig["vus_pr"], "vus_roc": main_fig["vus_roc"], "point_wise_average_precision": point_ap,
        "sensitivity": sensitivity, "no_skill": baselines,
        "event_wise_f0_5_corrected": f05,
        "comparator": "none: no published VUS-PR for ESA-ADB Mission1 channels 41-46 was found on 2026-11-14",
        "environment": env, "wall_clock_s": time.time() - t0,
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(rounded(record), indent=1) + "\n")
    card = {k: record[k] for k in ("model", "hf_revision", "series", "score", "window", "thresholds", "version", "vus_pr",
                                   "vus_roc", "point_wise_average_precision", "sensitivity", "no_skill",
                                   "event_wise_f0_5_corrected", "comparator")}
    card["_about"] = "What the satellite page's VUS-PR line shows; written by scripts/measure_satellite_vus_pr.py."
    card["tsb_ad"] = env["tsb_ad"]
    CARD.write_text(json.dumps(rounded(card, 4), indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("vus_pr", "vus_roc", "point_wise_average_precision", "sensitivity", "no_skill")}, indent=1))
    print(f"wrote {RECORD.relative_to(REPO)} and {CARD.relative_to(REPO)} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
