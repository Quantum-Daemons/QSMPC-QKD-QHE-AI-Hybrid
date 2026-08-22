#!/usr/bin/env python
"""Telemanom-style anomaly detection on the ESA Anomaly Dataset, Mission1 lightweight channels.

    T2 teacher   2 x LSTM(80) forecaster per channel + non-parametric dynamic thresholding
    T1 student   logistic regression over channel x timestep windows, evaluated under real CKKS

WHY THIS DATASET AND NOT SMAP/MSL
    The bundled NASA SMAP/MSL sample is kept for the browser demo, but it is not a defensible
    benchmark: essentially every result on it is reported with point-adjusted F1, and Kim et al.
    (AAAI-22, arXiv:2109.05257) showed that under point adjustment "even a random anomaly score
    can easily turn into a state-of-the-art TAD method". ESA-ADB is real ESA telemetry with
    event-level annotations and a published event-wise protocol.

    That attribution read "TSB-AD (AAAI 2022, arXiv:2109.05257)" until Round 40, five lines above the
    paragraph below documenting this very conflation. `references/REFERENCES.md` states that
    **TSB-AD has no arXiv ID at all** and that "an entry that carries an arXiv ID for TSB-AD is wrong
    by construction"; `src/qsh/models/metrics.py` has carried the quote correctly attributed to Kim
    et al. throughout. Fifth instance in this repository of a fix reaching one of two twins.

    The citation and the number were both WRONG until Round 33. This said TSB-AD "showed a uniform
    random score ranks 26th of 32 detectors", which could not be verified: TSB-AD's abstract describes
    40 detection algorithms, not 32, and identifies VUS-PR as the measure it recommends. The claim had
    propagated to seven places including the README and the technical report, and one of them
    strengthened it into "it beats six real detectors" - an inference no source supports. Kim et al. is
    the verified replacement and is the stronger statement: it is about any random score, not a rank
    in one table.

WHAT TO REPORT
    Event-wise F-beta at beta=0.5, which is what ESA-ADB and Telemanom actually report. Telemanom's
    frequently-quoted 0.71 is F0.5 under an event-overlap rule, not F1, and the two are routinely
    conflated. Point-adjusted F1 is computed too, purely so the page can show how much it inflates.

    The comparator is ESA-ADB **Supplementary Table 9** (anomalies alone), Mission1 lightweight
    channels 41-46, Telemanom-ESA-Pruned: corrected event-wise F0.5 = 0.968. Earlier rounds cited
    Table 2's 0.786, which is a real number correctly attributed but scores ALL events - anomalies
    PLUS annotated rare nominal events - and this project scores anomalies only. So 0.786 was the
    wrong row, and the lenient one. On the *full* Mission1 channel set the same method scores 0.061;
    that collapse is the honest headline and is printed alongside.

    CARE WITH THAT PAIRING, added Round 55. 0.061 is a TABLE 2 figure, so its matched partner is
    Table 2's 0.786 and NOT the 0.968 this paragraph just adopted - Supp. Table 9's own full-channel
    number is 0.039. The channel-scope collapse is a within-table comparison and only means anything
    if both halves come from one table and one event set.

    Two protocol gaps remain open and are recorded rather than closed: ESA-ADB's precision is
    multiplied by the time-level TNR (both values are computed here), and it splits each mission in
    half where this script tests on the last 30 percent - 65 annotated events in its lightweight
    test partition against 3 in the window scored here.

    A trivial baseline runs every time. On ESA-ADB Mission2 lightweight, a windowed isolation forest
    beats the deep detectors outright; a deep model that cannot beat it has not earned its complexity.

Usage:
    python scripts/train/satellite_telemanom.py [--resample 60s] [--epochs 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qsh.models import datasets, distil, export, metrics, train  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402

#: Telemanom's published configuration (KDD'18): 2 LSTM layers of 80 units, sequence length 250,
#: dropout 0.3, one model per channel.
LSTM_UNITS = 80
SEQ_LEN = 250
DROPOUT = 0.3

#: Non-parametric dynamic threshold pruning factor from the paper's best configuration.
PRUNE_P = 0.13

#: Fraction of the timeline held out. Spacecraft telemetry is a time series; a random split would
#: let the model see the future.
TEST_FRACTION = 0.30


class TelemanomLSTM:
    """Thin builder for the forecaster, kept separate so the ONNX export sees a plain module."""

    @staticmethod
    def build(n_channels: int):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(n_channels, LSTM_UNITS, num_layers=2,
                                    batch_first=True, dropout=DROPOUT)
                self.head = nn.Linear(LSTM_UNITS, n_channels)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        return Net()


def dynamic_threshold(errors: np.ndarray, prune_p: float = PRUNE_P) -> float:
    """Telemanom's non-parametric dynamic threshold with pruning.

    Sweep candidate thresholds at mu + z*sigma, pick the z maximising the paper's objective (the
    relative drop in mean and standard deviation per anomalous point removed), then prune sequences
    whose peak error is not at least ``prune_p`` above the next one down. The pruning step is what
    the paper credits for most of the false-positive reduction.
    """
    mu, sigma = float(errors.mean()), float(errors.std())
    if sigma == 0:
        return float(errors.max()) + 1.0
    best_z, best_score = 3.0, -np.inf
    for z in np.arange(1.5, 10.0, 0.25):
        thr = mu + z * sigma
        keep = errors[errors < thr]
        if keep.size == 0 or keep.size == errors.size:
            continue
        d_mu = (mu - keep.mean()) / mu if mu else 0.0
        d_sigma = (sigma - keep.std()) / sigma if sigma else 0.0
        n_anom = int((errors >= thr).sum())
        score = (d_mu + d_sigma) / (n_anom + n_anom ** 2)
        if score > best_score:
            best_z, best_score = z, score
    thr = mu + best_z * sigma
    # Pruning, per Telemanom section 3.2: walk the candidate peaks from largest down and keep
    # going while each successive drop is at least `prune_p` relative. At the first drop smaller
    # than that, the remaining (smaller) candidates are reclassified as nominal, so the threshold
    # sits at the last peak kept.
    #
    # The obvious transcription of this inverts it — returning as soon as a *large* drop is seen,
    # which sets the threshold near the very top of the distribution. That produced 131 flags
    # against 320 labelled anomaly intervals here, i.e. it discarded almost every true event.
    peaks = np.sort(errors[errors >= thr])[::-1]
    if peaks.size > 1:
        kept = peaks.size
        for i in range(peaks.size - 1):
            drop = (peaks[i] - peaks[i + 1]) / max(abs(peaks[i]), 1e-12)
            if drop < prune_p:
                kept = i + 1
                break
        return float(peaks[kept - 1])
    return float(thr)


def windows(x: np.ndarray, seq_len: int, stride: int = 1):
    """Sliding windows -> (n, seq_len, channels) inputs and (n, channels) next-step targets."""
    n = (len(x) - seq_len - 1) // stride + 1
    idx = np.arange(n) * stride
    xs = np.stack([x[i:i + seq_len] for i in idx]).astype(np.float32)
    ys = x[idx + seq_len].astype(np.float32)
    return xs, ys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", default="60s",
                    help="resampling cadence; raw telemetry is ~15M irregular samples per channel")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=10, help="window stride (10 keeps memory sane)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("models/satellite_telemanom"))
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    started = time.time()
    train.seed_everything(args.seed)

    print("loading ESA-ADB Mission1 lightweight channels (41-46) ...")
    raw = datasets.load_esa_channels()
    print(f"  raw: {raw.shape[0]:,} rows x {raw.shape[1]} channels, "
          f"{raw.index[0].date()} -> {raw.index[-1].date()}")
    # Imputation, stated because TRIPOD+AI item 11 requires it and because one half of it was a
    # look-ahead. `ffill` carries the last OBSERVED value forward across a gap, which is the standard
    # choice for telemetry and uses only the past. `bfill` carries a FUTURE value backwards, and it
    # was here until Round 34 - so any leading gap, and any gap `ffill` could not close, was filled
    # with data recorded after it. Dropping the leading rows instead costs a bounded prefix and uses
    # no future information; `ffill` handles every interior gap, so nothing else changes.
    df = raw.resample(args.resample).mean().ffill()
    leading = int(df.isna().any(axis=1).cumprod().sum())     # rows before the first full observation
    if leading:
        df = df.iloc[leading:]
        print(f"  dropped {leading:,} leading row(s) with no prior observation to carry forward "
              f"(previously back-filled from the future)")
    df = df.dropna()
    print(f"  resampled to {args.resample}: {df.shape[0]:,} rows (ffill only, no backward fill)")

    labels = datasets.load_esa_labels()
    mask = np.zeros(len(df), dtype=bool)
    idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index
    for _, row in labels.iterrows():
        mask |= (idx >= row["StartTime"]) & (idx <= row["EndTime"])
    print(f"  {len(labels)} anomaly intervals -> {mask.sum():,} anomalous samples "
          f"({100 * mask.mean():.3f}% of the timeline)")

    values = df.to_numpy(dtype=np.float32)
    cut = int(len(values) * (1.0 - TEST_FRACTION))
    # Fit the z-score on the TRAINING window only. Until Round 34 this computed mu and sigma over the
    # WHOLE series and split afterwards, so the test window contributed to the constants applied to
    # itself - textbook leakage, and on a temporal split it is look-ahead as well. The same repository
    # already gets this right in finance_aml.py (StandardScaler fitted on the fit window) and in
    # qcnn_variational.py (mu/sd from X[:cut]), so this was an oversight, not a policy.
    #
    # The effect is probably small on a long, roughly stationary telemetry series. "Probably small" is
    # not a measurement, and a project whose posture is that every number is attributable cannot leave
    # one contaminated by construction.
    mu, sigma = values[:cut].mean(0), values[:cut].std(0) + 1e-8
    values = (values - mu) / sigma
    print(f"  temporal split at {df.index[cut].date()}: train {cut:,} / test {len(values)-cut:,}"
          f"  (z-score fitted on the train window only)")

    xtr, ytr = windows(values[:cut], SEQ_LEN, args.stride)
    xte, yte = windows(values[cut:], SEQ_LEN, 1)
    print(f"  windows: train {xtr.shape}, test {xte.shape}")

    n_val = max(1, int(0.15 * len(xtr)))
    model = TelemanomLSTM.build(values.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Telemanom LSTM: 2 x LSTM({LSTM_UNITS}), seq {SEQ_LEN}, {n_params:,} parameters")

    tr_dl = DataLoader(TensorDataset(torch.tensor(xtr[:-n_val]), torch.tensor(ytr[:-n_val])),
                       batch_size=256, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.tensor(xtr[-n_val:]), torch.tensor(ytr[-n_val:])),
                       batch_size=512)

    # Forecasting, so the objective is regression error and the selection metric is negative MSE.
    def neg_mse(y_true, y_pred):
        return -float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    result = _train_forecaster(model, tr_dl, va_dl, args, neg_mse)
    print(f"  best val -MSE {result.best_metric:.6f} at epoch {result.best_epoch}")

    from safetensors.torch import load_file

    model.load_state_dict(load_file(result.checkpoint))

    # Prediction error per timestep -> dynamic threshold -> anomaly flags.
    errors = _forecast_errors(model, xte, yte)
    thr = dynamic_threshold(errors)
    pred = (errors >= thr).astype(int)
    truth = mask[cut + SEQ_LEN: cut + SEQ_LEN + len(pred)].astype(int)
    print(f"  dynamic threshold {thr:.4f} -> {pred.sum():,} flagged of {len(pred):,}")

    ev_f05 = metrics.event_wise_fbeta(truth, pred)
    # ESA-ADB's published table reports the CORRECTED score - event precision multiplied by the
    # time-level TNR - so an uncorrected number set beside its baseline compares two different
    # quantities, in the flattering direction. Both are recorded; the corrected one is the one that
    # is comparable to the citation.
    ev_f05_corr = metrics.corrected_event_wise_fbeta(truth, pred)
    ev_counts = metrics.event_wise_counts(truth, pred)
    pa_f1 = metrics.point_adjusted_f1(truth, pred)
    spec = get_spec("satellite_comms.telemanom")

    print("\n  teacher (Telemanom LSTM)")
    print(f"    event-wise F0.5         {ev_f05:.4f}   <- uncorrected")
    print(f"    corrected event-wise    {ev_f05_corr:.4f}   <- what ESA-ADB's table actually reports")
    # The denominator of that F0.5 is EVENTS, and this window holds a handful. Printed next to the
    # score because 320 anomaly intervals over the timeline reduces to 3 inside the scored window,
    # and the score alone gives no hint of which number it was divided by.
    print(f"    computed over           {ev_counts['n_true_events']} labelled event(s): "
          f"tp={ev_counts['tp']} fp={ev_counts['fp']} fn={ev_counts['fn']}")
    if ev_counts["n_true_events"] < 10:
        print(f"      ^ too few events to compare against the published "
              f"{spec.metric.baseline}: one event moves this score by tenths")
    print(f"    point-adjusted F1       {pa_f1:.4f}   <- companion; inflated by construction")
    print(f"    published lightweight   {spec.metric.baseline}  ({spec.metric.baseline_source})")
    # NAMES ITS TABLE, because the line above it does not come from the same one. 0.968 is Supp.
    # Table 9 (anomalies alone); 0.061 is Table 2 (all events). "The same method" was true of the
    # channel scope and false of the event set, and printing them adjacent implied otherwise.
    print("    published full-channel  0.061  <- Table 2, all events, all Mission1 channels")
    print("    published lightweight   0.786  <- Table 2, the SAME row as 0.061; scope is the only difference")

    # Trivial baseline. If this wins, the deep model has not earned its complexity.
    iso_f05 = _isolation_forest_baseline(xte, truth, args.seed)
    print(f"    windowed iForest F0.5   {iso_f05:.4f}   <- trivial baseline, run every time")

    # ---- T1 student -----------------------------------------------------------
    print("\n  distilling to the CKKS-linear student ...")
    flat_te = xte[:, -8:, :].reshape(len(xte), -1)      # last 8 timesteps x 6 channels = 48 features
    # Split so that BOTH halves contain labelled anomaly events. A plain 60/40 cut of a 0.75%-positive
    # timeline can leave the evaluation half with no events at all, and event-wise F-beta is then
    # undefined (nan) rather than zero — which would be reported as a metric if left unchecked.
    split = _split_with_events(truth, target=0.6)
    student = distil.distil_linear(pred[:split], flat_te[:split], seed=args.seed)
    student_pred = student.predict(flat_te[split:])
    student_export = distil.save_student(
        student, Path("models/satellite_telemanom_student"), n_features=int(flat_te.shape[1]),
        teacher_key="satellite_comms.telemanom", student_key="satellite_comms.student")
    print(f"  student persisted: {student_export['n_parameters']} params, "
          f"browser_runnable={student_export['browser_runnable']}")
    fidelity = distil.measure_fidelity(
        teacher_key="satellite_comms.telemanom", student_key="satellite_comms.student",
        teacher_pred=pred[split:], student_pred=student_pred, y_true=truth[split:],
        metric_fn=lambda y, p: metrics.event_wise_fbeta(y, p), metric_name="event_wise_f0_5",
        student_params=int(student.coef_.size + student.intercept_.size), teacher_params=n_params,
        # n_eval counts samples; this metric's denominator is events. On this slice those differ by
        # five orders of magnitude (307,601 samples, 2 events), so record both.
        n_scoring_units=int(metrics.event_wise_counts(truth[split:], pred[split:])["n_true_events"]),
        scoring_unit="labelled anomaly events",
    )
    print(f"    agreement with teacher  {fidelity.agreement:.4f}")
    print(f"    eval half holds {fidelity.n_scoring_units} labelled anomaly event(s) "
          f"in {fidelity.n_eval:,} samples")
    print(f"    event-wise F0.5 t/s     {fidelity.teacher_metric:.4f} / {fidelity.student_metric:.4f}")
    if fidelity.teacher_scored_zero:
        print("      ^ the teacher scored 0 here, so there is no fidelity claim to make: the delta "
              "compares the student against a model that caught nothing on this slice")

    # ---- artefacts ------------------------------------------------------------
    out = Path(result.checkpoint).parent
    report = export.verify_and_quantize(model, xte[:256], out, kind="torch")
    print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB -> int8 {report.int8_bytes/1e6:.2f} MB, "
          f"confident agreement {report.int8_confident_agreement:.4f}")

    (out / "feature_spec.json").write_text(json.dumps({
        "input": {"shape": [1, SEQ_LEN, int(values.shape[1])], "dtype": "float32"},
        "channels": list(datasets.ESA_MISSION1_LIGHTWEIGHT),
        "resample": args.resample,
        "normalisation": {"mean": mu.tolist(), "std": sigma.tolist()},
        "threshold": thr,
    }, indent=2) + "\n")

    train.record_metrics("satellite_comms.telemanom", {
        "event_wise_f0_5": round(float(ev_f05), 6),
        "corrected_event_wise_f0_5": round(float(ev_f05_corr), 6),
        "point_adjusted_f1": round(float(pa_f1), 6),
        "windowed_iforest_event_wise_f0_5": round(float(iso_f05), 6),
        # BOTH FROM ESA-ADB TABLE 2, and the literal is deliberate. `spec.metric.baseline` is 0.968,
        # which is Supplementary Table 9 - anomalies ALONE, with event precision multiplied by the
        # time-level TNR. `published_full_channel_f0_5` is 0.061, which is Table 2 - ALL events
        # excluding communication gaps. Sourcing one from the registry and hardcoding the other
        # paired two different tables and two different event sets under a "lightweight vs full
        # channels" label whose whole point is that only the CHANNEL SCOPE differs.
        #
        # Latent rather than live when found: the shipped `models/metrics.json` still holds the
        # consistent Table 2 pair (0.786 / 0.061) because this trainer has not run since the
        # baseline moved to 0.968 in Round 24. The next run would have written the mixed pair.
        #
        # This is the Round 24 category error reintroduced by a refactor - replacing a literal with
        # a registry read looks like removing duplication, and here it silently swapped which table
        # one half of a matched pair came from. Supp. Table 9's own full-channel partner is 0.039,
        # not 0.061, so the two are not interchangeable in either direction.
        "published_lightweight_f0_5": 0.786,
        "published_full_channel_f0_5": 0.061,
        "n_parameters": n_params,
        "n_anomaly_intervals": int(len(labels)),
        # The one that actually divides the score. n_anomaly_intervals above counts the WHOLE
        # timeline and is not the evaluation's sample size; these are.
        "n_events_scored": int(ev_counts["n_true_events"]),
        "event_wise_tp": int(ev_counts["tp"]),
        "event_wise_fp": int(ev_counts["fp"]),
        "event_wise_fn": int(ev_counts["fn"]),
        "anomaly_sample_rate": round(float(mask.mean()), 8),
        "resample": args.resample,
        "dataset": spec.dataset.name,
        "metric_note": (
            "Event-wise F-beta at beta=0.5 under the overlap rule, which is what ESA-ADB and "
            "Telemanom report. Telemanom's widely-quoted 0.71 is F0.5, not F1. point_adjusted_f1 is "
            "shown only to expose how much point adjustment inflates: Kim et al. (AAAI 2022) show "
            "that under point adjustment even a random anomaly score can reach state-of-the-art. Only "
            "labels whose ESA-ADB "
            "Category is 'Anomaly' count as positives; the 690 annotated rare *nominal* events do not."
        ),
        "baseline_gap_note": (
            "Read n_events_scored before comparing this to the baseline. The scored test window "
            "holds THREE labelled anomaly events, not the 320 that n_anomaly_intervals counts over "
            "the whole timeline - the annotated anomalies sit earlier in the mission, so a temporal "
            "split leaves almost nothing to score. Both recorded values decompose uniquely on three "
            "events: 0.714286 is tp=1 fp=0 fn=2 and 0.454545 is tp=1 fp=1 fn=2, so the 0.26 swing "
            "between two identical-config runs is ONE extra false-alarm run. The detector flags 2 "
            "points out of 441,600. Three protocol gaps to the citation remain open and are stated "
            "rather than closed: ESA-ADB multiplies event precision by the time-level TNR (recorded "
            "here as corrected_event_wise_f0_5), it splits each mission in half where this tests on "
            "the last 30 percent, and its lightweight test partition holds 65 events against 3 "
            "here. Workings: references/satellite-event-count-investigation.md."
        ),
        "sample_size_note": (
            "Read n_events_scored before comparing this to published_lightweight_f0_5. The "
            "denominator of an event-wise metric is EVENTS, and a temporal split of this corpus "
            "leaves very few of them in the scored window: 320 labelled intervals over the whole "
            "timeline reduce to a handful inside the test window, because the annotated anomalies "
            "are concentrated earlier in the mission. The recorded score is therefore a small-sample "
            "figure and one event either way moves it by tenths - it is NOT comparable to a "
            "published benchmark computed over the full annotated set, however similar the two "
            "decimals look. The number is reported as measured rather than reweighted, and the "
            "sample size is reported beside it so the comparison is not made silently."
        ),
        "git_sha": train.git_sha(),
        "export": {"fp32_bytes": report.fp32_bytes, "int8_bytes": report.int8_bytes,
                   "opset": report.opset},
        "wall_clock_s": round(time.time() - started, 1),
    })
    train.record_metrics("satellite_comms.student", fidelity.as_metrics())
    print(f"\n  wrote {out} and models/metrics.json in {time.time()-started:.0f}s")
    return 0


def _train_forecaster(model, tr_dl, va_dl, args, metric_fn):
    """Train the forecaster with an MSE objective (train.train_torch assumes classification)."""
    import torch

    cfg = train.TrainConfig(epochs=args.epochs, batch_size=256, lr=1e-3,
                            patience=6, seed=args.seed)
    dev = train._device()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.MSELoss()
    out_dir = Path.cwd() / "models" / "satellite_telemanom"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.safetensors"

    best, best_epoch, stale, history = -np.inf, -1, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                preds.append(model(xb.to(dev)).cpu().numpy())
                trues.append(yb.numpy())
        m = metric_fn(np.concatenate(trues), np.concatenate(preds))
        history.append({"epoch": epoch, "neg_mse": m})
        print(f"    epoch {epoch:3d}  val -MSE {m:.6f}")
        if m > best + cfg.min_delta:
            best, best_epoch, stale = m, epoch, 0
            train._save_safetensors(model, ckpt)
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    manifest = {"key": "satellite_comms.telemanom", "best_metric": best, "best_epoch": best_epoch,
                "epochs_run": len(history), "git_sha": train.git_sha(), "seed": cfg.seed,
                "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return train.TrainResult(key="satellite_comms.telemanom", best_metric=best,
                             best_epoch=best_epoch, epochs_run=len(history), history=history,
                             checkpoint=str(ckpt), manifest=manifest)


def _forecast_errors(model, xte, yte) -> np.ndarray:
    import torch

    dev = train._device()
    model.eval().to(dev)
    errs = []
    with torch.no_grad():
        for i in range(0, len(xte), 4096):
            xb = torch.tensor(xte[i:i + 4096]).to(dev)
            p = model(xb).cpu().numpy()
            errs.append(np.abs(p - yte[i:i + 4096]).mean(1))
    return np.concatenate(errs)


def _split_with_events(truth: np.ndarray, *, target: float = 0.6) -> int:
    """Split index near ``target`` that leaves at least one labelled event on each side."""
    starts = np.flatnonzero(np.diff(np.concatenate([[0], truth])) == 1)
    if starts.size < 2:
        return int(len(truth) * target)     # cannot satisfy it; caller reports whatever results
    want = int(len(truth) * target)
    # Snap to the event boundary closest to the target that still leaves events on both sides.
    candidates = starts[(starts > starts[0]) & (starts < starts[-1])]
    if candidates.size == 0:
        return int((starts[0] + starts[-1]) // 2)
    return int(candidates[np.argmin(np.abs(candidates - want))])


def _isolation_forest_baseline(xte, truth, seed) -> float:
    """Windowed isolation forest — the trivial detector ESA-ADB shows can beat deep models."""
    from sklearn.ensemble import IsolationForest

    feats = xte[:, -8:, :].reshape(len(xte), -1)
    sub = min(len(feats), 60_000)
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=seed, n_jobs=8)
    iso.fit(feats[:sub])
    pred = (iso.predict(feats) == -1).astype(int)
    return metrics.event_wise_fbeta(truth, pred)


if __name__ == "__main__":
    raise SystemExit(main())
