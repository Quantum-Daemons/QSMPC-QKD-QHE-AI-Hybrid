#!/usr/bin/env python
"""Telemanom-style anomaly detection on the ESA Anomaly Dataset, Mission1 lightweight channels.

    T2 teacher   one 2 x LSTM(80) forecaster over the six channels, multi-output as in ESA-ADB's Telemanom-ESA,
                 with Telemanom's non-parametric dynamic thresholding run per channel (qsh.models.telemanom_threshold)
    T1 student   a linear next-step forecaster evaluated under CKKS, trained from this model by
                 scripts/train/satellite_student_forecaster.py (Round 86; a logistic classifier until then)

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
    frequently-quoted 0.71 is printed in the "F0.5 score" column of Hundman et al. 2018's Table 2
    (arXiv:1802.04431), not as an F1, and the two are routinely conflated. It does not follow from
    that table's own precision and recall: the totals, 87.5% and 80.0%, give F0.5 = 0.859, and SMAP's
    85.5% and 85.5% give 0.855 for any beta. So this project quotes it as printed and compares with
    nothing derived from it. Point-adjusted F1 is computed too, purely so the page can show how much
    it inflates.

    The comparators are ESA-ADB's, Mission1 lightweight channels 41-46, Telemanom-ESA-Pruned, one per
    reading. Anomalies alone: **Supplementary Table 9** (Table 18 of the DMLR version), corrected
    event-wise F0.5 = 0.968, which the headline is set against. All events, anomalies PLUS annotated
    rare nominal events: Table 2's 0.786 (DMLR Table 4). Until Round 86 this project scored anomalies
    only, so Table 2 was then the wrong row, and the lenient one. Since Round 86 the record carries
    both readings, each against its own row. On the *full* Mission1 channel set the same method
    scores 0.061; that collapse is the honest headline and is printed alongside.

    CARE WITH THAT PAIRING, added Round 55. 0.061 is a TABLE 2 figure, so its matched partner is
    Table 2's 0.786 and NOT the 0.968 this paragraph just adopted - Supp. Table 9's own full-channel
    number is 0.039. The channel-scope collapse is a within-table comparison and only means anything
    if both halves come from one table and one event set.

    THE PROTOCOL SINCE ROUND 86 is ESA-ADB's Telemanom-ESA recipe on its own Mission1 split, read from
    kplabs-pl/ESA-ADB at commit 67194d3: the 30 s zero-order hold of its data preparation; a z-score from
    the unlabelled training rows; training only on windows that touch no label of any category, before
    2007-01-01, stopping early on the months from 2006-10-01; a 10-step forecast scored on its first
    step; Telemanom's thresholding per channel (qsh.models.telemanom_threshold); and ESA-ADB's corrected
    event-wise F0.5 (qsh.models.esa_scores), whose precision is multiplied by the time-level TNR. Until
    then this script tested on the last 30 percent by row, which held 3 labelled events, and set one
    threshold that flagged 2 points. The lightweight subsets carry no telecommands (ESA-ADB, section on
    lightweight subsets), so none are missing here.

    TWO ARMS, because one step of ESA-ADB's preparation reads the labels. It writes an Anomaly or Rare
    Event sample that the zero-order hold dropped back onto the grid (restore_annotated below), in the
    test half as in the training half, so a detector scored on that series has been shown where the
    annotated samples are. The replication arm applies it, as ESA-ADB's published numbers do; the
    project's own result is the plain zero-order hold, which a deployed detector could compute. Both
    arms use one trained model: every restored row is a labelled row, which no training window touches.

    A trivial baseline runs every time. On ESA-ADB Mission2 lightweight, a windowed isolation forest
    beats the deep detectors outright; a deep model that cannot beat it has not earned its complexity.

Usage:
    python scripts/train/satellite_telemanom.py [--resample 30s] [--max-epochs 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qsh.models import datasets, export, metrics, telemanom_threshold, train  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402

#: Telemanom's published configuration (KDD'18): 2 LSTM layers of 80 units, sequence length 250,
#: dropout 0.3. Telemanom fits one model per channel; this is ONE multi-output model over channels 41-46, which is
#: ESA-ADB's Telemanom-ESA design (the comment said "one model per channel" until Round 86).
LSTM_UNITS = 80
SEQ_LEN = 250
DROPOUT = 0.3

#: Fraction of the timeline held out when a caller passes no split date. Spacecraft telemetry is a time series; a
#: random split would let the model see the future.
TEST_FRACTION = 0.30

#: ESA-ADB's own Mission1 split since Round 86: its `84_months` dataset trains on the first half of the mission and
#: tests on the second, split at 2007-01-01 (`data/preprocessed/datasets.csv` in kplabs-pl/ESA-ADB), and its
#: experiments stop early on the last three months of the training half (`mission1_experiments.py`,
#: validation_date_split "2006-10-01"). The 70/30 cut by row left 3 labelled anomaly events to score.
ESA_ADB_SPLIT = "2007-01-01"
ESA_ADB_VALIDATION = "2006-10-01"

#: ESA-ADB's Telemanom-ESA recipe for Mission1 lightweight (TimeEval-algorithms/telemanom_esa/algorithm.py defaults
#: and mission1_experiments.py, read at commit 67194d3 on 2026-11-07), adopted in Round 86: a 10-step forecast of all
#: six channels scored on its first step; 1,000 batches of 70 random windows per epoch from the unlabelled training
#: stretches; Adam at Keras's default 1e-3; early stopping on the validation months with patience 20 and min_delta
#: 3e-4, keeping the LAST epoch's weights as ESA-ADB's run does.
HORIZON = 10
ESA_BATCH = 70
ESA_STEPS_PER_EPOCH = 1000
ESA_MAX_EPOCHS = 1000
ESA_PATIENCE = 20
ESA_MIN_DELTA = 3e-4
ESA_LR = 1e-3
#: Telemanom-ESA-Pruned's error floor in raw units (ESA-ADB Supp. Table 8; `scripts/reevaluate.py`: 0.05/7). It lifts
#: plain Telemanom-ESA from 0.090 to 0.968 on the same model, and the paper does not show it was chosen without the test
#: half, so this project reports it as the replication and chooses its own floor on the validation months.
PAPER_FLOOR = 0.007
#: The floors tried on the validation months, in raw units.
FLOOR_GRID = tuple(round(0.0005 * k, 4) for k in range(0, 41))
#: What ESA-ADB's lightweight anomaly score counts as a positive (its Supplementary Table 9).
ANOMALIES = {"Category": ["Anomaly"]}
#: ESA-ADB's Table 2 selection, all events but communication gaps, where Telemanom-ESA-Pruned reads 0.786.
ALL_EVENTS = {"Category": ["Rare Event", "Anomaly"]}
#: ESA-ADB's per-sample annotation codes (notebooks/data-prep/utils.py, AnnotationLabel). Its preparation gives every
#: category other than Anomaly and Rare Event the gap code.
_NOMINAL, _ANOMALY, _RARE, _GAP = 0, 1, 2, 3


class TelemanomLSTM:
    """Thin builder for the forecaster, kept separate so the ONNX export sees a plain module."""

    @staticmethod
    def build(n_channels: int, horizon: int = 1):
        """`horizon` 1 is the published model's form; ESA-ADB's recipe forecasts HORIZON steps (Round 86), with a dropout
        after the recurrent stack as Keras's Dropout layer after the second LSTM applies it."""
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.horizon = horizon
                self.lstm = nn.LSTM(n_channels, LSTM_UNITS, num_layers=2,
                                    batch_first=True, dropout=DROPOUT)
                if horizon > 1:
                    self.drop = nn.Dropout(DROPOUT)
                self.head = nn.Linear(LSTM_UNITS, n_channels * horizon)

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                if self.horizon == 1:
                    return self.head(last)
                return self.head(self.drop(last)).view(-1, self.horizon, n_channels)

        return Net()

    @staticmethod
    def keras_init(net) -> None:
        """Keras's initialisers, which ESA-ADB's Keras model starts from (Round 86).

        Glorot-uniform input kernels, orthogonal recurrent kernels, zero biases with the forget gate's set to 1
        (Keras's unit_forget_bias), and a glorot-uniform, zero-bias dense head. PyTorch's defaults draw every LSTM
        weight and bias from U(-1/sqrt(H), 1/sqrt(H)). Both libraries order the gates input, forget, cell, output.
        """
        import torch.nn as nn

        h = net.lstm.hidden_size
        for name, p in net.lstm.named_parameters():
            if name.startswith("weight_ih"):
                nn.init.xavier_uniform_(p)
            elif name.startswith("weight_hh"):
                nn.init.orthogonal_(p)
            else:
                nn.init.zeros_(p)
                if name.startswith("bias_ih"):
                    p.data[h:2 * h] = 1.0
        nn.init.xavier_uniform_(net.head.weight)
        nn.init.zeros_(net.head.bias)

    @staticmethod
    def step_one(net):
        """The forecast's first step only, which is what ESA-ADB scores and what the published graph serves."""
        import torch.nn as nn

        class StepOne(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x)[:, 0, :]

        return StepOne()


def label_mask(index, labels, resample: str) -> np.ndarray:
    """Which resampled bins an anomaly interval overlaps (Round 86).

    Each bin is labelled at its left edge and covers `resample` from there, so the bin at t overlaps an interval
    [start, end] when t <= end and t + resample > start. The rule until Round 86 marked a bin only when its
    left edge fell INSIDE the interval, so an anomaly shorter than a bin - most of Mission1's, which last 0 to
    18 seconds against a 300 s bin - usually marked nothing and was never scored: ESA-ADB lists 29 anomalies
    in the test half, and that rule counted 11.
    """
    import pandas as pd

    idx = index.tz_localize("UTC") if index.tz is None else index
    width = pd.Timedelta(resample)
    mask = np.zeros(len(idx), dtype=bool)
    for _, row in labels.iterrows():
        mask |= (idx <= row["EndTime"]) & (idx + width > row["StartTime"])
    return mask


def _utc_ns(x) -> np.ndarray:
    """Timestamps, with or without a time zone and at any resolution, as UTC int64 nanoseconds."""
    import pandas as pd

    idx = pd.DatetimeIndex(pd.to_datetime(x))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("ns").asi8.astype(np.int64)


def restore_annotated(values: np.ndarray, grid_index, names, series: dict, labels, resample: str) -> np.ndarray:
    """ESA-ADB's restoration of the annotated samples a zero-order hold drops, applied to `values` in place (Round 86).

    ESA-ADB's Mission1 preparation (notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py, read at commit
    67194d3 on 2026-11-07) labels each raw sample of a channel with its annotation's category, resamples by zero-order
    hold, and then, for every bin [t, t + resample) that holds more than one raw sample and whose LAST sample is
    nominal, writes the last Anomaly or Rare Event sample of the bin onto the grid point t + resample, the point that
    would otherwise carry the nominal sample. Communication gaps are not restored. Without it, an annotated sample that
    is not the last of its bin is absent from the series the forecaster reads.

    The rule reads the LABELS to decide which raw sample represents a bin, so a detector scored on restored data has
    been shown where the annotated samples are; scripts/train/satellite_telemanom.py uses it for its replication arm
    only. `values` is (rows, channels) on `grid_index`, its columns `names`; `series` holds each channel's own samples
    (qsh.models.datasets.load_esa_channel_series); `labels` carries ID, Channel, StartTime, EndTime and Category.
    Returns the mask of the rows it wrote.
    """
    import pandas as pd

    width = int(pd.Timedelta(resample).value)
    grid_ns = _utc_ns(grid_index)
    changed = np.zeros(len(grid_ns), dtype=bool)
    lo_ns, hi_ns = _utc_ns(labels["StartTime"]), _utc_ns(labels["EndTime"])
    for col, name in enumerate(names):
        s = series[name].sort_index()
        t, v = _utc_ns(s.index), s.to_numpy()
        # Each raw sample's code, written in the label file's order as ESA-ADB's loop writes it (inclusive slices).
        code = np.zeros(len(t), dtype=np.uint8)
        rows = np.flatnonzero((labels["Channel"] == name).to_numpy())
        for r in rows:
            cat = labels["Category"].iat[r]
            c = _ANOMALY if cat == "Anomaly" else _RARE if cat == "Rare Event" else _GAP
            code[np.searchsorted(t, lo_ns[r], "left"):np.searchsorted(t, hi_ns[r], "right")] = c
        b = t // width          # pd.Grouper's bins start at midnight; epoch alignment is the same for any width dividing a day
        first = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
        last = np.r_[first[1:], len(t)] - 1
        annotated = (code == _ANOMALY) | (code == _RARE)
        last_annotated = np.maximum.reduceat(np.where(annotated, np.arange(len(t)), -1), first)
        pick = (last > first) & (code[last] == _NOMINAL) & (last_annotated >= 0)
        when = (b[first[pick]] + 1) * width
        k = np.searchsorted(grid_ns, when)
        ok = k < len(grid_ns)
        ok[ok] = grid_ns[k[ok]] == when[ok]
        values[k[ok], col] = v[last_annotated[pick][ok]]
        changed[k[ok]] = True
    return changed


def windows(x: np.ndarray, seq_len: int, stride: int = 1):
    """Sliding windows -> (n, seq_len, channels) inputs and (n, channels) next-step targets."""
    n = (len(x) - seq_len - 1) // stride + 1
    idx = np.arange(n) * stride
    xs = np.stack([x[i:i + seq_len] for i in idx]).astype(np.float32)
    ys = x[idx + seq_len].astype(np.float32)
    return xs, ys


def window_view(x: np.ndarray, seq_len: int):
    """`windows(x, seq_len, 1)` as a read-only strided VIEW of `x`, not a copy (Round 86).

    At 30 s the test half is 7.4 million windows of 250 x 6, 44 GB as `windows` builds them; this is the same
    windows and targets at the cost of `x` itself, and callers copy one batch at a time.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    inputs = sliding_window_view(x[:-1], seq_len, axis=0).transpose(0, 2, 1)
    return inputs, x[seq_len:]


def _stamp(when: str, index):
    """`when` as a timestamp comparable with `index`, which may or may not carry a time zone."""
    import pandas as pd

    ts = pd.Timestamp(when)
    if index.tz is not None and ts.tz is None:
        ts = ts.tz_localize(index.tz)
    elif index.tz is None and ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts


def preprocess(raw, resample: str, split_at: str | None = None, hold: str = "mean"):
    """Resample, impute and z-score the raw channels exactly as training does.

    Returns ``(df, values, cut, mu, sigma)``: the resampled frame, its z-scored values, the index of
    the first test row (the first row at or after `split_at` when one is given, else the last TEST_FRACTION of
    the rows), and the constants fitted on the rows before it. A function rather than inline
    code since Round 71, because `scripts/prep_model_inputs.py` builds the browser's evaluation
    sample from it. Its own copy of these steps had drifted three ways: it back-filled, it fitted the
    z-score on the whole series, and it drew windows from the whole timeline, training rows included,
    while the page called them held-out.
    """
    # Imputation, stated because TRIPOD+AI item 11 requires it and because one half of it was a
    # look-ahead. `ffill` carries the last OBSERVED value forward across a gap, which is the standard
    # choice for telemetry and uses only the past. `bfill` carries a FUTURE value backwards, and it
    # was here until Round 34 - so any leading gap, and any gap `ffill` could not close, was filled
    # with data recorded after it. Dropping the leading rows instead costs a bounded prefix and uses
    # no future information; `ffill` handles every interior gap, so nothing else changes.
    # `hold="last"` is ESA-ADB's zero-order hold (Round 86): the grid point at t takes the last raw sample at or
    # before t (`reindex(method="ffill")` in its data preparation). "mean" averages each bin, which the published
    # 300 s model was trained on.
    if hold == "last":
        import pandas as pd

        grid = pd.date_range(raw.index[0].ceil(resample), raw.index[-1].floor(resample), freq=resample)
        df = raw.reindex(raw.index.union(grid)).ffill().reindex(grid)
    else:
        df = raw.resample(resample).mean().ffill()
    leading = int(df.isna().any(axis=1).cumprod().sum())     # rows before the first full observation
    if leading:
        df = df.iloc[leading:]
        print(f"  dropped {leading:,} leading row(s) with no prior observation to carry forward "
              f"(previously back-filled from the future)")
    df = df.dropna()
    print(f"  resampled to {resample}: {df.shape[0]:,} rows (ffill only, no backward fill)")

    values = df.to_numpy(dtype=np.float32)
    cut = (int(len(values) * (1.0 - TEST_FRACTION)) if split_at is None
           else int(df.index.searchsorted(_stamp(split_at, df.index))))
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
    return df, values, cut, mu, sigma


@dataclass
class Prepared:
    """Mission1's lightweight channels as ESA-ADB's Telemanom-ESA recipe prepares them (Round 86)."""

    df: object                  #: the grid, raw values, as preprocess() returns it
    raw_plain: np.ndarray       #: (rows, channels) float64, the zero-order hold
    raw_restored: np.ndarray    #: the same with restore_annotated applied: the replication arm's series
    restored: np.ndarray        #: rows restore_annotated wrote
    labels: object              #: anomaly label rows
    labels_all: object          #: every label row, with its Category
    mask: np.ndarray            #: rows an anomaly overlaps
    any_label: np.ndarray       #: rows any label overlaps, and every restored row
    cut: int                    #: first test row
    vstart: int                 #: first validation row
    mu: np.ndarray
    sigma: np.ndarray

    def zscore(self, raw: np.ndarray) -> np.ndarray:
        return ((raw - self.mu) / self.sigma).astype(np.float32)


def prepare(resample: str, split_at: str, validation_from: str, *, hold: str = "last") -> Prepared:
    """ESA-ADB's preparation of the six channels, the one copy that training and the evaluation sample both use.

    The zero-order hold on the `resample` grid; the anomaly and any-category label masks; ESA-ADB's restoration of
    annotated samples for the replication arm, every row of which is then marked labelled so that no training window
    and no scaling statistic reads one; and ESA-ADB's scaling, a z-score per channel from the UNLABELLED training rows
    before the validation months (channel.py concatenates the training stretches and excludes validation).
    """
    import pandas as pd

    print("loading ESA-ADB Mission1 lightweight channels (41-46) ...")
    series = datasets.load_esa_channel_series()
    raw = datasets.esa_common_axis(series)
    print(f"  raw: {raw.shape[0]:,} rows x {raw.shape[1]} channels, "
          f"{raw.index[0].date()} -> {raw.index[-1].date()}")
    df, _, cut, _, _ = preprocess(raw, resample, split_at=split_at, hold=hold)
    del raw
    raw_plain = df.to_numpy(dtype=np.float64)

    labels = datasets.load_esa_labels()
    mask = label_mask(df.index, labels, resample)
    types = pd.read_csv(datasets.esa_mission1_dir() / "anomaly_types.csv")[["ID", "Category"]]
    labels_all = datasets.load_esa_labels(anomalies_only=False).merge(types, on="ID", how="left")
    any_label = label_mask(df.index, labels_all, resample)
    raw_restored = raw_plain.copy()
    restored = restore_annotated(raw_restored, df.index, list(df.columns), series, labels_all, resample)
    any_label |= restored
    print(f"  {len(labels)} anomaly label rows ({labels['ID'].nunique()} anomalies) -> {mask.sum():,} anomalous "
          f"samples; {any_label.sum():,} samples carry a label of any category; ESA-ADB's restoration writes "
          f"{int(restored.sum()):,} annotated samples back onto the grid (replication arm only)")

    vstart = int(df.index.searchsorted(_stamp(validation_from, df.index)))
    nominal_fit = ~any_label[:vstart]
    mu = raw_plain[:vstart][nominal_fit].mean(0)
    sigma = raw_plain[:vstart][nominal_fit].std(0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    print(f"  split at {df.index[cut].date()}: train {cut:,} / test {len(df) - cut:,} at {resample}; validation "
          f"from {validation_from}; z-score from {int(nominal_fit.sum()):,} unlabelled training rows")
    return Prepared(df=df, raw_plain=raw_plain, raw_restored=raw_restored, restored=restored, labels=labels,
                    labels_all=labels_all, mask=mask, any_label=any_label, cut=cut, vstart=vstart, mu=mu, sigma=sigma)


def main() -> int:
    ap = argparse.ArgumentParser()
    # 300s, the published configuration. The default was 60s while the published model and its
    # record said 300s, so a retrain with no flags would have produced a different model at the
    # cadence Round 18 mistook for this one's (Round 71).
    # 30s since Round 86: Mission1's dominant sampling is 0.033 Hz (ESA-ADB, the section defining point
    # anomalies), and the 300 s bins averaged its many sub-minute anomalies away from the forecaster's input.
    ap.add_argument("--resample", default="30s",
                    help="resampling cadence; raw telemetry is ~15M irregular samples per channel")
    ap.add_argument("--max-epochs", type=int, default=ESA_MAX_EPOCHS,
                    help="ESA-ADB's 1000 by default; early stopping ends the run well before it")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-at", default=ESA_ADB_SPLIT,
                    help="first test timestamp; ESA-ADB's own Mission1 split by default")
    ap.add_argument("--hold", choices=["last", "mean"], default="last",
                    help="last: ESA-ADB's zero-order hold on the grid; mean: each bin's average")
    ap.add_argument("--validation-from", default=ESA_ADB_VALIDATION,
                    help="training windows whose target is on or after this stop the training early")
    ap.add_argument("--out", type=Path, default=Path("models/satellite_telemanom"))
    ap.add_argument("--export-only", action="store_true",
                    help="re-export the checkpoint in --out to ONNX without retraining, gated against "
                         "the held-out windows the original export saved as fixtures.npy")
    args = ap.parse_args()
    if args.export_only:
        return _export_only(args.out)

    import pandas as pd

    from qsh.models.esa_scores import esa_event_wise

    started = time.time()
    train.seed_everything(args.seed)

    prep = prepare(args.resample, args.split_at, args.validation_from, hold=args.hold)
    df, cut, vstart, mu, sigma = prep.df, prep.cut, prep.vstart, prep.mu, prep.sigma
    labels, labels_all, mask, any_label, restored = prep.labels, prep.labels_all, prep.mask, prep.any_label, prep.restored
    raw_plain, raw_restored = prep.raw_plain, prep.raw_restored
    z, z_restored = prep.zscore(raw_plain), prep.zscore(raw_restored)
    if not np.array_equal(z[~any_label], z_restored[~any_label]):
        raise SystemExit("the restoration wrote an unlabelled row; the two arms would not share a training set")

    length = SEQ_LEN + HORIZON
    train_idx = clean_starts(any_label, 0, vstart, length)
    val_idx = clean_starts(any_label, vstart, cut, length)
    print(f"  windows: {len(train_idx):,} unlabelled training starts, {len(val_idx):,} validation starts "
          f"(input {SEQ_LEN}, forecast {HORIZON} steps)")

    model, manifest = _esa_train(z, train_idx, val_idx, args.seed, args.out, max_epochs=args.max_epochs)
    n_params = manifest["n_parameters"]
    print(f"  {manifest['epochs_run']} epochs; last-epoch weights kept (val MSE {manifest['last_val_mse']:.6f}, "
          f"best {manifest['best_val_mse']:.6f})")

    # Each period is scored whole, as ESA-ADB's re-evaluation scores its test file: the first SEQ_LEN rows have no input
    # window inside the period and take the period's first forecast, as its reconstruction.csv pads them, so the
    # thresholding's batches start where ESA-ADB's do.
    v_times, t_times = df.index[vstart:cut], df.index[cut:]

    def period_labels(lo, hi):
        """ESA-ADB's scoring set: the six channels' labels that START and END inside the period (experiments.py)."""
        st = labels_all["StartTime"].dt.tz_localize(None) if labels_all["StartTime"].dt.tz is not None \
            else labels_all["StartTime"]
        en = labels_all["EndTime"].dt.tz_localize(None) if labels_all["EndTime"].dt.tz is not None \
            else labels_all["EndTime"]
        lo_n, hi_n = pd.Timestamp(lo).tz_localize(None), pd.Timestamp(hi).tz_localize(None)
        return labels_all[(st >= lo_n) & (en <= hi_n)]

    v_labels, t_labels = period_labels(v_times[0], v_times[-1]), period_labels(t_times[0], t_times[-1])
    n_val_anomalies = int(v_labels.loc[v_labels["Category"] == "Anomaly", "ID"].nunique())

    def score(errors, vals, floor):
        """Telemanom's flags at one error floor, and ESA-ADB's scores for them over the test half: anomalies alone,
        and all events but communication gaps (its Table 2 selection)."""
        per, flagged = telemanom_threshold.flag(errors, vals, telemanom_threshold.TelemanomConfig(min_error_value=floor))
        s_, s_all = (esa_event_wise(t_times, flagged.astype(int), t_labels, select=sel, full_range=(t_times[0], t_times[-1]))
                     for sel in (ANOMALIES, ALL_EVENTS))
        return per, flagged.astype(int), s_, s_all

    # Forecasts, errors in RAW units as ESA-ADB computes them (|y_hat_raw - y_raw|), and the values the thresholding's
    # scale checks read, per arm, for the validation months and the test half.
    arms: dict[str, dict] = {}
    for arm, zz, vals_raw in (("plain_zero_order_hold", z, raw_plain), ("restored_annotated_samples", z_restored,
                                                                        raw_restored)):
        v_err = np.abs(_padded_forecast(model, zz, vstart, cut) - zz[vstart:cut]) * sigma
        t_err = np.abs(_padded_forecast(model, zz, cut, len(zz)) - zz[cut:]) * sigma
        floor, table = _choose_floor(v_err, vals_raw[vstart:cut], v_times, v_labels, (v_times[0], v_times[-1]))
        runs = {}
        for name, f in (("validation_floor", floor), ("paper_floor", PAPER_FLOOR), ("no_floor", 0.0)):
            per, flagged, s_, s_all = score(t_err, vals_raw[cut:], f)
            runs[name] = {"floor": f, "per_channel": per, "flags": flagged, "esa": s_, "esa_all_events": s_all}
        # The third reading (the owner's choice, 2026-11-07): no floor, and a step flagged only where at least k of the six
        # channels flag it, k chosen on the validation months by the floor's rule. It keeps rare events the floor drops.
        per_v, _ = telemanom_threshold.flag(v_err, vals_raw[vstart:cut], telemanom_threshold.TelemanomConfig(min_error_value=0.0))
        k, k_table = _choose_agreement(per_v, v_times, v_labels, (v_times[0], v_times[-1]))
        per0 = runs["no_floor"]["per_channel"]
        flagged_k = (per0.sum(axis=1) >= k).astype(int)
        s_k, s_k_all = (esa_event_wise(t_times, flagged_k, t_labels, select=sel, full_range=(t_times[0], t_times[-1]))
                        for sel in (ANOMALIES, ALL_EVENTS))
        runs["agreement"] = {"floor": 0.0, "k": k, "per_channel": per0, "flags": flagged_k, "esa": s_k,
                             "esa_all_events": s_k_all}
        for name, r in runs.items():
            s_, s_all = r["esa"], r["esa_all_events"]
            print(f"    {arm:28s} {name:16s} floor {r['floor']:<7}{f' k {k}' if name == 'agreement' else ''} F0.5 "
                  f"{s_['f_0.50']:.4f}  P {s_['precision']:.4f}  R {s_['recall']:.4f}  tp {s_['tp']} fp {s_['fp']} "
                  f"fn {s_['fn']}  TNR {s_['tnr']}  | all events F0.5 {s_all['f_0.50']:.4f} (tp {s_all['tp']} "
                  f"fp {s_all['fp']} fn {s_all['fn']})")
        arms[arm] = {"floor_chosen_on_validation": floor, "floor_table": table, "runs": runs,
                     "agreement_k_chosen_on_validation": k, "agreement_table": k_table}
    del z_restored, raw_restored

    # The project's own result: the plain zero-order hold, at the floor chosen on the validation months.
    head = arms["plain_zero_order_hold"]["runs"]["validation_floor"]
    per_channel, pred, esa = head["per_channel"], head["flags"], head["esa"]
    floor = head["floor"]
    truth = mask[cut:].astype(int)
    xte, _ = window_view(z[cut:], SEQ_LEN)          # windows whose targets are the rows from cut + SEQ_LEN
    ev_f05 = _uncorrected_f05(esa)
    ev_f05_corr = esa["f_0.50"]
    pa_f1 = metrics.point_adjusted_f1(truth, pred)
    spec = get_spec("satellite_comms.telemanom")
    print(f"  Telemanom thresholding per channel at floor {floor} -> {pred.sum():,} flagged of {len(pred):,} "
          f"(per channel: {per_channel.sum(axis=0).tolist()}); the validation months hold {n_val_anomalies} "
          f"anomalies")
    print(f"    published lightweight   {spec.metric.baseline}  ({spec.metric.baseline_source[:90]}...)")
    iso = _isolation_forest_baseline(z, train_idx, length, cut, t_times, t_labels, args.seed)
    print(f"    windowed iForest        F0.5 {iso['f_0.50']:.4f} corrected, {_uncorrected_f05(iso):.4f} uncorrected "
          f"(tp {iso['tp']} fp {iso['fp']} fn {iso['fn']})   <- trivial baseline, run every time")

    # ---- T1 student -----------------------------------------------------------
    # Not here since Round 86. The logistic student this section distilled from the teacher's flags could not follow
    # the retrained teacher (0 against its 0.961 on the evaluation slice), and its replacement, a linear next-step
    # forecaster the page evaluates under CKKS and thresholds itself, is trained from the published teacher by
    # scripts/train/satellite_student_forecaster.py, so a teacher run no longer overwrites the student's record.

    # ---- artefacts ------------------------------------------------------------
    out = args.out
    # The published graph serves the forecast's first step, the one ESA-ADB scores, so its interface is unchanged.
    # int8_required=False: the LSTM's int8 graph is still built and measured, and shipped only if it passes (it
    # missed the 0.99 confident-agreement bar at 30 s in the Round 86 attempt that preceded this recipe).
    report = export.verify_and_quantize(TelemanomLSTM.step_one(model), np.ascontiguousarray(xte[:256]), out,
                                        kind="torch", int8_required=False)
    print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB -> int8 {report.int8_bytes/1e6:.2f} MB, "
          f"confident agreement {report.int8_confident_agreement:.4f}")

    (out / "feature_spec.json").write_text(json.dumps({
        "input": {"shape": [1, SEQ_LEN, int(z.shape[1])], "dtype": "float32"},
        "output": "the first step of a forecast trained over HORIZON steps",
        "horizon_trained": HORIZON,
        "hold": args.hold,
        "channels": list(datasets.ESA_MISSION1_LIGHTWEIGHT),
        "resample": args.resample,
        "normalisation": {"mean": mu.tolist(), "std": sigma.tolist()},
        "thresholding": {"method": "Telemanom (Hundman et al., KDD 2018) per channel, as ESA-ADB's TelemanomThresholding "
                                   "applies it (qsh.models.telemanom_threshold); a timestep is flagged when any channel "
                                   "flags it",
                         **{k: getattr(telemanom_threshold.DEFAULT, k)
                            for k in telemanom_threshold.TelemanomConfig.__dataclass_fields__},
                         "min_error_value": floor},
    }, indent=2) + "\n")

    def _brief(r: dict) -> dict:
        s_ = r["esa"]
        return {"floor": r["floor"], **({"k": r["k"]} if "k" in r else {}),
                "corrected_event_wise_f0_5": round(float(s_["f_0.50"]), 6),
                "event_wise_f0_5": round(float(_uncorrected_f05(s_)), 6),
                "precision": round(float(s_["precision"]), 6), "recall": round(float(s_["recall"]), 6),
                "tnr": None if s_["tnr"] is None else round(float(s_["tnr"]), 6),
                "tp": int(s_["tp"]), "fp": int(s_["fp"]), "fn": int(s_["fn"]),
                "flagged_points": int(r["flags"].sum()),
                "all_events": _all_events_brief(r["esa_all_events"])}

    esa_scores = {
        "scorer": ("qsh.models.esa_scores.esa_event_wise, a reimplementation of ESAScores in kplabs-pl/ESA-ADB (MIT, "
                   "commit 67194d3), checked against that repository's own test cases in tests/test_r86_esa_scores.py"),
        "selection": ("Category Anomaly; labels of channels 41-46 that start and end inside the scored period. Each "
                      "run's all_events block selects Rare Event and Anomaly instead, ESA-ADB's Table 2 reading, where "
                      "Telemanom-ESA-Pruned is 0.786 (published_lightweight_f0_5) and plain Telemanom-ESA 0.178"),
        "test_range": [str(t_times[0]), str(t_times[-1])],
        "validation_range": [str(v_times[0]), str(v_times[-1])],
        "n_validation_anomalies": n_val_anomalies,
        "paper_floor": PAPER_FLOOR,
        "paper_floor_source": ("ESA-ADB Supplementary Table 8, Telemanom-ESA-Pruned: min_error_value 0.05/7 in raw "
                               "units (scripts/reevaluate.py); plain Telemanom-ESA runs with 0"),
        "thresholding": ("every arm and floor uses Telemanom's thresholding as ESA-ADB's TelemanomThresholding applies "
                         "it for Telemanom-ESA-Pruned (qsh.models.telemanom_threshold); no_floor is that procedure at "
                         "floor 0, not Telemanom-ESA's own thresholding, which produced the benchmark's plain 0.090 "
                         "and differs in two details the module states"),
        "scored_rows": ("each period whole; its first 250 rows, which have no input window inside it, take the "
                        "period's first forecast, as ESA-ADB's reconstruction.csv pads them"),
        "restored_samples": int(restored.sum()),
        **{arm: {"floor_chosen_on_validation": a["floor_chosen_on_validation"],
                 **{name: _brief(r) for name, r in a["runs"].items()},
                 "validation_floor_table": a["floor_table"],
                 "agreement_k_chosen_on_validation": a["agreement_k_chosen_on_validation"],
                 "agreement_validation_table": a["agreement_table"]}
           for arm, a in arms.items()},
        "agreement_note": ("the third reading: no floor, and a step is flagged only where at least k channels flag it, "
                           "k chosen on the validation months by the floor's rule (the middle of the best plateau). "
                           "Those months hold 2 anomalies, so the choice is weakly determined, as the floor's is"),
        "windowed_iforest": {k: round(float(iso[k]), 6) if isinstance(iso[k], float) else iso[k]
                             for k in ("f_0.50", "precision", "recall", "tnr", "tp", "fp", "fn")},
    }
    replication = arms["restored_annotated_samples"]["runs"]["paper_floor"]["esa"]
    replication_all = arms["restored_annotated_samples"]["runs"]["paper_floor"]["esa_all_events"]
    train.record_metrics("satellite_comms.telemanom", {
        # ESA-ADB's event-wise F0.5 without its TNR correction, and with it, for the plain zero-order hold at the floor
        # chosen on the validation months; esa_scores holds the other floors and the replication arm.
        "event_wise_f0_5": round(float(ev_f05), 6),
        "corrected_event_wise_f0_5": round(float(ev_f05_corr), 6),
        "point_adjusted_f1": round(float(pa_f1), 6),
        "windowed_iforest_event_wise_f0_5": round(float(_uncorrected_f05(iso)), 6),
        "windowed_iforest_corrected_event_wise_f0_5": round(float(iso["f_0.50"]), 6),
        "esa_scores": esa_scores,
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
        # Label ROWS above (one per channel an anomaly touches); the distinct anomalies are these (Round 86: the
        # notes compared "320 intervals" with events scored, rows against events).
        "n_anomaly_events_timeline": int(labels["ID"].nunique()),
        "split": {"test_from": str(args.split_at), "validation_from": str(args.validation_from),
                  "source": "ESA-ADB's own Mission1 split (84_months, kplabs-pl/ESA-ADB datasets.csv and "
                            "mission1_experiments.py)"},
        "training": {k: manifest[k] for k in ("recipe", "epochs_run", "best_val_mse", "last_val_mse", "weights",
                                              "horizon", "batch", "steps_per_epoch", "patience", "min_delta",
                                              "learning_rate")}
                    | {"hold": args.hold, "training_windows": int(len(train_idx)),
                       "validation_windows": int(len(val_idx))},
        # The one that actually divides the score. n_anomaly_intervals above counts the WHOLE
        # timeline and is not the evaluation's sample size; these are.
        "n_events_scored": int(esa["n_selected_ids"]),
        "event_wise_tp": int(esa["tp"]),
        "event_wise_fp": int(esa["fp"]),
        "event_wise_fn": int(esa["fn"]),
        "anomaly_sample_rate": round(float(mask.mean()), 8),
        "resample": args.resample,
        "dataset": spec.dataset.name,
        "metric_note": (
            "ESA-ADB's event-wise F0.5 for anomalies alone, scored by a reimplementation of its ESAScores: an "
            "anomaly is detected when any flagged event intersects it, and a flagged event is a false positive "
            "only when it intersects no annotated event of any category. corrected_event_wise_f0_5 multiplies "
            "event precision by the time-level true-negative rate, as ESA-ADB reports it; event_wise_f0_5 does "
            "not. Telemanom's paper prints 0.71 under F0.5 in its Table 2, though its own precision and recall "
            "there (87.5% and 80.0% overall) give 0.86. point_adjusted_f1 is shown only to expose how much point "
            "adjustment inflates: Kim et al. (AAAI 2022) show that under point adjustment even a random anomaly "
            "score can reach state-of-the-art. Only labels whose ESA-ADB Category is 'Anomaly' count as "
            "positives; the 63 rare nominal events annotated on these six channels do not."
        ),
        # COMPUTED FROM THIS RUN since Round 71. Both notes carried their counts as literals ("THREE
        # labelled anomaly events", "flags 2 points out of 441,600"), true of the run that wrote
        # them and of no retrain after it, and this record is what the satellite page prints.
        "baseline_gap_note": (
            "Read n_events_scored before comparing this to the baseline. Since Round 86 the forecaster follows "
            "ESA-ADB's Telemanom-ESA recipe on its own Mission1 split: a 30 s zero-order hold, training only on "
            f"unlabelled windows before {args.split_at} with early stopping on the months from "
            f"{args.validation_from}, a 10-step forecast scored on its first step, and Telemanom's thresholding per "
            f"channel. The half after the split holds {esa['n_selected_ids']} labelled anomalies of the "
            f"{int(labels['ID'].nunique())} on these channels. This score is the plain zero-order hold at an error "
            f"floor of {floor} raw units, chosen on the validation months, which hold {n_val_anomalies} anomalies: "
            f"tp={esa['tp']} fp={esa['fp']} fn={esa['fn']}, {int(pred.sum()):,} of {len(pred):,} points flagged. "
            "The published 0.968 is Telemanom-ESA-Pruned, whose floor of 0.05/7 the paper does not show was "
            "chosen without the test half, and it is scored on ESA-ADB's prepared series, which writes annotated "
            "samples the zero-order hold dropped back onto the grid - a step that reads the labels. Replicating "
            "both gives "
            f"{replication['f_0.50']:.3f} (tp={replication['tp']} fp={replication['fp']} fn={replication['fn']}), and "
            f"{replication_all['f_0.50']:.3f} over all events but communication gaps, where Table 2 prints 0.786 "
            f"({replication_all['tp']} of {replication_all['n_selected_ids']} events here; the published recall "
            "0.424 is no whole number of events out of this count, so ESA-ADB's all-events set differs from this "
            "selection by at least one event); esa_scores holds every arm and floor. Remaining differences: "
            "PyTorch where ESA-ADB uses Keras, from Keras's initialisers, one seed, and ESA-ADB's per-file "
            "resampling at the split. Until Round 86 this model trained at 300 s on "
            "a 70/30 cut holding 3 events and flagged 2 of 441,600 points. Workings: "
            "references/satellite-event-count-investigation.md."
        ),
        "sample_size_note": (
            "Read n_events_scored before comparing this to a published figure. The denominator of an "
            f"event-wise metric is EVENTS: {esa['n_selected_ids']} anomalies in this test half, so one event "
            "either way moves the score by a visible step. n_anomaly_intervals counts label ROWS, one per "
            f"channel an anomaly touches ({len(labels)}); n_anomaly_events_timeline counts the anomalies "
            f"themselves ({int(labels['ID'].nunique())}). The number is reported as measured rather than "
            "reweighted, with the sample size beside it so the comparison is not made silently."
        ),
        "git_sha": train.git_sha(),
        "export": _export_block(report),
        "wall_clock_s": round(time.time() - started, 1),
    })
    print(f"\n  wrote {out} and models/metrics.json in {time.time()-started:.0f}s")
    return 0


def _all_events_brief(s: dict) -> dict:
    """ESA-ADB's Table 2 reading of one run: all events but communication gaps, corrected as ESA-ADB corrects it."""
    return {"corrected_event_wise_f0_5": round(float(s["f_0.50"]), 6), "precision": round(float(s["precision"]), 6),
            "recall": round(float(s["recall"]), 6), "tp": int(s["tp"]), "fp": int(s["fp"]), "fn": int(s["fn"]),
            "n_events": int(s["n_selected_ids"])}


def _uncorrected_f05(s: dict) -> float:
    """ESA-ADB's event-wise F0.5 from its counts, WITHOUT the time-level TNR factor on precision."""
    tp, fp = s["tp"], s["fp"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = s["recall"]
    return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


def clean_starts(any_label: np.ndarray, lo: int, hi: int, length: int) -> np.ndarray:
    """Start indices of the windows of `length` rows inside [lo, hi) that touch no labelled row (Round 86).

    ESA-ADB trains Telemanom-ESA only on stretches where no target channel carries any label - anomaly, rare event,
    gap or invalid - and a stretch shorter than the window plus the horizon yields nothing.
    """
    bad = np.r_[0, np.cumsum(any_label.astype(np.int64))]
    starts = np.arange(lo, max(lo, hi - length + 1))
    return starts[(bad[starts + length] - bad[starts]) == 0]


def _esa_train(z: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, seed: int, out_dir: Path, *,
               max_epochs: int = ESA_MAX_EPOCHS):
    """ESA-ADB's Telemanom-ESA training loop, on the GPU, over windows gathered from `z` by start index (Round 86).

    Each epoch draws ESA_STEPS_PER_EPOCH batches of ESA_BATCH windows from a fresh permutation of the training starts,
    as Keras's Sequence with steps_per_epoch=min(1000, len(generator)) does; the whole validation set is scored each
    epoch; training stops after ESA_PATIENCE epochs without an improvement of ESA_MIN_DELTA, and the LAST epoch's
    weights are kept, because ESA-ADB's EarlyStopping does not restore the best ones and its run saves the model after.
    """
    import torch

    torch.manual_seed(seed)
    dev = train._device()
    zt = torch.tensor(z, dtype=torch.float32, device=dev)
    n_ch = z.shape[1]
    model = TelemanomLSTM.build(n_ch, horizon=HORIZON)
    TelemanomLSTM.keras_init(model)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=ESA_LR, eps=1e-7)
    loss_fn = torch.nn.MSELoss()
    offs_in = torch.arange(SEQ_LEN, device=dev)
    offs_out = torch.arange(SEQ_LEN, SEQ_LEN + HORIZON, device=dev)
    tr = torch.tensor(train_idx, dtype=torch.long, device=dev)
    va = torch.tensor(val_idx, dtype=torch.long, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def gather(idx):
        return zt[idx[:, None] + offs_in], zt[idx[:, None] + offs_out]

    steps = min(ESA_STEPS_PER_EPOCH, int(np.ceil(len(train_idx) / ESA_BATCH)))
    best, stale, history = np.inf, 0, []
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=gen).to(dev)
        for s_ in range(steps):
            idx = tr[perm[s_ * ESA_BATCH:(s_ + 1) * ESA_BATCH]]
            x, y = gather(idx)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va), 8192):
                x, y = gather(va[i:i + 8192])
                total += float(torch.nn.functional.mse_loss(model(x), y, reduction="sum"))
                count += y.numel()
        val_loss = total / max(count, 1)
        history.append(val_loss)
        print(f"    epoch {epoch:4d}  val MSE {val_loss:.6f}")
        if val_loss < best - ESA_MIN_DELTA:
            best, stale = val_loss, 0
        else:
            stale += 1
            if stale >= ESA_PATIENCE:
                break
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.safetensors"
    train._save_safetensors(model, ckpt)        # the last epoch's weights
    manifest = {"key": "satellite_comms.telemanom", "recipe": "ESA-ADB Telemanom-ESA (Round 86)",
                "initialisation": "Keras's (glorot-uniform, orthogonal, unit forget bias)",
                "epochs_run": len(history), "best_val_mse": best, "last_val_mse": history[-1],
                "weights": "last epoch", "horizon": HORIZON, "batch": ESA_BATCH,
                "steps_per_epoch": steps, "patience": ESA_PATIENCE, "min_delta": ESA_MIN_DELTA,
                "learning_rate": ESA_LR, "git_sha": train.git_sha(), "seed": seed,
                "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return model.cpu(), manifest


def _padded_forecast(model, z: np.ndarray, first: int, last: int) -> np.ndarray:
    """Step-1 forecasts for every row of [first, last), from windows inside it; the first SEQ_LEN rows, which have none,
    repeat the first forecast, as ESA-ADB's batch_predict writes its reconstruction.csv (Round 86)."""
    pred = _forecast_step1(model, z, first + SEQ_LEN, last)
    return np.concatenate([np.repeat(pred[:1], SEQ_LEN, axis=0), pred])


def _forecast_step1(model, z: np.ndarray, first: int, last: int) -> np.ndarray:
    """Step-1 forecasts for the rows [first, last), each from the SEQ_LEN rows before it."""
    import torch

    dev = train._device()
    model.eval().to(dev)
    zt = torch.tensor(z, dtype=torch.float32, device=dev)
    offs = torch.arange(SEQ_LEN, device=dev)
    starts = torch.arange(first - SEQ_LEN, last - SEQ_LEN, device=dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(starts), 8192):
            out.append(model(zt[starts[i:i + 8192, None] + offs])[:, 0, :].cpu().numpy())
    return np.concatenate(out)


def _choose_agreement(per: np.ndarray, times, labels, full_range) -> tuple[int, list[dict]]:
    """The k, from 1 to the channel count, that scores best on ESA-ADB's corrected event-wise F0.5 over these labels when
    a step is flagged only where at least k channels flag it; on a tie, the middle of the best plateau, the floor's rule
    (Round 86, the owner's choice of a third reading). `per` is the per-channel flags at no floor, (rows, channels)."""
    from qsh.models.esa_scores import esa_event_wise

    count = per.sum(axis=1)
    table = []
    for k in range(1, per.shape[1] + 1):
        s = esa_event_wise(times, (count >= k).astype(int), labels, select=ANOMALIES, full_range=full_range)
        table.append({"k": k, "f_0.50": round(s["f_0.50"], 6), "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]})
    best = max(r["f_0.50"] for r in table)
    plateau = [r["k"] for r in table if r["f_0.50"] == best]
    return plateau[len(plateau) // 2], table


def _choose_floor(errors_raw, values_raw, times, labels, full_range):
    """The error floor, from FLOOR_GRID, that scores best on ESA-ADB's corrected event-wise F0.5 over the validation
    months; on a tie, the middle of the best plateau, so the choice does not sit on the edge of the grid (Round 86)."""
    from qsh.models.esa_scores import esa_event_wise

    table = []
    for floor in FLOOR_GRID:
        cfg = telemanom_threshold.TelemanomConfig(min_error_value=floor)
        _, flagged = telemanom_threshold.flag(errors_raw, values_raw, cfg)
        s = esa_event_wise(times, flagged.astype(int), labels, select={"Category": ["Anomaly"]},
                           full_range=full_range)
        table.append({"floor": floor, "f_0.50": round(s["f_0.50"], 6), "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]})
    best = max(r["f_0.50"] for r in table)
    plateau = [r["floor"] for r in table if r["f_0.50"] == best]
    return plateau[len(plateau) // 2], table


def _export_only(out: Path) -> int:
    """Re-export the published checkpoint without retraining it.

    Added in Round 71 for the static output batch (see qsh.ai.inference.export_onnx). The weights do
    not change, so neither do the metrics: only the `export` block is re-measured, and record_metrics
    labels every other field as carried forward from the run that measured it. The gate is the
    original one, against the same 256 held-out windows, which the training run saved as
    fixtures.npy precisely so a later export can be checked against them.
    """
    from safetensors.torch import load_file

    fixtures = np.load(out / "fixtures.npy")
    # Round 86: a checkpoint trained by ESA-ADB's recipe forecasts HORIZON steps and is served by its first step.
    horizon = int(json.loads((out / "manifest.json").read_text()).get("horizon", 1))
    model = TelemanomLSTM.build(int(fixtures.shape[2]), horizon=horizon)
    model.load_state_dict(load_file(str(out / "model.safetensors")))
    served = TelemanomLSTM.step_one(model) if horizon > 1 else model
    report = export.verify_and_quantize(served, fixtures, out, kind="torch", int8_required=horizon == 1)
    print(f"  re-exported {out}: fp32 {report.fp32_bytes:,} B, int8 {report.int8_bytes:,} B, "
          f"fp32 max abs error {report.fp32_max_abs_error:.3e} over {report.n_fixtures} fixtures")
    train.record_metrics("satellite_comms.telemanom", {"export": _export_block(report), "git_sha": train.git_sha()})
    return 0


def _export_block(report) -> dict:
    """The record's `export` block; an int8 graph built, measured and withheld says so (Round 86)."""
    block = {"fp32_bytes": report.fp32_bytes, "int8_bytes": report.int8_bytes, "opset": report.opset}
    if report.int8_agreement is not None:
        block["int8_agreement"] = round(float(report.int8_agreement), 6)
    if report.int8_skipped_reason:
        block["int8_skipped_reason"] = report.int8_skipped_reason
    return block


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


def _isolation_forest_baseline(z: np.ndarray, train_idx: np.ndarray, length: int, t_lo: int, t_times, t_labels,
                               seed: int) -> dict:
    """Windowed isolation forest, the trivial detector ESA-ADB shows can beat deep models, scored as the teacher is.

    Since Round 86 it is fitted on 60,000 of the unlabelled training windows, the forecaster's own training set, rather
    than on the first 60,000 test windows; its features are the eight rows ENDING at the scored timestamp, where they
    were the eight before it, which left a point anomaly undetectable at its own timestamp; and it is scored by
    ESA-ADB's rules. contamination=0.01 puts its threshold at the 99th percentile of the training windows' scores.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from sklearn.ensemble import IsolationForest

    from qsh.models.esa_scores import esa_event_wise

    last8 = sliding_window_view(z, 8, axis=0)                 # last8[i - 7] holds rows i-7..i
    rng = np.random.default_rng(seed)
    ends = rng.choice(train_idx, size=min(len(train_idx), 60_000), replace=False) + length - 1
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=seed, n_jobs=8)
    iso.fit(last8[ends - 7].reshape(len(ends), -1))
    feats = last8[t_lo - 7:].reshape(len(z) - t_lo, -1)
    pred = (iso.predict(feats) == -1).astype(int)
    return esa_event_wise(t_times, pred, t_labels, select=ANOMALIES, full_range=(t_times[0], t_times[-1]))


if __name__ == "__main__":
    raise SystemExit(main())
