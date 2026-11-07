"""Telemanom's non-parametric dynamic thresholding, per channel, as ESA-ADB applies it (Round 86).

Until Round 86, `scripts/train/satellite_telemanom.py` set ONE threshold over the whole test window's raw errors and
"pruned" individual sorted error values, stopping at the first small drop, which left the threshold near the single
largest error: the published forecaster flagged 2 of 441,600 points and scored event-wise F0.5 = 0. Telemanom's
thresholding (Hundman et al., KDD 2018, section 3.2) is a different procedure. This module implements it as ESA-ADB's
fork of TimeEval does (kplabs-pl/ESA-ADB, timeeval/metrics/thresholding.py, TelemanomThresholding, read at commit 67194d3
on 2026-11-07): the class its re-evaluation applies, with a floor of 0.05/7, to produce Telemanom-ESA-Pruned, the
benchmark's published 0.968. On random series with planted anomalies it flags exactly the rows that class flags. The
steps, with its defaults:

  1. the absolute one-step prediction error of each channel, smoothed by an exponentially weighted moving average with
     span `batch_size * smoothing_window_size * smoothing_perc` (70 * 30 * 0.05 = 105);
  2. the smoothed errors are walked in windows of `smoothing_window_size` batches of `batch_size` values, stepping one
     batch at a time, one window per batch, those near the end cut short by it;
  3. in each window, a threshold mean + z * sd for z in 2.5, 3.0, ..., 11.5 is chosen to maximise
     (relative fall in mean + relative fall in sd when the points above it are removed) /
     (number of anomalous sequences squared + number of anomalous points),
     subject to at most 5 sequences and fewer than half the window anomalous; each point above it is widened by
     `error_buffer` values either side, and the same search runs on the errors reflected about their mean;
  4. a point is flagged only if it also exceeds 5% of the window's 5th-to-95th percentile range of the channel's
     values, and only in the window's newest batch (the whole window, after the first `2 * window_size` values, for
     the first one);
  5. pruning on SEQUENCE maxima: the maxima of the window's anomalous sequences, sorted descending, with the largest
     non-anomalous error appended; any sequence whose maximum falls less than `p` (0.13) relative to the next one is
     dropped, unless a later drop of at least `p` follows it, which resets the list;
  6. a channel's flags are its anomalous sequences, each [first, last) as ESA-ADB writes them, so a sequence's last
     index is not flagged, and a timestep is anomalous if any channel flags it (ESA-ADB's `y_scores.max(axis=1)`).

Telemanom-ESA's own thresholding (TimeEval-algorithms/telemanom_esa/telemanom/errors.py), which produced the benchmark's
plain Telemanom-ESA, differs in two details: it replaces the first `window_size` smoothed errors with the mean of the
first 2 * window_size (`adjust_start` below), and its scale checks read the channel's values `window_size` rows BEFORE
the errors they are compared with, because it slices both with the same indices while its errors begin a window later.
This module keeps the values aligned with the errors.

The Telemanom code is Copyright (c) 2018, California Institute of Technology ("Caltech"), under a BSD-3-Clause-style
licence that permits commercial use; this module is a reimplementation of its procedure, not a copy of its code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TelemanomConfig:
    """Telemanom-ESA's defaults (its `manifest.json`); `min_error_value` 0 as ESA-ADB's Mission1 runs pass it."""

    batch_size: int = 70
    smoothing_window_size: int = 30
    smoothing_perc: float = 0.05
    error_buffer: int = 100
    p: float = 0.13
    #: The forecaster's input length. Telemanom ignores up to twice this many values at the start of the first window,
    #: where the smoothed error has too little history.
    window_size: int = 250
    sd_lim: float = 12.0
    min_error_value: float = 0.0
    #: errors.py's averaging of the first `window_size` smoothed errors; TimeEval's TelemanomThresholding does not.
    adjust_start: bool = False


#: The defaults, as one module-level instance (a call in an argument default runs once at import anyway).
DEFAULT = TelemanomConfig()


def smooth_errors(errors: np.ndarray, cfg: TelemanomConfig = DEFAULT) -> np.ndarray:
    """EWMA of the absolute error per channel (columns); with `adjust_start`, the first `window_size` rows take the mean
    of the first 2 * window_size."""
    import pandas as pd

    e = np.abs(np.asarray(errors, dtype=np.float64))
    if e.ndim == 1:
        e = e[:, None]
    span = int(cfg.batch_size * cfg.smoothing_window_size * cfg.smoothing_perc)
    e_s = pd.DataFrame(e).ewm(span=span).mean().to_numpy(copy=True)
    w = cfg.window_size
    if cfg.adjust_start and len(e_s) >= 2 * w:
        e_s[:w] = e_s[: 2 * w].mean(axis=0)
    return e_s


def _groups(idx: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive integers in a sorted unique index array, as (first, last)."""
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _widen(i_anom: np.ndarray, n: int, offsets: np.ndarray) -> np.ndarray:
    """Each index with every offset either side added, clipped to [0, n), sorted and unique."""
    if i_anom.size == 0:
        return i_anom.astype(int)
    grown = np.concatenate((i_anom, (i_anom[:, None] + offsets).ravel(), (i_anom[:, None] - offsets).ravel()))
    grown = grown[(grown >= 0) & (grown < n)]
    return np.unique(grown).astype(int)


def _find_epsilon(e_s: np.ndarray, mean: float, sd: float, cfg: TelemanomConfig) -> float:
    """The threshold that maximises Telemanom's objective over z in 2.5, 3.0, ..., below `sd_lim`."""
    epsilon = mean + cfg.sd_lim * sd
    if sd == 0.0:
        return epsilon
    best = -np.inf
    offsets = np.arange(1, cfg.error_buffer)            # the search widens by error_buffer - 1, as the reference does
    for z in np.arange(2.5, cfg.sd_lim, 0.5):
        eps = mean + sd * z
        pruned = e_s[e_s < eps]
        i_anom = _widen(np.flatnonzero(e_s >= eps), len(e_s), offsets)
        if i_anom.size == 0:
            break
        seqs = [g for g in _groups(i_anom) if g[0] != g[1]]
        mean_drop = (mean - pruned.mean()) / mean if mean else 0.0
        sd_drop = (sd - pruned.std()) / sd
        score = (mean_drop + sd_drop) / (len(seqs) ** 2 + i_anom.size)
        if score >= best and len(seqs) <= 5 and i_anom.size < 0.5 * len(e_s):
            best, epsilon = score, eps
    return epsilon


def _prune(seqs: list[tuple[int, int]], e_s: np.ndarray, non_anom_max: float, p: float) -> list[tuple[int, int]]:
    """Drop the sequences whose maximum is within `p` of the next one down, unless a large drop follows them."""
    if not seqs:
        return seqs
    maxima = np.array([e_s[a:b + 1].max() for a, b in seqs])
    ordered = np.append(np.sort(maxima)[::-1], non_anom_max)
    remove: list[int] = []
    for i in range(len(ordered) - 1):
        drop = (ordered[i] - ordered[i + 1]) / ordered[i] if ordered[i] else 0.0
        if drop < p:
            remove.extend(np.flatnonzero(maxima == ordered[i]).tolist())
        else:
            remove = []
    gone = set(remove)
    return [s for k, s in enumerate(seqs) if k not in gone]


def flag_channel(e_s: np.ndarray, values: np.ndarray, cfg: TelemanomConfig = DEFAULT) -> np.ndarray:
    """Telemanom's anomalous timesteps for one channel: smoothed errors `e_s` and the channel's values, both length n."""
    n = len(e_s)
    flags = np.zeros(n, dtype=bool)
    width = cfg.smoothing_window_size
    # One window per batch, as ESA-ADB's TelemanomThresholding enumerates them; the original Telemanom stopped at the last
    # full window and let it run to the end, which left up to a batch less one uncovered before the final batch.
    n_windows = int(np.ceil(n / cfg.batch_size))
    num_to_ignore = 0 if n < 1800 else cfg.window_size if n < 2500 else 2 * cfg.window_size
    found = np.array([], dtype=int)
    for w in range(n_windows):
        lo = w * cfg.batch_size
        hi = n if w == n_windows - 1 else min(n, width * cfg.batch_size + w * cfg.batch_size)
        win, y = e_s[lo:hi], values[lo:hi]
        mean, sd = float(win.mean()), float(win.std())
        inter = float(np.subtract(*np.percentile(y, [95, 5])))
        keep_all: list[np.ndarray] = []
        # Too small an error against the values' own spread, too short a window, or no error at all: nothing here.
        if not (sd > 0.05 * float(y.std()) or win.max() > 0.05 * inter) or len(win) < cfg.error_buffer \
                or not win.max() > cfg.min_error_value:
            continue
        for e in (win, mean + (mean - win)):           # the errors, and the errors reflected about their mean
            eps = _find_epsilon(e, mean, sd, cfg)
            i_anom = np.flatnonzero((e >= eps) & (e > 0.05 * inter))
            if i_anom.size == 0:
                continue
            i_anom = _widen(i_anom, len(e), np.arange(1, cfg.error_buffer + 1))
            i_anom = i_anom[i_anom >= num_to_ignore] if w == 0 else i_anom[i_anom >= len(e) - cfg.batch_size]
            if i_anom.size == 0:
                continue
            others = np.setdiff1d(np.arange(len(e)), np.concatenate((found - lo, i_anom)))
            non_anom_max = float(e[others].max()) if others.size else 0.0
            seqs = [g for g in _groups(i_anom) if g[0] != g[1]]
            if not seqs:
                # No sequence of two or more: ESA-ADB's prune_anoms returns before pruning, so these single points stay
                # and can join the next window's run when the sequences are formed at the end.
                keep_all.append(i_anom)
                continue
            seqs = _prune(seqs, e, non_anom_max, cfg.p)
            if not seqs:
                continue
            kept = np.concatenate([np.arange(a, b + 1) for a, b in seqs])
            keep_all.append(i_anom[np.isin(i_anom, kept)])
        if keep_all:
            found = np.union1d(found, np.concatenate(keep_all) + lo)
    for a, b in _groups(np.unique(found)):
        if a != b:                                     # single points are not sequences, as in the reference
            flags[a:b] = True                          # [first, last), as ESA-ADB writes a sequence
    return flags


def flag(errors: np.ndarray, values: np.ndarray, cfg: TelemanomConfig = DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel flags (n, channels) and their union (n,), from raw errors and the channels' values, both (n, channels)."""
    e_s = smooth_errors(errors, cfg)
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    per = np.column_stack([flag_channel(e_s[:, c], v[:, c], cfg) for c in range(e_s.shape[1])])
    return per, per.any(axis=1)
