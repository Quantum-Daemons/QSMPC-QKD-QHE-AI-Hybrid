"""ESA-ADB's corrected event-wise F-score, as its `ESAScores` computes it (Round 86).

A reimplementation of `timeeval/metrics/ESA_ADB_metrics.py::ESAScores.score` and `utils.convert_time_series_to_events`
in kplabs-pl/ESA-ADB (MIT), read at commit 67194d3 on 2026-11-07, without its `portion` dependency. The rules:

  - a prediction is a series of (timestamp, 0 or 1); each run of 1s is an event covering [t_first, t_next), the
    timestamp after the run, or [t_first, t_last] when the run reaches the end;
  - true positives and false negatives are counted per anomaly ID among the SELECTED labels (ESA-ADB's lightweight
    anomaly score selects Category == "Anomaly"): an ID is detected when any predicted event intersects any of its
    closed [StartTime, EndTime] intervals;
  - a predicted event is a false positive only when it matched no selected ID AND intersects no annotated event of
    ANY category, so a detection of a rare nominal event is neither credited nor penalised;
  - precision TP / (TP + FP) is multiplied by the time-based true-negative rate, 1 - (seconds of predicted events in
    nominal time) / (seconds of nominal time), nominal time being the full range minus every annotated event (the
    correction ESA-ADB takes from Sehili et al., arXiv:2308.13068); recall is TP / (TP + FN).

`portion` treats a point anomaly (StartTime == EndTime) as a closed interval of length zero, and so does this.
"""

from __future__ import annotations

import numpy as np


def _ns(x) -> np.ndarray:
    """Timestamps (index, series or array, with or without a time zone) as UTC int64 nanoseconds.

    `as_unit("ns")` because pandas 3 keeps each index at its own resolution, and `asi8` counts in it: the label file
    parses at microseconds, so without it a nanosecond grid and its labels were integers a thousandfold apart.
    """
    import pandas as pd

    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.integer):
        return x.astype(np.int64)
    idx = pd.DatetimeIndex(pd.to_datetime(x))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("ns").asi8.astype(np.int64)


def events_from_series(times, flags: np.ndarray) -> list[tuple[int, int, bool]]:
    """Runs of 1s as (lower, upper, upper_closed) in int64 nanoseconds."""
    t = _ns(times)
    f = np.asarray(flags).astype(bool)
    n = len(f)
    if n == 0:
        return []
    edges = np.diff(np.r_[0, f.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)     # ends are exclusive indices
    out = []
    for s, e in zip(starts, ends, strict=True):
        out.append((t[s], t[e - 1], True) if e == n else (t[s], t[e], False))
    return out


def _intersects(ev: tuple[np.int64, np.int64, bool], lo: np.int64, hi: np.int64) -> bool:
    """Does the predicted event intersect the CLOSED interval [lo, hi]?"""
    a, b, closed = ev
    return a <= hi and (lo <= b if closed else lo < b)


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def _overlap_seconds(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    """Total length, in seconds, of the intersection of two sorted disjoint interval lists (ns)."""
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total / 1e9


def esa_event_wise(times: np.ndarray, flags: np.ndarray, labels, *, select: dict | None = None,
                   beta: float = 0.5, full_range: tuple | None = None) -> dict:
    """ESA-ADB's corrected event-wise scores for one prediction series.

    `labels` is a DataFrame with ID, StartTime, EndTime and the label columns `select` filters on (e.g.
    {"Category": ["Anomaly"]}); every row counts as an annotated event for the false-positive and nominal-time rules.
    `full_range` defaults, as ESA-ADB's does, to the span of the labels and the prediction together.
    """
    import pandas as pd

    t_ns = _ns(times)
    lab = labels.assign(_lo=_ns(labels["StartTime"]), _hi=_ns(labels["EndTime"]))
    if full_range is None:
        full_lo, full_hi = min(int(lab["_lo"].min()), int(t_ns.min())), max(int(lab["_hi"].max()), int(t_ns.max()))
    else:
        full_lo, full_hi = (int(v) for v in _ns(pd.Series(list(full_range))))
    # ESA-ADB pads the prediction to the full range with its first and last values.
    f = np.asarray(flags).astype(bool)
    if t_ns[0] > full_lo:
        t_ns, f = np.r_[full_lo, t_ns], np.r_[f[0], f]
    if t_ns[-1] < full_hi:
        t_ns, f = np.r_[t_ns, full_hi], np.r_[f, f[-1]]
    events = events_from_series(t_ns, f)

    selected = lab
    for col, vals in (select or {}).items():
        selected = selected[selected[col].isin(list(np.atleast_1d(vals)))]
    tp = fn = redundant = 0
    matched = [False] * len(events)
    for _, g in selected.groupby("ID", sort=False):
        ivs = list(zip(g["_lo"].tolist(), g["_hi"].tolist(), strict=True))
        lo_all, hi_all = min(a for a, _ in ivs), max(b for _, b in ivs)
        hits = [0] * len(ivs)
        found = False
        for k, ev in enumerate(events):
            if ev[1] < lo_all or ev[0] > hi_all:
                continue
            inter = [_intersects(ev, a, b) for a, b in ivs]
            if not any(inter):
                continue
            matched[k] = True
            found = True
            for i, v in enumerate(inter):
                hits[i] += int(v)
        redundant += sum(h - 1 for h in hits if h > 1)
        tp += int(found)
        fn += int(not found)
    all_events = list(zip(lab["_lo"].tolist(), lab["_hi"].tolist(), strict=True))
    fp = sum(1 for k, ev in enumerate(events)
             if not matched[k] and not any(_intersects(ev, a, b) for a, b in all_events))
    precision = tp / (tp + fp) if tp + fp else 0.0
    alarming = tp / (tp + redundant) if tp + redundant else 0.0
    tnr = None
    if precision > 0:
        gt = _union([(int(a), int(b)) for a, b in all_events])
        nominal, cur = [], full_lo
        for a, b in gt:
            if a > cur:
                nominal.append((cur, min(a, full_hi)))
            cur = max(cur, b)
        if cur < full_hi:
            nominal.append((cur, full_hi))
        pred = _union([(int(a), int(b)) for a, b, _ in events])
        nominal_s = sum(b - a for a, b in nominal) / 1e9
        tnr = 1.0 - _overlap_seconds(nominal, pred) / nominal_s if nominal_s else 0.0
        precision *= tnr
    recall = tp / (tp + fn) if tp + fn else 0.0
    d = beta ** 2 * precision + recall
    fb = (1 + beta ** 2) * precision * recall / d if d else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "redundant_detections": redundant, "alarming_precision": alarming,
            "tnr": tnr, "precision": precision, "recall": recall, f"f_{beta:.2f}": fb, "n_selected_ids": tp + fn}
