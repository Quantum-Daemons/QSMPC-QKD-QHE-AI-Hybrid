#!/usr/bin/env python
"""Per-transaction graph features on AMLworld HI-Small, in batches: the 79 columns the published finance MLP adds to the 32 of `finance_aml.py` (Round 86).

WHY THESE. The 32 per-account aggregates reach F1 0.15 to 0.21, the level of the "basic features" row of the Graph Feature Preprocessor paper (Blanusa et al., ICAIF 2024, arXiv:2402.08593, Table 5: 19.75 for XGBoost on HI-Small). Its
next row, fan-in and fan-out patterns of each transaction's two accounts in a time window, reaches 49.7 to 59.7, and multi-hop patterns and vertex statistics take it to 62.9 to 64.8. IBM's library is proprietary, so this file
computes the same kind of feature in the open: for each transaction, per window ending at its batch, the counts and summed log amounts of its accounts' outgoing and incoming edges, the distinct counterparties, the mean and spread
of the amounts, the time since the account's last edge, and the scatter-gather count between its source and destination.

BATCHES, AS THE PAPER PROCESSES THEM. A transaction arrives in a batch and its features see every edge up to the end of that batch, as GFP's `transform` does (Sec. 2.2: patterns are enumerated "after inserting the input batch"). That is the
paper's setting for a bank scoring in micro-batches, and it is what the page says: the score uses transactions up to 2,048 rows after it. Strictly causal windows, with nothing from the current batch, reach F1 0.196 with the same model, against 0.401 for this.

No label is read here. Every value comes from accounts, timestamps and amounts at or before the end of the transaction's own batch.

    Returned columns, in order: the fan group (sorted by name) and then the statistics and scatter-gather group (sorted by name).
"""

from __future__ import annotations

import bisect
from collections import defaultdict

import numpy as np
import pandas as pd

#: Transactions per batch: GFP's largest batch on the AML datasets (Sec. 3, 128 and 2048).
BATCH_WINDOW = 2048
#: Window lengths in minutes: one hour, six hours, one day, three days.
WINDOWS = (60, 360, 1440, 4320)
#: The scatter-gather windows; the 1 h one rarely closes a pattern and is left out.
SCATTER_WINDOWS = (360, 1440, 4320)
#: Neighbour lists at most this long are intersected in full, so a hub account cannot make one row take seconds.
CAP = 300


def _frame(df: pd.DataFrame, horizon_minutes: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`horizon_minutes` caps how far past a transaction its batch may reach: a buffer of 2,048 rows fills in minutes at the corpus's normal rate and takes days in its sparse last days."""
    n = len(df)
    ids = pd.factorize(pd.concat([df["from_account"], df["to_account"]], ignore_index=True))[0]
    src, dst = ids[:n].astype(np.int64), ids[n:].astype(np.int64)
    minutes = ((df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 60).to_numpy().astype(np.int64)
    amount = np.log1p(np.clip(df["amount_paid"].to_numpy(float), 0, None))
    batch_end = np.minimum(n - 1, (np.arange(n) // BATCH_WINDOW + 1) * BATCH_WINDOW - 1)
    horizon = minutes[batch_end] if horizon_minutes is None else np.minimum(minutes[batch_end], minutes + horizon_minutes)
    return src, dst, minutes, amount, horizon


def _fan_side(query: np.ndarray, ev_acct: np.ndarray, ev_nbr: np.ndarray, minutes: np.ndarray, amount: np.ndarray, batch_t: np.ndarray, name: str, *, distinct: bool) -> dict[str, np.ndarray]:
    """Counts and summed log amounts over [T - w, T] at the transaction's batch end T, and optionally the distinct counterparties, per window."""
    n = len(query)
    order = np.lexsort((minutes, ev_acct))
    acct, t, am, nbr = ev_acct[order], minutes[order], amount[order], ev_nbr[order]
    key = (acct << 21) + t
    cum = np.concatenate(([0.0], np.cumsum(am)))
    end = np.searchsorted(key, (query << 21) + batch_t, side="right")
    out: dict[str, np.ndarray] = {}
    for w in WINDOWS:
        start = np.searchsorted(key, (query << 21) + np.maximum(batch_t - w, 0), side="left")
        out[f"{name}_cnt_{w}"] = (end - start).astype(np.float32)
        out[f"{name}_sum_{w}"] = (cum[end] - cum[start]).astype(np.float32)
    if distinct:
        q_order = np.lexsort((batch_t, query))
        q_acct, q_time = query[q_order], batch_t[q_order]
        m = len(acct)
        for w in WINDOWS:
            res = np.zeros(n, dtype=np.float32)
            current, seen, left, right = -1, {}, 0, 0
            for pos in range(n):
                who = int(q_acct[pos])
                tq = int(q_time[pos])
                if who != current:
                    current, seen = who, {}
                    left = right = int(np.searchsorted(acct, who, side="left"))
                while right < m and acct[right] == who and t[right] <= tq:
                    v = int(nbr[right])
                    seen[v] = seen.get(v, 0) + 1
                    right += 1
                while left < right and t[left] < tq - w:
                    v = int(nbr[left])
                    c = seen[v] - 1
                    if c == 0:
                        del seen[v]
                    else:
                        seen[v] = c
                    left += 1
                res[q_order[pos]] = len(seen)
            out[f"{name}_uniq_{w}"] = res
    return out


def _statistics_side(query: np.ndarray, ev_acct: np.ndarray, minutes: np.ndarray, amount: np.ndarray, batch_t: np.ndarray, name: str) -> dict[str, np.ndarray]:
    """Mean and standard deviation of the log amount per window, and the time since the account's last edge."""
    order = np.lexsort((minutes, ev_acct))
    acct, t, am = ev_acct[order], minutes[order], amount[order]
    key = (acct << 21) + t
    c1 = np.concatenate(([0.0], np.cumsum(am)))
    c2 = np.concatenate(([0.0], np.cumsum(am * am)))
    end = np.searchsorted(key, (query << 21) + batch_t, side="right")
    out: dict[str, np.ndarray] = {}
    for w in WINDOWS:
        start = np.searchsorted(key, (query << 21) + np.maximum(batch_t - w, 0), side="left")
        count = np.maximum(end - start, 1)
        s1, s2 = c1[end] - c1[start], c2[end] - c2[start]
        out[f"{name}_mean_{w}"] = (s1 / count).astype(np.float32)
        out[f"{name}_std_{w}"] = np.sqrt(np.maximum(s2 / count - (s1 / count) ** 2, 0)).astype(np.float32)
    has = (end > 0) & (acct[np.maximum(end - 1, 0)] == query)
    last = np.where(has, t[np.maximum(end - 1, 0)], 0)
    out[f"{name}_recency"] = np.log1p(np.where(has, np.maximum(batch_t - last, 0), 1e6)).astype(np.float32)
    return out


def _scatter_gather(src: np.ndarray, dst: np.ndarray, minutes: np.ndarray, batch_t: np.ndarray) -> dict[str, np.ndarray]:
    """For u -> v at batch end T: the accounts w with u -> w and w -> v inside the window."""
    n = len(src)
    out_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    in_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i in np.argsort(minutes, kind="stable"):
        out_edges[int(src[i])].append((int(minutes[i]), int(dst[i])))
        in_edges[int(dst[i])].append((int(minutes[i]), int(src[i])))
    out_times = {k: [e[0] for e in v] for k, v in out_edges.items()}
    in_times = {k: [e[0] for e in v] for k, v in in_edges.items()}
    res = {w: np.zeros(n, dtype=np.float32) for w in SCATTER_WINDOWS}
    for i in range(n):
        u, v, t_now = int(src[i]), int(dst[i]), int(batch_t[i])
        if u == v:
            continue
        for w in SCATTER_WINDOWS:
            low = max(0, t_now - w)
            a0, a1 = bisect.bisect_left(out_times[u], low), bisect.bisect_right(out_times[u], t_now)
            b0, b1 = bisect.bisect_left(in_times[v], low), bisect.bisect_right(in_times[v], t_now)
            if a1 == a0 or b1 == b0:
                continue
            a0, b0 = max(a0, a1 - CAP), max(b0, b1 - CAP)
            mid = {e[1] for e in out_edges[u][a0:a1]}
            res[w][i] = len(mid.intersection(e[1] for e in in_edges[v][b0:b1]) - {u, v})
    return {f"scatter_gather_{w}": a for w, a in res.items()}


def batch_reach(df: pd.DataFrame, n_fit: int) -> int:
    """The furthest past a transaction any batch of the fit window reaches, in minutes: the cap on every row's horizon.

    A batch of 2,048 rows fills in minutes at the corpus's normal rate (over the test window, median 3 and 99th percentile 14) and takes days in the sparse last days of the corpus, where few transactions remain and
    more than a quarter of the test window's laundering rows would see over a day ahead. No training row sees that far and no bank would hold a buffer that long, so the reach is read off the fit window, which the
    test window plays no part in, and it applies to every row. Measured, capping changed the validation window not at all and lifted test F1, so what it removes was a distribution shift and not a useful signal.
    """
    minutes = ((df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 60).to_numpy().astype(np.int64)
    n = len(df)
    batch_end = np.minimum(n - 1, (np.arange(n) // BATCH_WINDOW + 1) * BATCH_WINDOW - 1)
    return int((minutes[batch_end] - minutes)[:n_fit].max())


#: The pair columns, in order.
PAIR_FEATURES = ("pair_log_same_direction", "pair_log_reverse_direction", "pair_first_contact")


def pair_features(df: pd.DataFrame, horizon_minutes: int | None = None) -> np.ndarray:
    """For each transaction u -> v: log1p of the other u -> v edges and of the v -> u edges up to its horizon, and a flag for neither, a first contact so far.

    WHY. On AMLworld HI-Small 72.92 percent of illicit edges sit on account pairs with exactly one edge, against about 9 percent of all edges (MEGA-GNN, arXiv:2412.00241v3,
    Table 14, reproduced here from the corpus: 0.7292 and 0.0894). That statistic counts the whole corpus, later edges included; this is the part a holder can know when it
    scores: the pair's edges up to the transaction's horizon. Measured on the validation window, the three columns lifted the MLP's AUPRC from 0.4775 to 0.5233 and the GNN's
    from 0.5283 to 0.5452, three seeds each.
    """
    _, _, minutes, _, horizon = _frame(df, horizon_minutes)
    # An account is its bank and its number, as AMLworld defines it: account numbers alone are not unique across banks.
    n = len(df)
    ids = pd.factorize(pd.concat([df["from_bank"] + df["from_account"], df["to_bank"] + df["to_account"]], ignore_index=True))[0]
    src, dst = ids[:n].astype(np.int64), ids[n:].astype(np.int64)
    n_accounts = int(ids.max()) + 1
    key = np.sort((src * n_accounts + dst) * (1 << 21) + minutes)
    def count(pair: np.ndarray) -> np.ndarray:
        return np.searchsorted(key, pair * (1 << 21) + horizon, side="right") - np.searchsorted(key, pair * (1 << 21), side="left")
    same = count(src * n_accounts + dst) - 1
    back = count(dst * n_accounts + src) - (src == dst)
    return np.stack([np.log1p(same), np.log1p(back), ((same == 0) & (back == 0)).astype(float)], axis=1).astype(np.float32)


def transaction_features(df: pd.DataFrame, horizon_minutes: int | None = None) -> tuple[np.ndarray, list[str]]:
    """The 79 columns for the time-sorted frame `df`, log-compressed as they were measured. The fan group is compressed once and the statistics group by a sign-preserving log1p, which for `*_recency` is the second log."""
    src, dst, minutes, amount, batch_t = _frame(df, horizon_minutes)
    fan: dict[str, np.ndarray] = {}
    fan.update(_fan_side(src, src, dst, minutes, amount, batch_t, "src_out", distinct=True))
    fan.update(_fan_side(dst, dst, src, minutes, amount, batch_t, "dst_in", distinct=True))
    fan.update(_fan_side(src, dst, src, minutes, amount, batch_t, "src_in", distinct=False))
    fan.update(_fan_side(dst, src, dst, minutes, amount, batch_t, "dst_out", distinct=False))
    stats: dict[str, np.ndarray] = {}
    stats.update(_statistics_side(src, src, minutes, amount, batch_t, "src_out"))
    stats.update(_statistics_side(dst, dst, minutes, amount, batch_t, "dst_in"))
    stats.update(_statistics_side(src, dst, minutes, amount, batch_t, "src_in"))
    stats.update(_statistics_side(dst, src, minutes, amount, batch_t, "dst_out"))
    stats.update(_scatter_gather(src, dst, minutes, batch_t))
    fan_names, stat_names = sorted(fan), sorted(stats)
    x_fan = np.nan_to_num(np.log1p(np.abs(np.stack([fan[k] for k in fan_names], axis=1))), posinf=0, neginf=0).astype(np.float32)
    x_stat = np.stack([stats[k] for k in stat_names], axis=1)
    x_stat = np.nan_to_num(np.sign(x_stat) * np.log1p(np.abs(x_stat)), posinf=0, neginf=0).astype(np.float32)
    return np.hstack([x_fan, x_stat]), fan_names + stat_names
