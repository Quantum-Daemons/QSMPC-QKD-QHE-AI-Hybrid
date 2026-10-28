#!/usr/bin/env python
"""Train the finance / AML models on the full IBM AMLworld HI-Small corpus.

    T2 teacher   graph-feature-preprocessed gradient boosting (GFP + XGBoost)
    T1 student   logistic regression distilled from the teacher, evaluated under real CKKS

Dataset: 5,078,345 transactions, 5,177 labelled laundering — 1 in 981. That ratio is the whole
reason this script reports what it reports. Accuracy is 99.9% for a model that predicts "clean"
unconditionally, so the headline is minority-class F1 with AUPRC beside it, and the AUPRC baseline
(about 0.001) is printed so the lift is legible.

Baseline to beat: GFP+XGBoost minority-class F1 63.23 +/- 0.17 (NeurIPS 2023 D&B).
SOTA to match:    Multi-PNA+EU 68.16 +/- 2.65 (AAAI-24, vol. 38 no. 10, pp. 11838-11846).

The split is temporal, not random. Transactions are a time series and a random split leaks the
future into the past through the per-account aggregates, which inflates every number.

Usage:
    python scripts/train/finance_aml.py [--nrows N] [--out DIR]
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

#: Fraction of the timeline held out for testing. The paper's own protocol is temporal.
TEST_FRACTION = 0.30

#: Slice of the *training* window reserved for early stopping and for choosing the decision
#: threshold. The laundering rate in HI-Small rises through the timeline (1 in 6863 in the first
#: decile, 1 in 375 in the last), so both must come from a window the booster has not been fit on.
VAL_FRACTION = 0.20

#: Graph features are computed on the training window only; the test window reuses those tables.
#: Recomputing them over the full timeline would leak future edges into past rows.
#:
#: The AMLworld baseline uses IBM's Graph Feature Preprocessor, which ships in `snapml` under a
#: Proprietary licence and is therefore excluded by this project's commercial-OK-only rule. What is
#: reimplemented here is the set of graph patterns the literature describes as carrying the
#: laundering signal — degree and fan-in/fan-out, reciprocity, 2-cycles, scatter-gather, and
#: per-account amount and timing statistics. Expect a gap to the published number and report it.
FEATURES = [
    # edge-local
    "amount_paid_log", "amount_ratio", "currency_mismatch", "is_self_transfer",
    "hour_of_day", "day_of_week", "is_round_amount", "format_risk",
    # account degree / fan
    "src_out_degree", "src_unique_dst", "dst_in_degree", "dst_unique_src",
    "src_fan_out_ratio", "dst_fan_in_ratio",
    # account amount statistics
    "src_amount_mean_log", "src_amount_max_log", "dst_amount_mean_log",
    "amount_z_src", "amount_share_of_src_total",
    # graph patterns
    "is_reciprocal", "two_cycle_count", "scatter_gather_src", "scatter_gather_dst",
    "src_bank_diversity", "dst_bank_diversity", "is_cross_bank",
    # multi-hop temporal structure — the typologies the corpus actually injects
    "closes_cycle_2", "closes_cycle_3", "closes_cycle_4",
    "onward_hops", "src_burst_fanout", "dst_burst_fanin",
]

#: HI-Small_Patterns.txt names the injected typologies: CYCLE (up to 10 hops), GATHER-SCATTER,
#: BIPARTITE, FAN-OUT, SCATTER-GATHER, STACK, RANDOM, FAN-IN — 740 attempts in total. Every one of
#: those is a multi-hop temporal path, which is precisely what one-hop account aggregates cannot
#: see. Reachability is therefore computed up to this many hops on the account graph.
#:
#: Do NOT reach for this constant to close the gap to the published baseline. Raising it is the
#: obvious-looking move and the measurement says it is nearly worthless: the GFP paper
#: (arXiv:2402.08593) decomposes its own contribution and caps multi-hop cycle and scatter-gather
#: features at about 4% relative, against more than 30% for fan-in/fan-out, which is already
#: implemented here. The gap is architectural — per-transaction subgraph patterns versus per-account
#: scalar aggregates — and no value of this constant addresses that.
MAX_CYCLE_HOPS = 4

#: Window for burst fan-in / fan-out degree. The FAN-OUT attempts are described as "max k-degree"
#: and play out over a few days, so a daily window is the right granularity.
BURST_WINDOW = "24h"


def causal_account_features(full):
    """Prior-at-time per-account aggregates: for each row, only edges STRICTLY BEFORE it.

    This is the honest form of the same quantities `graph_features` builds with a whole-window
    ``groupby``. The difference is not cosmetic. A whole-window aggregate gives every transaction by
    an account the SAME degree, including activity that happened after it - measured on HI-Small, the
    median transaction has only 46.4% of its out-degree accrued by its own timestamp, so the feature
    mostly describes the account's future. Training rows therefore saw a leaky feature while test
    rows received the same table as a stale past, and the booster was fitted on one distribution and
    scored on another.

    Computing causally removes both problems at once: it leaks nothing (only the past is used), it
    needs no ``fit_on`` split, and it produces no unseen-account zeros, because an account's first
    transaction legitimately has a prior degree of 0 in train and test alike.

    Measured on a matched 9-10 feature harness, same booster and split:

        whole-window static (leaks across train+test)   AUPRC 0.1731
        causal prior-at-time (this function)            AUPRC 0.0627
        train-fitted static (what shipped before)       AUPRC 0.0432

    so causal is 1.45x the shipped approach, and the leaky arm's 2.76x over causal is the size of
    the leak rather than a result. Full workings: references/finance-gap-investigation.md.

    ``full`` must be the entire frame sorted by timestamp. Returns a DataFrame aligned to its index.
    """
    import pandas as pd

    src, dst = full["from_account"], full["to_account"]
    paid = full["amount_paid"].to_numpy(dtype=np.float64)
    recv = full["amount_received"].to_numpy(dtype=np.float64)
    out = pd.DataFrame(index=full.index)

    # Prior counts. cumcount is the number of earlier rows for that key, which is exactly "degree
    # before this transaction".
    out["src_out_degree"] = src.groupby(src).cumcount().to_numpy(dtype=np.float64)
    out["dst_in_degree"] = dst.groupby(dst).cumcount().to_numpy(dtype=np.float64)

    # Prior DISTINCT counterparties: mark the first occurrence of each (a,b) edge, then count how
    # many such firsts precede this row for that account. Cheap, and exact.
    pair = src.astype(str) + ">" + dst.astype(str)
    first_pair = ~pair.duplicated()
    out["src_unique_dst"] = (first_pair.groupby(src).cumsum() - first_pair.astype(int)
                             ).to_numpy(dtype=np.float64)
    rpair = dst.astype(str) + "<" + src.astype(str)
    first_rpair = ~rpair.duplicated()
    out["dst_unique_src"] = (first_rpair.groupby(dst).cumsum() - first_rpair.astype(int)
                             ).to_numpy(dtype=np.float64)

    # Prior amount statistics. Subtracting the current row from the cumulative sum is what makes
    # these strictly prior rather than inclusive.
    ps = pd.Series(paid, index=full.index)
    prior_sum = ps.groupby(src).cumsum().to_numpy(dtype=np.float64) - paid
    prior_n = np.maximum(out["src_out_degree"].to_numpy(), 1.0)
    src_mean = prior_sum / prior_n
    out["src_amount_mean_log"] = np.log1p(np.clip(src_mean, 0, None))
    # The running max is INCLUSIVE of the current row, unlike the counts and means above which are
    # strictly prior. That is deliberate and it is not a leak: the current transaction's own amount
    # is known at scoring time - it is already a feature - so "largest seen up to and including
    # now" is causally available. Stated because the inconsistency would otherwise look accidental,
    # and because the truncation test cannot distinguish inclusive-of-self from strictly-prior.
    out["src_amount_max_log"] = np.log1p(np.clip(
        ps.groupby(src).cummax().to_numpy(dtype=np.float64), 0, None))
    rs = pd.Series(recv, index=full.index)
    dst_prior_sum = rs.groupby(dst).cumsum().to_numpy(dtype=np.float64) - recv
    out["dst_amount_mean_log"] = np.log1p(np.clip(
        dst_prior_sum / np.maximum(out["dst_in_degree"].to_numpy(), 1.0), 0, None))

    out["src_fan_out_ratio"] = out["src_unique_dst"] / np.maximum(out["src_out_degree"], 1.0)
    out["dst_fan_in_ratio"] = out["dst_unique_src"] / np.maximum(out["dst_in_degree"], 1.0)

    # Prior dispersion, via the running second moment. Same subtract-self trick.
    sq = pd.Series(paid * paid, index=full.index)
    prior_sq = sq.groupby(src).cumsum().to_numpy(dtype=np.float64) - paid * paid
    var = np.maximum(prior_sq / prior_n - src_mean ** 2, 0.0)
    out["amount_z_src"] = (paid - src_mean) / np.maximum(np.sqrt(var), 1.0)
    out["amount_share_of_src_total"] = paid / np.maximum(prior_sum, 1.0)
    return out


#: Sliding window for the burst features, in hours. The cumulative aggregates above answer "how
#: active has this account ever been"; these answer "how active is it RIGHT NOW", and the pair
#: separates a steadily-busy account from one that woke up. 24h is one settlement day.
BURST_WINDOW_H = 24.0


def windowed_causal_features(full, *, window_h: float = BURST_WINDOW_H):
    """Per-transaction RECENT-history features, strictly prior, over a sliding time window.

    MEASURED AND NOT SHIPPED. Adding these six columns to the 32 shipped features moved
    minority-class F1 from 0.1519 to 0.1308 and AUPRC from 0.1233 to 0.1152, while AUROC ROSE from
    0.9526 to 0.9606 - the signature of a change that helps the majority class at a 1-in-981
    positive rate. Kept because it is a negative result worth keeping executable: the recency
    reading of the baseline gap is the obvious next idea, and re-running this costs 15 seconds
    against a day to re-derive it. See references/finance-gap-investigation.md.

    This is the architectural difference the baseline gap note names. The published reference
    computes per-transaction subgraph patterns over a dynamically updated graph; this project
    computed per-account scalar aggregates over all prior time. A cumulative degree cannot separate
    an account that has been busy for a year from one that sent nothing for a year and then sent
    forty transactions this morning - and the second is what a mule looks like.

    Each feature here is the same quantity as its cumulative twin, restricted to
    ``(t - window_h, t)``. Computed by binary search over per-account sorted timestamps, so it is
    O(n log n) and needs no graph library: for each row, find how many of that account's prior rows
    fall inside the window, and difference the account's prefix sums at those two positions.

    Strictly prior by construction - the searchsorted upper bound is the row's own position within
    its account, so its own edge is never counted. That is the same discipline as
    :func:`causal_account_features` and it is what makes the truncation test pass.
    """
    import pandas as pd

    out = pd.DataFrame(index=full.index)
    ts = full["timestamp"].to_numpy(dtype="datetime64[s]").astype("int64").astype(np.float64)
    window_s = window_h * 3600.0
    paid = full["amount_paid"].to_numpy(dtype=np.float64)

    for side, col, amt in (("src", "from_account", paid), ("dst", "to_account", paid)):
        keys = full[col].to_numpy()
        order = np.lexsort((ts, keys))            # group by account, ascending time within account
        k_sorted, t_sorted, a_sorted = keys[order], ts[order], amt[order]
        # Group boundaries: start index of each account's block in the sorted arrays.
        starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
        group_id = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, k_sorted.size]))
        pos_in_group = np.arange(k_sorted.size) - starts[group_id]
        # Prefix sums per account, so a window sum is one subtraction.
        csum = np.r_[0.0, np.cumsum(a_sorted)]
        base = csum[starts[group_id]]             # prefix at the account's first row

        # Left edge of the window, searched WITHIN the account's own block only.
        lo = np.empty(k_sorted.size, dtype=np.int64)
        for g, st in enumerate(starts):
            end = starts[g + 1] if g + 1 < starts.size else k_sorted.size
            blk = t_sorted[st:end]
            lo[st:end] = np.searchsorted(blk, blk - window_s, side="left")
        n_prior_window = pos_in_group - lo                      # excludes self
        sum_window = (csum[starts[group_id] + pos_in_group] - base) - \
                     (csum[starts[group_id] + lo] - base)

        cnt = np.empty(k_sorted.size)
        cnt[order] = n_prior_window
        smw = np.empty(k_sorted.size)
        smw[order] = sum_window
        out[f"{side}_burst_count_{int(window_h)}h"] = cnt
        out[f"{side}_burst_amount_log_{int(window_h)}h"] = np.log1p(np.clip(smw, 0, None))

    # Burstiness: recent activity relative to lifetime activity. 1.0 means everything this account
    # has ever done happened inside the window; near 0 means a long-established account.
    for side in ("src", "dst"):
        life = out.get(f"{side}_burst_count_{int(window_h)}h")
        deg = (full["from_account"] if side == "src" else full["to_account"])
        total_prior = deg.groupby(deg).cumcount().to_numpy(dtype=np.float64)
        out[f"{side}_burst_share"] = life / np.maximum(total_prior, 1.0)
    return out


def load_sorted(nrows: int | None = None):
    """The corpus in the order every window below is cut from: by timestamp, renumbered.

    Shared with `scripts/prep_model_inputs.py` since Round 71, with `temporal_windows`, so the
    browser's evaluation sample is drawn from the same test rows the metrics were measured on. It had
    been drawing from the whole corpus, training rows included.
    """
    return datasets.load_amlworld(nrows=nrows).sort_values("timestamp").reset_index(drop=True)


def temporal_windows(df):
    """``(fit, val, test, before_test)`` by position in the time-sorted frame.

    `before_test` is everything ahead of the test window, which is what the graph aggregates of a
    test row may use.
    """
    cut = int(len(df) * (1.0 - TEST_FRACTION))
    val_cut = int(cut * (1.0 - VAL_FRACTION))
    return df.iloc[:val_cut], df.iloc[val_cut:cut], df.iloc[cut:], df.iloc[:cut]


def graph_features(df, *, fit_on=None, causal=None):
    """Graph feature preprocessing: per-account aggregates joined back onto each edge.

    This is the 'GFP' of the AMLworld baseline — turn the transaction multigraph into tabular
    features a gradient booster can use, rather than running message passing. Fan-in / fan-out
    ratios are the features that actually carry the laundering signal: a mule account is
    characterised by many distinct counterparties relative to its transaction count.

    ``fit_on`` supplies the account tables computed from the training window. Passing the test
    frame with ``fit_on=train_frame`` is what keeps the split honest.
    """
    import pandas as pd

    source = fit_on if fit_on is not None else df

    out_agg = source.groupby("from_account", observed=True).agg(
        src_out_degree=("amount_paid", "size"),
        src_unique_dst=("to_account", "nunique"),
        src_amount_mean=("amount_paid", "mean"),
        src_amount_max=("amount_paid", "max"),
    )
    in_agg = source.groupby("to_account", observed=True).agg(
        dst_in_degree=("amount_received", "size"),
        dst_unique_src=("from_account", "nunique"),
        dst_amount_mean=("amount_received", "mean"),
    )

    f = pd.DataFrame(index=df.index)
    paid = df["amount_paid"].to_numpy(dtype=np.float64)
    recv = df["amount_received"].to_numpy(dtype=np.float64)
    f["amount_paid_log"] = np.log1p(np.clip(paid, 0, None))
    f["amount_ratio"] = np.where(paid > 0, recv / paid, 1.0)
    f["currency_mismatch"] = (df["recv_currency"].astype(str)
                              != df["pay_currency"].astype(str)).astype(np.float32)
    f["is_self_transfer"] = (df["from_account"] == df["to_account"]).astype(np.float32)
    ts = df["timestamp"]
    f["hour_of_day"] = ts.dt.hour.to_numpy(dtype=np.float32)
    f["day_of_week"] = ts.dt.dayofweek.to_numpy(dtype=np.float32)

    # Laundering structuring favours round figures; a genuine payment rarely lands on a clean 1000.
    f["is_round_amount"] = ((paid > 0) & (np.mod(paid, 1000.0) == 0)).astype(np.float32)
    f["is_cross_bank"] = (df["from_bank"].astype(str) != df["to_bank"].astype(str)).astype(np.float32)

    src = out_agg.reindex(df["from_account"]).to_numpy(dtype=np.float64)
    dst = in_agg.reindex(df["to_account"]).to_numpy(dtype=np.float64)
    src = np.nan_to_num(src)     # accounts unseen in the training window get zeros, which is correct:
    dst = np.nan_to_num(dst)     # at scoring time we genuinely know nothing about them
    f["src_out_degree"] = src[:, 0]
    f["src_unique_dst"] = src[:, 1]
    f["src_amount_mean_log"] = np.log1p(np.clip(src[:, 2], 0, None))
    f["src_amount_max_log"] = np.log1p(np.clip(src[:, 3], 0, None))
    f["dst_in_degree"] = dst[:, 0]
    f["dst_unique_src"] = dst[:, 1]
    f["dst_amount_mean_log"] = np.log1p(np.clip(dst[:, 2], 0, None))
    f["src_fan_out_ratio"] = src[:, 1] / np.maximum(src[:, 0], 1.0)
    f["dst_fan_in_ratio"] = dst[:, 1] / np.maximum(dst[:, 0], 1.0)

    # How unusual is this amount for this account, and how much of its throughput is it?
    src_std = source.groupby("from_account", observed=True)["amount_paid"].std()
    src_sum = source.groupby("from_account", observed=True)["amount_paid"].sum()
    std = np.nan_to_num(src_std.reindex(df["from_account"]).to_numpy(dtype=np.float64), nan=0.0)
    tot = np.nan_to_num(src_sum.reindex(df["from_account"]).to_numpy(dtype=np.float64), nan=0.0)
    f["amount_z_src"] = (paid - src[:, 2]) / np.maximum(std, 1.0)
    f["amount_share_of_src_total"] = paid / np.maximum(tot, 1.0)

    # Graph patterns. Reciprocity and 2-cycles are the cheapest structural laundering signatures:
    # value that comes straight back is layering, not commerce. Directed edges are counted once and
    # looked up in both orientations, which is O(E) rather than a join against the full edge list.
    edges = source.groupby(["from_account", "to_account"], observed=True).size()
    fwd = _lookup_pairs(edges, df["from_account"], df["to_account"])
    rev = _lookup_pairs(edges, df["to_account"], df["from_account"])
    f["is_reciprocal"] = (rev > 0).astype(np.float32)
    f["two_cycle_count"] = np.minimum(fwd, rev).astype(np.float32)

    # Scatter-gather: one account distributing to many that then re-converge. Approximated by the
    # counterparty breadth on each side relative to that side's transaction count.
    f["scatter_gather_src"] = src[:, 1] / np.maximum(dst[:, 1], 1.0)
    f["scatter_gather_dst"] = dst[:, 1] / np.maximum(src[:, 1], 1.0)

    # Spreading value across many banks is a classic placement signature.
    src_banks = source.groupby("from_account", observed=True)["to_bank"].nunique()
    dst_banks = source.groupby("to_account", observed=True)["from_bank"].nunique()
    f["src_bank_diversity"] = np.nan_to_num(
        src_banks.reindex(df["from_account"]).to_numpy(dtype=np.float64), nan=0.0)
    f["dst_bank_diversity"] = np.nan_to_num(
        dst_banks.reindex(df["to_account"]).to_numpy(dtype=np.float64), nan=0.0)

    # Multi-hop reachability: does value sent a -> b have a route back to a? A CYCLE attempt is
    # exactly that, and no amount of per-account aggregation can express it.
    cyc = _cycle_closures(source, df)
    for k in range(2, MAX_CYCLE_HOPS + 1):
        f[f"closes_cycle_{k}"] = cyc[k]
    f["onward_hops"] = cyc["onward"]

    # Burst degree: FAN-OUT attempts are described as "max k-degree" over a handful of days, so the
    # peak distinct-counterparty count inside a rolling day separates them from steady commerce.
    f["src_burst_fanout"] = _burst_degree(source, df, side="from")
    f["dst_burst_fanin"] = _burst_degree(source, df, side="to")

    # Payment format carries real signal but is categorical; encode it as the training-window
    # laundering rate for that format (target encoding fitted on the training window only).
    rate = source.groupby("payment_format", observed=True)["label"].mean()
    f["format_risk"] = df["payment_format"].map(rate).astype(np.float32).fillna(0.0)

    # Causal override. The 11 per-account SCALAR aggregates are replaced by their prior-at-time
    # form when supplied; the structural features below them - cycle closures, scatter-gather, bank
    # diversity, burst degree, format risk - stay on the training window, because recomputing a
    # sparse-matrix reachability power at every timestep is not tractable here. The split is stated
    # rather than blurred: `causal_account_features` documents which quantities it covers, and these
    # are exactly the ones the matched-harness experiment isolated.
    if causal is not None:
        for col in causal.columns:
            if col in f.columns:
                f[col] = causal.loc[df.index, col].to_numpy(dtype=np.float64)

    return f[FEATURES].to_numpy(dtype=np.float32)


def _cycle_closures(source, df) -> dict:
    """For each row a->b: is ``a`` reachable from ``b`` in k hops, for k = 2..MAX_CYCLE_HOPS.

    Built with boolean sparse matrix powers over the account graph. A^k has a 1 at (b, a) exactly
    when a k-hop path b -> ... -> a exists, so the edge a->b closes a (k+1)-cycle. Booleans keep the
    products small enough to hold even where hub accounts inflate the fill-in.

    ``onward`` is the destination's out-degree: how much further the value can travel. Terminal
    accounts score 0, which is what distinguishes a leaf payment from a layering hop.
    """
    import pandas as pd
    from scipy import sparse

    accounts = pd.Index(pd.unique(pd.concat([source["from_account"], source["to_account"]],
                                            ignore_index=True)))
    n = len(accounts)
    rows = accounts.get_indexer(source["from_account"])
    cols = accounts.get_indexer(source["to_account"])
    keep = (rows >= 0) & (cols >= 0)
    adj = sparse.csr_matrix((np.ones(keep.sum(), dtype=bool), (rows[keep], cols[keep])),
                            shape=(n, n), dtype=bool)

    # Rows of the frame we need to answer for: (b, a) with b the destination, a the source.
    q_from = accounts.get_indexer(df["to_account"])      # b
    q_to = accounts.get_indexer(df["from_account"])      # a
    known = (q_from >= 0) & (q_to >= 0)

    out: dict = {}
    power = adj
    for k in range(2, MAX_CYCLE_HOPS + 1):
        power = (power @ adj).astype(bool)
        hit = np.zeros(len(df), dtype=np.float32)
        if known.any():
            hit[known] = np.asarray(power[q_from[known], q_to[known]]).ravel().astype(np.float32)
        out[k] = hit

    onward = np.zeros(len(df), dtype=np.float32)
    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    onward[known] = deg[q_from[known]]
    out["onward"] = onward
    return out


def _burst_degree(source, df, *, side: str) -> np.ndarray:
    """Peak distinct-counterparty count for each account inside a rolling ``BURST_WINDOW``."""

    key, other = (("from_account", "to_account") if side == "from" else ("to_account", "from_account"))
    s = source[[key, other, "timestamp"]].sort_values("timestamp")
    # Distinct counterparties per account per calendar day, then the account's worst day. Exact
    # rolling-window distinct counts over 5M edges are far more expensive than this is worth, and
    # a daily bucket already separates a 16-degree fan-out from ordinary traffic.
    per_day = (s.assign(day=s["timestamp"].dt.floor(BURST_WINDOW))
                 .groupby([key, "day"], observed=True)[other].nunique())
    peak = per_day.groupby(level=0, observed=True).max()
    return np.nan_to_num(peak.reindex(df[key]).to_numpy(dtype=np.float64), nan=0.0).astype(np.float32)


def _lookup_pairs(edge_counts, left, right) -> np.ndarray:
    """Count of the directed edge (left -> right) for each row, 0 when absent."""
    import pandas as pd

    idx = pd.MultiIndex.from_arrays([left.to_numpy(), right.to_numpy()])
    return np.nan_to_num(edge_counts.reindex(idx).to_numpy(dtype=np.float64), nan=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None, help="cap rows (smoke runs only)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # Causal aggregates are the DEFAULT since Round 24: measured +88% minority-class F1 and +68%
    # AUPRC over the whole-window form, with no leakage (verified by a truncation test - every
    # feature is bit-identical whether or not later rows exist). --no-causal restores the old
    # behaviour for A/B work; it should not be used to produce shipped artifacts.
    ap.add_argument("--no-causal", dest="causal", action="store_false",
                    help="restore whole-window per-account aggregates (A/B only, worse and leaky "
                         "in the train/test-asymmetry sense; see causal_account_features)")
    ap.add_argument("--student-degree", type=int, default=1,
                    help="polynomial degree for the CKKS student's feature map. "
                         "The expansion is computed on plaintext before encryption, so "
                         "degree 2 costs no extra multiplicative depth -- only slots, and 560 of "
                         "4096 is affordable. MEASURED HERE AND REJECTED: degree 2 raised fidelity "
                         "(agreement 0.9471 -> 0.9651, disagreement on positives 0.3262 -> 0.1956) "
                         "and LOWERED the headline, minority-class F1 0.0185 -> 0.0115. It pays on "
                         "OPS-SAT and does not pay here, which is why this is a flag and not a "
                         "default -- check it per model rather than assuming it generalises.")
    ap.set_defaults(causal=True)
    args = ap.parse_args()

    import xgboost as xgb

    started = time.time()
    train.seed_everything(args.seed)

    print(f"loading AMLworld HI-Small{f' (first {args.nrows:,} rows)' if args.nrows else ''} ...")
    df = load_sorted(nrows=args.nrows)
    n_pos = int(df["label"].sum())
    print(f"  {len(df):,} transactions, {n_pos:,} laundering = 1 in {len(df)/max(n_pos,1):.0f}")

    # Three temporal windows, not two. The decision threshold has to be chosen somewhere the model
    # has not been fitted: picked on the fit window it lands around 0.98 (the booster separates its
    # own training data almost perfectly) and generalises to nothing. Picked on test it reports a
    # number nobody could achieve in deployment.
    fit, val, te, tr = temporal_windows(df)
    print(f"  temporal windows -> fit {len(fit):,} ({int(fit['label'].sum())} pos) | "
          f"val {len(val):,} ({int(val['label'].sum())} pos) | "
          f"test {len(te):,} ({int(te['label'].sum())} pos)")

    # One causal pass over the WHOLE sorted frame. It leaks nothing by construction - every value
    # uses only strictly-earlier rows - so the same block serves fit, val and test, and there is no
    # fit_on split to get wrong for these columns.
    cz = causal_account_features(df) if args.causal else None
    if cz is not None:
        print("  causal per-account aggregates: ON (prior-at-time, no fit_on needed)")
    Xfit = graph_features(fit, causal=cz)
    Xval = graph_features(val, fit_on=fit, causal=cz)
    Xte = graph_features(te, fit_on=tr, causal=cz)
    yfit = fit["label"].to_numpy(dtype=np.int8)
    yval = val["label"].to_numpy(dtype=np.int8)
    yte = te["label"].to_numpy(dtype=np.int8)
    Xtr, ytr = Xfit, yfit

    # scale_pos_weight is what makes a booster look at a 1-in-981 class at all.
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"  training XGBoost (scale_pos_weight={spw:.1f}) ...")
    teacher = xgb.XGBClassifier(
        n_estimators=600, max_depth=8, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=4, reg_lambda=2.0,
        scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
        device="cuda", n_jobs=16, random_state=args.seed,
    )
    teacher.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
    # Train on the GPU, score on the CPU. Leaving the booster on cuda while handing it host arrays
    # makes XGBoost copy every batch back and forth, and it is the CPU graph we export to ONNX.
    teacher.set_params(device="cpu")

    thr = _best_threshold(yval, teacher.predict_proba(Xval)[:, 1])
    proba = teacher.predict_proba(Xte)
    pred = (proba[:, 1] >= thr).astype(int)

    teacher_metrics = metrics.with_companion(yte, pred, proba, headline="minority_class_f1")
    teacher_metrics["decision_threshold"] = round(float(thr), 6)

    # ---- probability calibration ---------------------------------------------------------------
    # Fitted on VAL, scored on TEST. The val rows already pick `thr`, and reusing them for the
    # calibration map is sound precisely because the map's quality is measured on test, which neither
    # step has seen. Fitting and scoring on the same rows is what would flatter it.
    #
    # Platt rather than isotonic is a MEASURED tie-break since Round 49, not the size argument the
    # earlier note made: this calibration set is 710,969 cases, far past the ICML/UAI thresholds, and
    # on this split isotonic and Platt score within noise on Brier and log-loss. Isotonic loses on
    # what is not a proper scoring rule - it is only weakly monotone, and it cost 9.09% of AUPRC by
    # collapsing 1,388,669 distinct scores to 138. No reliability DIAGRAM is drawn - Riley et al. put
    # the floor for a flexible curve at about 200 events - but the CORP terms are SCALARS and
    # pool-adjacent-violators chooses no bin count, so they are reportable exactly where a curve is
    # not. Full workings: references/finance-calibration-isotonic-vs-platt.md.
    #
    # `on_logit=True` IS THE FIX FOR THE DEFECT ROUND 49 MEASURED AND LEFT IN PLACE. This call used
    # to hand `fit_platt` a PROBABILITY. Platt's map is linear in its argument, 90% of these scores
    # are under 1e-4, so the fitted sigmoid returned the one constant 0.00090148 across nine of ten
    # deciles - which is where the +8.7% log-loss this block used to report came from, and the 2.83x
    # ECE(20) the ablation measured alongside it (0.00183223 against 0.00064837 uncalibrated; this
    # block computes no ECE). Fitted on logit(s) instead, the same `fit_platt` - different a and b,
    # since they are the output of a fit on a different argument - gives Brier 0.00142199 and
    # log-loss 0.00818964. That beats the old map on both, and is within noise of isotonic
    # (0.00142218 / 0.00820025), which still WINS on CORP MCB, 9.34e-06 against 1.18e-05: the term
    # printed below as "the term calibration is supposed to move". Platt is kept on the ranking and
    # publication tie-breaks recorded above, not on a clean sweep of the scores. The flag lives in
    # `fit_platt` rather
    # than here so that `Platt.as_dict()` publishes which argument the floats belong to; a transform
    # applied here would leave the published formula describing a different function.
    from qsh.models import calibration as cal

    platt = cal.fit_platt(teacher.predict_proba(Xval)[:, 1], yval, on_logit=True)
    p_raw, p_cal = proba[:, 1], platt.apply(proba[:, 1])
    def _r(d: dict) -> dict:
        """Round the float terms; the CORP dict also carries strings and ints."""
        return {k: (round(v, 8) if isinstance(v, float) else v) for k, v in d.items()}

    teacher_metrics["calibration"] = {
        **platt.as_dict(),
        "fitted_on": "validation split; scored on the held-out test split",
        "brier_raw": round(cal.brier(p_raw, yte), 8),
        "brier_calibrated": round(cal.brier(p_cal, yte), 8),
        "log_loss_raw": round(cal.log_loss(p_raw, yte), 8),
        "log_loss_calibrated": round(cal.log_loss(p_cal, yte), 8),
        "corp_raw": _r(cal.corp_decomposition(p_raw, yte)),
        "corp_calibrated": _r(cal.corp_decomposition(p_cal, yte)),
        "reliability_diagram_defensible_here": (
            "YES on this split, and the first draft of this note said the opposite. Riley et al. "
            "(Stat Med 2021) put the floor for a flexible calibration curve at about 200 events; this "
            "test split carries 2,321. The 164-event figure this project quotes elsewhere is the 2D "
            "OVERLAY payload (8,192 sampled rows at a 0.02 base rate), not the test split, and "
            "carrying it over here would have declined a curve that the data supports - the same "
            "wrong-denominator mistake this repository already documents under 'ask what the metric "
            "was divided by'. The floor genuinely does bind on medical (about 42 events) and "
            "satellite (about 113). The CORP terms below are reported regardless, because PAV picks "
            "no bin count and so they are stable where a binned curve is not."),
        "does_not_move_ranking": (
            "Platt is STRICTLY monotone, so AUC and AUC-PR are unchanged by construction: measured "
            "Round 49, both are identical to the uncalibrated values to ten decimals, and that held "
            "for the map fitted on the probability and for the logit(s) map that now ships alike. "
            "The qualifier matters and was missing - isotonic is only WEAKLY monotone, and on this "
            "same split it moved AUROC by -0.00031674 and AUPRC by -0.01121590 (-9.09%). "
            "Calibration by a strictly monotone map fixes what a probability MEANS, never how the "
            "rows are ordered. ONE HONEST QUALIFIER ON THE MAP THAT SHIPS: logit() clips at 1e-12, "
            "so scores below 1e-12 or above 1-1e-12 would tie. None exist on this split (the raw "
            "range is 4.4e-11 to 0.99997), so no pair is tied and the AUCs above are exact; a "
            "retrain that saturates a leaf would need this re-checked rather than assumed."),
    }
    _c = teacher_metrics["calibration"]
    print("\n  calibration (Platt, fitted on val, scored on test)")
    print(f"    Brier     {_c['brier_raw']:.6f} -> {_c['brier_calibrated']:.6f}")
    print(f"    log-loss  {_c['log_loss_raw']:.6f} -> {_c['log_loss_calibrated']:.6f}")
    print(f"    CORP MCB  {_c['corp_raw']['mcb']:.6f} -> {_c['corp_calibrated']['mcb']:.6f}"
          "   <- the term calibration is supposed to move")
    print(f"    CORP DSC  {_c['corp_raw']['dsc']:.6f}   (discrimination; a monotone map cannot change it)")

    spec = get_spec("finance_aml.gbm")
    print("\n  teacher (GFP + XGBoost)")
    for k, v in teacher_metrics.items():
        print(f"    {k:22s} {v:.6f}" if isinstance(v, float) else f"    {k:22s} {v}")
    print(f"    published baseline     {spec.metric.baseline}  ({spec.metric.baseline_source})")
    print(f"    published ceiling      {spec.metric.ceiling}  ({spec.metric.ceiling_source})")

    # ---- T1 student: what actually runs under CKKS ----------------------------
    print("\n  distilling to the CKKS-linear student ...")
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    # SOFT targets: the teacher's probabilities, not its argmax. Hard labels discard precisely the
    # information distillation exists to transfer, and at 1-in-981 that is most of the signal.
    #
    # QUADRATIC expansion, computed on the plaintext input before encryption, so the encrypted step
    # stays one packed dot product at unchanged multiplicative depth. 32 features become 560, well
    # inside the 4096 slots a degree-8192 ring gives. Round 24 measured this student's ceiling over
    # ANY threshold at 0.0262 minority-class F1 against the teacher's 0.1519, and concluded the
    # binding constraint was capacity - so this tests whether the capacity that was missing is the
    # cipher's or the feature map's.
    # At degree 1 the student is a bare estimator and the CALLER owns the scaling, exactly as every
    # other student in this project does; at degree > 1 the returned Pipeline owns its own scaler and
    # must be fed raw features or it would z-score twice. Getting this backwards silently fits a
    # logistic model on unscaled 32-dimensional features - it converges to something, reports a
    # number, and that number was minority-class F1 0.0068 against 0.0185.
    Xtr_student = Xtr if args.student_degree > 1 else scaler.transform(Xtr)
    student = distil.distil_linear(
        teacher.predict_proba(Xtr), Xtr_student, seed=args.seed, soft=True,
        degree=args.student_degree)

    def _student_features(X):
        return X if args.student_degree > 1 else scaler.transform(X)

    # Tune the student's OWN threshold on validation, then fold it into the bias. Until Round 24
    # the student was scored at sklearn's default 0.5 while the teacher got a tuned cut - so the
    # reported student number was not the student's best operating point, it was an arbitrary one.
    # Folding matters because the encrypted path decides by argmax over two logits, which IS a 0.5
    # threshold: there is nowhere downstream to apply a tuned cut, so it has to live in the weights.
    # Measured: 0.0186 at 0.5, 0.0503 tuned, on identical weights.
    s_val = student.predict_proba(_student_features(Xval))[:, 1]
    s_thr = _best_threshold(yval, s_val)
    distil.fold_threshold_into_intercept(student[-1] if args.student_degree > 1 else student, s_thr)
    print(f"    student threshold {s_thr:.4f} folded into the bias "
          f"(the CKKS argmax has no other place to apply it)")
    student_pred = student.predict(_student_features(Xte))

    # Persist it. Until Round 19 the student was fit, measured and discarded -- so the one model
    # that actually runs under CKKS was the one model nobody could obtain or check.
    student_export = distil.save_student(
        student, Path("models/finance_aml_student"), n_features=len(FEATURES),
        teacher_key="finance_aml.gbm", student_key="finance_aml.student",
        scaler=(None if args.student_degree > 1 else scaler),
        fixtures=(Xte[:64] if args.student_degree > 1 else None))
    print(f"  student persisted: {student_export['n_parameters']} params, "
          f"browser_runnable={student_export['browser_runnable']}")

    fidelity = distil.measure_fidelity(
        teacher_key="finance_aml.gbm", student_key="finance_aml.student",
        teacher_pred=pred, student_pred=student_pred, y_true=yte,
        metric_fn=metrics.minority_class_f1, metric_name="minority_class_f1",
        student_params=int(_est(student).coef_.size + _est(student).intercept_.size),
        teacher_params=int(teacher.get_booster().trees_to_dataframe().shape[0]),
    )
    print(f"    agreement with teacher      {fidelity.agreement:.4f}")
    print(f"    disagreement on positives   {fidelity.disagreement_on_positives:.4f}"
          "   <- the number that matters at 1-in-981")
    print(f"    minority-F1 teacher/student {fidelity.teacher_metric:.4f} / {fidelity.student_metric:.4f}"
          f"  (delta {fidelity.metric_delta_pp:+.2f} pp)")

    # ---- artefacts ------------------------------------------------------------
    out = Path(args.out or (Path.cwd() / "models" / "finance_aml_gbm"))
    out.mkdir(parents=True, exist_ok=True)
    fixtures = Xte[np.random.default_rng(args.seed).choice(len(Xte), size=min(512, len(Xte)),
                                                           replace=False)]
    report = export.verify_and_quantize(teacher, fixtures, out, kind="xgboost",
                                        n_features=len(FEATURES))
    if report.int8_skipped_reason:
        print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB "
              f"(int8 skipped: {report.int8_skipped_reason})")
    else:
        print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB -> int8 {report.int8_bytes/1e6:.2f} MB "
              f"({report.compression:.1f}x), int8 agreement {report.int8_agreement:.4f}")

    (out / "feature_spec.json").write_text(json.dumps({
        "features": FEATURES,
        "decision_threshold": float(thr),
        "notes": "Graph aggregates are fitted on the training window only; unseen accounts score 0.",
    }, indent=2) + "\n")

    train.record_metrics("finance_aml.gbm", {
        **teacher_metrics,
        "dataset": spec.dataset.name,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "positive_rate": round(float(n_pos / len(df)), 8),
        "split": "temporal, inductive",
        "baseline_gap_note": (
            "This reaches minority-class F1 well below the published 0.6323 (GFP+XGBoost, AMLworld "
            "HI-Small, NeurIPS 2023 D&B). The cause is a feature ARCHITECTURE difference, not a "
            "tuning one: the gap is unreachable at any threshold, because measured AUPRC is far "
            "below what that F1 requires. IBM's Graph Feature Preprocessor computes per-transaction "
            "subgraph patterns over a dynamically updated graph; this project computes per-account "
            "scalar aggregates. A degree is one number per account, whereas a GFP feature is a "
            "pattern match in that transaction's own neighbourhood at the moment it arrives. GFP "
            "ships in `snapml` under a Proprietary licence and is excluded by this project's "
            "commercial-OK-only rule, but the licence is not itself the cause - the architecture is, "
            "and it is reimplementable without `snapml`, simply at more cost than renaming "
            f"aggregates. Reachability is computed to {MAX_CYCLE_HOPS} hops against injected CYCLE "
            "attempts of up to 10; that was previously recorded as a main cause and is NOT one - the "
            "GFP paper (arXiv:2402.08593) decomposes its own contribution and caps multi-hop cycle "
            "and scatter-gather features at about 4% relative, while fan-in/fan-out, which is "
            "reimplemented here, carries more than 30%. The evaluation is strictly inductive and "
            "temporally split. Per-account aggregates are causal (each row sees only edges strictly "
            "before it); the structural block - cycle closure, scatter-gather, bank diversity, burst "
            "degree - is still fitted on the training window, because recomputing sparse "
            "reachability at every timestep is not tractable here. Reported as measured; not "
            "adjusted to close the gap. Full workings: references/finance-gap-investigation.md."
        ),
        "git_sha": train.git_sha(),
        "data_digest": train.digest_array(Xtr[:10000], ytr[:10000]),
        # Mirror telecom_ids: when int8 is skipped there is no agreement to report, so record the
        # reason instead of a number. Emitting int8_agreement 1.0 beside int8_bytes 0 read as a
        # gate that passed rather than a step that never ran.
        "export": ({"fp32_bytes": report.fp32_bytes, "opset": report.opset,
                    "int8_skipped_reason": report.int8_skipped_reason}
                   if report.int8_skipped_reason else
                   {"fp32_bytes": report.fp32_bytes, "int8_bytes": report.int8_bytes,
                    "int8_agreement": report.int8_agreement, "opset": report.opset}),
        "wall_clock_s": round(time.time() - started, 1),
    })
    train.record_metrics("finance_aml.student", fidelity.as_metrics())
    print(f"\n  wrote {out} and models/metrics.json in {time.time()-started:.0f}s")
    return 0


def _est(student):
    """The estimator carrying the coefficients: a pipeline student keeps them on its last step."""
    return student[-1] if hasattr(student, "steps") else student


def _best_threshold(y, score) -> float:
    """Threshold maximising minority-class F1 on the given (training) split.

    ``precision_recall_curve`` returns n+1 precision/recall points against n thresholds, where the
    final point is the degenerate (recall 0, precision 1) end of the curve and has no threshold.
    Dropping that last point is what makes index i line up with thresholds[i].
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y, score)
    if thresholds.size == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


if __name__ == "__main__":
    raise SystemExit(main())
