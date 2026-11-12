#!/usr/bin/env python
"""Train the finance / AML models on the full IBM AMLworld HI-Small corpus.

    T2 teacher   graph-feature gradient boosting (this project's own graph features + XGBoost;
                 IBM's Graph Feature Preprocessor itself is excluded, see FEATURES below)
    T1 student   logistic regression distilled from the teacher, built for CKKS. The public page
                 encrypts it on the 8,192 test-window rows of the teacher's sample (since Round 84)

Dataset: 5,078,345 transactions, 5,177 labelled laundering — 1 in 981. That ratio is the whole
reason this script reports what it reports. Accuracy is 99.9% for a model that predicts "clean"
unconditionally, so the headline is minority-class F1 with AUPRC beside it, and the AUPRC baseline
(about 0.001) is printed so the lift is legible.

Baseline to beat: GFP+XGBoost minority-class F1 63.23 +/- 0.17 (NeurIPS 2023 D&B).
Best published:   Multi-FraudGT 75.81 +/- 0.75 on AML Small HI (MEGA, TMLR 08/2026,
                  arXiv:2412.00241v3, Table 16), the registry's ceiling; the paper's own MEGA-PNA
                  scores 74.01 +/- 1.55 there. Multi-PNA+EU (AAAI-24, 68.16 +/- 2.65) is no longer
                  the state of the art.

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
sys.path.insert(1, str(Path(__file__).resolve().parent))

from qsh.models import datasets, distil, export, metrics, train  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402

#: Fraction of the TRANSACTIONS, in time order, held out for testing (the last 30 percent of rows,
#: which spans a larger share of the calendar). The paper's own protocol is temporal too, but
#: 60/20/20 by transaction index, against 56/14/30 here once VAL_FRACTION is applied.
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
#: approximated here, as per-account aggregates, is the set of graph patterns the literature describes as carrying the
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
#: (arXiv:2402.08593) decomposes its own contribution, and its Table 5 (HI-Small, XGBoost, batch
#: 128) moves minority F1 from 19.75 to 56.88 with fan-in/fan-out, which is approximated here by
#: per-account aggregates, then only to 58.60 with multi-hop patterns ("up to 4%" in its text means F1 points, not a
#: relative gain). The gap is architectural - per-transaction subgraph patterns versus per-account
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


def model_inputs(df, fit, val, te, tr, causal):
    """The 114 inputs every finance model reads since Round 86, as fit, validation and test matrices, with their names and the batch reach.

    The 32 per-account aggregates of `graph_features`, then the 79 per-transaction window features and the 3 pair features of
    `finance_aml_fan`, the last two up to each transaction's horizon: the end of its batch of 2,048 and at most `batch_reach` minutes past it.
    One function, so the teacher, its CKKS student, the MLP and the GNN cannot drift apart on what a row is.
    """
    from finance_aml_fan import PAIR_FEATURES, batch_reach, pair_features, transaction_features

    reach = batch_reach(df, len(fit))
    extra, names = transaction_features(df, horizon_minutes=reach)
    tail = np.hstack([extra, pair_features(df, horizon_minutes=reach)])
    n_fit, n_val = len(fit), len(val)
    base = (graph_features(fit, causal=causal), graph_features(val, fit_on=fit, causal=causal), graph_features(te, fit_on=tr, causal=causal))
    parts = (tail[:n_fit], tail[n_fit:n_fit + n_val], tail[n_fit + n_val:])
    matrices = tuple(np.hstack([b, t]).astype(np.float32) for b, t in zip(base, parts, strict=True))
    return matrices, list(FEATURES) + names + list(PAIR_FEATURES), reach


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
    (Xfit, Xval, Xte), feature_names, reach = model_inputs(df, fit, val, te, tr, cz)
    print(f"  {len(feature_names)} inputs: per-account aggregates, per-transaction windows and pair counts, reaching at most {reach} minutes past each transaction")
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
    # step has seen. Fitting and scoring on the same rows is what would flatter it. `on_logit=True`
    # fits the map on logit(s): Round 49 measured that a map fitted on the probability itself returned
    # one constant across nine of ten deciles (references/finance-calibration-isotonic-vs-platt.md).
    # Every figure the record states about the map, isotonic regression beside it, and the ranking is
    # measured in this run by `teacher_calibration`; until Round 86 three of them were Round 49's text.
    teacher_metrics["calibration"] = teacher_calibration(
        teacher.predict_proba(Xval)[:, 1], yval, proba[:, 1], yte,
        fitted_by="scripts/train/finance_aml.py, in the run that fitted the teacher")
    _c = teacher_metrics["calibration"]
    print("\n  calibration (Platt, fitted on val, scored on test)")
    print(f"    Brier     {_c['brier_raw']:.6f} -> {_c['brier_calibrated']:.6f}")
    print(f"    log-loss  {_c['log_loss_raw']:.6f} -> {_c['log_loss_calibrated']:.6f}")
    print(f"    CORP MCB  {_c['corp_raw']['mcb']:.6f} -> {_c['corp_calibrated']['mcb']:.6f}"
          "   <- the term calibration is supposed to move")
    print(f"    CORP DSC  {_c['corp_raw']['dsc']:.6f}   (discrimination; a monotone map cannot change it)")

    spec = get_spec("finance_aml.gbm")
    print("\n  teacher (own graph features + XGBoost)")
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
        student, Path("models/finance_aml_student"), n_features=len(feature_names),
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
                                        n_features=len(feature_names))
    if report.int8_skipped_reason:
        print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB "
              f"(int8 skipped: {report.int8_skipped_reason})")
    else:
        print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB -> int8 {report.int8_bytes/1e6:.2f} MB "
              f"({report.compression:.1f}x), int8 agreement {report.int8_agreement:.4f}")

    (out / "feature_spec.json").write_text(json.dumps({
        "features": feature_names,
        "decision_threshold": float(thr),
        "batch_reach_minutes": reach,
        "notes": ("The first 32 are per-account aggregates, fitted on the training window only, where unseen accounts score 0; the rest are per-transaction "
                  "window features and pair counts up to the end of the transaction's batch of 2,048, at most batch_reach_minutes past it."),
    }, indent=2) + "\n")

    train.record_metrics("finance_aml.gbm", {
        **teacher_metrics,
        "dataset": spec.dataset.name,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "positive_rate": round(float(n_pos / len(df)), 8),
        "split": "temporal, inductive",
        "n_features": len(feature_names), "batch_reach_minutes": reach,
        "baseline_gap_note": teacher_gap_note(len(feature_names), teacher_metrics["minority_class_f1"], reach),
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
    # The student's calibration in the same record, so a retrain refits it: written separately, the next record of this
    # student would carry the previous weights' map forward and the page would apply it to new logits (Round 86).
    student_cal = student_calibration(
        student.decision_function(_student_features(Xval)), yval,
        student.decision_function(_student_features(Xte)), yte,
        fitted_by="scripts/train/finance_aml.py, in the run that fitted the student")
    train.record_metrics("finance_aml.student", {**fidelity.as_metrics(), "calibration": student_cal})
    print(f"\n  wrote {out} and models/metrics.json in {time.time()-started:.0f}s")
    return 0


def _est(student):
    """The estimator carrying the coefficients: a pipeline student keeps them on its last step."""
    return student[-1] if hasattr(student, "steps") else student


_STUDENT_ARGUMENT = (r"the student's P(laundering), $q = \sigma(w^\top z + c)$, where $w^\top z + c$ is the "
                     "decision logit the encrypted path decrypts, $z$ the z-scored features and $c$ the student's "
                     "intercept (`b` in its weights file). The tuned threshold is folded into $c$, so the operating "
                     "point is $q = 0.5$, a logit of 0")
_STUDENT_RAW_IS = ("the student's own probability, sigmoid of the decision logit. The folded threshold puts 0.5 at the "
                   "operating point, so it is not a risk.")
_STUDENT_WHY_TAIL = ("and it is the teacher's method, so the page's two decision curves differ by the "
                     "model and not by the map. scripts/measure_decision_curve.py records the student's window curve under both "
                     "maps.")


def network_calibration(logits_val, y_val, logits_te, y_te, *, fitted_by: str, model: str, platt=None) -> dict:
    """The finance MLP's or GNN's Platt map, by `student_calibration` itself, so the four finance records compare models and not maps (Round 86).

    Each network ends in two logits with the tuned threshold folded into the last bias, so P(laundering) is the softmax of
    the two, the sigmoid of their difference, and it is 0.5 at the operating point: not a risk, the `raw` arm. Trained with
    class weights, both put that raw probability well above the base rate on average, which the map fitted on validation
    corrects. The pages show the logits and the decision, not this probability.
    """
    lv, lt = np.asarray(logits_val, dtype=np.float64), np.asarray(logits_te, dtype=np.float64)
    return {"page_reads_this_map": False, **student_calibration(
        lv[:, 1] - lv[:, 0], y_val, lt[:, 1] - lt[:, 0], y_te, fitted_by=fitted_by,
        argument=(rf"the {model}'s P(laundering), $q = \sigma(z_1 - z_0)$, the softmax of its two logits $z_0, z_1$. The "
                  "tuned threshold is folded into the last layer's bias, so the operating point is $q = 0.5$"),
        raw_is=(f"the {model}'s own softmax probability. The folded threshold puts 0.5 at the operating point and the "
                "class weights of its training raise it, so it is not a risk."),
        why_tail=("and it is the method of the teacher's and the student's records, so the four finance records differ by "
                  "the model and not by the map."), noun=model, platt=platt)}


def student_calibration(logit_val, y_val, logit_te, y_te, *, fitted_by: str, argument: str = _STUDENT_ARGUMENT,
                        raw_is: str = _STUDENT_RAW_IS, why_tail: str = _STUDENT_WHY_TAIL, noun: str = "student",
                        platt=None) -> dict:
    """The CKKS student's Platt map, fitted over validation and scored on test (Round 86).

    The page draws a decision curve from the student's DECRYPTED logits, and a decision curve reads a risk against a
    threshold, so the student's output needs a map to a risk first. The decision logit is ``W . z + b`` with the tuned
    threshold already folded into ``b`` (`distil.fold_threshold_into_intercept`), which puts the student's emitted
    probability, ``sigmoid(logit)``, at 0.5 at the operating point: it is not a risk, and it is the `raw` arm below.

    Fitted on that probability with ``on_logit=True``, the teacher's convention, rather than on the logit itself. The
    map then takes what every surface holds: the student's ONNX graph emits the probability (float32, at a resolution of
    about 1.2e-7 near zero, far below any threshold here), and the private-inference figure takes the sigmoid of the
    decrypted logit. A map fitted on the logit, applied to a graph's probability, would be a different function. The
    windows are the teacher's map's: fitted on validation, which also chose the threshold, and scored on test, which
    neither step saw.

    Isotonic regression is measured beside it on the same windows rather than argued against. The teacher's comparison
    (Round 49) measured a different model, and its text, which `Platt.as_dict()` publishes, is replaced here so the
    student's record states nothing measured on the teacher. `noun` names the model in that sentence: the MLP and the GNN
    are distilled from IBM's Multi-PNA+EU, so "this student's ... not carried from the teacher" would name the wrong one.

    Called by `main()` in the run that fits the student, and by `scripts/finance_student_calibration.py` on the published
    weights. A record written without it would carry a previous student's map forward onto new weights.
    """
    from sklearn.isotonic import IsotonicRegression

    from qsh.models import calibration as cal

    lv, lt = np.asarray(logit_val, dtype=np.float64), np.asarray(logit_te, dtype=np.float64)
    yv, yt = np.asarray(y_val).astype(int), np.asarray(y_te).astype(int)
    pv, p_raw = 1.0 / (1.0 + np.exp(-lv)), 1.0 / (1.0 + np.exp(-lt))
    # None in a training run, which fits the map; a script rewriting a record from published weights passes the published
    # map, so the record keeps the floats the page applies (a refit moves them in the eighth digit).
    if platt is None:
        platt = cal.fit_platt(pv, yv, on_logit=True)
    p_cal = platt.apply(p_raw)
    # Isotonic depends on the order of its argument alone, so the logit and the probability fit the same map.
    p_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(lv, yv).predict(lt)
    clipped = int(((p_raw < cal.LOGIT_CLIP_EPS) | (p_raw > 1.0 - cal.LOGIT_CLIP_EPS)).sum())

    def _r(d: dict) -> dict:
        return {k: (round(v, 8) if isinstance(v, float) else v) for k, v in d.items()}

    brier_cal, ll_cal = cal.brier(p_cal, yt), cal.log_loss(p_cal, yt)
    brier_iso, ll_iso = cal.brier(p_iso, yt), cal.log_loss(p_iso, yt)
    auprc = {k: metrics.auprc(yt, p) for k, p in (("raw", p_raw), ("calibrated", p_cal), ("isotonic", p_iso))}
    record = platt.as_dict()

    def _vs(iso: float, platt_: float) -> str:
        """Isotonic against Platt in words, from the numbers, so the sentence is true of whatever run wrote it."""
        return "below" if iso < platt_ else "above" if iso > platt_ else "equal to"

    distinct, zeros = int(np.unique(p_iso).size), int((p_iso == 0).sum())
    return {
        **record,
        # The student's intercept is c here, not the weights file's `b`: the card prints this beside Platt's own b.
        "argument": argument,
        "test_rows_clipped_by_logit": clipped,
        "fitted_on": "validation split; scored on the held-out test split",
        "fitted_by": fitted_by,
        "raw_is": raw_is,
        "brier_raw": round(cal.brier(p_raw, yt), 8),
        "brier_calibrated": round(brier_cal, 8),
        "log_loss_raw": round(cal.log_loss(p_raw, yt), 8),
        "log_loss_calibrated": round(ll_cal, 8),
        "corp_raw": _r(cal.corp_decomposition(p_raw, yt)),
        "corp_calibrated": _r(cal.corp_decomposition(p_cal, yt)),
        "auprc_raw": round(auprc["raw"], 8),
        "auprc_calibrated": round(auprc["calibrated"], 8),
        "isotonic_measured": {
            "brier": round(brier_iso, 8), "log_loss": round(ll_iso, 8), "auprc": round(auprc["isotonic"], 8),
            "distinct_outputs": distinct, "rows_at_zero": zeros,
        },
        "why_not_isotonic": (
            f"Measured on this {noun}'s windows, not carried from the gradient-boosting teacher's record. Isotonic's Brier {brier_iso:.8f} is "
            f"{_vs(brier_iso, brier_cal)} Platt's {brier_cal:.8f} and its log-loss {ll_iso:.8f} is {_vs(ll_iso, ll_cal)} "
            f"{ll_cal:.8f}; lower is better on both. Platt is kept as a choice, not as the better score: it is strictly "
            "monotone, so the curve ranks rows as the student does (AUPRC "
            f"{auprc['calibrated']:.8f} against the raw {auprc['raw']:.8f}"
            + (f", with {clipped:,} test rows tied at the logit clip" if clipped else ", no row tied at the logit clip")
            + f"; isotonic's {auprc['isotonic']:.8f} comes from the test window's scores tied at {distinct} values, "
            f"{zeros:,} of them exactly zero), " + why_tail),
    }


def teacher_gap_note(n_features: int, minority_f1: float, reach: int) -> str:
    """The teacher card's "Gap to the published baseline", from its run's figures; `scripts/finance_gap_notes.py` rewrites a record with it.

    0.6323 is GFP + XGBoost at a batch of 128 transactions; these features are computed in batches of 2,048, where the GFP
    paper reports 64.77 (Blanusa et al., arXiv:2402.08593, Table 4), so the note names both (Round 86, read at the source).
    """
    return (
        f"On this project's split, 56/14/30 by time with nothing read past a transaction's horizon, the {n_features} inputs reach minority-class "
        f"F1 {minority_f1:.4f}, against the published 0.6323 for GFP + XGBoost on HI-Small (Altman et al., NeurIPS 2023 D&B), "
        "measured on the paper's 60/20/20 split at a batch of 128 transactions; at the batch of 2,048 these features use, the GFP paper reports 64.77 "
        "(Blanusa et al., arXiv:2402.08593, Table 4). The inputs are the 32 per-account aggregates, 79 per-transaction fan-in, fan-out, vertex-statistic and "
        f"scatter-gather features in the Graph Feature Preprocessor's style, computed in batches of 2,048 and at most {reach} minutes past the transaction, "
        "and 3 pair counts; there is no cycle enumeration. Until Round 86 the teacher read the 32 aggregates alone and reached 0.1519, the level of the "
        "GFP paper's basic-features row (Table 5: 19.75). Full workings: references/finance-gap-investigation.md.")


def teacher_calibration(p_val, y_val, p_te, y_te, *, fitted_by: str, platt=None) -> dict:
    """The teacher's Platt map, with everything its record says about it measured on the windows it was fitted and scored on.

    Fitted on validation, which also chose the threshold, and scored on test, which neither step saw. Until Round 86 the
    record also carried three paragraphs of text: `Platt.as_dict()`'s isotonic comparison and two notes, on the ranking
    and on the clip, all measured in Round 49 on a teacher that has since been retrained twice, and the Hub card printed
    the first of them beside the current map's own Brier score. Each is a sentence about this run's numbers now.

    `platt` is None in the trainer, which fits the map. `scripts/finance_teacher_calibration.py` passes the published map,
    so a record rewritten from the published graph describes the map the page applies rather than a refit of it.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    from qsh.models import calibration as cal

    pv, pt = np.asarray(p_val, dtype=np.float64), np.asarray(p_te, dtype=np.float64)
    yv, yt = np.asarray(y_val).astype(int), np.asarray(y_te).astype(int)
    if platt is None:
        platt = cal.fit_platt(pv, yv, on_logit=True)
    p_cal = platt.apply(pt)
    p_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(pv, yv).predict(pt)
    eps = platt.logit_clip_eps
    outside = (pt < eps) | (pt > 1.0 - eps)
    clipped = int(outside.sum())
    # The clip ties rows only where the scores hold them apart: a float32 graph already emits exactly 0.0 for every
    # probability below its resolution, so its clipped rows can sit at one value per side before the map sees them.
    clipped_values = sorted({float(v) for v in pt[outside]})
    sides = int((pt < eps).any()) + int((pt > 1.0 - eps).any())

    def _r(d: dict) -> dict:
        return {k: (round(v, 8) if isinstance(v, float) else v) for k, v in d.items()}

    arms = {"raw": pt, "calibrated": p_cal, "isotonic": p_iso}
    # The map is fitted on validation and scored on test, and the two windows need not share a laundering rate.
    val_brier = {"raw": cal.brier(pv, yv), "calibrated": cal.brier(platt.apply(pv), yv)}
    rate_val, rate_te = float(yv.mean()), float(yt.mean())
    brier = {k: cal.brier(p, yt) for k, p in arms.items()}
    loss = {k: cal.log_loss(p, yt) for k, p in arms.items()}
    auprc = {k: metrics.auprc(yt, p) for k, p in arms.items()}
    auroc = {k: float(roc_auc_score(yt, p)) for k, p in arms.items()}
    distinct_raw, distinct_iso = int(np.unique(pt).size), int(np.unique(p_iso).size)
    zeros, zero_events = int((p_iso == 0).sum()), int(((p_iso == 0) & (yt == 1)).sum())
    events = int(yt.sum())

    def _vs(a: float, b: float) -> str:
        return "below" if a < b else "above" if a > b else "equal to"

    def _moves(after: float, before: float) -> str:
        return "lowers" if after < before else "raises" if after > before else "leaves"

    ranks_kept = auroc["calibrated"] == auroc["raw"] and auprc["calibrated"] == auprc["raw"]
    record = platt.as_dict()
    return {
        **record,
        "fitted_on": "validation split; scored on the held-out test split",
        "fitted_by": fitted_by,
        "test_rows_clipped_by_logit": clipped,
        "brier_raw": round(brier["raw"], 8),
        "brier_calibrated": round(brier["calibrated"], 8),
        "log_loss_raw": round(loss["raw"], 8),
        "log_loss_calibrated": round(loss["calibrated"], 8),
        "corp_raw": _r(cal.corp_decomposition(pt, yt)),
        "corp_calibrated": _r(cal.corp_decomposition(p_cal, yt)),
        "auprc_raw": round(auprc["raw"], 8),
        "auprc_calibrated": round(auprc["calibrated"], 8),
        "auroc_raw": round(auroc["raw"], 8),
        "auroc_calibrated": round(auroc["calibrated"], 8),
        "isotonic_measured": {
            "brier": round(brier["isotonic"], 8), "log_loss": round(loss["isotonic"], 8),
            "auprc": round(auprc["isotonic"], 8), "auroc": round(auroc["isotonic"], 8),
            "distinct_outputs": distinct_iso, "rows_at_zero": zeros, "events_at_zero": zero_events,
        },
        "validation_brier": {"raw": round(val_brier["raw"], 8), "calibrated": round(val_brier["calibrated"], 8),
                             "laundering_rate": round(rate_val, 8)},
        "test_laundering_rate": round(rate_te, 8),
        "reliability_diagram_defensible_here": (
            f"{'YES' if min(events, len(yt) - events) >= 200 else 'NO'} on the test split, which carries {events:,} events "
            f"and {len(yt) - events:,} non-events. Van Calster et "
            "al. (J Clin Epidemiol 2016) ask for at least 200 of each before a flexible calibration curve, and Riley et al. "
            "(Stat Med 2021) show that such rules of thumb can still be too few. NO on the browser's evaluation sample, "
            "drawn from the same window with fewer than 200 events, which is why the page draws no curve there and says "
            "so. The CORP terms are reported regardless, because pool-adjacent-violators picks no bin count, so they are "
            "stable where a binned curve is not."),
        "does_not_move_ranking": (
            "Platt is strictly monotone, so it cannot reorder rows: on this test window AUROC "
            f"{auroc['calibrated']:.8f} against the raw {auroc['raw']:.8f} and AUPRC {auprc['calibrated']:.8f} against "
            f"{auprc['raw']:.8f}" + ("" if ranks_kept else ", which differ only where the map's float64 output ties")
            + f". logit() clips at {eps:g}, so scores outside [{eps:g}, 1 - {eps:g}] would tie under the map: "
            + (f"none exist here (the raw range is {pt.min():.3g} to {pt.max():.6f})." if not clipped
               else f"{clipped:,} test rows are there, every one at exactly " + " or ".join(f"{v:g}" for v in clipped_values)
                    + " in the scores, so they are tied before the map and the clip adds no tie." if len(clipped_values) == sides
               else f"{clipped:,} test rows are there, at {len(clipped_values):,} distinct values in the scores, which the clip "
                    "ties.")
            + f" Isotonic regression is only weakly monotone, and on the same rows moves AUROC to {auroc['isotonic']:.8f} "
            f"and AUPRC to {auprc['isotonic']:.8f}."),
        "why_not_isotonic": (
            f"Measured in the run that wrote this record, on the teacher's own windows. The map {_moves(brier['calibrated'], brier['raw'])} "
            f"the Brier score from {brier['raw']:.8f} to {brier['calibrated']:.8f} and {_moves(loss['calibrated'], loss['raw'])} "
            f"the log-loss from {loss['raw']:.8f} to {loss['calibrated']:.8f} on the test window, where {rate_te:.3%} of rows "
            f"are laundering; on the validation rows it was fitted on, {rate_val:.3%} of them laundering, it "
            f"{_moves(val_brier['calibrated'], val_brier['raw'])} the Brier score from {val_brier['raw']:.8f} to "
            f"{val_brier['calibrated']:.8f}. Isotonic regression fitted on the same "
            f"validation rows scores Brier {brier['isotonic']:.8f}, {_vs(brier['isotonic'], brier['calibrated'])} Platt's, and "
            f"log-loss {loss['isotonic']:.8f}, {_vs(loss['isotonic'], loss['calibrated'])} Platt's; lower is better on both. "
            "Platt is kept as a choice, not as the better score: it is strictly monotone and two floats the browser applies, "
            f"while isotonic ties the test window's {distinct_raw:,} distinct scores at {distinct_iso} values, {zeros:,} rows at "
            f"exactly zero, {zero_events} of them laundering, and it is the method of the student's and the networks' "
            "records, so the finance records differ by the model and not by the map."),
    }


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
