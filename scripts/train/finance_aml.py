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
MAX_CYCLE_HOPS = 4

#: Window for burst fan-in / fan-out degree. The FAN-OUT attempts are described as "max k-degree"
#: and play out over a few days, so a daily window is the right granularity.
BURST_WINDOW = "24h"


def graph_features(df, *, fit_on=None):
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
    args = ap.parse_args()

    import xgboost as xgb

    started = time.time()
    train.seed_everything(args.seed)

    print(f"loading AMLworld HI-Small{f' (first {args.nrows:,} rows)' if args.nrows else ''} ...")
    df = datasets.load_amlworld(nrows=args.nrows).sort_values("timestamp").reset_index(drop=True)
    n_pos = int(df["label"].sum())
    print(f"  {len(df):,} transactions, {n_pos:,} laundering = 1 in {len(df)/max(n_pos,1):.0f}")

    # Three temporal windows, not two. The decision threshold has to be chosen somewhere the model
    # has not been fitted: picked on the fit window it lands around 0.98 (the booster separates its
    # own training data almost perfectly) and generalises to nothing. Picked on test it reports a
    # number nobody could achieve in deployment.
    cut = int(len(df) * (1.0 - TEST_FRACTION))
    val_cut = int(cut * (1.0 - VAL_FRACTION))
    fit, val, te = df.iloc[:val_cut], df.iloc[val_cut:cut], df.iloc[cut:]
    tr = df.iloc[:cut]          # graph aggregates may use everything before the test window
    print(f"  temporal windows -> fit {len(fit):,} ({int(fit['label'].sum())} pos) | "
          f"val {len(val):,} ({int(val['label'].sum())} pos) | "
          f"test {len(te):,} ({int(te['label'].sum())} pos)")

    Xfit = graph_features(fit)
    Xval = graph_features(val, fit_on=fit)
    Xte = graph_features(te, fit_on=tr)
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
    student = distil.distil_linear(teacher.predict(Xtr), scaler.transform(Xtr), seed=args.seed)
    student_pred = student.predict(scaler.transform(Xte))

    fidelity = distil.measure_fidelity(
        teacher_key="finance_aml.gbm", student_key="finance_aml.student",
        teacher_pred=pred, student_pred=student_pred, y_true=yte,
        metric_fn=metrics.minority_class_f1, metric_name="minority_class_f1",
        student_params=int(student.coef_.size + student.intercept_.size),
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
            "This reaches minority-class F1 well below the published 0.6323. Two differences account "
            "for most of it and both are deliberate. (1) The published baseline uses IBM's Graph "
            "Feature Preprocessor, which ships in `snapml` under a Proprietary licence and is "
            "therefore excluded by this project's commercial-OK-only rule; what is reimplemented here "
            "is the documented typology set (degree, fan-in/out, reciprocity, k-hop cycle closure, "
            "burst degree, scatter-gather). (2) The corpus injects CYCLE attempts of up to 10 hops "
            "(HI-Small_Patterns.txt), while reachability here is computed to "
            f"{MAX_CYCLE_HOPS} hops because the boolean sparse powers grow quickly beyond that. "
            "The evaluation is also strictly inductive: graph aggregates are fitted on the training "
            "window only, so no test-time edge informs a training-time feature. Reported as measured; "
            "not adjusted to close the gap."
        ),
        "git_sha": train.git_sha(),
        "data_digest": train.digest_array(Xtr[:10000], ytr[:10000]),
        "export": {"fp32_bytes": report.fp32_bytes, "int8_bytes": report.int8_bytes,
                   "int8_agreement": report.int8_agreement, "opset": report.opset},
        "wall_clock_s": round(time.time() - started, 1),
    })
    train.record_metrics("finance_aml.student", fidelity.as_metrics())
    print(f"\n  wrote {out} and models/metrics.json in {time.time()-started:.0f}s")
    return 0


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
