#!/usr/bin/env python
"""Classify 9 network-attack classes on real 5G traffic, and measure what the drops are worth.

WHAT THIS IS
    The telecom use case's second T2 model, and the first one on REAL network data. The other one,
    `telecom_ids.qml`, classifies 8 QKD-LAYER attack types from synthetic per-session physical
    observables. This one classifies 9 NETWORK-LAYER attack types from 46 Argus flow statistics.
    There is no overlap in label space or feature space, so the two numbers on the telecom page are
    not a comparison and neither is the other's baseline. `registry.py` carries that sentence and
    the page prints it.

WHAT CHANGED IN ROUND 86: ONE COLUMN AND 281,529 ROWS CAME OUT
    From Round 50 this model read `IdleTime`, a column of the corpus's 113-column export that the
    52-column `Combined.csv` omits, and it scored accuracy 0.9829 against 0.7616 without it. Round 50
    took that for a flow statistic, on the argument that 17 distinct integer values cannot address a
    row. They do not have to. All 17 are multiples of 128, the spacing of a 32-bit float at this
    size, and they go with the attack sessions and the two base stations, one value per session
    file, so the column NAMES THE FILE, and each file is one attack. `file_label_counterfactual`
    below refits with it restored every run, beside the row-counter one, and the shipped model does
    not read it.

    Without that column the model scored 0.7616 and no model reading these columns could pass 0.7685,
    and that turned out not to be a shortage of information. THE FILE HOLDS ONE ATTACKER'S UDP FLOOD
    TWICE: 281,529 flows labelled Benign in the first base station's file and UDPFlood in the second's,
    equal on every column but the labels and the file-position columns. See
    `D.NIDD_CONTRADICTORY_COPIES`. The loader removes the Benign copy, and `contradictory_copies`
    refits without that removal every run.

    Two more measurements come with that, because the headline is a random split and the question is
    how far to believe it. `flow_level_ceiling` is the accuracy no model reading these 46 columns can
    beat on these rows. `cross_base_station` trains on one base station's file and tests on the
    other's.

THE LEAK COUNTERFACTUAL, kept from the previous round because it is still the honest control
    The corpus carries four integer row counters - `Unnamed: 0` (the CSV's own index), `Rank`, `Seq`
    and `Offset`. This script fits the SAME model again with them restored and records the result.

    That is a shortcut of the kind Engelen, Rimmer & Joosen (WTMC 2021) found in CIC-IDS2017, which
    this repository already cites as its reason for reporting per-class F1 before any aggregate. It
    is recorded as a measurement rather than narrated, so `leak_counterfactual_accuracy` in
    `models/metrics.json` is checkable and this paragraph is not the evidence.

    WHAT EARLIER ROUNDS GOT WRONG, recorded here because it was recorded wrong here. Round 49 read
    that counterfactual as evidence that the published 99.6-99.7% figures leaned on row counters.
    Round 50 answered that 0.9829 was reachable with every identifier dropped, and it kept
    `IdleTime`. Neither had the fact that decides it. On the rows as published, inputs that are equal
    on the two copies of the flood are capped at 0.7685, for the binary task too, so any higher score
    rests on a column that differs between the copies: the row index, `Rank`, `Seq`, `Offset` or
    `IdleTime`. The dataset paper's 9-class baselines (Samarakoon et al., arXiv:2212.01298) rank
    `Seq` and `Offset` among their top-10 inputs, and of Wang, Fok & Thing's 28 inputs exactly one,
    `Offset`, differs between the copies. `published_inputs_check` measures that with this project's
    booster, which is not their network. Wang, Fok & Thing's 99.61% / 99.71% are also BINARY
    accuracies.

WHAT IS MEASURED
    macro F1 across all 9 classes as the headline, with aggregate accuracy beside it. On these rows
    the two agree to within a tenth of a point - see METRIC_NOTE below, which is published with the
    metrics - and the per-class figures show what either alone would hide.

Usage:
    export QSH_DATA_ROOT=$PWD/data-full
    python scripts/train/telecom_nidd.py [--n-estimators 300] [--nrows N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from qsh.models import datasets as D  # noqa: E402
from qsh.models import export, metrics, train  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402

KEY = "telecom_ids.nidd"

#: Hoisted to module level so `scripts/check_generated_prose.py` can read them out of the AST with
#: `ast.literal_eval`. An f-string here is reported UNCHECKABLE and silently leaves the only gate
#: that compares a generator against the artifact it wrote. The pattern - define the constant here,
#: restate it as a literal inside the `record_metrics` dict - is the one at
#: `scripts/train/telecom_ids.py:283-285`; the `**record` entry has no static key and is skipped.

FILE_LABEL_NOTE = (
    "`IdleTime` holds 17 values, all multiples of 128, the spacing of a 32-bit float at this size, "
    "one for each attack session and base station. Read as Unix times they fall between 13 and 15 "
    "September 2022, and the dataset paper gives the captures as 26 and 29 August 2022, so the "
    "column does not measure anything about the flow: it names the file the flow came from, and each "
    "file is one attack. On the rows as published it was worth about 22 accuracy points, because it "
    "is one of the few columns that differs between the two copies of the flood described under "
    "`contradictory_copies`. On the cleaned rows it adds what is measured here, and that gain is "
    "still a model recognising the file. The shipped model does not read it."
)

CONTRADICTORY_COPIES_NOTE = (
    "The published file holds the same 281,529 flows twice. The first base station's UDP-flood file "
    "also holds the traffic of the second base station's attacker towards the victim, labelled "
    "Benign; the second base station's file holds those flows labelled UDPFlood. The two copies "
    "agree on every column but the labels and the file-position columns, and in the two capture "
    "files the 421,471 packets behind them carry identical timestamps, which two separate capture "
    "points could not produce. On these rows no model reading flow statistics can pass "
    "1 - 281,529 / 1,215,890 = 0.7685. The loader keeps the UDPFlood copy, which is what the traffic "
    "was, and removes the Benign one; `as_published` repeats the same fit without that removal."
)

FRAMING = (
    "Real 5G network traffic, 9 attack classes, 46 flow-statistic features. This is a "
    "NETWORK-LAYER model and a different task from the QKD-layer classifier on the same page - no "
    "shared labels, no shared features, so the two scores are not a comparison. "
    "TWO CORRECTIONS MADE IN ROUND 86 EXPLAIN THE NUMBERS. The published file holds one attacker's "
    "UDP flood twice, the same 281,529 flows labelled Benign in one base station's file and UDPFlood "
    "in the other's; the Benign copies are removed, leaving 934,361 flows. And the model no longer "
    "reads `IdleTime`, which names the attack-session file a flow came from. Accuracy is 0.9979 and "
    "macro F1 0.9972. On the rows as published the same fit scores accuracy 0.7616, and no model "
    "reading these columns could pass 0.7685 there (`contradictory_copies`). "
    "`file_label_counterfactual` and `leak_counterfactual_accuracy` restore bookkeeping columns and "
    "reach about 1.0000, which is not a better model but a leak. `cross_base_station` trains on one "
    "base station and tests on the other: accuracy 0.9973 and 0.9877, against 0.4199 and 0.6059 on "
    "the rows as published. Two weak spots remain. Trained on the second base station, SYNFlood on "
    "the first scores F1 0.22. And without the TTL and hop columns, UDPScan across base stations "
    "falls to F1 0.05 (`without_ttl_and_hops`)."
)

METRIC_NOTE = (
    "macro F1 and aggregate accuracy agree here, 0.9972 and 0.9979, and neither is the flattering "
    "one. They were far apart until the contradictory copies came out, 0.9275 against 0.7616, "
    "because the contradiction fell on the two largest classes. macro F1 stays the headline because "
    "the largest class is about 400 times the smallest. Read `per_class_f1` for what the two averages "
    "hide: SlowrateDoS is lowest at 0.9873, HTTPFlood is at 0.9934, and ICMPFlood, at 289 test rows, "
    "is at 0.9983. A linear model on the numeric columns reaches accuracy 0.9838 and macro F1 0.9691, "
    "so most of this is separable by a plane, which is what a corpus where each class came from its "
    "own tool against its own service should look like. The random split is the easier test: see "
    "`cross_base_station`."
)

SAMPLE_SIZE_NOTE = (
    "Trained and scored on 5G-NIDD BTS1_BTS2_fields_preserved.csv with the 281,529 Benign-labelled "
    "copies of one attacker's flood removed: 934,361 of its 1,215,890 flows, stratified 75/25 at "
    "seed 42, so the test split is 233,591 rows and every class keeps its proportion among the "
    "remaining rows. Row for row the file is the same corpus as Combined.csv, which this model used "
    "until Round 50; only the column count differs, 113 against 52. No subsampling and no class "
    "rebalancing: `ICMPFlood` is 1,155 rows in the corpus and 289 in the test split, which is small "
    "but is the real prevalence, and resampling it would report a per-class F1 for a distribution "
    "the deployed model never sees. The browser evaluation sample is a different matter and is NOT "
    "proportional - see the note in eval_sample.json, which states its per-class floor and why its "
    "macro F1 is not this one."
)

BASELINE_GAP_NOTE = (
    "There is no baseline number to have a gap to, and that is a deliberate registry decision "
    "rather than a missing measurement: the two figures usually quoted for this corpus (99.61% for "
    "a classical CNN, 99.71% for a hybrid quantum-classical Quan-ConvCNN, Wang/Fok/Thing) are "
    "ACCURACIES on the BINARY benign-versus-malicious task, and this model is judged on 9-class "
    "macro F1, so putting them in `baseline` would render an accuracy beside a macro F1 as though "
    "the two were the same scale. "
    "WHAT THE COMPARISON CAN AND CANNOT SAY. The 9-class anchors are the dataset paper's random "
    "forest at 99.48% accuracy on its top-10 features (Samarakoon et al., arXiv:2212.01298) and "
    "BiTAD (Future Internet 17(11) 482) at 0.9047. This model's 0.9979 is on 934,361 rows and its "
    "own stratified random split, and those figures are on the rows as published, so each is an "
    "anchor and not a head-to-head. "
    "WHAT THE ROWS SAY ABOUT THOSE FIGURES. On the 1,215,890 rows as published, inputs equal on the "
    "two copies of the flood cannot pass 0.7685, binary or nine-class, so a score above that rests "
    "on a column that differs between the copies. Those are `Unnamed: 0`, `Rank`, `Seq`, `Offset` "
    "and `IdleTime`. The dataset paper's top-10 inputs include `Seq` and `Offset`. Of "
    "Wang/Fok/Thing's 28 inputs exactly one, `Offset`, differs between the copies, and "
    "`published_inputs_check` fits this project's booster to them: on the binary task accuracy "
    "0.9599 with `Offset` and 0.7643 without it. That measures what those columns can do on these "
    "rows and does not rerun those papers' models. Wang/Fok/Thing's fall across base stations, from "
    "above 98% to 78.23% and 74.28% for the classical CNN, has the same shape here: with the copies "
    "in place this booster scores 0.4199 and 0.6059, and with them out 0.9973 and 0.9877."
)

#: The 28 inputs of Wang, Fok and Thing (arXiv:2504.20436, Table III). Exactly one, `Offset`, differs
#: between the two copies of the flood.
WANG_INPUTS = (
    "Dur", "sTtl", "dTtl", "TotPkts", "SrcPkts", "DstPkts", "TotBytes", "SrcBytes", "DstBytes",
    "Offset", "sMeanPktSz", "dMeanPktSz", "Load", "SrcLoad", "DstLoad", "Loss", "SrcLoss", "DstLoss",
    "SrcWin", "DstWin", "TcpRtt", "SynAck", "AckDat", "Rate", "SrcRate", "DstRate", "SrcTCPBase",
    "DstTCPBase",
)

#: The columns a device or a tool's defaults set, removed together to see what the scan classes lean on.
TTL_AND_HOPS = ("sTtl", "dTtl", "sHops", "dHops")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--nrows", type=int, default=None,
                    help="read only the first N rows; for a smoke test, never for a published run")
    ap.add_argument("--linear-rows", type=int, default=300_000,
                    help="rows for the linear floor, which is O(n) in memory at 9 classes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO / "models" / "telecom_nidd")
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split

    started = time.time()
    train.seed_everything(args.seed)
    spec = get_spec(KEY)

    print(f"loading 5G-NIDD BTS1_BTS2_fields_preserved.csv ({spec.dataset.licence}, "
          f"{spec.dataset.url})")
    x, y, feature_names, class_names = D.load_5g_nidd(nrows=args.nrows)
    assert x.shape[1] == len(D.NIDD_FEATURES) == 46, (
        f"loader returned {x.shape[1]} columns for {len(D.NIDD_FEATURES)} declared feature names - "
        "the ONNX input contract and the published feature_spec would disagree")
    assert not set(D.NIDD_FILE_LABEL) & set(feature_names), (
        "the file-label column is in the feature set, so this run would score by recognising each "
        "attack's file. Do not record it as this model.")
    if args.nrows is None:
        assert x.shape[0] == 1_215_890 - D.NIDD_CONTRADICTORY_COPIES, (
            f"{x.shape[0]:,} rows, so the contradictory copies were not removed as declared")
    counts = np.bincount(y, minlength=len(class_names))
    majority_baseline = float(counts.max() / counts.sum())
    print(f"  {x.shape[0]:,} flows x {x.shape[1]} features, {len(class_names)} classes")
    for name, n in sorted(zip(class_names, counts, strict=True), key=lambda kv: -kv[1]):
        print(f"      {name:16s} {n:>9,}  {n / counts.sum() * 100:5.2f}%")
    print(f"  majority-class baseline (predict '{class_names[int(counts.argmax())]}' always) "
          f"{majority_baseline:.4f}")

    # The CSV is NOT shuffled - it concatenates two base stations - so a `--nrows` head read
    # gets 6 of the 9 classes and nothing downstream would say so in those words. XGBoost's own
    # message for it is "Invalid classes inferred from unique values of `y`. Expected: [0 1 2 3 4 5],
    # got [0 2 4 6 7 8]", which is true and unhelpful. Say what it means instead.
    absent = [n for n, c in zip(class_names, counts, strict=True) if c == 0]
    if absent:
        raise SystemExit(
            f"{absent} have zero rows in this read, so a 9-class model cannot be fitted and macro F1 "
            f"over 9 classes would be meaningless.\n"
            f"The corpus is ordered by capture, not shuffled, so --nrows takes a head that is "
            f"missing classes rather than a sample of them. --nrows is a smoke-test switch; drop it "
            f"for any run whose numbers are recorded.")

    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=args.seed, stratify=y)

    # ---- the shipped model: 46 features, every trap removed -------------------------------------
    print(f"\n  training XGBoost ({args.n_estimators} trees, depth {args.max_depth}) on 46 features ...")
    teacher = _booster(args)
    teacher.fit(xtr, ytr)
    # Train on the GPU, score and export on the CPU. Leaving the booster on cuda while handing it
    # host arrays makes XGBoost copy every batch back and forth and warn about mismatched devices,
    # and it is the CPU graph that goes to ONNX. Same step as scripts/train/finance_aml.py.
    teacher.set_params(device="cpu")
    proba = teacher.predict_proba(xte)
    pred = proba.argmax(1)
    m = metrics.with_companion(yte, pred, proba, headline="macro_f1")
    per_class = {class_names[int(k)] if k.isdigit() and int(k) < len(class_names) else k: v
                 for k, v in m["per_class_f1"].items()}
    n_predicted = int(len(np.unique(pred)))

    print(f"\n  macro F1            {m['macro_f1']:.4f}   <- headline")
    print(f"  aggregate accuracy  {m['aggregate_accuracy']:.4f}   <- companion; see METRIC_NOTE")
    print(f"  classes predicted   {n_predicted}/{len(class_names)}")
    print("  per-class F1:")
    for name, f1 in sorted(per_class.items(), key=lambda kv: kv[1]):
        print(f"      {name:16s} {f1:.4f}")

    # ---- how much of the split is exact repeats: the near-duplicate flows of Flood et al. ---------
    seen = np.isin(_vectors(xte), _vectors(xtr))
    unseen_pred, unseen_true = pred[~seen], yte[~seen]
    print(f"\n  test rows whose feature vector is also in training: {seen.mean():.4f}")

    # ---- floors, asserted rather than hoped -----------------------------------------------------
    # A class the model never emits scores 0 by construction, and macro F1 then averages in a zero
    # that reads as poor discrimination rather than as an absent output. `telecom_ids.student`
    # carries exactly that defect and is documented as doing so, so this is not hypothetical.
    assert n_predicted == len(class_names), (
        f"the model emits only {n_predicted} of {len(class_names)} classes, so macro F1 has a floor "
        f"of zero on the ones it never predicts - do not publish this")
    if args.nrows is None:
        assert m["macro_f1"] > 0.99, f"macro F1 {m['macro_f1']:.4f} is far below the measured 0.997"
    assert m["aggregate_accuracy"] > majority_baseline, (
        f"accuracy {m['aggregate_accuracy']:.4f} does not beat the majority-class baseline "
        f"{majority_baseline:.4f} - this model would be worse than guessing the commonest label")
    # THE REGRESSIONS THIS GUARDS ARE SPECIFIC. About 0.76 is what this configuration scores when the
    # contradictory copies are back in the rows, and above 0.9995 is what it scores when a file label
    # or a row counter is back in the features; neither has any other symptom. Only on a full read: a
    # --nrows head is a different distribution and its score means nothing.
    if args.nrows is None:
        assert 0.99 < m["aggregate_accuracy"] < 0.9995, (
            f"accuracy {m['aggregate_accuracy']:.4f} against the measured 0.998. Near 0.76 means the "
            f"copies are back in the rows; near 1.0 means a file label or a row counter is back in "
            f"the features.")

    record: dict = {
        "macro_f1": round(float(m["macro_f1"]), 6),
        "per_class_f1": {k: round(float(v), 6) for k, v in per_class.items()},
        "aggregate_accuracy": round(float(m["aggregate_accuracy"]), 6),
        "majority_baseline": round(majority_baseline, 6),
        "n_classes": len(class_names),
        "n_classes_predicted": n_predicted,
        "n_features": int(x.shape[1]),
        "n_samples": int(x.shape[0]),
        "n_eval": int(len(yte)),
        "test_rows_unseen_in_training": {
            "share_of_test_rows_with_a_vector_also_in_training": round(float(seen.mean()), 6),
            "accuracy": round(float(metrics.accuracy(unseen_true, unseen_pred)), 6),
            "macro_f1": round(float(metrics.macro_f1(unseen_true, unseen_pred)), 6),
        },
        "class_counts": {n: int(c) for n, c in zip(class_names, counts, strict=True)},
        "dropped_columns": dict(D.NIDD_DROPPED),
    }

    # ---- THE FILE-LABEL COUNTERFACTUAL: what a model reading which file a flow came from would score
    # Loaded through the same loader with the same encoding, so the only difference is the one column.
    print(f"\n  refitting with {list(D.NIDD_FILE_LABEL)} restored (the file-label counterfactual) ...")
    xc, yc, _, _ = D.load_5g_nidd(nrows=args.nrows, with_file_label=True)
    assert np.array_equal(yc, y), "the two loads disagree on the labels; the counterfactual is void"
    c_acc, c_macro = _refit(args, xc, yc, seed=args.seed)
    print(f"    macro F1 {c_macro:.4f} (vs {m['macro_f1']:.4f})   accuracy {c_acc:.4f} (vs {m['aggregate_accuracy']:.4f})")
    record["file_label_counterfactual"] = {
        "restored": list(D.NIDD_FILE_LABEL),
        "n_features": int(xc.shape[1]),
        "macro_f1": round(c_macro, 6),
        "accuracy": round(c_acc, 6),
        "accuracy_gain": round(c_acc - float(m["aggregate_accuracy"]), 6),
        "what_this_means": FILE_LABEL_NOTE,
    }

    # ---- HOW FAR THE HEADLINE CAN BE TRUSTED: the ceiling of these columns, and the other base station
    record["flow_level_ceiling"] = _flow_level_ceiling(x, y, class_names)
    print(f"\n  ceiling of any per-flow model on these columns: accuracy {record['flow_level_ceiling']['accuracy']:.4f}")
    if args.nrows is None:
        cut = D.NIDD_SECOND_FILE_STARTS_AT_CLEANED
        record["cross_base_station"] = _cross_base_station(args, x, y, class_names, cut=cut)

        # ---- WHAT THE SCAN CLASSES LEAN ON: nmap's raw probes carry their own TTL -----------------
        print(f"\n  refitting without {list(TTL_AND_HOPS)} ...")
        keep = [i for i, n in enumerate(feature_names) if n not in TTL_AND_HOPS]
        t_acc, t_macro = _refit(args, x[:, keep], y, seed=args.seed)
        record["without_ttl_and_hops"] = {
            "removed": list(TTL_AND_HOPS),
            "accuracy": round(t_acc, 6),
            "macro_f1": round(t_macro, 6),
            "cross_base_station": _cross_base_station(args, x[:, keep], y, class_names, cut=cut),
        }

        # ---- THE SAME FIT ON THE ROWS AS PUBLISHED: what the copies do to every number above -------
        print(f"\n  refitting on the rows as published ({D.NIDD_CONTRADICTORY_COPIES:,} copies kept) ...")
        xp, yp, _, _ = D.load_5g_nidd(with_contradictory_copies=True)
        p_acc, p_macro = _refit(args, xp, yp, seed=args.seed)
        print(f"    macro F1 {p_macro:.4f}   accuracy {p_acc:.4f}")
        xpf, ypf, _, _ = D.load_5g_nidd(with_contradictory_copies=True, with_file_label=True)
        pf_acc, _ = _refit(args, xpf, ypf, seed=args.seed)
        record["contradictory_copies"] = {
            "rows_removed": D.NIDD_CONTRADICTORY_COPIES,
            "source": D.NIDD_COPY_SOURCE,
            "destination": D.NIDD_COPY_DESTINATION,
            "as_published": {
                "n_samples": int(len(yp)),
                "accuracy": round(p_acc, 6),
                "macro_f1": round(p_macro, 6),
                "accuracy_with_the_file_label": round(pf_acc, 6),
                "flow_level_ceiling": _flow_level_ceiling(xp, yp, class_names),
                "cross_base_station": _cross_base_station(args, xp, yp, class_names,
                                                          cut=D.NIDD_SECOND_FILE_STARTS_AT),
            },
            "what_this_means": CONTRADICTORY_COPIES_NOTE,
        }
        record["published_inputs_check"] = _published_inputs_check(args, class_names)

    # ---- THE COUNTERFACTUAL: the same fit with the four row identifiers restored ----------------
    # Loaded through the same loader with the same encoding rather than re-derived here, so the only
    # difference between the two fits is the four columns. A second drop list beside this one would
    # make part of the difference the script's own.
    print(f"\n  refitting with {list(D.NIDD_IDENTIFIERS)} restored (the leak counterfactual) ...")
    xi, yi, leak_names, _ = D.load_5g_nidd(nrows=args.nrows, with_identifiers=True)
    assert np.array_equal(yi, y), "the two loads disagree on the labels; the counterfactual is void"
    xitr, xite, yitr, yite = train_test_split(xi, yi, test_size=0.25, random_state=args.seed,
                                              stratify=yi)
    leaky = _booster(args)
    leaky.fit(xitr, yitr)
    leaky.set_params(device="cpu")
    leak_pred = leaky.predict_proba(xite).argmax(1)
    leak_macro = float(metrics.macro_f1(yite, leak_pred))
    leak_acc = float(metrics.accuracy(yite, leak_pred))
    print(f"    macro F1 {leak_macro:.4f} (vs {m['macro_f1']:.4f})   "
          f"accuracy {leak_acc:.4f} (vs {m['aggregate_accuracy']:.4f})")
    print(f"    -> restoring {len(D.NIDD_IDENTIFIERS)} integer row counters is worth "
          f"{(leak_acc - m['aggregate_accuracy']) * 100:+.1f} accuracy points")
    record.update({
        "leak_counterfactual_features": list(leak_names[-len(D.NIDD_IDENTIFIERS):]),
        "leak_counterfactual_macro_f1": round(leak_macro, 6),
        "leak_counterfactual_accuracy": round(leak_acc, 6),
        "leak_counterfactual_accuracy_gain": round(leak_acc - float(m["aggregate_accuracy"]), 6),
    })

    # ---- the trivial floors this model is actually measured against ----------------------------
    # The T2 model here is itself classical, so "the number to beat" is not a quantum comparison. It
    # is the majority-class predictor above, plus a linear floor - which also answers a standing
    # question for this use case: whether a CKKS-linear student would be viable on this corpus at
    # all, or whether it would collapse to a subset of the classes the way telecom's 8-class student
    # nearly does.
    record["linear_baseline"] = _linear_floor(xtr, ytr, xte, yte, class_names,
                                              n_rows=args.linear_rows, seed=args.seed)

    # ---- EDA: the same one-vs-rest leak check the evaluation samples run through --------------
    # Imported rather than reimplemented: `prep_model_inputs._warn_single_feature_leak` is the
    # threshold this project already publishes against, and a second copy would drift from it.
    #
    # IT FIRES A LOT HERE - roughly half the features clear 0.90 - and that is correct behaviour
    # rather than a problem to silence. Its own docstring anticipates the case ("a legitimately
    # dominant feature exists in some corpora ... a build that refuses to run is worse than one that
    # reports what it found"), and on a corpus where each attack class was produced by a different
    # tool against a different service, TTL, packet rate, TCP window and ACK timing separating those
    # classes is the detector working rather than a sampling artefact. What it must NOT still fire on
    # is the four identifiers, and it cannot, because they are not in this matrix - that contrast is
    # the whole point and `single_feature_leak_check` records it as data. The check could not tell
    # that `IdleTime` was a per-file label, and until Round 86 this comment argued it was not; what
    # told it was reading the values. The flagged list is published; a reader can check it.
    print("\n  single-feature leak check on the held-out split (a warning, not a gate):")
    record["single_feature_leak_check"] = _leak_check(xte, yte, feature_names, class_names,
                                                      seed=args.seed)

    record["framing"] = FRAMING
    record["metric_note"] = METRIC_NOTE
    record["sample_size_note"] = SAMPLE_SIZE_NOTE
    record["baseline_gap_note"] = BASELINE_GAP_NOTE
    record["dataset"] = spec.dataset.name
    record["dataset_attribution"] = spec.dataset.attribution
    record["git_sha"] = train.git_sha()
    record["wall_clock_s"] = round(time.time() - started, 1)

    # ---- artefacts -------------------------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures = xte[: min(256, len(xte))].astype(np.float32)
    report = export.verify_and_quantize(teacher, fixtures, args.out, kind="xgboost",
                                        n_features=int(x.shape[1]))
    print(f"\n  ONNX fp32 {report.fp32_bytes / 1e6:.2f} MB, domains {report.onnx_domains}"
          + (f" (int8 skipped: {report.int8_skipped_reason})" if report.int8_skipped_reason else ""))
    record["export"] = {"fp32_bytes": report.fp32_bytes, "opset": report.opset,
                        "onnx_domains": list(report.onnx_domains),
                        "int8_skipped_reason": report.int8_skipped_reason}

    (args.out / "feature_spec.json").write_text(json.dumps({
        "features": list(D.NIDD_FEATURES),
        "classes": list(D.NIDD_CLASSES),
        # Published so a consumer can encode a raw row itself. Without it the ten string columns
        # are unreachable: the ONNX graph takes floats and carries no vocabulary, so a caller that
        # guessed a different ordering would get a confident wrong class rather than an error.
        "categorical_vocabularies": {k: list(v) for k, v in D.NIDD_CATEGORICAL.items()},
        "missing_value": D.NIDD_MISSING,
        "dropped_columns": dict(D.NIDD_DROPPED),
        "note": (
            "46 of the 113 columns in BTS1_BTS2_fields_preserved.csv, the corpus's wide export. "
            "The 67 dropped columns are listed above WITH THE REASON each was dropped. They are 3 "
            "targets, 9 row identifiers, 10 host fingerprints, 5 columns bit-identical to `Dur`, "
            "1 file label, "
            "and 39 that hold exactly one state in this release - 30 of those are 100% null, "
            "including every SIntPkt/DIntPkt inter-packet-time and jitter column, which are EMPTY "
            "here and not merely uninformative. "
            "THE ROWS ARE 934,361 OF THE FILE'S 1,215,890: one attacker's UDP flood is in the file "
            "twice under contradictory labels and the Benign copy is removed "
            "(`contradictory_copies` in models/metrics.json). "
            "`Unnamed: 0` IS THE ONE TO CHECK FIRST. It is the CSV's own row index, and restoring "
            "the four row counters takes accuracy to about 1.0000, which is a leak rather than a "
            "result; `leak_counterfactual_*` in models/metrics.json re-measures it every run. "
            "A SEPARATE AND LEGITIMATE MATTER, stated because the numbers look alarming next to "
            f"that: {record['single_feature_leak_check']['n_features_flagged']} of the "
            f"{len(D.NIDD_FEATURES)} kept features still clear the 0.90 one-vs-rest AUC threshold "
            "after all 67 drops. Those are real flow statistics rather than bookkeeping columns: "
            "each attack class in this corpus was produced by a different tool against a different "
            "service, so packet rate, ACK timing, TTL and TCP window separating them is the signal "
            "an intrusion detector is meant to use. They are reported and kept. The row identifiers "
            "cleared the same threshold while describing only where a row sits in a file, and that "
            "is the distinction this file exists to make - see `single_feature_leak_check` in "
            "models/metrics.json for the full measured list. "
            "`categorical_vocabularies` is the ordinal encoding the ten string columns were fitted "
            "under; index into the list, and use `missing_value` for anything absent or unseen. "
            "THE LEADING AND TRAILING SPACES IN `Flgs`, `Dir` AND `TcpOpt` ARE PART OF THE VALUE: "
            "Argus writes them as fixed-width fields and the vocabulary is the raw cell, so look up "
            "the cell as read and do not strip it. The vocabulary is published rather than "
            "re-derived because one rebuilt from different rows would remap codes the ONNX "
            "thresholds were fitted against."),
    }, indent=2) + "\n")

    train.record_metrics(KEY, {
        **record,
        # Restated as literals so `scripts/check_generated_prose.py` can compare this generator to
        # the artifact it writes. The `**record` entry has no static key and is skipped by the AST
        # walk, so these four are what the gate actually reads.
        "framing": FRAMING,
        "metric_note": METRIC_NOTE,
        "sample_size_note": SAMPLE_SIZE_NOTE,
        "baseline_gap_note": BASELINE_GAP_NOTE,
    })

    print(f"\n  wrote {args.out.relative_to(REPO)} and models/metrics.json "
          f"in {time.time() - started:.1f}s")
    print("  NEXT: scripts/prep_model_inputs.py --usecase telecom_nidd, BEFORE publishing. A model "
          "that is pinned but ships no eval_sample.json renders "
          "'did not run - no substitute is shown' to a visitor who selects it.")
    return 0


def _booster(args):
    """The one configuration both fits use, so the counterfactual differs only in its columns."""
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        # `enable_categorical` is deliberately OFF and the ten string columns are ordinal-encoded
        # upstream in the loader. `export_xgboost` converts through
        # `FloatTensorType([None, n_features])`, so a categorical-aware booster cannot survive ONNX
        # export at all - the browser would feed floats to a model expecting category codes.
        # Measured cost of the encoding on the narrow export, where the comparison was run: macro F1
        # 0.926183 encoded against 0.926086 native, i.e. nothing.
        enable_categorical=False,
        device="cuda", n_jobs=16, random_state=args.seed,
    )


def _refit(args, x, y, *, seed: int) -> tuple[float, float]:
    """Fit `_booster` on a column subset of the already-encoded matrix; return (accuracy, macro F1).

    The split is recomputed with the same seed and the same stratification, so it is row-for-row the
    split the shipped fit was scored on. Reusing the matrix rather than reloading is what makes an
    ablation an ablation: the only difference between the two fits is the columns.
    """
    from sklearn.model_selection import train_test_split

    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    model = _booster(args)
    model.fit(xtr, ytr)
    model.set_params(device="cpu")
    pred = model.predict_proba(xte).argmax(1)
    return float(metrics.accuracy(yte, pred)), float(metrics.macro_f1(yte, pred))


def _vectors(x) -> np.ndarray:
    """Each row as one opaque value, so whole feature vectors can be compared."""
    rows = np.ascontiguousarray(x.astype(np.float32))
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def _flow_level_ceiling(x, y, class_names) -> dict:
    """The accuracy no model reading only these columns can beat on this corpus.

    Rows with the same feature vector get the same prediction, so the best any classifier can do is the
    most common label of each distinct vector. It is computed over all the rows given, so it is an upper
    bound for any split of them. The idea is the "near-duplicate flows" of Flood et al. (EuroS&P 2024),
    taken to exact identity across classes, and it is what exposed the copies in this file: on the rows
    as published it is 0.7685, exactly one minus the share of rows that are the Benign copies.
    """
    _, group = np.unique(_vectors(x), return_inverse=True)
    n_classes = len(class_names)
    pairs = group.astype(np.int64) * n_classes + y.astype(np.int64)
    unique_pairs, counts = np.unique(pairs, return_counts=True)
    groups = unique_pairs // n_classes
    best = np.zeros(int(groups.max()) + 1, dtype=np.int64)
    np.maximum.at(best, groups, counts)
    labels_per_group = np.bincount(groups)
    ambiguous = labels_per_group > 1
    rows_per_group = np.bincount(group)
    pair_names: dict[str, int] = {}
    # `groups` is sorted, so each ambiguous group's labels are one contiguous slice of `unique_pairs`.
    which = np.flatnonzero(ambiguous)
    for lo, hi in zip(np.searchsorted(groups, which, "left"), np.searchsorted(groups, which, "right"), strict=True):
        name = " / ".join(class_names[int(c)] for c in unique_pairs[lo:hi] % n_classes)
        pair_names[name] = pair_names.get(name, 0) + 1
    return {
        "accuracy": round(float(best.sum() / len(y)), 6),
        "distinct_feature_vectors": int(len(best)),
        "vectors_under_more_than_one_class": int(ambiguous.sum()),
        "share_of_rows_in_those_vectors": round(float(rows_per_group[ambiguous].sum() / len(y)), 6),
        "class_sets_of_those_vectors": dict(sorted(pair_names.items(), key=lambda kv: -kv[1])),
    }


def _cross_base_station(args, x, y, class_names, *, cut: int) -> dict:
    """Train on one base station's file and test on the other's, both ways.

    The dataset was captured at two base stations and the CSV concatenates their files; `cut` is where
    the second begins in the rows given. A stratified random split puts flows of the same attack session
    on both sides; this does not, and it is the fairer reading of what a model of this corpus would do
    elsewhere. On the rows as published the first direction trains on the flood labelled Benign and
    tests on it labelled UDPFlood, which is most of the fall Wang, Fok and Thing report on the binary
    task; on the cleaned rows the fall is small.
    """
    import xgboost as xgb  # noqa: F401  (the booster's own dependency; imported here for a clear error)
    from sklearn.metrics import f1_score

    assert len(y) > cut and set(np.unique(y[:cut])) == set(np.unique(y[cut:])) == set(range(len(class_names))), (
        "the two base-station files should each hold all nine classes; the file boundary has moved")
    out: dict = {}
    for name, (tr, te) in {"first_file_to_second": (np.arange(cut), np.arange(cut, len(y))),
                           "second_file_to_first": (np.arange(cut, len(y)), np.arange(cut))}.items():
        model = _booster(args)
        model.fit(x[tr], y[tr])
        model.set_params(device="cpu")
        pred = model.predict_proba(x[te]).argmax(1)
        per = f1_score(y[te], pred, average=None, labels=list(range(len(class_names))))
        out[name] = {
            "accuracy": round(float(metrics.accuracy(y[te], pred)), 6),
            "macro_f1": round(float(metrics.macro_f1(y[te], pred)), 6),
            "benign_f1": round(float(per[class_names.index("Benign")]), 6),
            "udpflood_f1": round(float(per[class_names.index("UDPFlood")]), 6),
            "per_class_f1": {c: round(float(v), 6) for c, v in zip(class_names, per, strict=True)},
            "lowest_class": class_names[int(np.argmin(per))],
        }
        print(f"    train {name}: accuracy {out[name]['accuracy']:.4f}, macro F1 {out[name]['macro_f1']:.4f}")
    return out


def _published_inputs_check(args, class_names) -> dict:
    """Wang, Fok and Thing's 28 inputs on the rows as published, with and without `Offset`.

    It is this project's booster and not their network, and it is here for one claim: inputs equal on
    the two copies of the flood are capped, and the one input that differs is what passes the cap.
    """
    from sklearn.model_selection import train_test_split

    x, y, names, _ = D.load_5g_nidd(with_identifiers=True, with_contradictory_copies=True)
    missing = [c for c in WANG_INPUTS if c not in names]
    assert not missing, f"{missing} are not among the columns this loader can return"
    where = [names.index(c) for c in WANG_INPUTS]
    without = [names.index(c) for c in WANG_INPUTS if c != "Offset"]
    binary = (y != class_names.index("Benign")).astype(np.int64)
    out: dict = {"inputs": list(WANG_INPUTS), "model": "this project's booster, 300 trees, a 70/30 split at seed 42"}
    for task, target in (("binary", binary), ("nine_classes", y)):
        row: dict = {"ceiling_without_offset": _flow_level_ceiling(x[:, without], target, ["0"] * (int(target.max()) + 1))["accuracy"]}
        for tag, cols in (("with_offset", where), ("without_offset", without)):
            xtr, xte, ytr, yte = train_test_split(x[:, cols], target, test_size=0.3, random_state=args.seed, stratify=target)
            model = _booster(args)
            model.fit(xtr, ytr)
            model.set_params(device="cpu")
            row[f"accuracy_{tag}"] = round(float(metrics.accuracy(yte, model.predict_proba(xte).argmax(1))), 6)
        out[task] = row
        print(f"    Wang inputs, {task}: {row}")
    return out


def _linear_floor(xtr, ytr, xte, yte, class_names, *, n_rows: int, seed: int) -> dict:
    """Multinomial logistic regression on the numeric columns: the floor, and a T1 feasibility probe.

    NOT a shipped model and not distilled from the booster - a plain fit, so it answers "how much of
    this is linear" rather than "how well does a student copy the teacher". Two things make it worth
    recording. It is the honest floor for a classical T2 model, since the majority-class predictor is
    a very low bar at 0.39. And it decides whether a CKKS-linear student is viable here later: a
    linear model that collapsed to a subset of the 9 classes would settle that question negatively
    before anyone spent a round on it.

    Numeric columns only, and standardised. The ten ordinal codes are nominal - `Proto` index 7 is
    `udp` and index 0 is `arp`, and nothing about that ordering is a magnitude - so feeding them to a
    linear model would assert a false ordering. A tree does not care, which is why the booster above
    takes all 46.
    """
    from sklearn.linear_model import LogisticRegression

    numeric = [i for i, n in enumerate(D.NIDD_FEATURES) if n not in D.NIDD_CATEGORICAL]
    take = min(n_rows, len(xtr))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xtr), size=take, replace=False)
    ztr, zte = xtr[np.ix_(idx, numeric)], xte[:, numeric]

    mean, scale = ztr.mean(axis=0), ztr.std(axis=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit((ztr - mean) / scale, ytr[idx])
    pred = model.predict((zte - mean) / scale)
    macro = float(metrics.macro_f1(yte, pred))
    acc = float(metrics.accuracy(yte, pred))
    n_predicted = int(len(np.unique(pred)))
    print(f"\n  linear floor ({len(numeric)} numeric features, {take:,} rows)")
    print(f"    macro F1 {macro:.4f}   accuracy {acc:.4f}   classes predicted "
          f"{n_predicted}/{len(class_names)}")
    return {
        "macro_f1": round(macro, 6),
        "accuracy": round(acc, 6),
        "n_classes_predicted": n_predicted,
        "n_features": len(numeric),
        "n_train_rows": int(take),
        "what_it_is": (
            f"multinomial logistic regression on the {len(numeric)} NUMERIC features only, standardised, fitted "
            "directly rather than distilled. The 10 ordinal codes are excluded because their integer "
            "order is nominal and a linear model would read it as a magnitude. Recorded as the "
            "classical floor and as a feasibility probe for a future CKKS-linear tier: it predicts "
            f"{n_predicted} of {len(class_names)} classes, so a linear student on this corpus does "
            "not collapse the way an 8-class one nearly does elsewhere in this project."),
    }


def _leak_check(xte, yte, feature_names, class_names, *, seed: int, n_rows: int = 100_000) -> dict:
    """Run `prep_model_inputs._warn_single_feature_leak` over a draw of the held-out split.

    Subsampled because that function takes the browser payload's nested-list shape and rank-sorts
    every column; 233,591 x 46 through a Python list is minutes of nothing useful. Rank-AUC is
    stable well below 100,000 rows, and the flags this is looking for sit at 0.99, not at 0.90.

    THE CHECK IS IMPORTED, NOT REIMPLEMENTED, and it now RETURNS its findings so this run can record
    them. The count is a published claim in feature_spec.json, and a claim measured nowhere is the
    drift this repository keeps a gate for; a second copy of the same rank-AUC and the same
    threshold living here would be the other failure it keeps a gate for. So the shared function
    grew a return value - every existing caller ignores it - rather than growing a twin.
    """
    from prep_model_inputs import LEAK_AUC_THRESHOLD, _warn_single_feature_leak  # type: ignore

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(yte), size=min(n_rows, len(yte)), replace=False)
    found = _warn_single_feature_leak("telecom_nidd", {
        "samples": np.asarray(xte[idx], dtype=float).tolist(),
        "labels": np.asarray(yte[idx], dtype=int).tolist(),
        "feature_names": list(feature_names),
        "class_names": list(class_names),
    })
    flagged = [{"feature": nm, "class": class_names[c] if c < len(class_names) else str(c),
                "one_vs_rest_auc": round(float(auc), 6)} for auc, nm, c in found]
    return {
        "n_rows_checked": int(len(idx)),
        "threshold": LEAK_AUC_THRESHOLD,
        "n_features_flagged": len(flagged),
        "n_features_total": len(feature_names),
        "flagged": flagged,
        "what_this_means": (
            "One-vs-rest rank-AUC per feature per class, at the 0.90 threshold "
            "scripts/prep_model_inputs.py already publishes against, measured AFTER all 67 drops. "
            "A LARGE NUMBER HERE IS EXPECTED ON THIS CORPUS AND IS NOT THE LEAK THIS MODEL IS ABOUT. "
            "The attacks are generated by distinct tools against distinct services, so a scan, a "
            "SYN flood and a slow-rate HTTP attack genuinely differ in TTL, packet rate, TCP window "
            "and ACK timing - these are the flow statistics an intrusion detector is supposed to "
            "use. The check's own docstring anticipates the case and warns rather than failing. "
            "The distinction that matters is against `leak_counterfactual_*`: `Unnamed: 0`, `Rank`, "
            "`Seq` and `Offset` cleared the same threshold while being bookkeeping columns that "
            "carry no information about the traffic at all, which is why those four are dropped and "
            "these are kept. The check cannot tell a file label from a statistic: `IdleTime` is "
            "out because its values name the attack session's file, which only reading the values "
            "showed, and `file_label_counterfactual` records what it was worth. Recorded rather "
            "than described so the distinction is checkable."),
    }


if __name__ == "__main__":
    raise SystemExit(main())
