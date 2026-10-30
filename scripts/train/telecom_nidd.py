#!/usr/bin/env python
"""Classify 9 network-attack classes on real 5G traffic, and measure what the drops are worth.

WHAT THIS IS
    The telecom use case's second T2 model, and the first one on REAL network data. The other one,
    `telecom_ids.qml`, classifies 8 QKD-LAYER attack types from synthetic per-session physical
    observables. This one classifies 9 NETWORK-LAYER attack types from 47 Argus flow statistics.
    There is no overlap in label space or feature space, so the two numbers on the telecom page are
    not a comparison and neither is the other's baseline. `registry.py` carries that sentence and
    the page prints it.

WHAT CHANGED IN ROUND 50: THE EXPORT, NOT THE MODEL
    Until Round 50 this trained on `Combined.csv`, the 52-column export, and scored accuracy 0.7608.
    The release also ships `BTS1_BTS2_fields_preserved.csv`: the same 1,215,890 rows, the same nine
    classes, 113 columns. Every one of Combined's 52 is present and informationally identical in the
    wide file, so Combined is a strict column SUBSET - and one of the columns it omits, `IdleTime`,
    is worth 22 accuracy points on its own.

    `idle_time_ablation` below refits this identical configuration with `IdleTime` removed and with
    all six of Combined's missing columns removed, every run. The first reproduces the old 0.76 and
    the second adds nothing to it, which is the whole finding: one flow statistic, not five.

    `IdleTime` IS NOT AN IDENTIFIER. It takes 17 distinct integer values over 1.2 million rows, so
    it cannot encode a row position, and it scores one-vs-rest AUC 0.9574 on `SlowrateDoS` - an
    attack whose definition is holding a connection open and idle. That is a dominant feature doing
    its job. The contrast with the row counters below is the point.

THE LEAK COUNTERFACTUAL, kept from the previous round because it is still the honest control
    The corpus carries four integer row counters - `Unnamed: 0` (the CSV's own index), `Rank`, `Seq`
    and `Offset`. This script fits the SAME model again with them restored and records the result.
    It reaches about 0.9997 accuracy, which is the published range for this corpus.

    That is the shape of result Engelen, Rimmer & Joosen (WTMC 2021) produced for CIC-IDS2017, which
    this repository already cites as its reason for reporting per-class F1 before any aggregate. It
    is recorded as a measurement rather than narrated, so `leak_counterfactual_accuracy` in
    `models/metrics.json` is checkable and this paragraph is not the evidence.

    WHAT ROUND 49 GOT WRONG, recorded here because it was recorded wrong here. That counterfactual
    was read as evidence that the published 99.6-99.7% figures leaned on row counters, on the
    grounds that nothing else could close a 24-point gap. Something else could: 0.9829 is reachable
    on this corpus with every identifier and every host fingerprint dropped. The counterfactual
    still shows what a row counter is worth; it never showed what the papers did. Nor does the
    refit settle the question the other way: the dataset paper's 9-class baselines (Samarakoon et
    al., arXiv:2212.01298) rank `Seq` and `Offset` among their top-10 inputs and Wang, Fok & Thing
    use `Offset`, so whether published figures lean on identifiers stays open. Wang, Fok & Thing's
    99.61% / 99.71% are also BINARY accuracies.

WHAT IS MEASURED, and why the usual reading is inverted here
    macro F1 across all 9 classes as the headline, with aggregate accuracy beside it. On every other
    model in this project the aggregate is the flattering one. Here it is still the opposite, though
    by far less than before - see METRIC_NOTE below, which is published with the metrics so the page
    cannot read backwards.

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

FRAMING = (
    "Real 5G network traffic, 9 attack classes, 47 flow-statistic features. This is a "
    "NETWORK-LAYER model and a different task from the QKD-layer classifier on the same page - no "
    "shared labels, no shared features, so the two scores are not a comparison. "
    "IT IS FITTED ON THE FIELDS-PRESERVED EXPORT (113 columns) AND WAS FITTED ON Combined.csv (52) "
    "UNTIL ROUND 50, which is the single most important thing to know about the numbers here: the "
    "same rows and the same classes, but the narrow export omits `IdleTime` and that one column is "
    "worth 22 points of accuracy. `idle_time_ablation` re-measures this every run by refitting "
    "without it, and the result reproduces what this model used to score. Two controls are recorded "
    "beside the headline and both are worth more than the headline. `idle_time_ablation` says the "
    "old weakness was a choice of export, not a property of the corpus. "
    "`leak_counterfactual_accuracy` says that restoring four integer row counters takes accuracy to "
    "about 0.9997, which is not a better model but a leak."
)

METRIC_NOTE = (
    "THE TWO NUMBERS STILL READ THE OPPOSITE WAY ROUND FROM EVERY OTHER MODEL IN THIS PROJECT, and "
    "the refit onto the wide export did NOT fix that - it shrank it. Everywhere else here the "
    "aggregate flatters and the per-class figure exposes it, which is why `aggregate_accuracy` is "
    "rendered as the companion 'shown so the gap is visible'. On this corpus macro F1 is still the "
    "OPTIMISTIC number and aggregate accuracy still the PESSIMISTIC one. What changed is the size "
    "of the disagreement: macro F1 minus accuracy was 0.1655 on the narrow export (0.9263 against "
    "0.7608) and is about 0.0121 now (0.9950 against 0.9829). "
    "THE MECHANISM IS UNCHANGED, which is the part a reader is most likely to get wrong. It is not "
    "that a small class now drags the macro average - the smallest class, ICMPFlood at 289 test "
    "rows, scores 1.0000 and lifts it. It is the same two majority classes as before: Benign at "
    "39.3% of rows and UDPFlood at 37.6% are the only pair that confuses at all, they carry 77% of "
    "the test set between them, so accuracy is dominated by them while a macro average over 9 "
    "classes is carried by the seven scoring 0.9999 or better. Adding `IdleTime` separated that "
    "pair almost completely rather than changing which pair it is. "
    "The residual is one directional confusion - UDPFlood predicted as Benign - and it is 1.71% of "
    "the test set against 23.92% before. Read `per_class_f1` and it is visible directly: the two "
    "lowest entries are Benign and UDPFlood, and every other class is at or above 0.9999. Neither "
    "figure is the flattering one to quote alone."
)

SAMPLE_SIZE_NOTE = (
    "Trained and scored on the whole of 5G-NIDD BTS1_BTS2_fields_preserved.csv - 1,215,890 flows, "
    "stratified 75/25 at seed 42, so the test split is 303,973 rows and every class keeps its "
    "corpus proportion. Row for row this is the same corpus as Combined.csv, which this model used "
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
    "WHAT THE COMPARISON CAN AND CANNOT SAY. This model scores about 0.9829 9-class accuracy with "
    "every row identifier and every host fingerprint dropped. Set beside binary accuracies, the "
    "1.3-point difference is not a gap on one task. The 9-class anchors are the dataset paper's "
    "random forest at 99.48% accuracy on its top-10 features (Samarakoon et al., "
    "arXiv:2212.01298) and BiTAD (Future Internet 17(11) 482) at 0.9047; this model sits between "
    "them, on its own stratified random split rather than a shared one, so each is an anchor and "
    "not a head-to-head. "
    "WHAT THE COUNTERFACTUAL SHOWS. Restoring the corpus's integer row counters takes accuracy to "
    "about 0.9997, recorded every run as `leak_counterfactual_accuracy`: that is what a row counter "
    "is worth on this corpus. It does not show what the published figures used, because 0.9829 is "
    "reachable here with no identifiers at all, using one additional flow statistic (`IdleTime`) "
    "that Combined.csv does not carry. Whether the published figures lean on identifiers is OPEN: "
    "the dataset paper's top-10 inputs include the sequence number `Seq` and `Offset`, "
    "Wang/Fok/Thing's inputs include `Offset`, both columns are dropped here, and no published "
    "ablation settles it."
)


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
    assert x.shape[1] == len(D.NIDD_FEATURES) == 47, (
        f"loader returned {x.shape[1]} columns for {len(D.NIDD_FEATURES)} declared feature names - "
        "the ONNX input contract and the published feature_spec would disagree")
    assert "IdleTime" in feature_names, (
        "`IdleTime` is not in the feature set, so this run is the narrow-export configuration under "
        "a wide-export label and will score about 0.76. Do not record it as this model.")
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

    # ---- the shipped model: 47 features, every trap removed -------------------------------------
    print(f"\n  training XGBoost ({args.n_estimators} trees, depth {args.max_depth}) on 47 features ...")
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

    print(f"\n  macro F1            {m['macro_f1']:.4f}   <- headline, and the OPTIMISTIC number here")
    print(f"  aggregate accuracy  {m['aggregate_accuracy']:.4f}   <- companion, and the PESSIMISTIC "
          "one; see METRIC_NOTE")
    print(f"  classes predicted   {n_predicted}/{len(class_names)}")
    print("  per-class F1:")
    for name, f1 in sorted(per_class.items(), key=lambda kv: kv[1]):
        print(f"      {name:16s} {f1:.4f}")

    # ---- floors, asserted rather than hoped -----------------------------------------------------
    # A class the model never emits scores 0 by construction, and macro F1 then averages in a zero
    # that reads as poor discrimination rather than as an absent output. `telecom_ids.student`
    # carries exactly that defect and is documented as doing so, so this is not hypothetical.
    assert n_predicted == len(class_names), (
        f"the model emits only {n_predicted} of {len(class_names)} classes, so macro F1 has a floor "
        f"of zero on the ones it never predicts - do not publish this")
    assert m["macro_f1"] > 0.80, f"macro F1 {m['macro_f1']:.4f} is far below the measured 0.995"
    assert m["aggregate_accuracy"] > majority_baseline, (
        f"accuracy {m['aggregate_accuracy']:.4f} does not beat the majority-class baseline "
        f"{majority_baseline:.4f} - this model would be worse than guessing the commonest label")
    # THE REGRESSION THIS GUARDS IS SPECIFIC: 0.76 is what this configuration scores on the narrow
    # export, and it is the number a wrong CSV at the fetch path produces without any other symptom.
    # Only on a full read - a --nrows head is a different distribution and its score means nothing.
    if args.nrows is None:
        assert m["aggregate_accuracy"] > 0.95, (
            f"accuracy {m['aggregate_accuracy']:.4f} against the measured 0.9829. Anything near 0.76 "
            f"means the fit ran on Combined.csv's column set, i.e. without `IdleTime`.")

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
        "class_counts": {n: int(c) for n, c in zip(class_names, counts, strict=True)},
        "dropped_columns": dict(D.NIDD_DROPPED),
    }

    # ---- THE ABLATION: what the wide export is actually worth, and which column carries it -------
    # This is the model's most important single fact and it is measured here rather than quoted.
    # Two refits, both on columns taken out of the SAME encoded matrix `x` so nothing but the column
    # subset differs - no second CSV read, no second drop list, no second encoding to disagree with.
    print("\n  ablating the columns Combined.csv does not carry (the Round 50 finding) ...")
    ablations = {}
    for tag, drop in (("without_idle_time", ("IdleTime",)),
                      ("without_all_six_columns_combined_lacks", D.NIDD_ABSENT_FROM_COMBINED)):
        keep = [i for i, n in enumerate(feature_names) if n not in drop]
        a_acc, a_macro = _refit(args, x[:, keep], y, seed=args.seed)
        ablations[tag] = {
            "dropped": list(drop),
            "n_features": len(keep),
            "macro_f1": round(a_macro, 6),
            "accuracy": round(a_acc, 6),
            "accuracy_cost": round(float(m["aggregate_accuracy"]) - a_acc, 6),
        }
        print(f"    -{len(drop)} column(s) {list(drop)}")
        print(f"      macro F1 {a_macro:.4f}   accuracy {a_acc:.4f}   "
              f"({a_acc - float(m['aggregate_accuracy']):+.4f} against the shipped fit)")
    record["idle_time_ablation"] = {
        **ablations,
        "what_this_means": (
            "Removing `IdleTime` ALONE costs essentially the same accuracy as removing all six "
            "columns Combined.csv omits, and both land on the score this model published while it "
            "was fitted on Combined.csv. The other five - `Dir`, `Flgs`, `TcpOpt`, `sVlan`, `dVlan` "
            "- are worth 0.0006 accuracy between them. So the model's former weakness was one "
            "absent flow statistic, not the corpus and not the method. "
            "THAT IS REDUNDANCY AND NOT EMPTINESS, stated because the shorter version would be "
            "false. Two of those five are individually strong - `Dir` scores one-vs-rest AUC 0.9811 "
            "on ICMPFlood and `TcpOpt` 0.9433 on SlowrateDoS, both above the 0.90 threshold "
            "`single_feature_leak_check` uses, and both appear in its flagged list. What the "
            "ablation shows is that the other 41 columns already carry what they carry. "
            "`IdleTime` IS NOT AN IDENTIFIER and the distinction is the same one "
            "`leak_counterfactual_*` exists to draw: it takes 17 distinct integer values over "
            "1,215,890 rows, so it cannot encode a row position, and it separates `SlowrateDoS` - "
            "an attack defined by holding a connection idle - at one-vs-rest AUC 0.9574. A row "
            "counter that scores that well describes where a row sits in a file. This describes "
            "what the traffic did."),
    }

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
    # the whole point and `single_feature_leak_check` records it as data. `IdleTime` DOES appear in
    # the flagged list, at 0.9574 on SlowrateDoS, and that is the same distinction one level down:
    # 17 distinct values over 1.2 million rows cannot address a row, so it is a dominant feature and
    # not a counter. The flagged list is published; a reader can check that claim against it.
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
            "47 of the 113 columns in BTS1_BTS2_fields_preserved.csv. THIS IS THE WIDE EXPORT AND "
            "THE MODEL WAS FITTED ON Combined.csv (52 columns) UNTIL ROUND 50; the two carry the "
            "same 1,215,890 rows, and the six columns the narrow one lacks are the difference "
            "between accuracy 0.76 and 0.98. One of them does all of it: `idle_time_ablation` in "
            "models/metrics.json refits without `IdleTime` alone every run and lands on the old "
            "score. "
            "The 66 dropped columns are listed above WITH THE REASON each was dropped. They are 3 "
            "targets, 9 row identifiers, 10 host fingerprints, 5 columns bit-identical to `Dur`, "
            "and 39 that hold exactly one state in this release - 30 of those are 100% null, "
            "including every SIntPkt/DIntPkt inter-packet-time and jitter column, which are EMPTY "
            "here and not merely uninformative. "
            "`Unnamed: 0` IS THE ONE TO CHECK FIRST. It is the CSV's own row index and leaving it "
            "in takes accuracy to 0.9997, which is the range published for this corpus and is a "
            "leak rather than a result; `leak_counterfactual_*` in models/metrics.json re-measures "
            "it every run. "
            "A SEPARATE AND LEGITIMATE MATTER, stated because the numbers look alarming next to "
            f"that: {record['single_feature_leak_check']['n_features_flagged']} of the "
            f"{len(D.NIDD_FEATURES)} kept features still clear the 0.90 one-vs-rest AUC threshold "
            "after all 66 drops. Those are real flow statistics rather than bookkeeping columns: "
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
    takes all 47.
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
            "multinomial logistic regression on the 37 NUMERIC features only, standardised, fitted "
            "directly rather than distilled. The 10 ordinal codes are excluded because their integer "
            "order is nominal and a linear model would read it as a magnitude. Recorded as the "
            "classical floor and as a feasibility probe for a future CKKS-linear tier: it predicts "
            f"{n_predicted} of {len(class_names)} classes, so a linear student on this corpus does "
            "not collapse the way an 8-class one nearly does elsewhere in this project."),
    }


def _leak_check(xte, yte, feature_names, class_names, *, seed: int, n_rows: int = 100_000) -> dict:
    """Run `prep_model_inputs._warn_single_feature_leak` over a draw of the held-out split.

    Subsampled because that function takes the browser payload's nested-list shape and rank-sorts
    every column; 303,973 x 39 through a Python list is minutes of nothing useful. Rank-AUC is
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
            "scripts/prep_model_inputs.py already publishes against, measured AFTER all 66 drops. "
            "A LARGE NUMBER HERE IS EXPECTED ON THIS CORPUS AND IS NOT THE LEAK THIS MODEL IS ABOUT. "
            "The attacks are generated by distinct tools against distinct services, so a scan, a "
            "SYN flood and a slow-rate HTTP attack genuinely differ in TTL, packet rate, TCP window "
            "and ACK timing - these are the flow statistics an intrusion detector is supposed to "
            "use. The check's own docstring anticipates the case and warns rather than failing. "
            "The distinction that matters is against `leak_counterfactual_*`: `Unnamed: 0`, `Rank`, "
            "`Seq` and `Offset` cleared the same threshold while being bookkeeping columns that "
            "carry no information about the traffic at all, which is why those four are dropped and "
            "these are kept. `IdleTime` is in this list and is the case that shows the rule is not "
            "circular - it clears the threshold on SlowrateDoS, it is the single most valuable "
            "column in the corpus by `idle_time_ablation`, and it takes 17 distinct integer values "
            "over 1.2 million rows, so no amount of it can address a row. Recorded rather than "
            "described so the distinction is checkable."),
    }


if __name__ == "__main__":
    raise SystemExit(main())
