#!/usr/bin/env python
"""Classify 9 network-attack classes on real 5G traffic, and measure what the drops are worth.

WHAT THIS IS
    The telecom use case's second T2 model, and the first one on REAL network data. The other one,
    `telecom_ids.qml`, classifies 8 QKD-LAYER attack types from synthetic per-session physical
    observables. This one classifies 9 NETWORK-LAYER attack types from 39 Argus flow statistics.
    There is no overlap in label space or feature space, so the two numbers on the telecom page are
    not a comparison and neither is the other's baseline. `registry.py` carries that sentence and
    the page prints it.

THE FINDING, which is the reason to ship this rather than the score
    5G-NIDD's 52 columns include three integer row counters - `Unnamed: 0` (the CSV's own index),
    `Seq` (the Argus flow sequence number) and `Offset` (a byte offset into the capture). This
    script fits the SAME model twice, changing nothing but whether those three columns are present,
    and records both. Restoring them takes aggregate accuracy from about 0.76 to about 1.00.

    That is the shape of result Engelen, Rimmer & Joosen (WTMC 2021) produced for CIC-IDS2017, which
    this repository already cites as its reason for reporting per-class F1 before any aggregate. It
    is recorded as a measurement rather than narrated, so `leak_counterfactual_accuracy` in
    `models/metrics.json` is checkable and this paragraph is not the evidence.

    WHAT IT DOES NOT ESTABLISH, stated because the stronger version is the tempting one. The
    published 99.6-99.7% accuracies on this corpus are reproduced by the counterfactual fit to
    within a fraction of a point. That does NOT show the papers reporting them used these columns;
    their feature set is not stated in what was read here. It is an open question about the
    comparison, not a refutation - and asserting otherwise would repeat the error that got 5G-NIDD
    rejected in the first place, which was a confident conclusion drawn from a source that did not
    say it.

WHAT IS MEASURED, and why the usual reading is inverted here
    macro F1 across all 9 classes as the headline, with aggregate accuracy beside it. On every other
    model in this project the aggregate is the flattering one. Here it is the opposite - see
    METRIC_NOTE below, which is published with the metrics so the page cannot read backwards.

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
#: `scripts/train/telecom_ids.py:280-282`; the `**record` entry has no static key and is skipped.

FRAMING = (
    "Real 5G network traffic, 9 attack classes, 39 flow-statistic features. This is a "
    "NETWORK-LAYER model and a different task from the QKD-layer classifier on the same page - no "
    "shared labels, no shared features, so the two scores are not a comparison. The reason it is "
    "published is not its score but the counterfactual recorded beside it: the identical "
    "configuration refitted with the corpus's three integer row identifiers restored reproduces "
    "the published 99.6-99.7% accuracy range, and dropping them costs roughly 24 accuracy points. "
    "Read `leak_counterfactual_accuracy` against `aggregate_accuracy` - that pair is the result."
)

METRIC_NOTE = (
    "THE TWO NUMBERS READ THE OPPOSITE WAY ROUND FROM EVERY OTHER MODEL IN THIS PROJECT, and this "
    "note exists because the page would otherwise imply the reverse. Everywhere else here the "
    "aggregate flatters and the per-class figure exposes it, which is why `aggregate_accuracy` is "
    "rendered as the companion 'shown so the gap is visible'. On this corpus macro F1 is the "
    "OPTIMISTIC number and aggregate accuracy the PESSIMISTIC one. The reason is the class "
    "distribution: seven minority classes are near-perfectly separated while the two majority "
    "classes - Benign at 39.3% of rows and UDPFlood at 37.6% - mutually confuse, so a macro average "
    "over 9 classes is carried by the seven that work and an accuracy over rows is dominated by the "
    "two that do not. The residual is one confusion pair, not diffuse weakness; read `per_class_f1` "
    "and it is visible directly. Neither figure is the flattering one to quote alone."
)

SAMPLE_SIZE_NOTE = (
    "Trained and scored on the whole of 5G-NIDD Combined.csv - 1,215,890 flows, stratified 75/25 at "
    "seed 42, so the test split is 303,973 rows and every class keeps its corpus proportion. No "
    "subsampling and no class rebalancing: `ICMPFlood` is 1,155 rows in the corpus and 289 in the "
    "test split, which is small but is the real prevalence, and resampling it would report a "
    "per-class F1 for a distribution the deployed model never sees. The browser evaluation sample "
    "is a different matter and is NOT proportional - see the note in eval_sample.json, which states "
    "its per-class floor and why its macro F1 is not this one."
)

BASELINE_GAP_NOTE = (
    "There is no baseline number to have a gap to, and that is a deliberate registry decision "
    "rather than a missing measurement: the two figures usually quoted for this corpus (99.61% for "
    "a CNN, 99.71% for a hybrid quantum-classical Quan-ConvCNN, Wang/Fok/Thing) are ACCURACIES, "
    "and this model is judged on macro F1, so putting them in `baseline` would render an accuracy "
    "beside a macro F1 as though the two were the same scale. "
    "What can be said is measured here and recorded above. Fitting this identical configuration "
    "with the corpus's three integer row identifiers restored - `Unnamed: 0`, `Seq`, `Offset`, all "
    "three of which trip this project's own single-feature leak threshold on their own - reproduces "
    "that 99.6-99.7% range. Removing them costs roughly 24 points of accuracy. The published "
    "figures are therefore an anchor rather than a target that has been beaten or missed. "
    "THE LIMIT OF THAT CLAIM: it shows the range is reachable from row counters on this corpus. It "
    "does NOT show the cited papers used them, because their feature set is not stated in what was "
    "read. This is an open question about the comparison, not a refutation of the papers."
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

    print(f"loading 5G-NIDD Combined.csv ({spec.dataset.licence}, {spec.dataset.url})")
    x, y, feature_names, class_names = D.load_5g_nidd(nrows=args.nrows)
    assert x.shape[1] == len(D.NIDD_FEATURES) == 39, (
        f"loader returned {x.shape[1]} columns for {len(D.NIDD_FEATURES)} declared feature names - "
        "the ONNX input contract and the published feature_spec would disagree")
    counts = np.bincount(y, minlength=len(class_names))
    majority_baseline = float(counts.max() / counts.sum())
    print(f"  {x.shape[0]:,} flows x {x.shape[1]} features, {len(class_names)} classes")
    for name, n in sorted(zip(class_names, counts, strict=True), key=lambda kv: -kv[1]):
        print(f"      {name:16s} {n:>9,}  {n / counts.sum() * 100:5.2f}%")
    print(f"  majority-class baseline (predict '{class_names[int(counts.argmax())]}' always) "
          f"{majority_baseline:.4f}")

    # Combined.csv is NOT shuffled - it concatenates two base stations - so a `--nrows` head read
    # gets 6 of the 9 classes and nothing downstream would say so in those words. XGBoost's own
    # message for it is "Invalid classes inferred from unique values of `y`. Expected: [0 1 2 3 4 5],
    # got [0 2 4 6 7 8]", which is true and unhelpful. Say what it means instead.
    absent = [n for n, c in zip(class_names, counts, strict=True) if c == 0]
    if absent:
        raise SystemExit(
            f"{absent} have zero rows in this read, so a 9-class model cannot be fitted and macro F1 "
            f"over 9 classes would be meaningless.\n"
            f"Combined.csv is ordered by capture, not shuffled, so --nrows takes a head that is "
            f"missing classes rather than a sample of them. --nrows is a smoke-test switch; drop it "
            f"for any run whose numbers are recorded.")

    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=args.seed, stratify=y)

    # ---- the shipped model: 39 features, all four traps removed ---------------------------------
    print(f"\n  training XGBoost ({args.n_estimators} trees, depth {args.max_depth}) on 39 features ...")
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
    assert m["macro_f1"] > 0.80, f"macro F1 {m['macro_f1']:.4f} is far below the measured 0.93"
    assert m["aggregate_accuracy"] > majority_baseline, (
        f"accuracy {m['aggregate_accuracy']:.4f} does not beat the majority-class baseline "
        f"{majority_baseline:.4f} - this model would be worse than guessing the commonest label")

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

    # ---- THE COUNTERFACTUAL: the same fit with the three row identifiers restored ---------------
    # Loaded through the same loader with the same encoding rather than re-derived here, so the only
    # difference between the two fits is the three columns. A second drop list beside this one would
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
    print(f"    -> restoring 3 integer row counters is worth "
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
    # IT FIRES A LOT HERE - measured: 21 of the 39 features clear 0.90 - and that is correct
    # behaviour rather than a problem to silence. Its own docstring anticipates the case ("a
    # legitimately dominant feature exists in some corpora ... a build that refuses to run is worse
    # than one that reports what it found"), and on a corpus where each attack class was produced by
    # a different tool against a different service, TTL, packet rate, TCP window and ACK timing
    # separating those classes is the detector working rather than a sampling artefact. What it must
    # NOT still fire on is the three identifiers, and it cannot, because they are not in this matrix
    # - that contrast is the whole point and `single_feature_leak_check` records it as data.
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
        # Published so a consumer can encode a raw row itself. Without it the five string columns
        # are unreachable: the ONNX graph takes floats and carries no vocabulary, so a caller that
        # guessed a different ordering would get a confident wrong class rather than an error.
        "categorical_vocabularies": {k: list(v) for k, v in D.NIDD_CATEGORICAL.items()},
        "missing_value": D.NIDD_MISSING,
        "dropped_columns": dict(D.NIDD_DROPPED),
        "note": (
            "39 of the corpus's 52 columns. The 13 dropped ones are listed above WITH THE REASON "
            "each was dropped, because three of them are the point of this model: `Unnamed: 0`, "
            "`Seq` and `Offset` are integer row counters, and refitting this identical "
            "configuration with them restored reproduces the 99.6-99.7% accuracies published for "
            "this corpus while dropping them costs about 24 points. Those figures are recorded as "
            "`leak_counterfactual_*` in models/metrics.json and are re-measured on every run. "
            "A SEPARATE AND LEGITIMATE MATTER, stated because the numbers look alarming next to "
            f"that: {record['single_feature_leak_check']['n_features_flagged']} of the "
            f"{len(D.NIDD_FEATURES)} kept features still clear the 0.90 one-vs-rest AUC threshold "
            "after all 13 drops, led by `Rate` at 0.9941 on ICMPFlood and `AckDat` at 0.9928 on "
            "HTTPFlood. Those are real flow statistics rather than bookkeeping columns: each attack "
            "class in this corpus was produced by a different tool against a different service, so "
            "packet rate, ACK timing, TTL and TCP window separating them is the signal an intrusion "
            "detector is meant to use. They are reported and kept. The three identifiers cleared the "
            "same threshold while describing only where a row sits in a file, and that is the "
            "distinction this file exists to make - see `single_feature_leak_check` in "
            "models/metrics.json for the full measured list. "
            "`categorical_vocabularies` is the ordinal encoding the five string columns were fitted "
            "under; index into the list, and use `missing_value` for anything absent or unseen. It "
            "is published rather than re-derived because a vocabulary rebuilt from different rows "
            "would remap codes the ONNX thresholds were fitted against."),
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
        # `enable_categorical` is deliberately OFF and the five string columns are ordinal-encoded
        # upstream in the loader. `export_xgboost` converts through
        # `FloatTensorType([None, n_features])`, so a categorical-aware booster cannot survive ONNX
        # export at all - the browser would feed floats to a model expecting category codes.
        # Measured cost of the encoding: macro F1 0.926183 against 0.926086 native, i.e. nothing.
        enable_categorical=False,
        device="cuda", n_jobs=16, random_state=args.seed,
    )


def _linear_floor(xtr, ytr, xte, yte, class_names, *, n_rows: int, seed: int) -> dict:
    """Multinomial logistic regression on the numeric columns: the floor, and a T1 feasibility probe.

    NOT a shipped model and not distilled from the booster - a plain fit, so it answers "how much of
    this is linear" rather than "how well does a student copy the teacher". Two things make it worth
    recording. It is the honest floor for a classical T2 model, since the majority-class predictor is
    a very low bar at 0.39. And it decides whether a CKKS-linear student is viable here later: a
    linear model that collapsed to a subset of the 9 classes would settle that question negatively
    before anyone spent a round on it.

    Numeric columns only, and standardised. The five ordinal codes are nominal - `Proto` index 7 is
    `udp` and index 0 is `arp`, and nothing about that ordering is a magnitude - so feeding them to a
    linear model would assert a false ordering. A tree does not care, which is why the booster above
    takes all 39.
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
            "multinomial logistic regression on the 34 NUMERIC features only, standardised, fitted "
            "directly rather than distilled. The 5 ordinal codes are excluded because their integer "
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
            "scripts/prep_model_inputs.py already publishes against, measured AFTER all 13 drops. "
            "A LARGE NUMBER HERE IS EXPECTED ON THIS CORPUS AND IS NOT THE LEAK THIS MODEL IS ABOUT. "
            "The attacks are generated by distinct tools against distinct services, so a scan, a "
            "SYN flood and a slow-rate HTTP attack genuinely differ in TTL, packet rate, TCP window "
            "and ACK timing - these are the flow statistics an intrusion detector is supposed to "
            "use. The check's own docstring anticipates the case and warns rather than failing. "
            "The distinction that matters is against `leak_counterfactual_*`: `Unnamed: 0`, `Seq` "
            "and `Offset` cleared the same threshold while being bookkeeping columns that carry no "
            "information about the traffic at all, which is why those three are dropped and these "
            "are kept. Recorded rather than described so the distinction is checkable."),
    }


if __name__ == "__main__":
    raise SystemExit(main())
