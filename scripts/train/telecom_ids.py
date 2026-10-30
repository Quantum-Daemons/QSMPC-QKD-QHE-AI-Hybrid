#!/usr/bin/env python
"""Compare classical and quantum intrusion detectors on QKD-attack telemetry.

    classical   RandomForest — the baseline every quantum result has to beat
    quantum     variational QNN (PennyLane) and a fidelity quantum-kernel SVC

WHAT THIS IS AND IS NOT
    This is an **architecture comparison, not a performance claim**, and the numbers are reported
    that way. Three reasons, all of them citable:

      * The data is **synthetic**, and the reason stated here was too broad. This docstring used to
        say "No commercial-OK real network-attack corpus exists". 5G-NIDD is one - CC BY 4.0,
        access_type Open, at Fairdata/Metax, DOI 10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361 -
        and this project rejected it until Round 47 on a mirror that carries no licence string.
        What survives is narrower and is the actual constraint: **no commercial-OK corpus of real
        QKD-LAYER attacks exists**, because that needs a physical QKD testbed under attack.
        UNSW-NB15 needs the authors' agreement for commercial use; CIC-IDS2017's provider pages
        state conflicting terms and the Engelen-corrected redistribution states none; and 5G-NIDD
        is network-layer flow statistics, a different problem rather than a substitute for the
        labels this script trains on. The class structure follows Al-Kuwari et al.
        (arXiv:2509.14282, 16 Sep 2025) - 8 classes = 7 attack types + normal.
      * Published QML-IDS results are **mixed, not uniformly below a random forest**. An earlier
        revision of this docstring claimed they were, and the very paper cited above refutes it:
        Al-Kuwari et al. Table VIII reports a hybrid QLSTM at 94.7% accuracy / 94.7% F1 against
        their own random-forest baseline on the same generated data. What survives is the narrower
        point: quantum results in this area are usually obtained on heavily subsampled and
        dimension-reduced inputs, so they are hard to compare with a forest given the full feature
        set. `registry.py` carries the corrected note and the use-case page prints it.
      * arXiv:2512.05069, a hybrid quantum-classical autoencoder IDS, was **withdrawn by its own
        authors** (v2, 2026-04-02) for evaluation "insufficient to fully support the paper's
        conclusions". That is the honest state of the field and it is printed here.

    So the classical baseline runs every time and is reported first. A quantum model that does not
    beat it has not earned its qubits, and saying so is the point of the exercise.

WHAT IS MEASURED
    Per-class F1 across all 8 classes, macro F1, and — deliberately alongside — aggregate accuracy,
    which is the metric that makes a broken detector look fine. Engelen et al. (WTMC 2021) showed
    aggregate random-forest accuracy on CIC-IDS2017 sits at 0.99 both before and after correcting
    the labels, while Bot F1 moves 0.60 -> 0.99. The aggregate is blind to the fix.

Usage:
    python scripts/train/telecom_ids.py [--n-per-class 300] [--quantum]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qsh.models import export, metrics, train  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402
from qsh.qml_ids.attacks import ATTACKS, FEATURES, generate  # noqa: E402

#: Hoisted to module level so `scripts/check_generated_prose.py` can read them out of the AST.
#: They were assigned by subscript onto an already-built dict, which the gate cannot see - so two
#: published claims sat outside the only check that compares a generator to its artifact. The gate
#: now reports how many fields are beyond its reach; these were two of the three.
CLASSICAL_RF_ATTRIBUTION_NOTE = (
        "Impurity importance (training split, biased toward high-cardinality features) and "
        "permutation importance (held-out split, scored on macro F1, 20 repeats, std reported). "
        "Read the permutation figure where they disagree. These rank which features of a SYNTHETIC "
        "corpus this forest needs, not signals from a real QKD link. Two planted features are "
        "nearly redundant: sift_ratio separates detector_blinding perfectly on its own yet costs "
        "almost nothing to remove, because click_asymmetry separates the same class; and "
        "dark_count_rate is within one std of zero. Exact linear SHAP does not apply to a tree "
        "ensemble, which is why these two measures are used instead."
    )

FRAMING = (
        "Architecture comparison, not a performance claim. The published model is the random "
        "forest; the quantum-kernel SVC and the variational QNN are recorded beside it and not "
        "published. The data is synthetic because no commercial-OK real QKD-attack corpus exists. "
        "Published QML-IDS results are mixed rather than uniformly below a random forest: "
        "Al-Kuwari et al. (arXiv:2509.14282) report a hybrid QLSTM at 94.7% against their own "
        "random forest, while arXiv:2512.05069 was withdrawn by its own authors for insufficient "
        "evaluation. Per-class F1 is the headline; aggregate accuracy is "
        "shown beside it because Engelen et al. (WTMC 2021) demonstrated it stays at 0.99 across a "
        "label correction that moved Bot F1 from 0.60 to 0.99."
    )


STUDENT_METRIC_NOTE = (
            "Read `n_classes_predicted` before `macro_f1`. A linear student that stops predicting a "
            "class scores 0 on it by construction, and macro F1 then averages in a zero that looks "
            "like poor discrimination rather than an absent output - this repo's own QNN carries "
            "exactly that defect and is documented as doing so, so the check is not hypothetical. "
            "`agreement` is recorded but is NOT evidence the minority classes survived: agreement on "
            "the finance student rose from 0.884 to 0.947 across a change that left minority-class F1 "
            "unmoved, which is why per-class F1 is the headline here and the aggregate is a "
            "companion.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--quantum", action="store_true",
                    help="also fit the variational QNN and quantum-kernel SVC (slow)")
    ap.add_argument("--n-quantum", type=int, default=120,
                    help="samples per class for the quantum models; the kernel is O(n^2) circuits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("models/telecom_qml_ids"))
    ap.add_argument("--student-out", type=Path, default=Path("models/telecom_ids_student"),
                    help="the T1 CKKS-linear student distilled from the forest")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    started = time.time()
    train.seed_everything(args.seed)
    spec = get_spec("telecom_ids.qml")

    print(f"generating QKD-attack telemetry: {len(ATTACKS)} classes x {args.n_per_class} "
          "(SYNTHETIC — see the module docstring)")
    # `generate` returns (X, y, LABEL names) - its docstring says so. An earlier revision here
    # unpacked the third value as `feature_names` and wrote it into feature_spec.json as
    # "features", so the published spec advertised eight features named after the eight ATTACK
    # CLASSES for a model that takes seven. Nothing failed: the shapes are never compared, and
    # `zip(names, values)` truncates to the shorter side instead of raising, so the attribution
    # added in this round silently labelled qber_mean as "normal" and dropped click_asymmetry.
    # The real feature axis is `FEATURES`. Two guards, deliberately: the assert below names the
    # problem in one line, and `strict=True` on each zip makes any future length divergence raise
    # at the point of use rather than truncate. Truncation is what made the original wrong and
    # plausible at the same time.
    X, y, class_names = generate(n_per_class=args.n_per_class, seed=args.seed)
    assert list(class_names) == list(ATTACKS), "generate() no longer returns the class names"
    assert X.shape[1] == len(FEATURES), (
        f"FEATURES has {len(FEATURES)} names for {X.shape[1]} columns - a silent mislabel of "
        "every per-feature number downstream")
    print(f"  {X.shape[0]} samples x {X.shape[1]} features ({', '.join(FEATURES)}); "
          f"classes: {', '.join(ATTACKS)}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    # ---- classical baseline, always ------------------------------------------
    rf = RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=8)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    rf_metrics = metrics.with_companion(yte, rf_pred, rf.predict_proba(Xte), headline="macro_f1")

    print("\n  classical RandomForest (the number to beat)")
    print(f"    macro F1            {rf_metrics['macro_f1']:.4f}")
    print(f"    aggregate accuracy  {rf_metrics['aggregate_accuracy']:.4f}"
          "   <- companion; looks fine even when a class is broken")
    print("    per-class F1:")
    for name, f1 in sorted(rf_metrics["per_class_f1"].items(), key=lambda kv: kv[1]):
        label = ATTACKS[int(name)] if name.isdigit() and int(name) < len(ATTACKS) else name
        print(f"      {label:24s} {f1:.4f}")

    record: dict = {
        "classical_rf_macro_f1": round(rf_metrics["macro_f1"], 6),
        "classical_rf_per_class_f1": {
            (ATTACKS[int(k)] if k.isdigit() and int(k) < len(ATTACKS) else k): round(v, 6)
            for k, v in rf_metrics["per_class_f1"].items()},
        "aggregate_accuracy": round(rf_metrics["aggregate_accuracy"], 6),
        "n_classes": len(ATTACKS),
        "n_samples": int(X.shape[0]),
        "data": "synthetic, class structure per arXiv:2509.14282 (Al-Kuwari et al., 2025)",
    }

    # ---- attribution for the classical baseline --------------------------------
    # The published RandomForest carries this use case's headline number and had NO attribution of
    # any kind, for four rounds, while the browser showed exact linear SHAP over a DIFFERENT model
    # it trains in the tab. Exact SHAP does not apply to a tree ensemble, so the two standard tree
    # attributions are computed here instead, at training time, on the estimator that produced the
    # number.
    #
    # Both, not one, because they disagree in a way that is informative. Impurity importance is
    # computed on the TRAINING split and is biased toward high-cardinality features; permutation
    # importance is measured on the HELD-OUT split and answers "how much worse does the model get
    # without this feature". Where they disagree, the second is the one to trust.
    #
    # `scoring="f1_macro"` and not the default accuracy. This module's own WHAT IS MEASURED section
    # is an argument against aggregate accuracy on this corpus: it sits at 0.99 across a label
    # correction that moves a per-class F1 from 0.60 to 0.99. Permuting one feature of an 8-class
    # problem scoring 0.9867 would barely move the aggregate, so the default would have reported
    # every feature as unimportant.
    perm = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=args.seed,
                                  scoring="f1_macro", n_jobs=8)
    record["classical_rf_impurity_importance"] = {
        name: round(float(v), 6) for name, v in zip(FEATURES, rf.feature_importances_, strict=True)}
    record["classical_rf_permutation_importance"] = {
        name: round(float(v), 6) for name, v in zip(FEATURES, perm.importances_mean, strict=True)}
    record["classical_rf_permutation_importance_std"] = {
        name: round(float(v), 6) for name, v in zip(FEATURES, perm.importances_std, strict=True)}
    # WHAT THESE NUMBERS ARE AN ATTRIBUTION OF, stated because it is easy to over-read. The corpus
    # is synthetic, so this is a statement about `qsh.qml_ids.attacks` and about which planted
    # features the forest actually NEEDS - not about which signal matters on a real QKD link.
    #
    # It earns its place by being more than a restatement of the generator. Measured: the two
    # rankings disagree, and where they do the permutation figure is the informative one.
    # `sift_ratio` is 5th by impurity (0.1318) but 6th by permutation (+0.0054, i.e. removing it
    # costs almost nothing) - because `click_asymmetry` separates the same class equally well, so
    # either alone suffices and neither is individually necessary. `dark_count_rate` is +0.0001
    # against a std of 0.0010: **not distinguishable from zero**, even though the generator does
    # vary it 3x for detector_blinding. Two of seven planted features are redundant to this model.
    #
    # It also corroborates a falsifiable prediction this project wrote down before measuring:
    # a correct attribution must NOT name `sift_ratio` or `dark_count_rate` as a dominant driver.
    # They rank last of seven. That is the check working, not a coincidence.
    record["classical_rf_attribution_note"] = CLASSICAL_RF_ATTRIBUTION_NOTE

    print("\n  attribution (top 4 by permutation importance on macro F1)")
    for name, v in sorted(record["classical_rf_permutation_importance"].items(),
                          key=lambda kv: -kv[1])[:4]:
        sd = record["classical_rf_permutation_importance_std"][name]
        imp = record["classical_rf_impurity_importance"][name]
        print(f"    {name:24s} perm {v:+.4f} +/- {sd:.4f}   impurity {imp:.4f}")

    # ---- quantum models, opt-in ----------------------------------------------
    if args.quantum:
        from qsh.qml_ids.detector import evaluate_quantum, evaluate_quantum_svc

        print(f"\n  variational QNN (PennyLane), {args.n_quantum}/class ...")
        qnn = evaluate_quantum(n_per_class=args.n_quantum, seed=args.seed)
        print(f"    accuracy {qnn.accuracy:.4f}  |  macro F1 {qnn.macro_f1:.4f}  "
              f"|  binary (attack vs normal) F1 {qnn.binary_f1:.4f}  |  {qnn.n_params} parameters")

        print(f"  quantum-kernel SVC, {min(args.n_quantum, 60)}/class (O(n^2) circuit evaluations) ...")
        qsvc = evaluate_quantum_svc(n_per_class=min(args.n_quantum, 60), seed=args.seed)
        print(f"    accuracy {qsvc.accuracy:.4f}  |  macro F1 {qsvc.macro_f1:.4f}  "
              f"|  binary F1 {qsvc.binary_f1:.4f}")

        record.update({
            "quantum_qnn_accuracy": round(float(qnn.accuracy), 6),
            "quantum_qnn_macro_f1": round(float(qnn.macro_f1), 6),
            "quantum_qnn_binary_f1": round(float(qnn.binary_f1), 6),
            "quantum_qnn_n_params": int(qnn.n_params),
            "quantum_svc_accuracy": round(float(qsvc.accuracy), 6),
            "quantum_svc_macro_f1": round(float(qsvc.macro_f1), 6),
            "quantum_svc_binary_f1": round(float(qsvc.binary_f1), 6),
        })
        # Compare like with like: macro F1 against macro F1, on the same 8-class problem.
        beat = max(qnn.macro_f1, qsvc.macro_f1) > rf_metrics["macro_f1"]
        record["quantum_beats_classical"] = bool(beat)
        print(f"\n  does either quantum model beat the random forest? {'yes' if beat else 'NO'}"
              f"{'' if beat else '  <- expected, and the reason this is framed as an architecture comparison'}")
    else:
        print("\n  (quantum models skipped; pass --quantum to fit them)")

    record["framing"] = FRAMING
    record["dataset"] = spec.dataset.name
    record["git_sha"] = train.git_sha()
    record["wall_clock_s"] = round(time.time() - started, 1)

    # ---- artefacts ------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures = Xte[: min(256, len(Xte))].astype(np.float32)
    report = export.verify_and_quantize(rf, fixtures, args.out, kind="sklearn",
                                        n_features=X.shape[1])
    print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB"
          + (f" (int8 skipped: {report.int8_skipped_reason})" if report.int8_skipped_reason else ""))
    record["export"] = {"fp32_bytes": report.fp32_bytes, "opset": report.opset,
                        "int8_skipped_reason": report.int8_skipped_reason}

    (args.out / "feature_spec.json").write_text(json.dumps({
        "features": list(FEATURES),
        "classes": list(ATTACKS),
        "note": (
            "8 classes = 7 attack types + normal traffic (ATTACKS[0] == 'normal'), not 'eight "
            "attack scenarios'. 'features' are the QKD telemetry statistics the model takes as "
            "input, from FEATURES in src/qsh/qml_ids/attacks.py; they are NOT the class labels. "
            "An earlier revision of this generator wrote the class labels here, because it "
            "unpacked generate()'s third return value - documented as label_names - into a "
            "variable called feature_names. Generated from the source, and now true: the JSON was "
            "once corrected by hand while this generator kept writing the wrong value, so the "
            "next retrain reintroduced it."),
    }, indent=2) + "\n")

    train.record_metrics("telecom_ids.qml", {
        **record,
        # Restated as literals so the gate can compare generator to artifact. The `**record`
        # entry has no static key and is skipped; these two are what it reads.
        "classical_rf_attribution_note": CLASSICAL_RF_ATTRIBUTION_NOTE,
        "framing": FRAMING,
    })

    # ---- T1: the CKKS-linear student, so this use case HAS an encrypted tier -------------------
    student_record = distil_linear_student(rf, Xtr, ytr, Xte, yte, args.student_out, seed=args.seed)
    train.record_metrics("telecom_ids.student", {
        **student_record,
        # As above: `**` has no static key and is skipped, so this literal is what the
        # prose gate compares against the artifact.
        "metric_note": STUDENT_METRIC_NOTE,
    })

    print(f"\n  wrote {args.out}, {args.student_out} and models/metrics.json "
          f"in {time.time()-started:.1f}s")
    return 0


def distil_linear_student(teacher, Xtr, ytr, Xte, yte, out: Path, *, seed: int) -> dict:
    """Distil the 8-class RandomForest to a linear model that can run under CKKS.

    WHY THIS EXISTS. `runs_encrypted=True` appeared on five ModelSpecs and none of them was telecom,
    so this was the one vertical with no encrypted tier at all - which meant the project's standing
    per-industry question ("is the private-inference artifact class right for this industry?") had no
    artifact to judge here, and `references/model-choice-vs-2026-practice.md` answered a
    model-QUALITY question in its place without noticing the substitution.

    SOFT TARGETS, and this is the part that matters. The teacher's *probabilities* are the training
    signal, not its argmax. Distilling on hard labels was a real defect in this repo once already -
    `references/finance-gap-investigation.md` records it and `finance_aml.py` carries the fix - and on
    an 8-class problem it is worse than on a binary one: the argmax discards exactly the minority-class
    information that macro F1 is computed from. sklearn's LogisticRegression takes no soft labels, so
    each training row is replicated once per class with `sample_weight` equal to the teacher's
    probability for that class. Weighted cross-entropy over those copies IS cross-entropy against the
    soft target, exactly - no approximation, and it costs n_train x 8 rows, which is nothing here.

    THE ARGMAX IS NOT ENCRYPTED, and the caller must say so. CKKS has no native comparison. What runs
    under encryption is 8 dot products - 56 weights and 8 biases, 8 slots of a ring of 8192 - and the
    client decrypts all 8 class scores and takes the maximum in the clear. That is the mainstream
    construction (it is what Concrete-ML's deployment model does) and it is honest here because in a
    single-tab demo the client IS the user, so nothing is revealed that was not already theirs. The
    alternatives were both rejected on measured grounds rather than by preference: a homomorphic argmax
    costs 12.83 s for 128 elements (Mazzone et al., USENIX Security 2025), and handing the argmax to
    the MPC round that `mpc/secret_inference.py:secure_argmax` already implements would buy an
    architecture demonstration and not a security property, because `docs/threat-model.md` records that
    both MPC parties share one JS heap.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    n_classes = len(ATTACKS)
    soft = teacher.predict_proba(Xtr)
    if soft.shape[1] != n_classes:                       # a teacher that never saw a class
        raise SystemExit(f"teacher emits {soft.shape[1]} columns for {n_classes} classes - the "
                         f"training split is missing a class and macro F1 would be meaningless")

    # Standardise on the training split only, and ship mean/scale so the browser reproduces it.
    mean, scale = Xtr.mean(axis=0), Xtr.std(axis=0)
    scale[scale == 0] = 1.0
    Ztr, Zte = (Xtr - mean) / scale, (Xte - mean) / scale

    rep_X = np.repeat(Ztr, n_classes, axis=0)
    rep_y = np.tile(np.arange(n_classes), len(Ztr))
    rep_w = soft.reshape(-1)
    keep = rep_w > 1e-6                                  # drop zero-weight copies; sklearn ignores them anyway
    # No `multi_class=` argument: it was removed in sklearn 1.7, and multinomial (softmax) is the
    # default for a multiclass fit with lbfgs, which is the objective the soft targets require.
    student = LogisticRegression(max_iter=4000, C=10.0, random_state=seed)
    student.fit(rep_X[keep], rep_y[keep], sample_weight=rep_w[keep])

    scores = Zte @ student.coef_.T + student.intercept_  # exactly what the CKKS path computes
    pred = scores.argmax(axis=1)
    st = metrics.with_companion(yte, pred, None, headline="macro_f1")
    n_predicted = int(len(np.unique(pred)))
    teacher_pred = teacher.predict(Xte)
    agreement = float((pred == teacher_pred).mean())

    print("\n  T1 CKKS-linear student (distilled from the forest on SOFT targets)")
    print(f"    macro F1            {st['macro_f1']:.4f}   (teacher {record_macro(teacher, Xte, yte):.4f})")
    print(f"    classes predicted   {n_predicted}/{n_classes}"
          + ("" if n_predicted == n_classes else "   <- COLLAPSED, see below"))
    print(f"    agreement w/teacher {agreement:.4f}   <- NOT evidence the minority classes survived")
    print("    per-class F1:")
    for name, f1 in sorted(st["per_class_f1"].items(), key=lambda kv: kv[1]):
        label = ATTACKS[int(name)] if name.isdigit() and int(name) < len(ATTACKS) else name
        print(f"      {label:24s} {f1:.4f}")

    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "kind": "logistic-linear-multiclass",
        "note": (
            "The CKKS-linear student for telecom. What runs under homomorphic encryption is 8 dot "
            "products over 7 standardised features - W is 8x7 and b is 8. The ARGMAX IS NOT "
            "ENCRYPTED: the client decrypts all 8 class scores and takes the maximum in the clear, "
            "because CKKS has no native comparison. That is the standard construction and it is "
            "honest in a single-tab demo where the client is the user, but it must not be described "
            "as deciding the class under encryption. Distilled from the forest's PROBABILITIES, not "
            "its argmax - on 8 classes the argmax discards precisely the minority-class signal that "
            "macro F1 is computed from."),
        "teacher_key": "telecom_ids.qml",
        "student_key": "telecom_ids.student",
        "n_features": int(Xtr.shape[1]),
        "features": list(FEATURES),
        "input": {"preprocessing": ["subtract mean", "divide by scale"],
                  "mean": [float(v) for v in mean], "scale": [float(v) for v in scale]},
        "W": [[float(w) for w in row] for row in student.coef_],
        "b": [float(v) for v in student.intercept_],
        "n_parameters": int(student.coef_.size + student.intercept_.size),
        "classes": list(ATTACKS),
    }, indent=2) + "\n"
    (out / "weights.json").write_text(payload)

    # The browser copy is BUNDLED rather than fetched from the Hub, following `genomics.prs` - the
    # other tier that runs under CKKS with `hf_repo=None`. 64 floats do not need a pinned revision,
    # and bundling means the encrypted tier works on a fresh clone with no network. Written from the
    # same `payload` string, so the two cannot drift: one serialisation, two destinations.
    web = Path("web/public/data/telecom")
    web.mkdir(parents=True, exist_ok=True)
    (web / "student_weights.json").write_text(payload)

    rec = {
        "macro_f1": round(float(st["macro_f1"]), 6),
        "per_class_f1": {
            (ATTACKS[int(k)] if k.isdigit() and int(k) < len(ATTACKS) else k): round(v, 6)
            for k, v in st["per_class_f1"].items()},
        "aggregate_accuracy": round(float(st["aggregate_accuracy"]), 6),
        "n_classes_predicted": n_predicted,
        "n_classes": n_classes,
        "n_eval": int(len(yte)),
        "n_parameters": int(student.coef_.size + student.intercept_.size),
        "agreement": round(agreement, 6),
        "distilled_on": "teacher predict_proba (soft targets), weighted one-copy-per-class",
        "argmax_location": "client, in plaintext, after decrypting all 8 class scores",
        "metric_note": STUDENT_METRIC_NOTE,
    }
    if n_predicted < n_classes:
        rec["collapse_warning"] = (
            f"predicts only {n_predicted} of {n_classes} classes, so macro F1 has a floor of zero on "
            f"the {n_classes - n_predicted} it never emits")
    return rec


def record_macro(model, Xte, yte) -> float:
    """The teacher's macro F1, recomputed so the student's print is directly comparable."""
    from sklearn.metrics import f1_score
    return float(f1_score(yte, model.predict(Xte), average="macro"))


if __name__ == "__main__":
    raise SystemExit(main())
