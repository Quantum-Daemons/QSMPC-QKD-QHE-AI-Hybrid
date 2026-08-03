#!/usr/bin/env python
"""Compare classical and quantum intrusion detectors on QKD-attack telemetry.

    classical   RandomForest — the baseline every quantum result has to beat
    quantum     variational QNN (PennyLane) and a fidelity quantum-kernel SVC

WHAT THIS IS AND IS NOT
    This is an **architecture comparison, not a performance claim**, and the numbers are reported
    that way. Three reasons, all of them citable:

      * The data is **synthetic**. No commercial-OK real network-attack corpus exists: UNSW-NB15
        needs the authors' agreement for commercial use, and neither CIC-IDS2017 nor the
        Engelen-corrected redistribution states a licence at all. The class structure follows
        Al-Kuwari et al. (arXiv:2509.14282, 16 Sep 2025) — 8 classes = 7 attack types + normal.
      * Every published QML-IDS number we could verify sits **below a random forest on the same
        data**, obtained on heavily subsampled and dimension-reduced inputs.
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
from qsh.qml_ids.attacks import ATTACKS, generate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--quantum", action="store_true",
                    help="also fit the variational QNN and quantum-kernel SVC (slow)")
    ap.add_argument("--n-quantum", type=int, default=120,
                    help="samples per class for the quantum models; the kernel is O(n^2) circuits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("models/telecom_qml_ids"))
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    started = time.time()
    train.seed_everything(args.seed)
    spec = get_spec("telecom_ids.qml")

    print(f"generating QKD-attack telemetry: {len(ATTACKS)} classes x {args.n_per_class} "
          "(SYNTHETIC — see the module docstring)")
    X, y, feature_names = generate(n_per_class=args.n_per_class, seed=args.seed)
    print(f"  {X.shape[0]} samples x {X.shape[1]} features; classes: {', '.join(ATTACKS)}")

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

    record["framing"] = (
        "Architecture comparison, not a performance claim. The data is synthetic because no "
        "commercial-OK real attack corpus exists. Every published QML-IDS figure we could verify "
        "sits below a random forest on the same data, and arXiv:2512.05069 was withdrawn by its own "
        "authors for insufficient evaluation. Per-class F1 is the headline; aggregate accuracy is "
        "shown beside it because Engelen et al. (WTMC 2021) demonstrated it stays at 0.99 across a "
        "label correction that moved Bot F1 from 0.60 to 0.99."
    )
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
        "features": list(feature_names),
        "classes": list(ATTACKS),
        "note": "8 classes = 7 attack types + normal traffic, not 'eight attack scenarios'.",
    }, indent=2) + "\n")

    train.record_metrics("telecom_ids.qml", record)
    print(f"\n  wrote {args.out} and models/metrics.json in {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
