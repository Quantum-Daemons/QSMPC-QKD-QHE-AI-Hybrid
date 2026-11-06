"""The telecom quantum comparison, like for like: the forest on each quantum model's own split and labels (Round 86).

`scripts/train/telecom_ids.py` recorded `quantum_beats_classical` as the best quantum macro F1 against the forest's.
The two were not the same quantity. The quantum models are scored on BINARY attack-against-normal labels, so their
macro F1 averages two classes, while the forest's averages eight. Their data differ too: the variational QNN trains on
`generate(120, seed)` and the kernel SVC on `generate(60, seed)`, each split by the config's `ids.test_fraction`, and
the forest on `generate(300, seed)` with a 25% split. This script reproduces each quantum model's split. It checks the
split against the record: the kernel SVC is re-fitted and must give its recorded figures, and the QNN's recorded
figures must be the constant "attack" predictor's on its split, which is what they are. It then scores, on exactly
those rows and labels:

  - a random forest trained on the binary labels (the trainer's hyperparameters, 200 trees, seed 42);
  - the same forest trained on the eight classes, with its predictions collapsed to attack against normal;
  - the constant "attack" predictor.

It writes `docs/measurements/telecom-quantum-like-for-like-<date>.json`, and records the comparison under
`telecom_ids.qml` in `models/metrics.json` (`quantum_comparison_like_for_like`, and `quantum_beats_classical` on that
basis), through `record_metrics`, without retraining or republishing anything.

    .venv/bin/python scripts/measure_telecom_quantum_like_for_like.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

KEY = "telecom_ids.qml"
#: The trainer's `--seed` default and its `--n-quantum` default, and the kernel SVC's cap (`min(n_quantum, 60)`).
SEED, N_QNN, N_QSVC = 42, 120, 60


def main() -> int:
    from qsh.config import get_config
    from qsh.models import registry, train
    from qsh.qml_ids.detector import classical_on_quantum_split, evaluate_quantum_svc

    rec = registry.achieved(KEY)
    tf = float(get_config().ids.test_fraction)
    qnn, qsvc = classical_on_quantum_split(N_QNN, SEED, tf), classical_on_quantum_split(N_QSVC, SEED, tf)

    # The QNN's recorded figures are the constant "attack" predictor's on its split; required, not assumed.
    const = qnn["constant_attack"]
    recorded_qnn = {"accuracy": rec["quantum_qnn_accuracy"], "macro_f1": rec["quantum_qnn_macro_f1"],
                    "attack_f1": rec["quantum_qnn_binary_f1"]}
    if any(abs(const[k] - recorded_qnn[k]) > 5e-6 for k in const):
        raise SystemExit(f"the QNN's record {recorded_qnn} is not the constant predictor's {const} on its split")
    # The kernel SVC re-fitted on its split must give its recorded figures.
    svc = evaluate_quantum_svc(n_per_class=N_QSVC, seed=SEED)
    got = {"accuracy": round(svc.accuracy, 6), "macro_f1": round(svc.macro_f1, 6), "attack_f1": round(svc.binary_f1, 6)}
    recorded_svc = {"accuracy": rec["quantum_svc_accuracy"], "macro_f1": rec["quantum_svc_macro_f1"],
                    "attack_f1": rec["quantum_svc_binary_f1"]}
    if any(abs(got[k] - recorded_svc[k]) > 5e-6 for k in got):
        raise SystemExit(f"the kernel SVC re-fitted gives {got}; the record says {recorded_svc}")
    qnn["quantum"] = {"model": "variational QNN", **recorded_qnn,
                      "is_the_constant_attack_predictor": True}
    qsvc["quantum"] = {"model": "quantum-kernel SVC", **recorded_svc}

    best_gap = max(qnn["quantum"]["macro_f1"] - qnn["forest_trained_binary"]["macro_f1"],
                   qsvc["quantum"]["macro_f1"] - qsvc["forest_trained_binary"]["macro_f1"])
    beats = bool(best_gap > 0)
    measured = dt.date.today().isoformat()
    out = {
        "measured": measured, "script": "scripts/measure_telecom_quantum_like_for_like.py",
        "basis": ("binary attack-against-normal macro F1, each quantum model against a forest trained and scored on "
                  "that model's own split and labels"),
        "why": ("the record compared the quantum models' binary macro F1 with the forest's 8-class macro F1, on "
                "different draws (`telecom_ids.py` called that 'the same 8-class problem')"),
        "test_fraction": tf, "seed": SEED, "qnn_split": qnn, "qsvc_split": qsvc,
        "quantum_beats_classical": beats,
    }
    path = REPO / "docs" / "measurements" / f"telecom-quantum-like-for-like-{measured}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    payload = dict(rec)
    payload["quantum_comparison_like_for_like"] = {
        "basis": out["basis"],
        "qnn": {"quantum_macro_f1": qnn["quantum"]["macro_f1"],
                "forest_macro_f1": qnn["forest_trained_binary"]["macro_f1"],
                "constant_attack_macro_f1": qnn["constant_attack"]["macro_f1"], "test_rows": qnn["test_rows"]},
        "qsvc": {"quantum_macro_f1": qsvc["quantum"]["macro_f1"],
                 "forest_macro_f1": qsvc["forest_trained_binary"]["macro_f1"],
                 "constant_attack_macro_f1": qsvc["constant_attack"]["macro_f1"], "test_rows": qsvc["test_rows"]},
        "measured_by": f"{out['script']} at git {train.git_sha()} ({path.relative_to(REPO)})",
    }
    payload["quantum_beats_classical"] = beats
    train.record_metrics(KEY, payload)
    for name, s in (("QNN", qnn), ("kernel SVC", qsvc)):
        print(f"{name} split ({s['test_rows']} test rows): quantum {s['quantum']['macro_f1']}, forest binary "
              f"{s['forest_trained_binary']['macro_f1']}, forest 8-class read binary "
              f"{s['forest_trained_8_class_read_binary']['macro_f1']}, constant {s['constant_attack']['macro_f1']}")
    print(f"quantum_beats_classical, like for like: {beats}; wrote {path.relative_to(REPO)} and the record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
