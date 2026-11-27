#!/usr/bin/env python3
"""The DeFi page's QCNN on Aave v3 liquidations: its arms, comparators and record, as pre-registered
(`references/defi-qcnn-aave-v3-preregistration.md` and its addendum; Round 86, batch 8).

    python -I scripts/train/defi_qcnn_comparison.py arm qcnn8 --seeds 0-19     # the trainer; GPU on the owner's approval
    python -I scripts/train/defi_qcnn_comparison.py arm qcnn4 --seeds 0-19
    python -I scripts/train/defi_qcnn_comparison.py arm noent --seeds 0-4
    python -I scripts/train/defi_qcnn_comparison.py comparators                # CPU
    python -I scripts/train/defi_qcnn_comparison.py record

Reads `scripts/data/defi_qcnn_inputs.py`'s artifacts; the intervals, the F1 at the validation-best threshold and the
average precision are the finance comparison's own functions, and the classical surrogate's features the trainer's.
Writes docs/measurements/defi-qcnn-aave-v3-2026-11-27.json and web/public/data/defi/qcnn_comparison.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

from finance_qcnn_comparison import f1_at_val_best, intervals, seeds_of  # noqa: E402

PREREG = REPO / "references" / "defi-qcnn-aave-v3-preregistration.md"
PREREG_SHA256 = "af2cc309b6aa3d69b45c3474a6c8d39ce6eadd332baa87a502a98e2166902110"  # with its two addenda, 9031341
WORK = REPO.parent / "tools_external" / "defi_qcnn_work"
RECORD = REPO / "docs" / "measurements" / "defi-qcnn-aave-v3-2026-11-27.json"
SITE = REPO / "web" / "public" / "data" / "defi" / "qcnn_comparison.json"
TRAINER = REPO / "scripts" / "train" / "qcnn_variational.py"
COMMON = ["--arch", "tree", "--layers", "3", "--epochs", "120", "--threads", "4", "--select-on", "ap", "--save-scores",
          "--encoding-scale", "0.5"]
ARMS = {"qcnn8": {"usecase": "defi_aave_v3", "qubits": 8, "flags": []},
        "qcnn4": {"usecase": "defi_aave_v3_k4", "qubits": 4, "flags": []},
        "noent": {"usecase": "defi_aave_v3", "qubits": 8, "flags": ["--no-entangle"]}}
ENCODING_SCALE = 0.5
ARM_LABELS = {
    "qcnn8_shipped": ("QCNN, eight wires: the seed best on validation", "quantum circuit, simulated classically"),
    "qcnn8_ensemble": ("QCNN, eight wires: the 20 seeds' ensemble", "quantum circuit, simulated classically"),
    "qcnn4_ensemble": ("QCNN, four wires: the 20 seeds' ensemble", "quantum circuit, simulated classically"),
    "noent_ensemble": ("QCNN without entanglement: the 5 seeds' ensemble", "product circuit, simulated classically"),
    "logistic_regression": ("Logistic regression on the same inputs", "classical"),
    "gradient_boosting": ("Gradient-boosted trees on the same inputs", "classical"),
    "classical_surrogate": ("Logistic regression on the circuit's Pauli products", "classical surrogate"),
    "gradient_boosting_all_inputs": ("Gradient-boosted trees on all 80 inputs, the whole fit window (reference)", "classical"),
}


def check_prereg() -> str:
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    return got


def load_inputs() -> dict:
    from qsh.models import datasets

    z = np.load(datasets.artifacts("defi_qcnn_inputs.npz"))
    X, y, part = z["X"].astype(np.float64), z["y"].astype(np.int64), z["part"]
    return {"X": X, "y": y, "fit": np.flatnonzero(part == 0), "val": np.flatnonzero(part == 1),
            "te": np.flatnonzero(part == 2),
            "meta": json.loads(datasets.artifacts("defi_qcnn_inputs.json").read_text())}


def run_arm(arm: str, seeds: list[int]) -> int:
    spec = ARMS[arm]
    for s in seeds:
        out = WORK / arm / f"seed{s}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "weights.json").exists() and (out / "test_logits.npy").exists():
            print(f"{arm} seed {s}: done already, kept")
            continue
        t0 = time.time()
        cmd = [sys.executable, "-I", str(TRAINER), "--usecase", spec["usecase"], "--qubits", str(spec["qubits"]), *COMMON,
               *spec["flags"], "--seed", str(s), "--out", str(out)]
        with open(out / "train.log", "w") as log:
            code = subprocess.run(cmd, cwd=REPO, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{arm} seed {s}: exit {code} in {time.time() - t0:.0f} s", flush=True)
        if code:
            return code
    return 0


def _pauli_products():
    spec = importlib.util.spec_from_file_location("qcnn_variational", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._pauli_products


def run_comparators() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score as ap

    from qsh.models import datasets

    check_prereg()
    d = load_inputs()
    X, y, fit, val, te = d["X"], d["y"], d["fit"], d["val"], d["te"]
    t0 = time.time()
    out = WORK / "comparators"
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    def keep(name: str, sv: np.ndarray, st: np.ndarray, **extra) -> None:
        np.save(out / f"{name}_val.npy", np.asarray(sv, dtype=np.float32))
        np.save(out / f"{name}_test.npy", np.asarray(st, dtype=np.float32))
        res[name] = {"validation_ap": round(float(ap(y[val], sv)), 6), **extra}
        print(f"  {name}: validation AP {res[name]['validation_ap']:.4f}", flush=True)

    lr = LogisticRegression(max_iter=2000).fit(X[fit], y[fit])
    keep("logistic_regression", lr.decision_function(X[val]), lr.decision_function(X[te]))
    best = None
    for depth in (3, 5):
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=depth, early_stopping=False,
                                            random_state=0).fit(X[fit], y[fit])
        v = float(ap(y[val], gb.decision_function(X[val])))
        if best is None or v > best[0]:
            best = (v, depth, gb)
    _, depth, gb = best
    keep("gradient_boosting", gb.decision_function(X[val]), gb.decision_function(X[te]), chosen={"max_depth": depth})
    # The circuit's own hypothesis class: logistic regression on the 3^8 products of 1, sin and cos of each wire's angle,
    # c pi tanh(z), with the trainer's function and estimator; scored in blocks.
    pauli = _pauli_products()
    theta = ENCODING_SCALE * np.pi * np.tanh(X)
    sur = LogisticRegression(max_iter=2000).fit(pauli(theta[fit]), y[fit])

    def blocks(rows: np.ndarray) -> np.ndarray:
        return np.concatenate([sur.decision_function(pauli(theta[rows[i:i + 50_000]])) for i in range(0, len(rows), 50_000)])

    keep("classical_surrogate", blocks(val), blocks(te), n_features=int(pauli(theta[:1]).shape[1]))
    # The reference row: the booster on every one of the 80 inputs and the whole fit window, its depth chosen the same way.
    a = np.load(datasets.artifacts("defi_qcnn_all_inputs.npz"))
    best = None
    for depth in (3, 5):
        gw = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=depth, early_stopping=False,
                                            random_state=0).fit(a["fit"], a["y_fit"])
        v = float(ap(a["y_val"], gw.decision_function(a["val"])))
        if best is None or v > best[0]:
            best = (v, depth, gw)
    _, depth, gw = best
    if not (np.array_equal(a["y_val"], y[val]) and np.array_equal(a["y_test"], y[te])):
        raise SystemExit("the all-input windows are not the eight-input windows")
    keep("gradient_boosting_all_inputs", gw.decision_function(a["val"]), gw.decision_function(a["test"]),
         chosen={"max_depth": depth}, inputs=int(a["fit"].shape[1]), rows=int(len(a["y_fit"])))
    res["wall_clock_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


def run_record() -> int:
    from sklearn.metrics import average_precision_score as ap
    from sklearn.metrics import roc_auc_score as auc

    got = check_prereg()
    d = load_inputs()
    yv, yt = d["y"][d["val"]], d["y"][d["te"]]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    arms: dict = {}
    scores: dict = {}
    for arm, seeds in (("qcnn8", range(20)), ("qcnn4", range(20)), ("noent", range(5))):
        per, val_p, test_p = [], [], []
        for s in seeds:
            o = WORK / arm / f"seed{s}"
            vl, tl = np.load(o / "val_logits.npy").astype(np.float64), np.load(o / "test_logits.npy").astype(np.float64)
            per.append({"seed": s, "val_ap": round(float(ap(yv, vl)), 6), "test_ap": round(float(ap(yt, tl)), 6),
                        "test_auc": round(float(auc(yt, tl)), 6)})
            val_p.append(sig(vl))
            test_p.append(sig(tl))
        ens_v, ens_t = np.mean(val_p, axis=0), np.mean(test_p, axis=0)
        best = max(per, key=lambda r: (r["val_ap"], -r["seed"]))
        arms[arm] = {"seeds": per, "mean_test_ap": round(float(np.mean([r["test_ap"] for r in per])), 6),
                     "sd_test_ap": round(float(np.std([r["test_ap"] for r in per], ddof=1)), 6),
                     "ensemble_test_ap": round(float(ap(yt, ens_t)), 6), "ensemble_test_auc": round(float(auc(yt, ens_t)), 6),
                     "ensemble_minority_f1": round(f1_at_val_best(yv, ens_v, yt, ens_t), 6), "best_on_validation": best["seed"]}
        scores[f"{arm}_ensemble"] = ens_t
        if arm == "qcnn8":
            o = WORK / arm / f"seed{best['seed']}"
            vl, tl = np.load(o / "val_logits.npy").astype(np.float64), np.load(o / "test_logits.npy").astype(np.float64)
            arms[arm]["shipped_seed"] = {"seed": best["seed"], "test_ap": best["test_ap"], "test_auc": best["test_auc"],
                                         "minority_f1": round(f1_at_val_best(yv, vl, yt, tl), 6)}
            scores["qcnn8_shipped"] = tl
    comp = json.loads((WORK / "comparators" / "summary.json").read_text())
    for name in [k for k in comp if k != "wall_clock_s"]:
        st = np.load(WORK / "comparators" / f"{name}_test.npy").astype(np.float64)
        sv = np.load(WORK / "comparators" / f"{name}_val.npy").astype(np.float64)
        arms[name] = {**comp[name], "test_ap": round(float(ap(yt, st)), 6), "test_auc": round(float(auc(yt, st)), 6),
                      "minority_f1": round(f1_at_val_best(yv, sv, yt, st), 6)}
        scores[name] = st
    ci = intervals(yt, scores, "qcnn8_shipped")
    export = json.loads((WORK / "export_check.json").read_text()) if (WORK / "export_check.json").exists() else None
    prevalence = float(yt.mean())
    ship = arms["qcnn8"]["shipped_seed"]
    signal = bool(ship["test_ap"] >= 2 * prevalence)
    record = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
        "inputs": {k: d["meta"][k] for k in ("features", "abs_auc_minus_half_on_fit", "counts", "fit_set_rule", "archive")},
        "test_prevalence": round(prevalence, 8), "twice_prevalence": round(2 * prevalence, 8),
        "arms": arms, "intervals": ci,
        "criteria": {"1_signal": signal, "2_export_check": export,
                     "met": bool(signal and export is not None and export.get("criterion_met") and export.get("seed") == ship["seed"]),
                     "comparators_above_the_shipped_seed": sorted(
                         k for k, v in arms.items() if k not in ("qcnn8", "qcnn4", "noent") and v["test_ap"] > ship["test_ap"])},
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    rows = []
    for key, (label, kind) in ARM_LABELS.items():
        if key not in ci:
            continue
        arm = key.removesuffix("_ensemble").removesuffix("_shipped")
        test_ap = (ship["test_ap"] if key == "qcnn8_shipped" else
                   arms[arm]["ensemble_test_ap"] if key.endswith("_ensemble") else arms[key]["test_ap"])
        rows.append({"key": key, "label": label + (f" (seed {ship['seed']})" if key == "qcnn8_shipped" else ""), "kind": kind,
                     "test_ap": test_ap, "ap_ci95": ci[key]["ap_ci95"], "shipped_minus_this_ci95": ci[key]["ship_minus_this_ci95"]})
    SITE.parent.mkdir(parents=True, exist_ok=True)
    SITE.write_text(json.dumps({
        "preregistration": record["preregistration"]["file"], "record": str(RECORD.relative_to(REPO)),
        "task": "a borrow whose account is liquidated within 30 days, on Aave v3 (Polygon), windows by time",
        "features": d["meta"]["features"], "test_rows": int(len(yt)), "test_positive": int(yt.sum()),
        "prevalence": record["test_prevalence"], "twice_prevalence": record["twice_prevalence"], "rows": rows,
        "spreads": {a: {"mean_test_ap": arms[a]["mean_test_ap"], "sd_test_ap": arms[a]["sd_test_ap"], "seeds": len(arms[a]["seeds"])}
                    for a in ("qcnn8", "qcnn4", "noent")},
        "shipped_seed": ship, "criterion_1_signal": signal, "criteria_met": record["criteria"]["met"],
        "comparators_above_the_shipped_seed": [ARM_LABELS.get(k, (k,))[0] for k in record["criteria"]["comparators_above_the_shipped_seed"]],
        "statement": ("Every circuit here acts on at most eight wires, so its state is at most 256 amplitudes and is simulated "
                      "exactly on a classical computer; QCNNs on classical data of this kind are classically simulable "
                      "(Bermejo et al., PRX Quantum 7, 020304, 2026). No quantum advantage is claimed."),
        "licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": ("Spadea, F. and Seneviratne, O., From Risk to Rescue: An Agentic Survival Analysis Framework for "
                        "Liquidation Prevention Dataset, Figshare, doi:10.6084/m9.figshare.31058368.v1, CC BY 4.0."),
        "modifications": ("Figures on the test window of a split by time of the archive's Borrow-to-Liquidated table, from "
                          "models fitted here on eight of its 80 numeric inputs; no rows, accounts or identifiers are shipped."),
    }, indent=1) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)} and {SITE.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("arm", choices=sorted(ARMS))
    a.add_argument("--seeds", default="0-19")
    sub.add_parser("comparators")
    sub.add_parser("record")
    args = ap.parse_args()
    if args.cmd == "arm":
        check_prereg()
        return run_arm(args.arm, seeds_of(args.seeds))
    return run_comparators() if args.cmd == "comparators" else run_record()


if __name__ == "__main__":
    raise SystemExit(main())
