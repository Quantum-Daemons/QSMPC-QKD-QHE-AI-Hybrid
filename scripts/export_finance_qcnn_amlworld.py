#!/usr/bin/env python3
"""The finance page's QCNN becomes the AMLworld circuit the comparison would ship (Round 86, batch 6).

`references/finance-qcnn-amlworld-comparison-preregistration.md`, criterion 1: the page's finance QCNN becomes the arm-1
seed with the best validation average precision if its test average precision is at least twice the prevalence and the
export check holds (`scripts/check_finance_qcnn_export.py`); the owner chose on 2026-11-26 to make the change in this
batch. This writes the served artifact and everything that reads it, and the record of each step:

- THE ARTIFACT, as trained, in the units the pages already feed. The circuit was trained on the fit window's
  standardised inputs, clipped at 10 (`scripts/data/aml_qcnn_inputs.py`). The served file folds that standardisation into
  its own `feature_mean` and `feature_std` and declares `input_space: "raw"`, so every reader (the browser's circuit, the
  tree contraction, the Pauli form under CKKS, the Python tier) takes a transaction's raw values, as it does on the other
  pages. The clip is not carried: tanh(10) is 1 - 4.1e-9, and the record measures what dropping it changes.
- THE DECISION, at the threshold best on the validation window. The trainer decides at a logit of 0, which a fit sample of
  one laundering transaction to four others puts far below the window's prevalence (minority-class F1 0.064 on the test
  window). The served bias carries the logit that maximises validation minority-class F1, the comparison's own rule,
  chosen on the validation window and never on the test window.
- THE COMPARATORS, at the same rule: the trainer's logistic regression on the same tanh features and the classical
  surrogate on the circuit's 6,561 Pauli products, refitted on the same fit sample and thresholded on validation, so the
  page sets like against like.
- THE HELD-OUT SAMPLE: the 8,192 test-window transactions `finance/student_heldout.json` already ships (a draw enriched to
  2 percent laundering), at the circuit's eight inputs, with the figures the trainer's circuit gives on them, which the
  tab's re-score reproduces; and a byte-identical copy under `data/finance/` for the Python tier, whose image ships
  `data/` and not `web/public/`.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/export_finance_qcnn_amlworld.py
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts" / "train"), str(REPO / "src")]

WEIGHTS = REPO / "models" / "qcnn_finance" / "weights.json"
SITE_WEIGHTS = REPO / "web" / "public" / "data" / "finance" / "qcnn_weights.json"
SITE_HELD = REPO / "web" / "public" / "data" / "finance" / "qcnn_heldout.json"
DATA_HELD = REPO / "data" / "finance" / "qcnn_heldout.json"
STUDENT_HELD = REPO / "web" / "public" / "data" / "finance" / "student_heldout.json"
RECORD = REPO / "docs" / "measurements" / "finance-qcnn-page-export-2026-11-26.json"
BLOCK = 20_000


def _comparison():
    spec = importlib.util.spec_from_file_location("finance_qcnn_comparison", REPO / "scripts" / "train" / "finance_qcnn_comparison.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def val_best(yv: np.ndarray, sv: np.ndarray) -> float:
    """The score maximising validation minority-class F1, the comparison's `f1_at_val_best` rule."""
    from sklearn.metrics import precision_recall_curve

    p, r, thr = precision_recall_curve(yv, sv)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    return float(thr[int(np.argmax(f))])


def at(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    from sklearn.metrics import average_precision_score, f1_score

    pred = (s > t).astype(int)
    return {"accuracy": round(float((pred == y).mean()), 6), "minority_f1": round(float(f1_score(y, pred, zero_division=0)), 6),
            "average_precision": round(float(average_precision_score(y, s)), 6)}


def pauli(theta: np.ndarray) -> np.ndarray:
    import qcnn_variational as Q

    return Q._pauli_products(theta).astype(np.float32)


def main() -> int:
    import qcnn_variational as Q
    import torch
    from sklearn.linear_model import LogisticRegression

    from qsh.models import datasets
    from qsh.qhe import qcnn_tree

    started = time.time()
    F = _comparison()
    F.check_prereg()
    check = json.loads((F.WORK / "export_check.json").read_text())
    if not check["criterion_met"]:
        raise SystemExit("the export check did not pass; criterion 1 does not hold and nothing ships")
    seed = int(check["seed"])
    w = json.loads((F.WORK / "qcnn1" / f"seed{seed}" / "weights.json").read_text())
    meta = json.loads(datasets.artifacts("aml_qcnn_inputs.json").read_text())
    if list(meta["features"]) != list(w["feature_names"]):
        raise SystemExit("the seed's inputs are not the pre-registered eight")
    d = F.load_inputs()
    Z, y = d["X"], d["y"]
    fit, val, te = d["fit"], d["val"], d["te"]
    yv, yt = y[val], y[te]
    c_enc = float(w["encoding_scale"])

    # ---- the circuit's own logits (the trainer's, on the GPU), its threshold, its figures ------------------------------
    o = F.WORK / "qcnn1" / f"seed{seed}"
    vl, tl = np.load(o / "val_logits.npy").astype(np.float64), np.load(o / "test_logits.npy").astype(np.float64)
    t_q = val_best(yv, vl)
    qcnn_test = at(yt, tl, t_q)
    print(f"QCNN seed {seed}: validation threshold logit {t_q:.6f}; test {qcnn_test}", flush=True)

    # ---- the trainer's comparators, refitted on the same fit sample and thresholded on validation -------------------
    Xs = np.tanh(Z)
    lr = LogisticRegression(max_iter=2000).fit(Xs[fit], y[fit])
    lv, lt = lr.decision_function(Xs[val]), lr.decision_function(Xs[te])
    lr_test = at(yt, lt, val_best(yv, lv))
    print(f"logistic regression on the tanh features: test {lr_test}", flush=True)
    theta = c_enc * np.pi * Xs
    # ONE THREAD. On 6,561 products the solver's path depends on the order its sums are taken in, and a multithreaded BLAS
    # takes them in an order that varies with the machine's load: two runs of 2026-11-26 gave minority-class F1 0.271299
    # and 0.274374. Pinned, the figure is a function of the inputs, and the file says whether the solver converged.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        sur = LogisticRegression(max_iter=2000).fit(pauli(theta[fit]), y[fit])
        sv = np.concatenate([sur.decision_function(pauli(theta[val[i:i + BLOCK]])) for i in range(0, len(val), BLOCK)])
        st = np.concatenate([sur.decision_function(pauli(theta[te[i:i + BLOCK]])) for i in range(0, len(te), BLOCK)])
    sur_iter = int(np.max(sur.n_iter_))
    sur_test = at(yt, st, val_best(yv, sv))
    print(f"classical surrogate on the 6,561 Pauli products: test {sur_test}; {sur_iter} iterations", flush=True)

    # ---- the served artifact -----------------------------------------------------------------------------------------
    centre, scale_ = np.asarray(meta["centre"], dtype=np.float64), np.asarray(meta["scale"], dtype=np.float64)
    served = copy.deepcopy(w)
    served.update({
        "usecase": "finance", "trained_as": "finance_amlworld", "input_space": "raw",
        "feature_mean": [float(v) for v in centre], "feature_std": [float(v) for v in scale_],
        "bias": float(w["bias"]) - t_q,
        "test_accuracy": qcnn_test["accuracy"], "test_minority_f1": qcnn_test["minority_f1"],
        "test_auprc": qcnn_test["average_precision"],
        "classical_lr_accuracy": lr_test["accuracy"], "classical_lr_minority_f1": lr_test["minority_f1"],
        "classical_lr_auprc": lr_test["average_precision"],
        "classical_surrogate": {**w["classical_surrogate"], "test_accuracy": sur_test["accuracy"],
                                "test_minority_f1": sur_test["minority_f1"], "test_auprc": sur_test["average_precision"],
                                "solver_iterations": sur_iter, "converged": sur_iter < 2000, "threads": 1,
                                "threshold": "the score maximising validation minority-class F1, as the circuit's"},
        "decision_threshold": {"rule": "the logit maximising minority-class F1 on the validation window (710,969 "
                                       "transactions), the comparison's own rule; never chosen on the test window",
                               "logit": round(t_q, 9), "folded_into_bias": True,
                               "trainer_bias": float(w["bias"]),
                               "at_the_trainers_logit_0": {"test_accuracy": w["test_accuracy"],
                                                           "test_minority_f1": w["test_minority_f1"]}},
        "standardisation": {"folded": ("the fit window's centre and scale of each input, from scripts/data/aml_qcnn_inputs.py, "
                                       "are this file's feature_mean and feature_std; the trainer's input was the same "
                                       "standardised value clipped at 10, and the clip is not carried (see `clip_effect`)"),
                            "clip": float(meta["clip"])},
        "selection": {**w.get("selection", {}), "seed": seed, "of_seeds": len(list((F.WORK / "qcnn1").glob("seed*/val_logits.npy"))),
                      "rule": "the highest validation average precision of arm 1's twenty seeds (the comparison's criterion 1)"},
        "comparison": {"preregistration": "references/finance-qcnn-amlworld-comparison-preregistration.md",
                       "record": "docs/measurements/finance-qcnn-amlworld-comparison-2026-11-26.json",
                       "export_check": {k: check[k] for k in ("test_rows", "page_vs_state_vector", "criterion_met")}},
        "replaces": ("the PaySim QCNN of Round 84 (four wires, PaySim's four columns), which the finance page ran until "
                     "2026-11-26"),
        "note": (w.get("note", "") + " SERVED (Round 86, batch 6): the fit window's standardisation folded into "
                 "feature_mean and feature_std, and the validation-best threshold into the bias; see `standardisation`, "
                 "`decision_threshold` and `clip_effect`."),
    })

    # ---- what folding and dropping the clip change, on the whole test window -----------------------------------------
    trained_dec = tl > t_q
    clipped = np.abs(Z[te]) >= float(meta["clip"]) - 1e-12
    rows_clipped = np.flatnonzero(clipped.any(axis=1))
    zinf = Z[te][rows_clipped].copy()
    zinf[clipped[rows_clipped]] = np.sign(zinf[clipped[rows_clipped]]) * 1e6          # tanh -> +-1: the clip's furthest reach
    sv_w = copy.deepcopy(served)
    sv_w.update({"feature_mean": [0.0] * 8, "feature_std": [1.0] * 8})               # z in, the served circuit otherwise
    ro_at_clip = qcnn_tree.readout(sv_w, Z[te][rows_clipped]) if len(rows_clipped) else np.zeros(0)
    ro_beyond = qcnn_tree.readout(sv_w, zinf) if len(rows_clipped) else np.zeros(0)
    lg = lambda r: r * float(served["scale"]) + float(served["bias"])  # noqa: E731
    clip_effect = {"test_rows_with_a_clipped_input": int(len(rows_clipped)),
                   "largest_readout_change_up_to_the_limit": float(np.max(np.abs(ro_beyond - ro_at_clip))) if len(rows_clipped) else 0.0,
                   "decisions_that_could_change": int(np.sum((lg(ro_beyond) > 0) != (lg(ro_at_clip) > 0)))}
    served["clip_effect"] = clip_effect
    print(f"clip: {clip_effect}", flush=True)
    # The served circuit on the standardised rows (its own moments applied to raw = centre + scale * z) gives the trained
    # circuit's decisions at the served threshold: the readouts are the trainer's to float32 rounding (2.1e-6, the check).
    page_ro = qcnn_tree.readout(sv_w, Z[te])
    served_dec = lg(page_ro) > 0
    agreement = {"test_rows": int(len(te)), "decisions_differing_from_the_trained_circuit_at_the_threshold":
                 int(np.sum(served_dec != trained_dec))}
    print(f"served against trained at the threshold: {agreement}", flush=True)

    text = json.dumps(served, indent=1) + "\n"
    WEIGHTS.write_text(text)
    SITE_WEIGHTS.write_text(text)

    # ---- the held-out sample: the page's 8,192 test-window rows, at the circuit's eight inputs ---------------------------
    sh = json.loads(STUDENT_HELD.read_text())
    cols = [sh["feature_names"].index(n) for n in served["feature_names"]]
    rows = np.asarray([[r[j] for j in cols] for r in sh["samples"]], dtype=np.float64)
    labels = np.asarray(sh["labels"], dtype=int)
    x = c_enc * np.tanh((rows - centre) / scale_)
    with torch.no_grad():
        z = Q.circuit(torch.tensor(x, dtype=torch.float32), torch.tensor(served["theta"], dtype=torch.float32), 8,
                      readout_qubits=0, layers=int(served["layers"]), arch="tree")
    logit = z.numpy().astype(np.float64) * float(served["scale"]) + float(served["bias"])
    sample = at(labels, logit, 0.0)
    held = {
        "usecase": "finance", "reads": ["finance/qcnn_weights.json"], "feature_names": list(served["feature_names"]),
        "input_space": "raw", "class_names": served.get("class_names"),
        "split": (f"the {len(labels):,} test-window transactions finance/student_heldout.json ships, a draw enriched to 2 "
                  "percent laundering; the circuit never trained, chose a checkpoint or a threshold on them. Its figures over "
                  "the whole test window, 1,523,504 transactions, are the weights file's"),
        "rows": [[float(v) for v in r] for r in rows], "labels": [int(v) for v in labels],
        "expected": {"test_accuracy": sample["accuracy"], "test_minority_f1": sample["minority_f1"],
                     "note": "the trainer's circuit on these rows, at the served threshold: what the tab's re-score reproduces"},
        "provenance": served["provenance"],
        "licence": sh["licence"], "licence_url": sh["licence_url"], "attribution": sh["attribution"],
        "modifications": ("The circuit's eight of the 114 computed inputs, unrounded as finance/student_heldout.json ships "
                          "them, for the same 8,192 rows and in their order. Enhanced Data shared under the same "
                          "CDLA-Sharing-1.0."),
    }
    htext = json.dumps(held, separators=(",", ":")) + "\n"
    SITE_HELD.write_text(htext)
    DATA_HELD.write_text(htext)
    print(f"held-out sample: {sample}", flush=True)

    RECORD.write_text(json.dumps({
        "preregistration": {"file": "references/finance-qcnn-amlworld-comparison-preregistration.md",
                            "sha256": F.PREREG_SHA256},
        "seed": seed, "export_check": check, "decision_threshold_logit": t_q,
        "test_window": {"qcnn": qcnn_test, "logistic_regression_tanh_features": lr_test,
                        "classical_surrogate": sur_test, "prevalence": round(float(yt.mean()), 8)},
        "clip_effect": clip_effect, "served_against_trained": agreement, "held_out_sample": sample,
        "wall_clock_s": round(time.time() - started, 1),
    }, indent=1) + "\n")
    print(f"wrote {WEIGHTS.relative_to(REPO)}, {SITE_WEIGHTS.relative_to(REPO)}, {SITE_HELD.relative_to(REPO)}, "
          f"{DATA_HELD.relative_to(REPO)} and {RECORD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
