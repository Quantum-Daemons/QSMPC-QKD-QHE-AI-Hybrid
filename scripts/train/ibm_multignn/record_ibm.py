#!/usr/bin/env python3
"""The record of IBM's Multi-GNN, trained here, in `models/metrics.json` (Round 86).

Reads what the runs left, recomputes nothing a run already measured, and writes one record per protocol through
`train.record_metrics`, which owns that file:
- per seed, from IBM's own training log: the epoch IBM's loop keeps (a validation F1 higher than every earlier
  epoch's, never the first; numbered as its `save_model` stores it, the loop's index plus one) and the validation
  and test F1 its loop logged there, in training mode, as the paper's figures were made;
- per seed, from `precise_bn.py`'s output: the checkpoint with BatchNorm statistics recomputed on the fit window
  (the owner's choice of 2026-11-16), its validation and test probabilities, the threshold that maximises
  validation F1 (`finance_aml._best_threshold`, the rule this project's models use) and the test F1, AUPRC and
  AUROC there; and, for the causal protocol, the same checkpoint's test F1 under the plain batches of 2,048;
- the selected seed (the best logged validation F1 of the five, checked here) and its export's report;
- under the causal protocol, the time input (`QSH_CAUSAL_TIME`, the owner's choice of 2026-11-16): the selected one
  must have the best mean logged validation F1 of those recorded beside it (`--compare`), whose five seeds the record
  carries too, and the runs with IBM's absolute time input, stopped after three seeds, may be recorded as the defect.

The runs' files are named as the queue names them, in the work folder: `logs/ibm_<model>_<run>_seed<s>_saved.log`
and `<model>_<run>_fitbn_seed<s>[_batchcut]/`, the run being `project`, or `causal` with the time input after it
(`causal_lag`; the absolute runs are plain `causal`). Run with this repository's environment, after the runs:
    python scripts/train/ibm_multignn/record_ibm.py --key finance_aml.gnn --protocol ibm --work <work> \
        --export <artifact dir> --selected 4
    python scripts/train/ibm_multignn/record_ibm.py --key finance_aml.gnn_causal --protocol causal --work <work> \
        --time-input lag --compare drop --absolute-seeds 1,2,3 --export <artifact dir> --selected 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "train")]

from finance_aml import _best_threshold  # noqa: E402

from qsh.models import train  # noqa: E402

LINE = re.compile(r"\] (Validation|Test) F1: ([0-9.]+)")
SEEDS = (1, 2, 3, 4, 5)


def logged(path: Path) -> dict:
    """IBM's loop as it ran: the epoch it keeps, by its own rule, and what it logged there."""
    va, te = [], []
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if m:
            (va if m.group(1) == "Validation" else te).append(float(m.group(2)))
    n = min(len(va), len(te))
    best, at = 0.0, None
    for ep in range(1, n):
        if va[ep] > best:
            best, at = va[ep], ep
    if at is None:
        raise SystemExit(f"{path}: no epoch after the first improved on a validation F1 of 0")
    return {"epochs_run": n, "kept_epoch": at + 1, "logged_validation_f1": va[at], "logged_test_f1": te[at]}


def scored(d: Path) -> dict:
    """The recomputed statistics' probabilities, at the threshold tuned on validation, and the saved ones' for the record."""
    va, te = np.load(d / "val.npz"), np.load(d / "test.npz")
    out = {}
    for key, tag in (("probs", ""), ("probs_saved_bn", "saved_bn_")):
        pv, pt, yt = va[key].astype(np.float64), te[key].astype(np.float64), te["labels"]
        thr = _best_threshold(va["labels"], pv)
        out[f"{tag}decision_threshold"] = round(float(thr), 6)
        out[f"{tag}test_minority_f1"] = round(float(f1_score(yt, pt >= thr)), 6)
        out[f"{tag}test_auprc"] = round(float(average_precision_score(yt, pt)), 6)
        if not tag:
            out["test_auroc"] = round(float(roc_auc_score(yt, pt)), 6)
            out["test_accuracy"] = round(float(((pt >= thr).astype(int) == yt).mean()), 6)
            out["validation_minority_f1"] = round(float(f1_score(va["labels"], pv >= thr)), 6)
            out["n_test_scored"] = int(len(yt))
            out["test_positives_scored"] = int(yt.sum())
            out["n_validation_scored"] = int(len(va["labels"]))
    rep = json.loads((d / "bn_report.json").read_text())
    out["batchnorm"] = {"fit_batches": rep["fit_batches"], "saved_num_batches_tracked": rep["saved_num_batches_tracked"],
                        "test_decisions_at_0_5_changed": rep["windows"]["test"]["decisions_at_0.5_changed"]}
    out["checkpoint_sha256"] = hashlib.sha256((d / "checkpoint_fit_bn.tar").read_bytes()).hexdigest()
    out["source_checkpoint_sha256"] = rep["source_checkpoint_sha256"]
    if (d / "train.npz").exists():
        out["fit_window_probs_sha256"] = hashlib.sha256((d / "train.npz").read_bytes()).hexdigest()
    return out


def batch_cut(d: Path, thr: float) -> dict:
    """The same checkpoint under the plain batches of 2,048 (the owner's choice of 2026-11-15), at the same threshold."""
    te = np.load(d / "test.npz")
    pt, yt = te["probs"].astype(np.float64), te["labels"]
    return {"test_minority_f1": round(float(f1_score(yt, pt >= thr)), 6),
            "test_auprc": round(float(average_precision_score(yt, pt)), 6), "n_test_scored": int(len(yt))}


def spread(vals: list[float]) -> dict:
    return {"mean": round(st.mean(vals), 6), "sample_sd": round(st.stdev(vals), 6), "n": len(vals)}


#: The causal protocol's time inputs, as the record states them (the export's change note says the same).
TIME_TEXT = {"absolute": "IBM's absolute timestamp, scaled by the fit window's statistics",
             "drop": "the time input set to zero, so time only orders and cuts the batches",
             "lag": ("each transfer's time before the end of its batch, over the fit window's span (443,520 s), clipped "
                     "to [-1, 0]")}


def runs(work: Path, model: str, run: str, seeds: tuple[int, ...], causal: bool) -> list[dict]:
    """Every seed of one kind of run, from its training log and its precise_bn.py output (and plain batches if causal)."""
    out = []
    for s in seeds:
        fit = work / f"{model}_{run}_fitbn_seed{s}"
        row = {"seed": s, **logged(work / "logs" / f"ibm_{model}_{run}_seed{s}_saved.log"), **scored(fit)}
        if causal:
            row["plain_batches_of_2048"] = batch_cut(fit.with_name(fit.name + "_batchcut"), row["decision_threshold"])
        out.append(row)
    return out


def summary(rows: list[dict], causal: bool) -> dict:
    return {"logged_validation_f1": spread([r["logged_validation_f1"] for r in rows]),
            "logged_test_f1": spread([r["logged_test_f1"] for r in rows]),
            "test_minority_f1": spread([r["test_minority_f1"] for r in rows]),
            "test_auprc": spread([r["test_auprc"] for r in rows]),
            **({"plain_batches_of_2048_test_minority_f1":
                spread([r["plain_batches_of_2048"]["test_minority_f1"] for r in rows])} if causal else {})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=("finance_aml.gnn", "finance_aml.gnn_causal"))
    ap.add_argument("--protocol", required=True, choices=("ibm", "causal"))
    ap.add_argument("--model", default="pna", choices=("pna", "gin"))
    ap.add_argument("--work", type=Path, required=True, help="the folder the runs wrote: logs/ and the precise_bn.py outputs")
    ap.add_argument("--time-input", default="absolute", choices=tuple(TIME_TEXT),
                    help="the causal protocol's time input of the selected runs (QSH_CAUSAL_TIME)")
    ap.add_argument("--compare", action="append", default=[], choices=("drop", "lag"),
                    help="another time input trained the same way, five seeds, recorded beside the selected one")
    ap.add_argument("--absolute-seeds", default="", help="the causal runs with IBM's absolute time input, as 1,2,3")
    ap.add_argument("--export", type=Path, required=True, help="export_for_browser.py's output for the selected seed")
    ap.add_argument("--selected", type=int, required=True)
    a = ap.parse_args()
    causal = a.protocol == "causal"
    if causal == (a.time_input == "absolute"):
        raise SystemExit("the causal protocol's runs are recorded by their time input, drop or lag; IBM's protocol keeps IBM's")
    if (a.compare or a.absolute_seeds) and not causal:
        raise SystemExit("--compare and --absolute-seeds are the causal protocol's")
    if a.time_input in a.compare:
        raise SystemExit("--compare names a time input other than the selected one")
    run = lambda mode: "project" if not causal else ("causal" if mode == "absolute" else f"causal_{mode}")  # noqa: E731

    seeds = runs(a.work, a.model, run(a.time_input), SEEDS, causal)
    others = {m: runs(a.work, a.model, run(m), SEEDS, causal) for m in a.compare}
    means = {m: round(st.mean([r["logged_validation_f1"] for r in rows]), 6)
             for m, rows in {a.time_input: seeds, **others}.items()}
    if max(means, key=means.get) != a.time_input:
        raise SystemExit(f"{a.time_input} was asked for, but the mean logged validation F1s are {means}")
    best = max(seeds, key=lambda r: r["logged_validation_f1"])
    if best["seed"] != a.selected:
        raise SystemExit(f"seed {a.selected} was asked for, but seed {best['seed']} has the best logged validation F1")
    sel = next(r for r in seeds if r["seed"] == a.selected)
    export = json.loads((a.export / "export_report.json").read_text())
    if export["checkpoint_sha256"] != sel["checkpoint_sha256"]:
        raise SystemExit("the export was made from another checkpoint than the selected seed's recomputed one")
    if abs(export["decision_threshold"] - sel["decision_threshold"]) > 1e-6:
        raise SystemExit("the export's threshold is not the one tuned on validation from the recomputed statistics")
    if export.get("time_input", "absolute") != a.time_input:
        raise SystemExit(f"the export's time input is {export.get('time_input')!r}, not {a.time_input!r}")

    record = {
        "dataset": "IBM AMLworld HI-Small", "protocol": a.protocol,
        "split": ("temporal, this project's windows by position (fit 56, validation 14, test 30 percent); "
                  + ("IBM's loader samples each transfer's neighbours from its window's whole graph"
                     if not causal else
                     "each mini-batch one batch of 2,048 in time order, split where it spans more than 17 minutes, "
                     "with every sampled transfer later than it dropped, and every window scaled by the fit window's "
                     f"statistics; the time input {TIME_TEXT[a.time_input]}")),
        "time_input": a.time_input,
        "minority_class_f1": sel["test_minority_f1"], "auprc": sel["test_auprc"], "auroc": sel["test_auroc"],
        "accuracy": sel["test_accuracy"], "decision_threshold": sel["decision_threshold"],
        "auprc_baseline": round(sel["test_positives_scored"] / sel["n_test_scored"], 8),
        "n_test": sel["n_test_scored"], "n_validation": sel["n_validation_scored"],
        "selected_seed": a.selected, "kept_epoch": sel["kept_epoch"],
        "logged_test_f1": sel["logged_test_f1"], "logged_validation_f1": sel["logged_validation_f1"],
        "five_seeds": summary(seeds, causal),
        "seeds": seeds,
        "batchnorm": {"statistics": "recomputed on the fit window, a plain average over one pass of the protocol's fit-window loader",
                      "why": "IBM's loop never calls model.eval(), so the checkpoint it saves carries statistics last set by the test pass",
                      **sel["batchnorm"]},
        "metric_note": ("minority_class_f1, auprc and decision_threshold are the exported model's, scored in eval mode with the "
                        "recomputed statistics at the threshold that maximises validation F1; logged_test_f1 is the figure IBM's "
                        "own loop logs at the kept epoch, in training mode (dropout on, batch statistics), as the paper's figures are."),
        "export": {"opset": export["opset"], "fp32_bytes": export["graph_bytes"],
                   "max_abs_logit_difference_onnx_vs_torch_on_seeds": export["max_abs_logit_difference_onnx_vs_torch_on_seeds"],
                   "checkpoint_sha256": export["checkpoint_sha256"],
                   **({"time_input_range": export["time_input_range"]} if "time_input_range" in export else {}),
                   **({"against_the_recorded_scoring": export["against_the_recorded_scoring"]}
                      if export.get("against_the_recorded_scoring") else {}),
                   **({"seconds_before_batch_end": export["seconds_before_batch_end"]}
                      if "seconds_before_batch_end" in export else {}),
                   "int8_skipped_reason": (f"not quantised: the 16-transfer sample's graph is {export['graph_bytes'] / 1e6:.2f} MB "
                                           "in fp32, and no int8 build was checked against it")},
        "code": {"repository": "https://github.com/IBM/Multi-GNN", "commit": "252b0252afca109d1d216c411c59ff70753b25fc",
                 "licence": "Apache-2.0",
                 "patches": ["scripts/train/ibm_multignn/project_split.patch"]
                            + (["scripts/train/ibm_multignn/causal.patch"] if causal else []),
                 "paper": "Egressy et al., 'Provably Powerful Graph Neural Networks for Directed Multigraphs', AAAI 2024, arXiv:2306.11586"},
        "git_sha": train.git_sha(),
    }
    if causal:
        record["time_input_selection"] = {
            "rule": ("the time input whose five seeds have the best mean logged validation F1, then the seed with the best "
                     "logged validation F1 among its five (the owner's choice of 2026-11-16)"),
            "mean_logged_validation_f1": means}
        record["other_time_inputs"] = {m: {"time_input_note": TIME_TEXT[m], "five_seeds": summary(rows, causal), "seeds": rows}
                                       for m, rows in others.items()}
        if a.absolute_seeds:
            abs_seeds = tuple(int(x) for x in a.absolute_seeds.split(","))
            rows = runs(a.work, a.model, run("absolute"), abs_seeds, causal)
            record["absolute_time_input"] = {
                "time_input_note": TIME_TEXT["absolute"],
                "why_stopped": ("stopped after these seeds, its test inputs lying outside every value training saw (the owner's "
                                "choice of 2026-11-16)"),
                "seeds": rows}
    train.record_metrics(a.key, record)
    print(json.dumps({k: record[k] for k in ("minority_class_f1", "auprc", "auroc", "decision_threshold", "selected_seed",
                                             "logged_test_f1", "time_input")}, indent=1))
    print(json.dumps(record["five_seeds"], indent=1))
    if causal:
        print(json.dumps(record["time_input_selection"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
