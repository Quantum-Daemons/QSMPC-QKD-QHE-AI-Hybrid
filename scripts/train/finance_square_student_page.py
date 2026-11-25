#!/usr/bin/env python3
"""F7b: the square network on the finance page's fully encrypted path, run as pre-registered.

`references/finance-square-student-page-preregistration.md` (committed before any of its seeds was trained, the owner's
choices of 2026-11-25) asks whether the page's fully encrypted path should run F7's square network in place of the
published linear student. F7 (`scripts/train/finance_square_student.py`) already gave the two models' accuracy, so the
accuracy margins were chosen after F7 and are stated as such there; what this measures beside them is whether new seeds
reproduce F7's accuracy and whether the network is fit for the page: its time and memory under CKKS in a Web Worker in
a browser.

Three stages, in order, into one record:

    python scripts/train/finance_square_student_page.py train   --work W   # seeds 5 to 9 on the GPU; criteria 1, 2
    python scripts/train/finance_square_student_page.py ckks    --work W   # every test row in node, A2 and B; criterion 3
    python scripts/train/finance_square_student_page.py browser --work W   # headless Chrome, three runs a set; criterion 4
    python scripts/train/finance_square_student_page.py export  --work W   # the page's file, after the run (below)

`export` follows the owner's choice of 2026-11-25, after the result: the square network replaces the linear student on
the finance page. It is not part of the pre-registration. It writes `web/public/data/finance/square_student_weights.json`
(the exported seed's folded weights, the scaling and clip, the parameter sets the criteria offered, the record's figures)
with a Platt map fitted on the validation window by `finance_aml.network_calibration`, the finance records' method, so
the page's decision curve and probability can read this network as they read the linear student; the map is written to
the record too, under `after_the_run`, so that it is never read as pre-registered.

The network, its recipe, windows, inputs and parameter sets are F7's, imported from F7's trainer, so the two runs differ
by their seeds alone. The control is the published linear student, unchanged, scored on the same test window as F7 scored
it. The browser stage runs `web/scripts/measure_square_tabular_browser.mjs` on the rows the page ships
(`web/public/data/finance/student_heldout.json`), whose 114 inputs it checks are the trainer's, in the same order.
"""

from __future__ import annotations

import argparse
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

import finance_square_student as f7  # noqa: E402

PREREG = REPO / "references" / "finance-square-student-page-preregistration.md"
PREREG_SHA256 = "388a1668b82b8e9011539511efcbc3887b053a2e0e60022040c4054eee60492f"
RECORD = REPO / "docs" / "measurements" / "finance-square-student-page-2026-11-25.json"
SHIPPED = REPO / "web" / "public" / "data" / "finance" / "student_heldout.json"
SEEDS = (5, 6, 7, 8, 9)
MEAN_MARGIN, EACH_MARGIN = 0.15, 0.10            # criterion 1, over the linear student's test AUPRC
BROWSER_MS, BROWSER_BYTES = 30_000, 1.5e9        # criterion 4: 30 s keys to last decryption; 1.5 GB (10^9 bytes)
RUNS = 3
PAGE_FILE = REPO / "web" / "public" / "data" / "finance" / "square_student_weights.json"


def check_preregistration() -> None:
    if f7.sha256(PREREG) != PREREG_SHA256:
        raise SystemExit(f"{PREREG.relative_to(REPO)} is not the pre-registered text (SHA-256 {f7.sha256(PREREG)[:12]})")


def stage_train(work: Path) -> int:
    import torch
    from finance_aml import (
        _best_threshold,
        causal_account_features,
        load_sorted,
        model_inputs,
        temporal_windows,
    )
    from finance_aml_mlp import CLIP, DEVICE, N_FEATURES, _positive_scores, _scaled, scaling_of
    from sklearn.metrics import average_precision_score

    from qsh.models import distil, metrics

    started = time.time()
    df = load_sorted()
    fit_w, val_w, te_w, tr_w = temporal_windows(df)
    y_fit, y_val, y_te = (d["label"].to_numpy(dtype=np.int8) for d in (fit_w, val_w, te_w))
    (x_fit, x_val, x_te), features, _ = model_inputs(df, fit_w, val_w, te_w, tr_w, causal_account_features(df))
    if x_fit.shape[1] != N_FEATURES:
        raise SystemExit(f"expected {N_FEATURES} features, got {x_fit.shape[1]}")
    centre, scale = scaling_of("standard", x_fit)
    zf, zv, zt = _scaled(x_fit, centre, scale), _scaled(x_val, centre, scale), _scaled(x_te, centre, scale)
    labels = torch.tensor(y_fit, dtype=torch.long, device=DEVICE)
    work.mkdir(parents=True, exist_ok=True)
    f7_record = json.loads(f7.RECORD.read_text())
    record = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
              "follows": {"preregistration": f7_record["preregistration"], "record": str(f7.RECORD.relative_to(REPO))},
              "data": {"corpus": "IBM AMLworld HI-Small (CDLA-Sharing-1.0, synthetic)", "inputs": N_FEATURES,
                       "fit": int(len(y_fit)), "val": int(len(y_val)), "test": int(len(y_te)),
                       "test_positive": int(y_te.sum()), "scaling": "standard, from the fit window", "clip": CLIP},
              "arms": {"square": {}}, "environment": f7.environment()}
    for seed in SEEDS:
        t0 = time.time()
        model, val_auprc, epoch = f7.fit("square", zf, labels, zv, y_val, seed)
        s_val, s_te = _positive_scores(model, zv), _positive_scores(model, zt)
        thr = _best_threshold(y_val, s_val)
        pred = s_te >= thr
        row = {"val_auprc": round(val_auprc, 6), "test_auprc": round(float(average_precision_score(y_te, s_te)), 6),
               "test_minority_f1": round(float(metrics.minority_class_f1(y_te, pred.astype(int))), 6),
               "predicts_both_classes": bool(0 < pred.sum() < len(pred)), "threshold": thr,
               "best_epoch": int(epoch), "seconds": round(time.time() - t0, 1)}
        distil.fold_threshold_into_torch_head(model, thr)
        p = f7.folded(model)
        np.savez(work / f"square_seed{seed}_folded.npz", **p)
        with torch.no_grad():
            torch_logits = torch.cat([model(zt[i:i + 262_144]) for i in range(0, len(zt), 262_144)]).double().cpu().numpy()
        mine = f7.plain_logits(zt.cpu().numpy(), p)
        row["folded_matches_torch_max_abs"] = float(np.max(np.abs(mine - torch_logits)))
        row["decisions_at_argmax_equal_threshold"] = bool(np.array_equal(mine.argmax(1) == 1, pred))
        record["arms"]["square"][str(seed)] = row
        print(f"  square seed {seed}: {row}", flush=True)
    # The control: the published linear student, unchanged, scored on the same test window as F7 scored it.
    rec = json.loads((REPO / "models" / "metrics.json").read_text())
    lin = json.loads((REPO / "models" / "finance_aml_student" / "weights.json").read_text())
    zl = (x_te - np.asarray(lin["input"]["mean"])) / np.asarray(lin["input"]["scale"])
    s_lin = 1 / (1 + np.exp(-(zl @ np.asarray(lin["W"][0]) + lin["b"][0])))
    linear = round(float(average_precision_score(y_te, s_lin)), 6)
    record["control"] = {"model": "finance_aml.student", "test_auprc": linear,
                         "test_minority_f1": rec["finance_aml.student"]["student_metric"],
                         "f7_scored_it": f7_record["references"]["published_linear_student"]["test_auprc"]}
    sq = record["arms"]["square"]
    aucs = [r["test_auprc"] for r in sq.values()]
    export = max(sq, key=lambda s: sq[s]["val_auprc"])
    record["criteria"] = {
        "linear_test_auprc": linear,
        "mean_test_auprc": round(float(np.mean(aucs)), 6), "sd_test_auprc": round(float(np.std(aucs, ddof=1)), 6),
        "1_accuracy": bool(np.mean(aucs) >= linear + MEAN_MARGIN and all(a >= linear + EACH_MARGIN for a in aucs)),
        "1_margins": {"mean_at_least": round(linear + MEAN_MARGIN, 6), "each_at_least": round(linear + EACH_MARGIN, 6)},
        "2_no_collapse": bool(all(r["test_auprc"] >= f7.FLOOR and r["predicts_both_classes"] for r in sq.values())),
        "export_seed": int(export)}
    # The exported seed's test AUPRC with a 95 % percentile interval over 2,000 resamples of test rows (seed 0), as F7.
    p = dict(np.load(work / f"square_seed{export}_folded.npz"))
    z_np = zt.cpu().numpy()
    lg = f7.plain_logits(z_np, p)
    s_exp = 1 / (1 + np.exp(-(lg[:, 1] - lg[:, 0])))
    rng = np.random.default_rng(0)
    boots = [float(average_precision_score(y_te[idx], s_exp[idx]))
             for idx in (rng.integers(0, len(y_te), len(y_te)) for _ in range(2000))]
    record["export_bootstrap"] = {"auprc": round(float(average_precision_score(y_te, s_exp)), 6),
                                  "ci95": [round(float(np.percentile(boots, 2.5)), 6), round(float(np.percentile(boots, 97.5)), 6)],
                                  "resamples": len(boots)}
    record["references"] = {
        "f7_square_test_auprc_by_seed": {k: v["test_auprc"] for k, v in f7_record["arms"]["square"].items()},
        "f7_control_relu_mean_test_auprc": f7_record["criteria"]["mean_test_auprc"]["control"],
        "published_mlp": f7_record["references"]["published_mlp"],
        "published_teacher": f7_record["references"]["published_teacher"]}
    np.save(work / "z_test.npy", z_np.astype(np.float32))
    np.savez(work / "scaling.npz", centre=centre, scale=scale)
    (work / "features.json").write_text(json.dumps(features))
    record["wall_clock_s"] = round(time.time() - started, 1)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["criteria"], indent=1))
    return 0


def stage_browser(work: Path) -> int:
    """Criterion 4: one block of the shipped rows per set, three fresh Web Workers each, in headless Chrome."""
    from finance_aml_mlp import CLIP

    record = json.loads(RECORD.read_text())
    seed = record["criteria"]["export_seed"]
    shipped = json.loads(SHIPPED.read_text())
    features = json.loads((work / "features.json").read_text())
    if shipped["feature_names"] != features or shipped.get("input_space") != "raw":
        raise SystemExit(f"{SHIPPED.relative_to(REPO)} does not hold the trainer's 114 raw inputs in its order")
    p = dict(np.load(work / f"square_seed{seed}_folded.npz"))
    sc = dict(np.load(work / "scaling.npz"))
    w = f7.weights_json(p, sc["centre"], sc["scale"], CLIP, list(f7.PARAMS), int(seed),
                        record["arms"]["square"][str(seed)]["threshold"])
    (work / "page_weights.json").write_text(json.dumps(w))
    out = work / "browser.json"
    load = os.getloadavg()
    if load[0] > 1.5:
        raise SystemExit(f"the machine is not quiet (load average {load[0]:.2f}); criterion 4 is timed, so wait")
    r = subprocess.run(["node", str(REPO / "web" / "scripts" / "measure_square_tabular_browser.mjs"),
                        str(work / "page_weights.json"), str(SHIPPED), str(out), str(RUNS)],
                       cwd=REPO / "web", capture_output=True, text=True, timeout=3600)
    print(r.stdout, end="")
    if r.returncode:
        raise SystemExit(f"the browser harness failed: {r.stderr[-2000:]}")
    b = json.loads(out.read_text())
    # The module's float64 plaintext logits on the shipped rows, against NumPy's on the same scaled rows.
    x = np.asarray(shipped["samples"], dtype=np.float64)
    z = np.clip((x - sc["centre"]) / np.where(sc["scale"] == 0, 1, sc["scale"]), -CLIP, CLIP)
    mine = f7.plain_logits(z, p)
    sets = {}
    for name, runs in b["sets"].items():
        n = runs[0]["rows"]
        js = np.asarray(runs[0]["plain"], dtype=np.float64).reshape(-1, 2)
        sets[name] = {
            "rows": n, "blocks": runs[0]["blocks"],
            "runs": [{k: v for k, v in run.items() if k not in ("plain", "ok")} for run in runs],
            "js_plain_matches_numpy_max_abs": float(np.max(np.abs(js - mine[:n]))),
            "slowest_total_ms": max(run["total_ms"] for run in runs),
            "largest_wasm_memory_bytes": max(run["wasm_memory_bytes"] for run in runs),
            "criterion_4": bool(all(run["total_ms"] <= BROWSER_MS and run["wasm_memory_bytes"] is not None
                                    and run["wasm_memory_bytes"] <= BROWSER_BYTES and run["decisions_differ"] == 0
                                    for run in runs))}
        print(f"  browser {name}: slowest {sets[name]['slowest_total_ms'] / 1000:.2f} s, largest memory "
              f"{sets[name]['largest_wasm_memory_bytes'] / 1e9:.3f} GB, criterion 4 {sets[name]['criterion_4']}", flush=True)
    record["browser"] = {"browser": b["browser"], "user_agent": b["user_agent"], "runs_per_set": b["runs_per_set"],
                         "budget": {"total_ms": BROWSER_MS, "wasm_memory_bytes": BROWSER_BYTES},
                         "rows_from": str(SHIPPED.relative_to(REPO)), "sets": sets,
                         "machine": "the machine F7's record names (i5-13600K, 20 threads), headless, no other job"}
    c = record["criteria"]
    c["4_page"] = {k: v["criterion_4"] for k, v in sets.items()}
    offered = [k for k in f7.PARAMS if c.get("3_fidelity", {}).get(k) and c["4_page"].get(k)]
    c["sets_offered"] = offered
    c["adopted"] = bool(c["1_accuracy"] and c["2_no_collapse"] and offered)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: c[k] for k in ("1_accuracy", "2_no_collapse", "3_fidelity", "4_page", "sets_offered", "adopted")
                      if k in c}, indent=1))
    return 0


def page_file(w: dict, record: dict, calibration: dict) -> dict:
    """The finance page's copy of the exported seed: what `clientFHESquareTabular.ts` reads, the sets the criteria
    offered, the calibration fitted after the run, and the record's figures the page quotes."""
    c, b = record["criteria"], record["browser"]["sets"]
    seed = str(c["export_seed"])
    arm = record["arms"]["square"][seed]
    keep = ("kind", "n_in", "width", "centre", "scale", "clip", "w1", "b1", "w2", "b2", "classes", "decision")
    return {
        "_generated_by": "scripts/train/finance_square_student_page.py export; do not edit by hand",
        **{k: w[k] for k in keep},
        "ckks": {k: w["ckks"][k] for k in c["sets_offered"]},
        "export_seed": int(seed),
        "calibration": calibration,
        "record": str(RECORD.relative_to(REPO)),
        "preregistration": record["preregistration"],
        "figures": {
            "test_auprc": arm["test_auprc"], "test_minority_f1": arm["test_minority_f1"],
            "test_auprc_ci95": record["export_bootstrap"]["ci95"],
            "mean_test_auprc": c["mean_test_auprc"], "sd_test_auprc": c["sd_test_auprc"],
            "linear_test_auprc": c["linear_test_auprc"], "linear_test_minority_f1": record["control"]["test_minority_f1"],
            "test_rows": record["data"]["test"], "test_positive": record["data"]["test_positive"],
            "ckks": {k: {kk: record["ckks"][k][kk] for kk in ("max_abs_logit_error", "decisions_differ", "rows")}
                     for k in c["sets_offered"]},
            "browser": {k: {"rows": b[k]["rows"], "slowest_total_ms": b[k]["slowest_total_ms"],
                            "largest_wasm_memory_bytes": b[k]["largest_wasm_memory_bytes"]} for k in c["sets_offered"]},
            "browser_name": record["browser"]["browser"],
            "sets_not_offered": {k: "slowest run {:.1f} s, over the 30 s budget".format(b[k]["slowest_total_ms"] / 1000)
                                 for k in b if k not in c["sets_offered"]},
        },
        "note": ("F7b's exported seed (references/finance-square-student-page-preregistration.md): dense 114 -> 128 with "
                 "batch norm folded, the square, dense 128 -> 2, the validation threshold folded into the last bias, "
                 "so the decision is the argmax of the two logits. CKKS evaluates it end to end, one row a slot. The "
                 "Platt map was fitted on the validation window after the pre-registered run, on the owner's choice to "
                 "replace the linear student on the page, and is not part of the pre-registration."),
    }


def stage_export(work: Path) -> int:
    from finance_aml import (
        causal_account_features,
        load_sorted,
        model_inputs,
        network_calibration,
        temporal_windows,
    )
    from finance_aml_mlp import _scaled, scaling_of

    record = json.loads(RECORD.read_text())
    if not record.get("criteria", {}).get("adopted"):
        raise SystemExit("F7b was not adopted, so the page does not change")
    seed = record["criteria"]["export_seed"]
    df = load_sorted()
    fit_w, val_w, te_w, tr_w = temporal_windows(df)
    y_val, y_te = (d["label"].to_numpy(dtype=np.int8) for d in (val_w, te_w))
    (x_fit, x_val, _x_te), features, _ = model_inputs(df, fit_w, val_w, te_w, tr_w, causal_account_features(df))
    centre, scale = scaling_of("standard", x_fit)
    sc = dict(np.load(work / "scaling.npz"))
    if not (np.array_equal(centre, sc["centre"]) and np.array_equal(scale, sc["scale"])
            and features == json.loads((work / "features.json").read_text())):
        raise SystemExit("the windows rebuilt here are not the ones the run scaled")
    p = dict(np.load(work / f"square_seed{seed}_folded.npz"))
    lv = f7.plain_logits(_scaled(x_val, centre, scale).cpu().numpy(), p)
    lt = f7.plain_logits(np.load(work / "z_test.npy").astype(np.float64), p)
    calibration = network_calibration(lv, y_val, lt, y_te, model="square network",
                                      fitted_by="scripts/train/finance_square_student_page.py export, after the run")
    calibration["fitted_after_the_preregistered_run"] = True
    # network_calibration writes False for the MLP and the compact GNN, whose pages show logits; this network replaces
    # the linear student, whose decision curve and probability the finance page reads through its map.
    calibration["page_reads_this_map"] = True
    record["after_the_run"] = {
        "why": ("the owner's choice of 2026-11-25, after the result: the square network replaces the linear student on "
                "the finance page, whose decision curve and probability read a Platt map"),
        "calibration": calibration}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    w = json.loads((work / "page_weights.json").read_text())
    PAGE_FILE.write_text(json.dumps(page_file(w, record, calibration), separators=(",", ":")) + "\n")
    print(f"wrote {PAGE_FILE.relative_to(REPO)} ({PAGE_FILE.stat().st_size / 1e3:.0f} kB); Brier "
          f"{calibration['brier_raw']:.8f} -> {calibration['brier_calibrated']:.8f}, MCB "
          f"{calibration['corp_raw']['mcb']:.3e} -> {calibration['corp_calibrated']['mcb']:.3e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("train", "ckks", "browser", "export"))
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    # Absolute: the node harnesses run from web/, and a relative work path would resolve there.
    args.work = args.work.resolve()
    check_preregistration()
    if args.stage == "train":
        return stage_train(args.work)
    if args.stage == "browser":
        return stage_browser(args.work)
    if args.stage == "export":
        return stage_export(args.work)
    z = np.load(args.work / "z_test.npy", mmap_mode="r")
    raw = args.work / "z_test.npy.f32"
    if not raw.exists() or raw.stat().st_size != z.size * 4:
        np.ascontiguousarray(z, dtype=np.float32).tofile(raw)
    return f7.stage_ckks(args.work, record_path=RECORD)


if __name__ == "__main__":
    raise SystemExit(main())
