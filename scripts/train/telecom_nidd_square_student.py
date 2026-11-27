#!/usr/bin/env python3
"""The square network for the telecom page's encrypted 5G-NIDD path, run as pre-registered (Round 86, batch 12).

`references/telecom-nidd-square-student-preregistration.md` (committed before any network below was trained, on the
owner's choices of 2026-11-27 and 2026-11-28) asks whether the telecom page's encrypted 5G-NIDD path should offer a
square network beside, or in place of, the linear student. The re-run of batch 9 already gave this recipe's accuracy on
seeds 0 to 4, so the accuracy margins were chosen after it and are stated as such there; what this measures beside them
is whether new seeds reproduce it and whether the network is fit for the page: its fidelity under CKKS, and its time and
memory in a Web Worker in a browser.

Four stages, in order, into one record:

    python scripts/train/telecom_nidd_square_student.py train   --work W   # seeds 5 to 9 on the GPU; criteria 1, 2, 5
    python scripts/train/telecom_nidd_square_student.py ckks    --work W   # 2,048 test flows in node, A2 and B; criterion 3
    python scripts/train/telecom_nidd_square_student.py browser --work W   # headless Chrome, three runs a set; criterion 4
    python scripts/train/telecom_nidd_square_student.py verdict --work W   # the pre-registered rules, read off the record

Everything the re-run fixed is the re-run's, imported from `scripts/train/telecom_nidd_clean_protocols.py`: the loader,
P2, F30, the scores (FLamby's metric among them), the held-out draw, the linear student's procedure (the control, which
must come out as the re-run's record has it) and the square MLP's (`mlp_arm`, which with `weights=True` also returns
what it kept). The two parameter sets are the finance square network's, imported from its trainer. The work directory
holds what is not committed: the exported seed's weights and the flows the two encrypted stages read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

import finance_square_student as f7  # noqa: E402  (the two parameter sets, unchanged)

PREREG = REPO / "references" / "telecom-nidd-square-student-preregistration.md"
PREREG_SHA256 = "e23f2515e064475d558ccf07fa6695071e10843d168caa88db1f0bfebd00cdb4"  # committed in 73f176f
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-square-student-2026-11-28.json"
SEEDS = (5, 6, 7, 8, 9)
MEAN_MARGIN, EACH_MARGIN = 0.015, 0.010     # criterion 1, over the control's test FLamby metric
MACRO_F1_FLOOR = 0.90                       # criterion 2
FIDELITY = 1e-3                             # criterion 3, the largest decrypted-against-plaintext logit difference
BOOTSTRAP, BOOTSTRAP_SEED = 2000, 0         # criterion 5
BROWSER_MS, BROWSER_BYTES = 30_000, 1.5e9   # criterion 4: 30 s keys to last decryption; 1.5 GB (10^9 bytes)
RUNS = 3


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _module("telecom_nidd_clean_protocols", REPO / "scripts" / "train" / "telecom_nidd_clean_protocols.py")


def check_preregistration() -> None:
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.relative_to(REPO)} hashes to {got[:12]}, not the committed {PREREG_SHA256[:12]}: "
                         "run nothing")


def data():
    """The re-run's data under P2 on F30, checked against the re-run's record."""
    X, y, feat, classes, seq, fid, station = C.load()
    fit, val, test = C.protocols(y, seq, fid, station)["P2"]
    cols = C.input_sets(feat)["F30"]
    rerun = json.loads(C.RECORD.read_text())
    if [int(len(a)) for a in (fit, val, test)] != rerun["parts"]["P2"] or int(len(y)) != rerun["rows"]:
        raise SystemExit("P2 computed here is not the re-run's: run nothing")
    return X, y, feat, classes, station, fit, val, test, cols, rerun


def scaled(X: np.ndarray, rows: np.ndarray, cols: list[int], w: dict) -> np.ndarray:
    """The client's scaling in float64, as `scaleRow` computes it: (x - centre) / scale, clipped."""
    return np.clip((X[np.ix_(rows, cols)] - np.asarray(w["centre"])) / np.asarray(w["scale"]), -C.S.CLIP, C.S.CLIP)


def logits64(Z: np.ndarray, w: dict) -> np.ndarray:
    h = Z @ np.asarray(w["w1"]).T + np.asarray(w["b1"])
    return (h * h) @ np.asarray(w["w2"]).T + np.asarray(w["b2"])


def class_f1s(y_true: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Each class present in the part, by name."""
    return {classes[int(c)]: round(C.class_f1(y_true, pred, int(c)), 6) for c in np.unique(y_true)}


def flamby(y_true: np.ndarray, pred: np.ndarray, station: np.ndarray, k: int) -> float:
    """FLamby's metric as `scores` computes it (each station's macro-F1 over its classes present, then the mean)."""
    per = []
    for s in (0, 1):
        m = station == s
        if not m.any():
            continue
        ys, ps = y_true[m], pred[m]
        tc, pc = np.bincount(ys, minlength=k), np.bincount(ps, minlength=k)
        tp = np.bincount(ys[ys == ps], minlength=k)
        present = tc > 0
        per.append(float(np.mean(2 * tp[present] / (tc[present] + pc[present]))))
    return float(np.mean(per))


def bootstrap(y_true: np.ndarray, pred: np.ndarray, station: np.ndarray, k: int) -> dict:
    """Criterion 5: a 95 percent percentile interval of the test FLamby metric over test flows."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    m = len(y_true)
    vals = np.empty(BOOTSTRAP)
    for b in range(BOOTSTRAP):
        idx = rng.integers(0, m, m)
        vals[b] = flamby(y_true[idx], pred[idx], station[idx], k)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED, "low": round(float(lo), 6), "high": round(float(hi), 6),
            "method": "percentile; the test flows resampled with replacement, each station's macro-F1 over the classes "
                      "present in its resample, then their mean"}


def stage_train(work: Path, device: str) -> int:
    import torch

    check_preregistration()
    X, y, feat, classes, station, fit, val, test, cols, rerun = data()
    k = len(classes)
    control = C.student_arm(X, y, k, cols, fit, val, test, station)
    recorded = rerun["student"]["F30/P2"]
    if control["test"] != recorded["test"] or control["l2"] != recorded["l2"]:
        raise SystemExit("the control is not the re-run's linear student under P2 on F30: run nothing")
    print(f"control (the re-run's linear student): test FLamby {control['test']['flamby_metric']}", flush=True)
    arms, kept = {}, {}
    for seed in SEEDS:
        r = C.mlp_arm(X, y, k, cols, fit, val, test, station, seed, device, weights=True)
        w, pred = r.pop("_weights"), r.pop("_pred")
        # The bootstrap's statistic on the sample itself is the scores' figure (which rounds each station to 6 places).
        if abs(flamby(y[test], pred["test"], station[test], k) - r["test"]["flamby_metric"]) > 2e-6:
            raise SystemExit("the bootstrap's FLamby metric is not the scores' on the sample itself: run nothing")
        p64 = logits64(scaled(X, test, cols, w), w).argmax(1)
        st = {f"station_{s + 1}": station[test] == s for s in (0, 1)}
        r["test_class_f1"] = class_f1s(y[test], pred["test"], classes)
        r["validation_class_f1"] = class_f1s(y[val], pred["validation"], classes)
        r["float64"] = {"note": "the kept weights in float64 NumPy, as the page computes them; the figures above are the "
                                "trained network's own (float32 on the device)",
                        "test": C.scores(y[test], p64, k, st),
                        "test_decisions_differing_from_the_device": int(np.sum(p64 != pred["test"]))}
        arms[str(seed)] = r
        kept[seed] = (w, pred["test"])
        print(f"seed {seed}: epoch {r['epoch']}, validation FLamby {r['validation']['flamby_metric']}, test FLamby "
              f"{r['test']['flamby_metric']}, macro-F1 {r['test']['macro_f1']}", flush=True)
    export = max(SEEDS, key=lambda s: (arms[str(s)]["validation"]["flamby_metric"], -s))
    w, pred = kept[export]
    weights = {"kind": "dense-bn-square-dense", "n_in": len(cols), "width": int(np.asarray(w["w1"]).shape[0]),
               "centre": np.asarray(w["centre"]).tolist(), "scale": np.asarray(w["scale"]).tolist(),
               "clip": float(C.S.CLIP), "w1": np.asarray(w["w1"]).tolist(), "b1": np.asarray(w["b1"]).tolist(),
               "w2": np.asarray(w["w2"]).tolist(), "b2": np.asarray(w["b2"]).tolist(), "classes": list(classes),
               "ckks": {name: f7.PARAMS[name] for name in f7.PARAMS},
               "decision": {"rule": "the first class with the largest logit"},
               "inputs": [feat[c] for c in cols], "export_seed": int(export),
               "batch_norm": "none: the re-run's recipe has none, so nothing is folded"}
    work.mkdir(parents=True, exist_ok=True)
    (work / "weights.json").write_text(json.dumps(weights))
    sq = [arms[str(s)]["test"]["flamby_metric"] for s in SEEDS]
    ctrl = control["test"]["flamby_metric"]
    c1 = bool(float(np.mean(sq)) >= ctrl + MEAN_MARGIN and min(sq) >= ctrl + EACH_MARGIN)
    c2 = bool(all(arms[str(s)]["test"]["macro_f1"] >= MACRO_F1_FLOOR and min(arms[str(s)]["test_class_f1"].values()) > 0
                  for s in SEEDS))
    record = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
        "data": {"corpus": "5G-NIDD (CC BY 4.0)", "protocol": "P2", "inputs": "F30",
                 "input_names": [feat[c] for c in cols], "classes": list(classes), "rows": int(len(y)),
                 "parts": [int(len(fit)), int(len(val)), int(len(test))],
                 "from": "the re-run's loader, splits and inputs (scripts/train/telecom_nidd_clean_protocols.py)"},
        "device": {"requested": device, "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else None,
                   "cuda": torch.version.cuda, "python": platform.python_version()},
        "control": {"what": "the re-run's linear student under P2 on F30, refitted and found equal to its record",
                    **control},
        "arms": {"square": arms},
        "export_seed": int(export),
        "export_bootstrap": bootstrap(y[test], pred, station[test], k),
        "weights_sha256": hashlib.sha256((work / "weights.json").read_bytes()).hexdigest(),
        "criteria": {"1_accuracy": c1, "1_figures": {"control_test_flamby": ctrl, "square_mean": round(float(np.mean(sq)), 6),
                                                     "square_min": min(sq), "mean_margin": MEAN_MARGIN,
                                                     "each_margin": EACH_MARGIN},
                     "2_no_collapse": c2},
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"export_seed": export, **record["criteria"], "bootstrap": record["export_bootstrap"]}, indent=1))
    return 0


CKKS_HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3], DIR = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export * from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquareTabular"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const w = JSON.parse(readFileSync(DIR + "/weights.json", "utf8"));
const buf = readFileSync(DIR + "/z.f64");
const z = new Float64Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const rows = z.length / w.n_in;
const out = { node: process.version, sets: {} };
for (const set of Object.keys(w.ckks)) {
  const r = await m.runSquareTabular(w, z, rows, set);
  writeFileSync(`${DIR}/dec_${set}.f64`, Buffer.from(r.logits.buffer));
  writeFileSync(`${DIR}/plain_${set}.f64`, Buffer.from(r.plain.buffer));
  const { logits, plain, ...rest } = r;
  out.sets[set] = rest;
}
console.log(JSON.stringify(out));
"""


def flows(work: Path):
    """The 2,048 test flows the encrypted stages read, drawn as the shipped student's held-out rows were."""
    X, y, feat, classes, station, fit, val, test, cols, _ = data()
    w = json.loads((work / "weights.json").read_text())
    take = C.held_out_draw(y[test], len(classes))
    rows = test[take]
    return X, y, classes, cols, w, rows


def margin_rule(dec: np.ndarray, plain: np.ndarray) -> dict:
    """Criterion 3's rule on one run: the largest |decrypted - plaintext| logit difference, and the argmax against the
    plaintext one on every flow whose two largest plaintext logits differ by more than twice it."""
    err = float(np.max(np.abs(dec - plain)))
    top = np.sort(plain, axis=1)
    clear = (top[:, -1] - top[:, -2]) > 2 * err
    differ = dec.argmax(1) != plain.argmax(1)
    return {"max_abs_error": err, "decisions_differ": int(differ.sum()), "flows_within_twice_the_error": int((~clear).sum()),
            "decisions_differ_on_the_other_flows": int(differ[clear].sum()),
            "meets": bool(err <= FIDELITY and not differ[clear].any())}


def stage_ckks(work: Path) -> int:
    check_preregistration()
    record = json.loads(RECORD.read_text())
    X, y, classes, cols, w, rows = flows(work)
    if hashlib.sha256((work / "weights.json").read_bytes()).hexdigest() != record["weights_sha256"]:
        raise SystemExit("the work directory's weights are not the record's exported seed: run nothing")
    k = len(classes)
    z = scaled(X, rows, cols, w)
    ref = logits64(z, w)
    labels = y[rows]
    web = REPO / "web"
    sets = {}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "weights.json").write_text(json.dumps(w))
        z.astype(np.float64).tofile(d / "z.f64")
        (d / "harness.mjs").write_text(CKKS_HARNESS)
        out = subprocess.run(["node", str(d / "harness.mjs"), str(web), str(d / "bundle.mjs"), str(d)], cwd=web,
                             check=True, capture_output=True, text=True, timeout=3600).stdout
        got = json.loads(out.strip().splitlines()[-1])
        for name, run in got["sets"].items():
            dec = np.fromfile(d / f"dec_{name}.f64").reshape(-1, k)
            plain = np.fromfile(d / f"plain_{name}.f64").reshape(-1, k)
            rule = margin_rule(dec, plain)
            sets[name] = {**rule, "rows": int(len(rows)), "blocks": run["blocks"],
                          "module_max_abs_error": run["maxAbsError"], "module_decisions_differ": run["decisionsDiffer"],
                          "plaintext_matches_numpy_max_abs": float(np.max(np.abs(plain - ref))),
                          "encrypted_accuracy": round(float(np.mean(dec.argmax(1) == labels)), 6),
                          "plaintext_accuracy": round(float(np.mean(plain.argmax(1) == labels)), 6),
                          "key_ms": run["keyMs"], "key_bytes": run["keyBytes"], "per_block": run["per_block"]}
            print(f"  ckks {name}: max |err| {rule['max_abs_error']:.3e}, {rule['decisions_differ']} decisions differ "
                  f"({rule['flows_within_twice_the_error']} flows within twice the error), meets {rule['meets']}",
                  flush=True)
    record["ckks"] = {"node": got["node"], "rows": int(len(rows)), "class_counts": {classes[c]: int(np.sum(labels == c))
                                                                                     for c in range(k)},
                      "draw": "the shipped student's held-out draw from P2's test part (held_out_draw)",
                      "module": "web/src/lib/clientFHESquareTabular.ts (runSquareTabular), one flow a slot", "sets": sets}
    record["criteria"]["3_fidelity"] = {name: s["meets"] for name, s in sets.items()}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    # What the browser stage reads: the raw inputs, which the worker scales as the client would.
    raw = X[np.ix_(rows, cols)].astype(np.float64)
    (work / "rows.json").write_text(json.dumps({"samples": raw.tolist(), "feature_names": w["inputs"],
                                                "input_space": "raw", "labels": labels.astype(int).tolist()}))
    return 0


def stage_browser(work: Path) -> int:
    check_preregistration()
    record = json.loads(RECORD.read_text())
    if hashlib.sha256((work / "weights.json").read_bytes()).hexdigest() != record["weights_sha256"]:
        raise SystemExit("the work directory's weights are not the record's exported seed: run nothing")
    w = json.loads((work / "weights.json").read_text())
    shipped = json.loads((work / "rows.json").read_text())
    k = len(w["classes"])
    load = os.getloadavg()
    if load[0] > 1.5:
        raise SystemExit(f"the machine is not quiet (load average {load[0]:.2f}); criterion 4 is timed, so wait")
    out = work / "browser.json"
    r = subprocess.run(["node", str(REPO / "web" / "scripts" / "measure_square_tabular_browser.mjs"),
                        str(work / "weights.json"), str(work / "rows.json"), str(out), str(RUNS)],
                       cwd=REPO / "web", capture_output=True, text=True, timeout=3600)
    print(r.stdout, end="")
    if r.returncode:
        raise SystemExit(f"the browser harness failed: {r.stderr[-2000:]}")
    b = json.loads(out.read_text())
    x = np.asarray(shipped["samples"], dtype=np.float64)
    ref = logits64(np.clip((x - np.asarray(w["centre"])) / np.asarray(w["scale"]), -w["clip"], w["clip"]), w)
    sets = {}
    for name, runs in b["sets"].items():
        n = runs[0]["rows"]
        plain = np.asarray(runs[0]["plain"], dtype=np.float64).reshape(-1, k)
        top = np.sort(plain, axis=1)
        gap = top[:, -1] - top[:, -2]
        per = []
        for run in runs:
            decided = np.asarray(run["decided"])
            clear = gap > 2 * run["max_abs_error"]
            differ = decided != plain.argmax(1)
            per.append({**{key: v for key, v in run.items() if key not in ("plain", "ok", "decided")},
                        "decisions_differ_on_flows_clear_of_twice_the_error": int(differ[clear].sum()),
                        "meets": bool(run["total_ms"] <= BROWSER_MS and run["wasm_memory_bytes"] is not None
                                      and run["wasm_memory_bytes"] <= BROWSER_BYTES and run["max_abs_error"] <= FIDELITY
                                      and not differ[clear].any())})
        sets[name] = {"rows": n, "blocks": runs[0]["blocks"], "runs": per,
                      "js_plain_matches_numpy_max_abs": float(np.max(np.abs(plain - ref[:n]))),
                      "slowest_total_ms": max(run["total_ms"] for run in runs),
                      "largest_wasm_memory_bytes": max(run["wasm_memory_bytes"] for run in runs),
                      "criterion_4": bool(all(p["meets"] for p in per))}
        print(f"  browser {name}: slowest {sets[name]['slowest_total_ms'] / 1000:.2f} s, largest memory "
              f"{sets[name]['largest_wasm_memory_bytes'] / 1e9:.3f} GB, criterion 4 {sets[name]['criterion_4']}", flush=True)
    record["browser"] = {"browser": b["browser"], "user_agent": b["user_agent"], "runs_per_set": b["runs_per_set"],
                         "budget": {"total_ms": BROWSER_MS, "wasm_memory_bytes": BROWSER_BYTES},
                         "rows": "the ckks stage's 2,048 flows, raw, scaled in the worker", "sets": sets,
                         "machine": f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical cores, "
                                    "headless, no other job"}
    record["criteria"]["4_page"] = {name: s["criterion_4"] for name, s in sets.items()}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    return 0


def stage_verdict() -> int:
    """The pre-registration's rules, applied to the record as it stands."""
    record = json.loads(RECORD.read_text())
    c = record["criteria"]
    if "3_fidelity" not in c or "4_page" not in c:
        raise SystemExit("the ckks and browser stages have not both run")
    offered = [name for name in f7.PARAMS if c["3_fidelity"].get(name) and c["4_page"].get(name)]
    c["sets_offered"] = offered
    c["adopted"] = bool(c["1_accuracy"] and c["2_no_collapse"] and offered)
    c["outcome"] = (f"criteria met under {', '.join(offered)}: the network may be offered on the page; beside or in place "
                    "of the linear student is the owner's choice" if c["adopted"] else
                    "a negative finding: the page does not change")
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(c, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("train", "ckks", "browser", "verdict"))
    ap.add_argument("--work", type=Path, required=True, help="outside the repository: the weights and the flows")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    work = args.work.resolve()
    if REPO in work.parents or work == REPO:
        raise SystemExit("the work directory must be outside the repository")
    if args.stage == "train":
        return stage_train(work, args.device)
    if args.stage == "ckks":
        return stage_ckks(work)
    if args.stage == "browser":
        return stage_browser(work)
    return stage_verdict()


if __name__ == "__main__":
    raise SystemExit(main())
