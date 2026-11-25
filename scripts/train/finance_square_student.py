#!/usr/bin/env python
"""The pre-registered fully encrypted finance student with a square activation (F7).

`references/finance-square-student-preregistration.md`, committed before any network of this design was trained: one
hidden layer 128 wide (dense, batch norm, x^2, dense) against the same network with a ReLU, by the published MLP's recipe
(`finance_aml_mlp.fit_one`) on its windows and inputs, seeds 0 to 4; criteria on test AUPRC, collapse, and CKKS fidelity
on every test row. Each stage refuses to run unless the pre-registration's SHA-256 is the one recorded here.

    python scripts/train/finance_square_student.py train --work W     # the GPU, on the owner's choice of 2026-11-25
    python scripts/train/finance_square_student.py ckks --work W      # the page's module under node-seal, every test row

`train` writes the record `docs/measurements/finance-square-student-<date>.json` (criteria 1 and 2) and, in W, each seed's
folded float64 network and the scaled test inputs the `ckks` stage reads; `ckks` adds criterion 3 to the record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "train")]

PREREG = REPO / "references" / "finance-square-student-preregistration.md"
PREREG_SHA256 = "9ac9039309cbf6bfa53b76fcbe749f817a3d25839419670e99413eb14d49759e"
RECORD = REPO / "docs" / "measurements" / "finance-square-student-2026-11-25.json"
WIDTH = 128
SEEDS = (0, 1, 2, 3, 4)
MARGIN, FLOOR, MAX_LOGIT_ERR = 0.02, 0.15, 1e-3        # criteria 1, 2 and 3
SHARDS = 10                                             # node processes at once in the ckks stage
PARAMS = {
    "A2": {"N": 8192, "coeff_mod_bits": [48, 34, 34, 34, 52], "scale_bits": 34},
    "B": {"N": 16384, "coeff_mod_bits": [60, 40, 40, 40, 60], "scale_bits": 40},
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_preregistration() -> None:
    if sha256(PREREG) != PREREG_SHA256:
        raise SystemExit(f"{PREREG.relative_to(REPO)} is not the pre-registered text (SHA-256 {sha256(PREREG)[:12]})")


def environment() -> dict:
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()}


def network(arm: str):
    """dense 114 -> 128, batch norm, x^2 (square) or ReLU (control), dense 128 -> 2."""
    from torch import nn

    from finance_aml_mlp import N_FEATURES

    class Square(nn.Module):
        def forward(self, x):
            return x * x

    return nn.Sequential(nn.Linear(N_FEATURES, WIDTH), nn.BatchNorm1d(WIDTH), Square() if arm == "square" else nn.ReLU(),
                         nn.Linear(WIDTH, 2))


def fit(arm: str, zf, yf, zv, yv: np.ndarray, seed: int):
    """`finance_aml_mlp.fit_one`'s loop with this network: class-weighted cross-entropy on the labels, AdamW, early
    stopping on validation AUPRC. The same constants, imported, so the recipes cannot drift apart."""
    import torch
    from sklearn.metrics import average_precision_score
    from torch import nn

    from finance_aml_mlp import BATCH, DEVICE, EPOCHS, LR, PATIENCE, WEIGHT_DECAY, _positive_scores

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = network(arm).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n, pos = len(zf), int(yf.sum())
    w = torch.tensor([n / (2 * (n - pos)), n / (2 * pos)], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    best, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(zf[b]), yf[b])
            loss.backward()
            opt.step()
        model.eval()
        score = float(average_precision_score(yv, _positive_scores(model, zv)))
        if score > best + 1e-5:
            best, best_state, best_epoch, stale = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, best_epoch


def folded(model) -> dict:
    """The square network's inference weights in float64, batch norm folded into the first layer (exact in eval mode)."""
    import torch

    with torch.no_grad():
        lin1, bn, lin2 = model[0], model[1], model[3]
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return {"w1": (lin1.weight * s[:, None]).double().cpu().numpy(),
                "b1": ((lin1.bias - bn.running_mean) * s + bn.bias).double().cpu().numpy(),
                "w2": lin2.weight.double().cpu().numpy(), "b2": lin2.bias.double().cpu().numpy()}


def plain_logits(z: np.ndarray, p: dict) -> np.ndarray:
    """The folded square network in float64 on scaled, clipped rows: (W1 z + b1)^2 through W2, b2."""
    h = z.astype(np.float64) @ p["w1"].T + p["b1"]
    return (h * h) @ p["w2"].T + p["b2"]


def stage_train(work: Path) -> int:
    import torch
    from sklearn.metrics import average_precision_score

    from finance_aml import _best_threshold, causal_account_features, load_sorted, model_inputs, temporal_windows
    from finance_aml_mlp import CLIP, DEVICE, N_FEATURES, _positive_scores, _scaled, scaling_of
    from qsh.models import distil, metrics

    started = time.time()
    df = load_sorted()
    fit_w, val_w, te_w, tr_w = temporal_windows(df)
    y_fit, y_val, y_te = (d["label"].to_numpy(dtype=np.int8) for d in (fit_w, val_w, te_w))
    (x_fit, x_val, x_te), features, _ = model_inputs(df, fit_w, val_w, te_w, tr_w, causal_account_features(df))
    if x_fit.shape[1] != N_FEATURES:
        raise SystemExit(f"expected {N_FEATURES} features, got {x_fit.shape[1]}")
    print(f"  windows: fit {len(y_fit):,} ({int(y_fit.sum())} pos), val {len(y_val):,} ({int(y_val.sum())}), "
          f"test {len(y_te):,} ({int(y_te.sum())})", flush=True)
    centre, scale = scaling_of("standard", x_fit)
    zf, zv, zt = _scaled(x_fit, centre, scale), _scaled(x_val, centre, scale), _scaled(x_te, centre, scale)
    labels = torch.tensor(y_fit, dtype=torch.long, device=DEVICE)
    work.mkdir(parents=True, exist_ok=True)
    record = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
              "data": {"corpus": "IBM AMLworld HI-Small (CDLA-Sharing-1.0, synthetic)", "inputs": N_FEATURES,
                       "fit": int(len(y_fit)), "val": int(len(y_val)), "test": int(len(y_te)),
                       "test_positive": int(y_te.sum()), "scaling": "standard, from the fit window", "clip": CLIP},
              "arms": {"control": {}, "square": {}}, "environment": environment()}
    for arm in ("control", "square"):
        for seed in SEEDS:
            t0 = time.time()
            model, val_auprc, epoch = fit(arm, zf, labels, zv, y_val, seed)
            s_val, s_te = _positive_scores(model, zv), _positive_scores(model, zt)
            thr = _best_threshold(y_val, s_val)
            pred = s_te >= thr
            row = {"val_auprc": round(val_auprc, 6), "test_auprc": round(float(average_precision_score(y_te, s_te)), 6),
                   "test_minority_f1": round(float(metrics.minority_class_f1(y_te, pred.astype(int))), 6),
                   "predicts_both_classes": bool(0 < pred.sum() < len(pred)), "threshold": thr,
                   "best_epoch": int(epoch), "seconds": round(time.time() - t0, 1)}
            if arm == "square":
                distil.fold_threshold_into_torch_head(model, thr)
                p = folded(model)
                np.savez(work / f"square_seed{seed}_folded.npz", **p)
                with torch.no_grad():
                    torch_logits = torch.cat([model(zt[i:i + 262_144]) for i in range(0, len(zt), 262_144)]).double().cpu().numpy()
                mine = plain_logits(zt.cpu().numpy(), p)
                row["folded_matches_torch_max_abs"] = float(np.max(np.abs(mine - torch_logits)))
                row["decisions_at_argmax_equal_threshold"] = bool(np.array_equal(mine.argmax(1) == 1, pred))
            record["arms"][arm][str(seed)] = row
            print(f"  {arm} seed {seed}: {row}", flush=True)
    sq, ct = record["arms"]["square"], record["arms"]["control"]
    means = {a: float(np.mean([r["test_auprc"] for r in record["arms"][a].values()])) for a in ("control", "square")}
    sds = {a: float(np.std([r["test_auprc"] for r in record["arms"][a].values()], ddof=1)) for a in ("control", "square")}
    export = max(sq, key=lambda s: sq[s]["val_auprc"])
    record["criteria"] = {
        "mean_test_auprc": {k: round(v, 6) for k, v in means.items()},
        "sd_test_auprc": {k: round(v, 6) for k, v in sds.items()},
        "1_accuracy": bool(means["square"] >= means["control"] - MARGIN),
        "2_no_collapse": bool(all(r["test_auprc"] >= FLOOR and r["predicts_both_classes"] for r in sq.values())),
        "export_seed": int(export)}
    # The exported seed's test AUPRC with a 95 % percentile interval over 2,000 resamples of test rows (seed 0).
    p = dict(np.load(work / f"square_seed{export}_folded.npz"))
    z_np = zt.cpu().numpy()
    lg = plain_logits(z_np, p)
    s_exp = 1 / (1 + np.exp(-(lg[:, 1] - lg[:, 0])))
    rng = np.random.default_rng(0)
    boots = [float(average_precision_score(y_te[idx], s_exp[idx]))
             for idx in (rng.integers(0, len(y_te), len(y_te)) for _ in range(2000))]
    record["export_bootstrap"] = {"auprc": round(float(average_precision_score(y_te, s_exp)), 6),
                                  "ci95": [round(float(np.percentile(boots, 2.5)), 6), round(float(np.percentile(boots, 97.5)), 6)],
                                  "resamples": len(boots)}
    # References, not criteria: the published MLP's recorded seeds, the linear student and the teacher, same window.
    rec = json.loads((REPO / "models" / "metrics.json").read_text())
    lin = json.loads((REPO / "models" / "finance_aml_student" / "weights.json").read_text())
    zl = (x_te - np.asarray(lin["input"]["mean"])) / np.asarray(lin["input"]["scale"])
    s_lin = 1 / (1 + np.exp(-(zl @ np.asarray(lin["W"][0]) + lin["b"][0])))
    record["references"] = {
        "published_mlp": {"test_auprc_by_seed": [r["test_auprc"] for r in rec["finance_aml.mlp_large"]["seeds"]
                                                 if r["scaling"] == "standard"],
                          "note": "114 -> 128 -> 64 -> 2 with ReLU; its ReLUs run in garbled circuits on the page"},
        "published_linear_student": {"test_auprc": round(float(average_precision_score(y_te, s_lin)), 6),
                                     "test_minority_f1": rec["finance_aml.student"]["student_metric"]},
        "published_teacher": {"test_auprc": rec["finance_aml.gbm"]["auprc"],
                              "test_minority_f1": rec["finance_aml.gbm"]["minority_class_f1"]}}
    # What the ckks stage reads: the scaled, clipped test inputs (float32, as the page encrypts them) and the scaling.
    np.save(work / "z_test.npy", z_np.astype(np.float32))
    np.savez(work / "scaling.npz", centre=centre, scale=scale)
    (work / "features.json").write_text(json.dumps(features))
    record["wall_clock_s"] = round(time.time() - started, 1)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record["criteria"], indent=1))
    return 0


HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
const [WEB, OUT, DIR, SET, ROWS, OFFSET, COUNT, TAG] = process.argv.slice(2);
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
const buf = readFileSync(ROWS);
const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const off = Number(OFFSET), rows = Number(COUNT);
const z = all.subarray(off * w.n_in, (off + rows) * w.n_in);
const r = await m.runSquareTabular(w, z, rows, SET, (done, total) => {
  if (done % 25 === 0 || done === total) console.error(`${TAG}: block ${done} of ${total}`);
});
writeFileSync(DIR + `/plain_${TAG}.f64`, Buffer.from(r.plain.buffer));
writeFileSync(DIR + `/dec_${TAG}.f64`, Buffer.from(r.logits.buffer));
const { logits, plain, ...rest } = r;
console.log(JSON.stringify(rest));
"""


def weights_json(p: dict, centre: np.ndarray, scale: np.ndarray, clip: float, sets: list[str], seed: int,
                 threshold: float) -> dict:
    return {"kind": "dense-bn-square-dense", "n_in": int(p["w1"].shape[1]), "width": int(p["w1"].shape[0]),
            "centre": centre.astype(float).tolist(), "scale": scale.astype(float).tolist(), "clip": float(clip),
            "w1": p["w1"].tolist(), "b1": p["b1"].tolist(), "w2": p["w2"].tolist(), "b2": p["b2"].tolist(),
            "classes": ["licit", "laundering"],
            "ckks": {k: PARAMS[k] for k in sets},
            "decision": {"threshold": threshold,
                         "rule": "laundering when logit1 > logit0; the validation threshold is folded into b2"},
            "export_seed": seed}


def stage_ckks(work: Path) -> int:
    """Every test row under each pre-registered set, through the page's module, one row a slot."""
    from finance_aml_mlp import CLIP

    record = json.loads(RECORD.read_text())
    seed = record["criteria"]["export_seed"]
    arm = record["arms"]["square"][str(seed)]
    p = dict(np.load(work / f"square_seed{seed}_folded.npz"))
    sc = dict(np.load(work / "scaling.npz"))
    z = np.load(work / "z_test.npy", mmap_mode="r")
    mine = plain_logits(np.asarray(z), p)
    record["ckks"] = {}
    for name in PARAMS:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "weights.json").write_text(json.dumps(weights_json(p, sc["centre"], sc["scale"], CLIP, list(PARAMS),
                                                                    int(seed), arm["threshold"])))
            (d / "harness.mjs").write_text(HARNESS)
            load_before = os.getloadavg()
            t0 = time.time()
            # node-seal's wasm is single-threaded: the rows are cut into SHARDS contiguous runs of whole blocks, each in a
            # node process of its own with its own keys, all at once. Every row is still decrypted and compared.
            block = PARAMS[name]["N"] // 2
            n = len(z)
            per = -(-(-(-n // block)) // SHARDS) * block
            cuts = [(o, min(per, n - o)) for o in range(0, n, per)]
            procs = [subprocess.Popen(["node", "--max-old-space-size=8192", str(d / "harness.mjs"), str(REPO / "web"),
                                       str(d / f"bundle_{k}.mjs"), str(d), name, str(work / "z_test.npy.f32"), str(o),
                                       str(c), f"s{k}"], cwd=REPO / "web", stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
                     for k, (o, c) in enumerate(cuts)]
            runs = []
            for k, pr in enumerate(procs):
                so, se = pr.communicate(timeout=4 * 3600)
                if pr.returncode:
                    raise SystemExit(f"shard {k} of set {name} failed: {se[-2000:]}")
                runs.append(json.loads(so.strip().splitlines()[-1]))
            plain_js = np.concatenate([np.fromfile(d / f"plain_s{k}.f64", dtype=np.float64) for k in range(len(cuts))]).reshape(-1, 2)
            dec = np.concatenate([np.fromfile(d / f"dec_s{k}.f64", dtype=np.float64) for k in range(len(cuts))]).reshape(-1, 2)
            pb = [r["per_block"] for r in runs]
            run = {"blocks": int(sum(r["blocks"] for r in runs)), "shards": len(cuts),
                   "per_block": {**pb[0], **{k: float(np.median([b[k] for b in pb]))
                                             for k in ("encrypt_ms", "evaluate_ms", "decrypt_ms")}},
                   "key_ms_median": float(np.median([r["keyMs"] for r in runs]))}
        err = np.abs(dec - plain_js)
        differ = int((dec.argmax(1) != plain_js.argmax(1)).sum())
        res = {**PARAMS[name], "modulus_bits": sum(PARAMS[name]["coeff_mod_bits"]), "rows": int(len(dec)),
               "blocks": run["blocks"], "max_abs_logit_error": float(err.max()),
               "median_abs_logit_error": float(np.median(err)), "decisions_differ": differ,
               "js_plain_matches_numpy_max_abs": float(np.max(np.abs(plain_js - mine))),
               "largest_abs_plain_logit": float(np.max(np.abs(mine))),
               "per_block": run["per_block"], "shards": run["shards"], "key_ms_median": run["key_ms_median"],
               "per_block_note": f"medians over {run['shards']} node processes running at once, one a shard of whole "
                                 "blocks, on the 20-thread i5-13600K, so each figure is a block's time under that load",
               "seconds": round(time.time() - t0, 1),
               "load_average_before": [round(x, 2) for x in load_before],
               "load_average_after": [round(x, 2) for x in os.getloadavg()]}
        res["criterion_3"] = bool(res["max_abs_logit_error"] <= MAX_LOGIT_ERR and differ == 0)
        record["ckks"][name] = res
        print(f"  CKKS {name}: max |err| {res['max_abs_logit_error']:.3e}, decisions differ {differ} of {len(dec):,}, "
              f"{res['seconds']:.0f} s", flush=True)
    record["criteria"]["3_fidelity"] = {k: v["criterion_3"] for k, v in record["ckks"].items()}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("train", "ckks"))
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    check_preregistration()
    if args.stage == "train":
        return stage_train(args.work)
    # The harness reads raw float32 rows; numpy's .npy header is stripped once.
    z = np.load(args.work / "z_test.npy", mmap_mode="r")
    raw = args.work / "z_test.npy.f32"
    if not raw.exists() or raw.stat().st_size != z.size * 4:
        np.ascontiguousarray(z, dtype=np.float32).tofile(raw)
    return stage_ckks(args.work)


if __name__ == "__main__":
    raise SystemExit(main())
