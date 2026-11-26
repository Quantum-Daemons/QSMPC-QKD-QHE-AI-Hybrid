#!/usr/bin/env python3
"""The 5G-NIDD federated model's Hub artefacts and registry record (Round 86, batch 6).

On the owner's choice of 2026-11-26 to show the federation on the page and publish its model. The model is the one
scripts/train/telecom_nidd_federated.py trained as pre-registered (references/telecom-nidd-federated-preregistration.md)
and wrote into web/public/data/telecom_nidd/federated.json; nothing here refits it. What this adds is what a Hub repo
needs and the site file does not carry: the model's own standardisation (the two stations' combined statistics,
recomputed exactly as the run computed them and checked against the run's record), a graph built from the weights
(`qsh.models.linear_onnx`, with the clip the inputs carry), and an evaluation sample of the 2,048 held-out flows' raw
values, which the site ships only standardised.

Writes models/telecom_nidd_federated/weights.json, onnx/model.onnx and eval_sample.json; with --metrics, also the
registry's record in models/metrics.json, with the page's CKKS figures measured by its own code in node. --check holds
the folder to the site file and the record without the corpus.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/export_nidd_federated.py [--metrics]
       python -I scripts/export_nidd_federated.py --check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models.registry import DS_5G_NIDD  # noqa: E402

KEY = "telecom_ids.nidd_federated"
OUT = REPO / "models" / "telecom_nidd_federated"
SITE = REPO / "web" / "public" / "data" / "telecom_nidd" / "federated.json"
HELD = REPO / "web" / "public" / "data" / "telecom_nidd" / "student_heldout.json"
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-federated-2026-11-26.json"
CLIP = 20.0
HELD_OUT_ROWS = 2048
KEYS = ("station_1", "station_2")
SPLIT = ("the CKKS student's 2,048 held-out flows, a class-stratified draw from the seed-42 test split, each named by "
         "its base station")


def corpus_parts() -> dict:
    """The run's split, stations and the federated standardisation, recomputed as the trainer computes them."""
    from sklearn.model_selection import train_test_split

    from qsh.models import datasets as D

    x, y, names, classes = D.load_5g_nidd()
    cut = D.NIDD_SECOND_FILE_STARTS_AT_CLEANED
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42, stratify=y)
    xtr, xte, yte = x[itr], x[ite], y[ite]
    numeric = [i for i, n in enumerate(D.NIDD_FEATURES) if n not in D.NIDD_CATEGORICAL]
    st_tr = (itr >= cut).astype(int)
    parts = []
    for s in (0, 1):
        a = xtr[st_tr == s][:, numeric]
        parts.append((int(np.sum(st_tr == s)), a.mean(axis=0).astype(np.float64), a.std(axis=0).astype(np.float64)))
    n_all = sum(n for n, _, _ in parts)
    c_f = sum(n * c for n, c, _ in parts) / n_all
    s_f = np.sqrt(sum(n * (s_ * s_ + (c - c_f) ** 2) for n, c, s_ in parts) / n_all)
    s_f[s_f == 0] = 1.0
    centre = xtr[:, numeric].mean(axis=0).astype(np.float64)
    scale = xtr[:, numeric].std(axis=0).astype(np.float64)
    scale[scale == 0] = 1.0
    # The student's held-out draw, as its trainer draws it.
    rng = np.random.default_rng(0)
    take: list[int] = []
    for c in range(len(classes)):
        idx = np.flatnonzero(yte == c)
        take.extend(rng.choice(idx, size=max(1, round(HELD_OUT_ROWS * len(idx) / len(yte))), replace=False).tolist())
    take_a = np.sort(np.asarray(take[:HELD_OUT_ROWS]))
    raw = xte[take_a][:, numeric].astype(np.float64)
    return {"features": [D.NIDD_FEATURES[i] for i in numeric], "classes": list(classes), "c_f": c_f, "s_f": s_f,
            "centre": centre, "scale": scale, "raw": raw, "labels": yte[take_a].astype(int),
            "station": (ite[take_a] >= cut).astype(int)}


def weights_file(site: dict, p: dict) -> dict:
    f = site["federated"]
    return {
        "kind": "logistic-linear-multiclass", "student_key": KEY,
        "note": ("The 5G-NIDD detector the corpus's two base stations trained together by FedAvg under pairwise masks, "
                 "as pre-registered: 9 dot products over 36 flow features, standardised by the two stations' combined "
                 "statistics and clipped to [-20, 20]. Under CKKS all 9 scores are decrypted and the argmax is taken in the "
                 "clear. `temperature` divides the logits for calibrated probabilities and does not change the class."),
        "n_features": len(p["features"]), "features": p["features"],
        "input": {"preprocessing": ["subtract mean", "divide by scale", "clip"], "mean": p["c_f"].tolist(),
                  "scale": p["s_f"].tolist(), "clip": CLIP},
        "W": f["W"], "b": f["b"], "temperature": f["temperature"], "n_parameters": 9 * len(p["features"]) + 9,
        "classes": p["classes"],
        "trained": {"algorithm": "FedAvg under pairwise masks agreed by X25519", "rounds": f["rounds"],
                    "local_iterations": f["local_iterations"], "station_weights": f["weights"], "l2": f["l2"],
                    "preregistration": site["preregistration"], "record": site["record"]},
        # Fitted on 5G-NIDD (CC BY 4.0), so the weights carry the corpus's licence and credit, as every set of weights fitted
        # on a CC BY corpus here does and as the card always said (the owner's choice of 2026-11-26; this said Apache-2.0).
        "licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": DS_5G_NIDD.attribution,
        "modifications": ("Fitted by FedAvg across the corpus's two base stations on its training flows, after the flows "
                          "published twice under contradictory labels lose their Benign copy; the 36 numeric columns are "
                          "standardised by the two stations' combined statistics and clipped to [-20, 20]. No flow ships in "
                          "this file."),
    }


def graph_logits(weights: dict, raw: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    from qsh.models.linear_onnx import linear_student_graph

    sess = ort.InferenceSession(linear_student_graph(weights).SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": raw.astype(np.float32)})[0]


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    per = []
    for c in range(k):
        tp, fp, fn = (np.sum((pred == c) & (y == c)), np.sum((pred == c) & (y != c)), np.sum((pred != c) & (y == c)))
        per.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(per))


def sample_file(weights: dict, p: dict) -> dict:
    raw32 = [[float(str(np.float32(v))) for v in row] for row in p["raw"]]
    logits = graph_logits(weights, np.asarray(raw32))
    pred, y = logits.argmax(1), p["labels"]
    held = np.asarray(json.loads(HELD.read_text())["rows"], dtype=np.float64)
    page = (held @ np.asarray(weights["W"]).T + np.asarray(weights["b"])).argmax(1)
    return {
        "_generated_by": "scripts/export_nidd_federated.py; do not edit by hand",
        "task": "classification",
        "provenance": ("5G-NIDD, real traffic from a 5G test network (University of Oulu): " + SPLIT + ", the rows "
                       "web/public/data/telecom_nidd/student_heldout.json ships standardised, here as raw values."),
        "split": SPLIT,
        "licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": DS_5G_NIDD.attribution,
        "modifications": ("The flows published twice under contradictory labels lose their Benign copy; the 36 numeric "
                          "columns of 2,048 test flows, a class-stratified draw, each value the shortest decimal of its "
                          "float32; the graph standardises and clips them."),
        "input": {"shape": [1, len(p["features"])], "dtype": "float32", "preprocessing": [],
                  "layout": "samples[i] is one flow's 36 raw numeric features, in the order below; the graph standardises and clips"},
        "feature_names": p["features"], "samples": raw32, "labels": y.tolist(), "station": p["station"].tolist(),
        "class_names": p["classes"],
        "expected": {"graph": "model.onnx", "accuracy": float(np.mean(pred == y)),
                     "macro_f1": macro_f1(y, pred, len(p["classes"])),
                     "majority_baseline": float(np.bincount(y).max() / len(y)),
                     "same_class_as_the_page": int(np.sum(pred == page)),
                     "note": ("Measured here with onnxruntime on exactly these rows through the graph built from "
                              "weights.json. `same_class_as_the_page` counts the rows whose class equals the one the page "
                              "computes from its own copy, the rows standardised by the student's statistics, which the "
                              "stations' combined statistics reproduce to a few parts in ten million.")},
    }


CKKS_HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { niddFederated } from ${JSON.stringify(resolve(WEB, "src/lib/realData"))};
export { encryptedHeldOutAgreement } from ${JSON.stringify(resolve(WEB, "src/lib/clientFHE"))};`, resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
globalThis.fetch = async (url) => {
  const p = join(WEB, "public", String(url).replace(/^\/+/, ""));
  try { const t = readFileSync(p, "utf8"); return { ok: true, status: 200, json: async () => JSON.parse(t) }; }
  catch { return { ok: false, status: 404, json: async () => { throw new Error("404"); } }; }
};
const m = await import(`file://${OUT}`);
const fed = await m.niddFederated();
const r = await m.encryptedHeldOutAgreement(fed.held);
let flips = 0;
r.plot.points.forEach((q) => { if (!q.agree && Math.abs(q.plain) > 2 * r.batch.maxAbsError) flips++; });
console.log(JSON.stringify({ rows: r.plot.rows, ciphertexts: r.plot.ciphertexts, max_abs_logit_error: r.batch.maxAbsError,
  decisions_differ: r.mismatches, flips_beyond_twice_the_error: flips, encrypted_accuracy: r.encryptedAccuracy }));
"""


def ckks_figures() -> dict:
    web = REPO / "web"
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "harness.mjs"
        script.write_text(CKKS_HARNESS)
        out = subprocess.run(["node", str(script), str(web), str(Path(tmp) / "bundle.mjs")], cwd=web, check=True,
                             capture_output=True, text=True, timeout=900).stdout
    return json.loads(out.strip().splitlines()[-1])


def registry_record(site: dict, record: dict, sample: dict) -> dict:
    fed, pooled = record["federated"], record["pooled"]
    return {
        "flamby_macro_f1": fed["flamby_metric"], "pooled_flamby_macro_f1": pooled["flamby_metric"],
        "station_1_alone_flamby_macro_f1": record["alone"][0]["flamby_metric"],
        "station_2_alone_flamby_macro_f1": record["alone"][1]["flamby_metric"],
        "station_1_macro_f1": fed["test"]["station_1"]["macro_f1"], "station_2_macro_f1": fed["test"]["station_2"]["macro_f1"],
        "union_macro_f1": fed["union_test"]["macro_f1"], "union_accuracy": fed["union_test"]["accuracy"],
        "ece_station_1": fed["test"]["station_1"]["ece"], "ece_station_2": fed["test"]["station_2"]["ece"],
        "n_eval": int(fed["union_test"]["rows"]), "scoring_unit": "flows", "student_params": 333,
        "metric_name": "flamby_macro_f1",
        "metric_note": ("`flamby_macro_f1` is the mean of the two base stations' test macro F1, each station weighted "
                        "equally; the `*_alone_*` and `pooled_*` figures are the pre-registered comparison on the same rows. "
                        "`ckks` is the page's own module in node on the 2,048 held-out flows."),
        "rounds": fed["rounds"], "local_iterations": fed["local_iterations"], "masked_messages": fed["masked_messages"],
        "l2": fed["l2"], "temperature": fed["temperature"], "criteria": record["criteria"],
        "eval_sample": {k: sample["expected"][k] for k in ("accuracy", "macro_f1", "same_class_as_the_page")},
        "ckks": ckks_figures(), "preregistration": record["preregistration"],
    }


def check() -> int:
    site = json.loads(SITE.read_text())
    w = json.loads((OUT / "weights.json").read_text())
    f = site["federated"]
    assert w["W"] == f["W"] and w["b"] == f["b"] and w["temperature"] == f["temperature"], "weights.json is not the site's model"
    sample = json.loads((OUT / "eval_sample.json").read_text())
    held = json.loads(HELD.read_text())
    assert sample["labels"] == held["labels"] and sample["station"] == site["heldout_station"], "the sample is not the page's rows"
    logits = graph_logits(w, np.asarray(sample["samples"]))
    pred, y = logits.argmax(1), np.asarray(sample["labels"])
    assert float(np.mean(pred == y)) == sample["expected"]["accuracy"], "the graph does not reproduce the sample's accuracy"
    print(f"{OUT.relative_to(REPO)} holds: the site's model, the page's rows, the graph's accuracy "
          f"{sample['expected']['accuracy']:.6f} on them")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", action="store_true", help="also write the registry's record (models/metrics.json)")
    ap.add_argument("--check", action="store_true", help="hold the folder to the site file without the corpus")
    args = ap.parse_args()
    if args.check:
        return check()
    from qsh.models.linear_onnx import write_linear_student_graph

    site, record = json.loads(SITE.read_text()), json.loads(RECORD.read_text())
    p = corpus_parts()
    # The run's standardisation, exactly: the same figures the record checked, and the same shipped-row difference.
    s = record["standardisation"]
    got = float(np.max(np.abs(p["c_f"] - p["centre"]) / p["scale"]))
    assert got == s["max_centre_difference_in_deviations"], (got, s["max_centre_difference_in_deviations"])
    dz = float(np.max(np.abs(np.clip((p["raw"] - p["c_f"]) / p["s_f"], -CLIP, CLIP)
                             - np.clip((p["raw"] - p["centre"]) / p["scale"], -CLIP, CLIP))))
    assert dz == s["max_abs_difference_on_the_shipped_rows"], (dz, s["max_abs_difference_on_the_shipped_rows"])
    assert p["station"].tolist() == site["heldout_station"], "the held-out draw is not the run's"
    w = weights_file(site, p)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights.json").write_text(json.dumps(w, indent=1) + "\n")
    write_linear_student_graph(w, OUT / "onnx" / "model.onnx")
    sample = sample_file(w, p)
    (OUT / "eval_sample.json").write_text(json.dumps(sample, separators=(",", ":")) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}: weights.json, onnx/model.onnx, eval_sample.json (accuracy "
          f"{sample['expected']['accuracy']:.6f}, macro F1 {sample['expected']['macro_f1']:.6f}, "
          f"{sample['expected']['same_class_as_the_page']} of {len(sample['labels'])} rows the page's class)")
    if args.metrics:
        from qsh.models import train

        payload = registry_record(site, record, sample)
        prev = json.loads((REPO / "models" / "metrics.json").read_text()).get(KEY) or {}
        if "hf_revision" in prev:                          # a pin written by a publish is kept
            payload["hf_revision"] = prev["hf_revision"]
        train.record_metrics(KEY, payload)
        print(f"models/metrics.json: {KEY}, FLamby macro F1 {payload['flamby_macro_f1']}, CKKS {payload['ckks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
