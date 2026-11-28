#!/usr/bin/env python3
"""The four published BreastMNIST models on BUS-BRA, an external test, as pre-registered (Round 86, batch 12):
references/medical-busbra-external-test-preregistration.md.

Reads the deposit's archive in place (no file is extracted), checks the pre-registration's SHA-256 and every graph's
against its pin first, scores each model on every BUS-BRA image and on its own published BreastMNIST sample, runs both
students' CKKS paths in node on every BUS-BRA image, and writes docs/measurements/medical-busbra-external-<date>.json,
aggregate figures only. The graphs are gitignored: `QSH_PUBLISHED_GRAPHS` names a checkout that holds them.

    QSH_DATA_ROOT=... QSH_PUBLISHED_GRAPHS=<checkout> python scripts/check_busbra_external.py --work DIR
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets as D  # noqa: E402

PREREG = REPO / "references" / "medical-busbra-external-test-preregistration.md"
PREREG_SHA256 = "33b3e5abc0c15ef590a0d3e6536abbc750393ae7515495b7459e82b1a06b38c2"
WEB = REPO / "web"
#: name: (registry key, models directory, graph, input side). ResNet-18 from both its graphs: the int8 build the page
#: prefers and the fp32 graph its recorded figure comes from.
MODELS = {
    "medical_fl.resnet18 (int8)": ("medical_fl.resnet18", "medical_breastmnist_resnet18", "model_int8.onnx", 28),
    "medical_fl.resnet18 (fp32)": ("medical_fl.resnet18", "medical_breastmnist_resnet18", "model.onnx", 28),
    "medical_fl.resnet50": ("medical_fl.resnet50", "medical_breastmnist_resnet50", "model.onnx", 28),
    "medical_fl.student": ("medical_fl.student", "medical_breastmnist_student", "model.onnx", 14),
    "medical_fl.square_student": ("medical_fl.square_student", "medical_breastmnist_square_student", "model.onnx", 14),
}
KEYS = {v[0] for v in MODELS.values()}
BOOT, MIN_DEVICE, CHUNK, CKKS_LIMIT = 2000, 100, 250, 1e-3
MALIGNANT = 0                    # BreastMNIST's class 0; benign ("normal, benign") is class 1

HARNESS = r"""
// The page's two CKKS modules for the BreastMNIST students, on a list of 14x14 images, one JSON line per image.
import { createRequire } from "node:module";
import { readFileSync, writeFileSync, appendFileSync } from "node:fs";
import { resolve } from "node:path";
const [WEB, OUT, INPUT, RESULT] = process.argv.slice(2);
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { encryptedStudentInference } from ${JSON.stringify(resolve(WEB, "src/lib/clientFHE"))};
export { runSquareStudent } from ${JSON.stringify(resolve(WEB, "src/lib/clientFHESquare"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}' },
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" }, () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const job = JSON.parse(readFileSync(INPUT, "utf8"));
writeFileSync(RESULT, "");
for (let i = 0; i < job.images.length; i++) {
  const x = job.images[i];
  const img = Array.from({ length: 14 }, (_, y) => x.slice(y * 14, (y + 1) * 14));
  const s = await m.encryptedStudentInference(job.student, img);
  const row = { i: job.offset + i, student: { logits: s.logits, pred: s.prediction } };
  for (const set of job.sets) {
    const q = await m.runSquareStudent(x, job.square, set);
    row[set] = { logits: q.logits, pred: q.predicted };
  }
  appendFileSync(RESULT, JSON.stringify(row) + "\n");
}
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins() -> dict[str, dict[str, str]]:
    """Each model's pinned file digests, from the shipped config the page reads."""
    out: dict[str, dict[str, str]] = {}

    def walk(o) -> None:
        if isinstance(o, dict):
            if o.get("key") in KEYS and "file_sha256" in o:
                out[o["key"]] = o["file_sha256"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads((WEB / "src" / "lib" / "config.defaults.json").read_text()))
    return out


def graph_path(directory: str, graph: str) -> Path:
    here = REPO / "models" / directory / "onnx" / graph
    if here.exists():
        return here
    other = os.environ.get("QSH_PUBLISHED_GRAPHS")
    if other and (Path(other) / "models" / directory / "onnx" / graph).exists():
        return Path(other) / "models" / directory / "onnx" / graph
    raise SystemExit(f"models/{directory}/onnx/{graph} is not here and QSH_PUBLISHED_GRAPHS holds none: run nothing")


def load_busbra(path: Path) -> tuple[np.ndarray, list[dict]]:
    """Every image at 28x28 in 8 bits, as pre-registered, and its metadata row, in `bus_data.csv` order."""
    from PIL import Image
    from skimage.transform import resize

    with zipfile.ZipFile(path) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open("BUSBRA/bus_data.csv"), encoding="utf-8")))
        out = np.zeros((len(rows), 28, 28), dtype=np.uint8)
        for k, r in enumerate(rows):
            img = np.asarray(Image.open(io.BytesIO(z.read(f"BUSBRA/Images/{r['ID']}.png"))).convert("L"), dtype=np.float64)
            small = resize(img, (28, 28), order=3, anti_aliasing=True, preserve_range=True)
            out[k] = np.clip(np.rint(small), 0, 255).astype(np.uint8)
    return out, rows


def normalise(u8: np.ndarray) -> np.ndarray:
    return ((u8.astype(np.float32) / 255.0 - 0.5) / 0.5)[:, None]


def to14(x28: np.ndarray) -> np.ndarray:
    from qsh.ai.cnn_layers import downsample

    return np.asarray(downsample(x28, 14), dtype=np.float32)


def auc(score: np.ndarray, pos: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(pos, score)) if 0 < pos.sum() < len(pos) else None


def ap(score: np.ndarray, pos: np.ndarray) -> float | None:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(pos, score)) if 0 < pos.sum() < len(pos) else None


def decide(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits, axis=1)          # a tie is class 0, malignant, as the page's rule takes it


def figures(logits: np.ndarray, label: np.ndarray, patient: list[str] | None) -> dict:
    """AUC and AUPRC of the malignant class with patient-resampled intervals, accuracy and the confusion matrix."""
    score, pos = logits[:, MALIGNANT] - logits[:, 1 - MALIGNANT], (label == MALIGNANT).astype(int)
    pred = decide(logits)
    out = {"images": int(len(label)), "malignant": int(pos.sum()), "auc": auc(score, pos), "auprc": ap(score, pos),
           "accuracy": float(np.mean(pred == label)),
           "confusion_rows_true_cols_predicted_malignant_benign": [
               [int(np.sum((label == t) & (pred == p))) for p in (MALIGNANT, 1 - MALIGNANT)] for t in (MALIGNANT, 1 - MALIGNANT)]}
    if patient is not None:
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(patient):
            groups.setdefault(c, []).append(i)
        index = [np.array(v) for v in groups.values()]
        rng = np.random.default_rng(0)
        a, p = [], []
        for _ in range(BOOT):
            pick = np.concatenate([index[j] for j in rng.integers(0, len(index), len(index))])
            va, vp = auc(score[pick], pos[pick]), ap(score[pick], pos[pick])
            if va is not None:
                a.append(va)
                p.append(vp)
        out.update({"patients": len(index),
                    "auc_ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a else None,
                    "auprc_ci95": [float(np.percentile(p, 2.5)), float(np.percentile(p, 97.5))] if p else None,
                    "resamples_used": len(a)})
    return out


def numpy_student(x14: np.ndarray, w: dict) -> np.ndarray:
    """The float64 network of a student's published weights.json: conv (zero-padded), its activation, a 2x2 average
    pool, flattened channel-major, then each dense layer with its own activation."""
    W = np.array(w["conv"]["weight"], dtype=np.float64)[:, 0]
    b = np.array(w["conv"]["bias"], dtype=np.float64)
    pad, side = w["conv"]["padding"], x14.shape[-1]
    xp = np.pad(x14.astype(np.float64), ((0, 0), (pad, pad), (pad, pad)))
    conv = np.zeros((len(x14), W.shape[0], side, side))
    for c in range(W.shape[0]):
        for i in range(3):
            for j in range(3):
                conv[:, c] += W[c, i, j] * xp[:, i:i + side, j:j + side]
        conv[:, c] += b[c]
    act = w["conv"]["activation"]
    conv = np.maximum(conv, 0.0) if act == "relu" else conv ** 2 if act == "square" else conv
    v = conv.reshape(len(x14), W.shape[0], side // 2, 2, side // 2, 2).mean(axis=(3, 5)).reshape(len(x14), -1)
    for L in w["dense"]:
        v = v @ np.array(L["weight"], dtype=np.float64).T + np.array(L["bias"], dtype=np.float64)
        if L["activation"] == "relu":
            v = np.maximum(v, 0.0)
    return v


def ckks(x14: np.ndarray, student: dict, square: dict, work: Path) -> list[dict]:
    """Every image through the page's CKKS modules in node, in chunks of 250, one process per chunk."""
    script = work / "busbra_ckks_harness.mjs"
    script.write_text(HARNESS)
    sets = sorted(square["ckks"])
    rows: list[dict] = []
    flat = x14.reshape(len(x14), -1).astype(np.float64).tolist()
    for start in range(0, len(flat), CHUNK):
        job, res = work / f"ckks_job_{start}.json", work / f"ckks_result_{start}.jsonl"
        job.write_text(json.dumps({"student": student, "square": square, "sets": sets, "offset": start,
                                   "images": flat[start:start + CHUNK]}))
        subprocess.run(["node", str(script), str(WEB), str(work / "busbra_ckks_bundle.mjs"), str(job), str(res)],
                       cwd=WEB, check=True, timeout=3600)
        rows += [json.loads(line) for line in res.read_text().splitlines()]
        print(f"  CKKS: {len(rows)} of {len(flat)} images", flush=True)
    return rows


def agreement(enc: np.ndarray, plain: np.ndarray) -> dict:
    diff = float(np.abs(enc - plain).max())
    margin = np.abs(plain[:, 0] - plain[:, 1])
    changed = decide(enc) != decide(plain)
    return {"largest_abs_logit_difference": diff, "decisions_changed": int(changed.sum()),
            "decisions_changed_where_margin_exceeds_twice_the_difference": int((changed & (margin > 2 * diff)).sum()),
            "meets_the_page_rule": bool(diff <= CKKS_LIMIT and not (changed & (margin > 2 * diff)).any())}


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap_.add_argument("--work", type=Path, required=True, help="a directory outside the repository for the node stage")
    args = ap_.parse_args()
    got = sha256(PREREG)
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got[:12]}, not the committed {PREREG_SHA256[:12]}: run nothing")
    if REPO in args.work.resolve().parents or args.work.resolve() == REPO:
        raise SystemExit("--work must be outside the repository")
    args.work.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort

    pinned = pins()
    graphs = {}
    for name, (key, directory, graph, _) in MODELS.items():
        p = graph_path(directory, graph)
        want = pinned.get(key, {}).get(f"onnx/{graph}")
        if sha256(p) != want:
            raise SystemExit(f"{name}: {graph} is not the pinned file: run nothing")
        graphs[name] = p
    for key in ("medical_fl.student", "medical_fl.square_student"):
        w = REPO / "models" / MODELS[key][1] / "weights.json"
        if sha256(w) != pinned[key]["weights.json"]:
            raise SystemExit(f"{key}: weights.json is not the pinned file: run nothing")

    path = Path(D.data_root()) / "raw" / "bus-bra" / "BUSBRA.zip"
    u8, meta = load_busbra(path)
    label = np.array([MALIGNANT if r["Pathology"] == "malignant" else 1 - MALIGNANT for r in meta])
    patient, device = [r["Case"] for r in meta], [r["Device"] for r in meta]
    x28 = normalise(u8)
    x14 = to14(x28)
    recorded = json.loads((REPO / "models" / "metrics.json").read_text())
    models: dict[str, dict] = {}
    logits_of: dict[str, np.ndarray] = {}
    for name, (key, directory, graph, side) in MODELS.items():
        sess = ort.InferenceSession(str(graphs[name]), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        x = x28 if side == 28 else x14
        lg = np.concatenate([sess.run(None, {inp: x[i:i + 256]})[0] for i in range(0, len(x), 256)]).astype(np.float64)
        logits_of[name] = lg
        sample = json.loads((REPO / "models" / directory / "eval_sample.json").read_text())
        xs = np.array(sample["samples"], dtype=np.float32)
        xs = normalise(xs.reshape(-1, 28, 28).astype(np.uint8)) if side == 28 else xs.reshape(-1, 1, 14, 14)
        bm = sess.run(None, {inp: xs})[0].astype(np.float64)
        entry = {"registry_key": key, "graph": f"models/{directory}/onnx/{graph}", "graph_sha256": sha256(graphs[name]),
                 "busbra": figures(lg, label, patient),
                 "breastmnist_test_same_graph": figures(bm, np.array(sample["labels"]).ravel(), None),
                 "registry_recorded": {k: recorded.get(key, {}).get(k) for k in ("auc", "accuracy")},
                 "per_device": {}}
        for dev in sorted(set(device)):
            idx = np.array([i for i, d in enumerate(device) if d == dev])
            if len(idx) >= MIN_DEVICE:
                entry["per_device"][dev] = figures(lg[idx], label[idx], [patient[i] for i in idx])
            else:
                entry["per_device"][dev] = {"images": int(len(idx)), "malignant": int((label[idx] == MALIGNANT).sum()),
                                            "patients": len({patient[i] for i in idx}), "counts_only": True}
        models[name] = entry
        print(f"  {name}: AUC {entry['busbra']['auc']:.4f} on BUS-BRA, {entry['breastmnist_test_same_graph']['auc']:.4f} "
              f"on BreastMNIST's test split", flush=True)

    weights = {k: json.loads((REPO / "models" / MODELS[k][1] / "weights.json").read_text())
               for k in ("medical_fl.student", "medical_fl.square_student")}
    plain = {k: numpy_student(x14[:, 0], w) for k, w in weights.items()}
    for k in plain:
        models[k]["float64_network_vs_graph_largest_abs_logit_difference"] = float(np.abs(plain[k] - logits_of[k]).max())
    rows = ckks(x14[:, 0], weights["medical_fl.student"], weights["medical_fl.square_student"], args.work)
    if [r["i"] for r in rows] != list(range(len(x14))):
        raise SystemExit("the CKKS stage did not return every image once, in order")
    models["medical_fl.student"]["ckks"] = {"page_context": agreement(np.array([r["student"]["logits"] for r in rows]),
                                                                      plain["medical_fl.student"])}
    models["medical_fl.square_student"]["ckks"] = {
        s: agreement(np.array([r[s]["logits"] for r in rows]), plain["medical_fl.square_student"])
        for s in sorted(weights["medical_fl.square_student"]["ckks"])}

    import onnxruntime
    import PIL
    import skimage
    import sklearn

    record = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
              "deposit": {"zenodo": "8231412", "licence": "CC-BY-4.0 on the record; the archive's LICENSE.txt asks that "
                                                          "work using it cite doi:10.1002/mp.16812",
                          "archive_sha256": sha256(path)},
              "images": len(meta), "patients": len(set(patient)), "malignant": int((label == MALIGNANT).sum()),
              "devices": {d: device.count(d) for d in sorted(set(device))},
              "tools": {"onnxruntime": onnxruntime.__version__, "scikit-image": skimage.__version__,
                        "pillow": PIL.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__},
              "models": models}
    out = REPO / "docs" / "measurements" / f"medical-busbra-external-{date.today().isoformat()}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: {"busbra_auc": v["busbra"]["auc"], "ckks": v.get("ckks")} for k, v in models.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
