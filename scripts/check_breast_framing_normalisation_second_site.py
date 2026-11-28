#!/usr/bin/env python3
"""Three checks of the published BreastMNIST models, as pre-registered (Round 86, batch 13):
references/medical-breast-framing-normalisation-second-site-preregistration.md.

Arm S scores every BrEaST-Lesions-USG image through the first external test's protocol (its CKKS stage included); arm F
scores BUS-BRA and BrEaST cropped to each lesion's box at three margins; arm N adapts, without labels, the inputs'
statistics (all five graphs) and the ResNets' batch-normalisation statistics (their unfused checkpoints) on a seeded
half of each dataset's patients and scores the other half. Reads both archives in place, checks the pre-registration's
SHA-256 and every graph's and checkpoint's first, and writes
docs/measurements/medical-breast-framing-normalisation-second-site-<date>.json, aggregate figures only.

    QSH_DATA_ROOT=... QSH_PUBLISHED_GRAPHS=<checkout> python scripts/check_breast_framing_normalisation_second_site.py --work DIR
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import zipfile
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "references" / "medical-breast-framing-normalisation-second-site-preregistration.md"
PREREG_SHA256 = "e986882ad2ff663711cd55321327b8e1812909613cf088c14f5feb337583c081"
BREAST_ZIP_SHA256 = "c32e49cdfa54042e065582d0cb35978b5fc5e5b5e461b33bf773fb50cfdad355"
BREAST_SHEET_SHA256 = "89a8874496a6f1f93960390b963f7369b90f7e72ee08a738f2d17e37f5eff8bf"
#: arch: (models directory, the unfused checkpoint's SHA-256), the files the pre-registration names.
CHECKPOINTS = {
    "resnet18": ("medical_breastmnist_resnet18", "6c1499d461865522c8e6c7034b6509ff6fdd6a60ff359b2dd889d6070e66ef85"),
    "resnet50": ("medical_breastmnist_resnet50", "b17d62629369e3222753150d60dcabd022e83e502e5c86c3819898e3ecf6df50"),
}
GRAPH_OF_ARCH = {"resnet18": "medical_fl.resnet18 (fp32)", "resnet50": "medical_fl.resnet50"}
MARGINS = (0.0, 0.12, 0.5)
CHECKPOINT_LIMIT, BN_BATCH = 1e-4, 64


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("check_busbra_external", REPO / "scripts" / "check_busbra_external.py")   # also puts src/ on the path
from qsh.models import datasets as D  # noqa: E402

MALIGNANT = B.MALIGNANT


# ---------------------------------------------------------------------------------------------------------- reading
def _xlsx_rows(path: Path) -> list[list]:
    """The first sheet of an .xlsx as rows, with the standard library only."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
    out = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).find("m:sheetData", ns).findall("m:row", ns):
        vals = {}
        for c in r.findall("m:c", ns):
            col = 0
            for ch in re.match(r"[A-Z]+", c.get("r")).group(0):
                col = col * 26 + (ord(ch) - 64)
            v, t = c.find("m:v", ns), c.get("t")
            if t == "inlineStr":
                vals[col - 1] = "".join(x.text or "" for x in c.iter("{%s}t" % ns["m"]))
            elif v is not None:
                vals[col - 1] = shared[int(v.text)] if t == "s" else v.text
        out.append([vals.get(i) for i in range(max(vals) + 1)] if vals else [])
    return out


def _to28(gray: np.ndarray) -> np.ndarray:
    from skimage.transform import resize

    small = resize(gray, (28, 28), order=3, anti_aliasing=True, preserve_range=True)
    return np.clip(np.rint(small), 0, 255).astype(np.uint8)


def _crop(gray: np.ndarray, box: tuple[float, float, float, float] | None, m: float) -> np.ndarray:
    """The image cropped to `box` = (x, y, width, height) widened by m x width left and right and m x height top and
    bottom, clipped to the image; the whole image when there is no box."""
    if box is None:
        return gray
    x, y, w, h = box
    H, W = gray.shape
    x0, x1 = max(0, math.floor(x - m * w)), min(W, math.ceil(x + w + m * w))
    y0, y1 = max(0, math.floor(y - m * h)), min(H, math.ceil(y + h + m * h))
    return gray[y0:y1, x0:x1]


def load_busbra(path: Path) -> dict:
    from PIL import Image

    with zipfile.ZipFile(path) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open("BUSBRA/bus_data.csv"), encoding="utf-8")))
        grays = [np.asarray(Image.open(io.BytesIO(z.read(f"BUSBRA/Images/{r['ID']}.png"))).convert("L"), dtype=np.float64)
                 for r in rows]
    return {"name": "BUS-BRA", "gray": grays, "box": [tuple(json.loads(r["BBOX"])) for r in rows],
            "label": np.array([MALIGNANT if r["Pathology"] == "malignant" else 1 - MALIGNANT for r in rows]),
            "patient": [r["Case"] for r in rows], "archive_sha256": B.sha256(path)}


def load_breast(root: Path) -> dict:
    from PIL import Image

    zpath = root / "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip"
    sheet = root / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    if B.sha256(zpath) != BREAST_ZIP_SHA256 or B.sha256(sheet) != BREAST_SHEET_SHA256:
        raise SystemExit("BrEaST-Lesions-USG is not the archive and sheet the pre-registration names: run nothing")
    rows = _xlsx_rows(sheet)
    hdr = rows[0]
    col = {k: hdr.index(k) for k in ("CaseID", "Image_filename", "Mask_tumor_filename", "Classification")}
    prefix = "BrEaST-Lesions_USG-images_and_masks/"
    grays, boxes, labels, patients, alpha_below = [], [], [], [], 0
    with zipfile.ZipFile(zpath) as z:
        for r in rows[1:]:
            im = Image.open(io.BytesIO(z.read(prefix + r[col["Image_filename"]])))
            if im.mode in ("RGBA", "LA") and int(np.asarray(im)[..., -1].min()) < 255:
                alpha_below += 1
            grays.append(np.asarray(im.convert("L"), dtype=np.float64))
            masks = r[col["Mask_tumor_filename"]]
            box = None
            if masks and masks != "None":
                on = None
                for name in masks.split("&"):
                    mk = np.asarray(Image.open(io.BytesIO(z.read(prefix + name))).convert("L")) > 127
                    on = mk if on is None else (on | mk)
                ys, xs = np.nonzero(on)
                if len(xs):
                    box = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
            boxes.append(box)
            labels.append(MALIGNANT if r[col["Classification"]] == "malignant" else 1 - MALIGNANT)
            patients.append(str(r[col["CaseID"]]))
    return {"name": "BrEaST-Lesions-USG", "gray": grays, "box": boxes, "label": np.array(labels), "patient": patients,
            "archive_sha256": BREAST_ZIP_SHA256, "sheet_sha256": BREAST_SHEET_SHA256,
            "images_with_alpha_below_255": alpha_below,
            "classification": {k: sum(1 for r in rows[1:] if r[col["Classification"]] == k)
                               for k in sorted({r[col["Classification"]] for r in rows[1:]})},
            "images_without_a_tumour_box": sum(b is None for b in boxes)}


# ---------------------------------------------------------------------------------------------------------- scoring
def scores(sessions: dict, x28: np.ndarray) -> dict[str, np.ndarray]:
    """Every graph's logits on preprocessed 28x28 inputs (N, 1, 28, 28); the students get the area resize."""
    x14 = B.to14(x28)
    out = {}
    for name, (sess, side) in sessions.items():
        inp = sess.get_inputs()[0].name
        x = x28 if side == 28 else x14
        out[name] = np.concatenate([sess.run(None, {inp: x[i:i + 256].astype(np.float32)})[0]
                                    for i in range(0, len(x), 256)]).astype(np.float64)
    return out


def figs(logits: dict[str, np.ndarray], label: np.ndarray, patient: list[str]) -> dict:
    return {k: B.figures(v, label, patient) for k, v in logits.items()}


def build_checkpoint(arch: str, mb, base: Path):
    import torch
    from safetensors.torch import load_file

    directory, want = CHECKPOINTS[arch]
    p = base / "models" / directory / "model.safetensors"
    if not p.exists() or B.sha256(p) != want:
        raise SystemExit(f"{arch}: {p} is not the checkpoint the pre-registration names: run nothing")
    m = mb.build_resnet(arch)
    m.load_state_dict(load_file(str(p)))
    m.eval()
    torch.set_grad_enabled(False)
    return m


def torch_logits(model, x28: np.ndarray) -> np.ndarray:
    import torch

    return np.concatenate([model(torch.from_numpy(x28[i:i + 256].astype(np.float32))).numpy()
                           for i in range(0, len(x28), 256)]).astype(np.float64)


def adapt_bn(model, x_adapt: np.ndarray):
    """Every BatchNorm2d's running statistics re-estimated on `x_adapt`: train mode, momentum None (the cumulative
    average over batches), batches of 64 in order, no gradient, every weight fixed; returned in eval mode."""
    import torch

    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None
    model.train()
    with torch.no_grad():
        for i in range(0, len(x_adapt), BN_BATCH):
            model(torch.from_numpy(x_adapt[i:i + BN_BATCH].astype(np.float32)))
    model.eval()
    return model


def halves(patient: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ids = sorted(set(patient), key=lambda s: int(float(s)))
    order = np.random.default_rng(0).permutation(len(ids))
    adapt = {ids[j] for j in order[: len(ids) // 2]}
    a = np.array([i for i, p in enumerate(patient) if p in adapt])
    t = np.array([i for i, p in enumerate(patient) if p not in adapt])
    return a, t


# ---------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, required=True, help="a directory outside the repository for the node stage")
    args = ap.parse_args()
    got = B.sha256(PREREG)
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got[:12]}, not the committed {PREREG_SHA256[:12]}: run nothing")
    if REPO in args.work.resolve().parents or args.work.resolve() == REPO:
        raise SystemExit("--work must be outside the repository")
    args.work.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort

    pinned = B.pins()
    sessions, graph_sha = {}, {}
    for name, (key, directory, graph, side) in B.MODELS.items():
        p = B.graph_path(directory, graph)
        if B.sha256(p) != pinned.get(key, {}).get(f"onnx/{graph}"):
            raise SystemExit(f"{name}: {graph} is not the pinned file: run nothing")
        sessions[name] = (ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]), side)
        graph_sha[name] = B.sha256(p)
    weights = {}
    for key in ("medical_fl.student", "medical_fl.square_student"):
        w = REPO / "models" / B.MODELS[key][1] / "weights.json"
        if B.sha256(w) != pinned[key]["weights.json"]:
            raise SystemExit(f"{key}: weights.json is not the pinned file: run nothing")
        weights[key] = json.loads(w.read_text())

    # The checkpoints first: each must reproduce its published graph on the published sample, or arm N2 does not run.
    base = Path(os.environ.get("QSH_PUBLISHED_GRAPHS") or REPO)
    mb = _load("medical_breastmnist", REPO / "scripts" / "train" / "medical_breastmnist.py")
    checkpoint_check, run_n2 = {}, True
    for arch, (directory, sha) in CHECKPOINTS.items():
        m = build_checkpoint(arch, mb, base)
        sample = json.loads((REPO / "models" / directory / "eval_sample.json").read_text())
        xs = B.normalise(np.array(sample["samples"], dtype=np.float32).reshape(-1, 28, 28).astype(np.uint8))
        g = scores({GRAPH_OF_ARCH[arch]: sessions[GRAPH_OF_ARCH[arch]]}, xs)[GRAPH_OF_ARCH[arch]]
        t = torch_logits(m, xs)
        diff, same = float(np.abs(t - g).max()), bool((B.decide(t) == B.decide(g)).all())
        checkpoint_check[arch] = {"checkpoint_sha256": sha, "graph": GRAPH_OF_ARCH[arch], "largest_abs_logit_difference": diff,
                                  "decisions_same": same, "images": len(xs)}
        run_n2 &= diff <= CHECKPOINT_LIMIT and same
    print("checkpoints:", json.dumps(checkpoint_check), flush=True)

    data_root = Path(D.data_root()) / "raw"
    sets = {"busbra": load_busbra(data_root / "bus-bra" / "BUSBRA.zip"), "breast": load_breast(data_root / "breast-lesions-usg")}
    record: dict = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
                    "graphs": graph_sha, "checkpoint_check": checkpoint_check, "arm_n2_ran": run_n2, "datasets": {}}

    # BreastMNIST's training split for N1's target statistics.
    npz = data_root / "medmnist" / "breastmnist.npz"
    train = B.normalise(np.load(npz)["train_images"].astype(np.uint8))
    mu_b, sd_b = float(train.mean()), float(train.std())
    record["breastmnist_train"] = {"file_sha256": B.sha256(npz), "images": int(len(train)), "mean": mu_b, "sd": sd_b}

    for key, ds in sets.items():
        label, patient = ds["label"], ds["patient"]
        whole = B.normalise(np.stack([_to28(g) for g in ds["gray"]]))
        entry: dict = {"images": int(len(label)), "patients": len(set(patient)), "malignant": int((label == MALIGNANT).sum()),
                       "archive_sha256": ds["archive_sha256"]}
        for k in ("sheet_sha256", "images_with_alpha_below_255", "classification", "images_without_a_tumour_box"):
            if k in ds:
                entry[k] = ds[k]
        lw = scores(sessions, whole)
        entry["whole"] = figs(lw, label, patient)
        print(f"{ds['name']} whole: " + ", ".join(f"{k} {v['auc']:.3f}" for k, v in entry["whole"].items()), flush=True)

        # Arm S: BrEaST in full, beside each graph's BreastMNIST figure, with the students' float64 and CKKS checks.
        if key == "breast":
            same_graph = {}
            for name, (key_, directory, graph, side) in B.MODELS.items():
                sample = json.loads((REPO / "models" / directory / "eval_sample.json").read_text())
                xs = np.array(sample["samples"], dtype=np.float32)
                xs = B.normalise(xs.reshape(-1, 28, 28).astype(np.uint8)) if side == 28 else xs.reshape(-1, 1, 14, 14)
                sess = sessions[name][0]
                bm = sess.run(None, {sess.get_inputs()[0].name: xs})[0].astype(np.float64)
                same_graph[name] = B.figures(bm, np.array(sample["labels"]).ravel(), None)
            x14 = B.to14(whole)
            plain = {k: B.numpy_student(x14[:, 0], w) for k, w in weights.items()}
            rows = B.ckks(x14[:, 0], weights["medical_fl.student"], weights["medical_fl.square_student"], args.work)
            if [r["i"] for r in rows] != list(range(len(x14))):
                raise SystemExit("the CKKS stage did not return every image once, in order")
            entry["arm_s"] = {
                "breastmnist_test_same_graph": same_graph,
                "float64_network_vs_graph_largest_abs_logit_difference": {k: float(np.abs(plain[k] - lw[k]).max()) for k in plain},
                "ckks": {"medical_fl.student": {"page_context": B.agreement(np.array([r["student"]["logits"] for r in rows]),
                                                                            plain["medical_fl.student"])},
                         "medical_fl.square_student": {s: B.agreement(np.array([r[s]["logits"] for r in rows]),
                                                                      plain["medical_fl.square_student"])
                                                       for s in sorted(weights["medical_fl.square_student"]["ckks"])}}}

        # Arm F: framing at each margin.
        entry["arm_f"] = {}
        for m in MARGINS:
            x = B.normalise(np.stack([_to28(_crop(g, b, m)) for g, b in zip(ds["gray"], ds["box"], strict=True)]))
            f = figs(scores(sessions, x), label, patient)
            entry["arm_f"][f"m={m}"] = {k: {**v, "auc_change_from_whole": (v["auc"] - entry["whole"][k]["auc"])
                                            if v["auc"] is not None and entry["whole"][k]["auc"] is not None else None}
                                        for k, v in f.items()}
            print(f"{ds['name']} framed m={m}: " + ", ".join(f"{k} {v['auc']:.3f}" for k, v in f.items()), flush=True)

        # Arm N: a seeded patient half adapts, the other is scored.
        a, t = halves(patient)
        lab_t, pat_t = label[t], [patient[i] for i in t]
        n: dict = {"adaptation_images": int(len(a)), "test_images": int(len(t)),
                   "test_malignant": int((lab_t == MALIGNANT).sum()), "test_patients": len(set(pat_t))}
        n["n0"] = figs({k: v[t] for k, v in lw.items()}, lab_t, pat_t)
        mu_a, sd_a = float(whole[a].mean()), float(whole[a].std())
        x1 = (whole[t] - mu_a) * (sd_b / sd_a) + mu_b
        n["n1"] = {"adaptation_mean": mu_a, "adaptation_sd": sd_a, "models": figs(scores(sessions, x1), lab_t, pat_t)}
        if run_n2:
            n["n2"] = {}
            for arch in CHECKPOINTS:
                plain_ck = torch_logits(build_checkpoint(arch, mb, base), whole[t])
                adapted = torch_logits(adapt_bn(build_checkpoint(arch, mb, base), whole[a]), whole[t])
                n["n2"][arch] = {"unadapted_checkpoint": B.figures(plain_ck, lab_t, pat_t),
                                 "adapted": B.figures(adapted, lab_t, pat_t),
                                 "batches": math.ceil(len(a) / BN_BATCH)}
        entry["arm_n"] = n
        print(f"{ds['name']} N: n0 " + ", ".join(f"{k} {v['auc']:.3f}" for k, v in n["n0"].items())
              + " | n1 " + ", ".join(f"{k} {v['auc']:.3f}" for k, v in n["n1"]["models"].items())
              + (" | n2 " + ", ".join(f"{k} {v['unadapted_checkpoint']['auc']:.3f}->{v['adapted']['auc']:.3f}"
                                      for k, v in n.get("n2", {}).items())), flush=True)
        record["datasets"][key] = entry

    import onnxruntime
    import PIL
    import skimage
    import sklearn
    import torch

    record["tools"] = {"onnxruntime": onnxruntime.__version__, "scikit-image": skimage.__version__, "pillow": PIL.__version__,
                       "scikit-learn": sklearn.__version__, "numpy": np.__version__, "torch": torch.__version__}
    record["licences"] = {"BUS-BRA": "CC-BY-4.0 (Zenodo 8231412); cite doi:10.1002/mp.16812",
                          "BrEaST-Lesions-USG": "CC-BY-4.0 (TCIA, doi:10.7937/9WKK-Q141; TCIA Data Usage Policy); "
                                                "descriptor doi:10.1038/s41597-024-02984-z"}
    out = REPO / "docs" / "measurements" / f"medical-breast-framing-normalisation-second-site-{date.today().isoformat()}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote", out.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
