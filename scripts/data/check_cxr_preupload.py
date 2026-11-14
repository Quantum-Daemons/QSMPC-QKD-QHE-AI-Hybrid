#!/usr/bin/env python3
"""The chest X-ray model's checks before a Hub upload (Round 86, the owner's choice of 2026-11-14).

The owner's rule: before any Hugging Face upload, check the data's collection, cleansing, EDA, scaling, calibration and
feature extraction, with the resources on this machine. This reads the NIH files, the prepared 224 px set and the saved
FedAvg ResNet-18s of `scripts/train/medical_cxr_fl.py` (seeds 1 to 3, the hospitals' split fixed), and writes one record:

  collection   the twelve archives and five files against MANIFEST.tsv, bytes and SHA-256 recomputed; the entries against
               the official lists (every image in one list, no patient in both); the views; the original sizes NIH
               resized to 1024 px; every PNG decoded again, its colour mode counted
  cleansing    the pixel-identical groups and what the training drops; images near uniform, dark or bright at 224 px, and
               each image's correlation with its view's mean image (an inverted or rotated film scores low); ages and
               sexes as recorded; the hospitals as trained, rebuilt from the split seed, with no patient in two of them
  eda          by split and by hospital: images, patients, the AP share, effusion overall and by view, sex and age band,
               follow-ups, images per patient, and the findings effusion is recorded with
  scaling      the normalisation measured on 2,000 training images against all of them, and rounded; the published
               model's scores in bfloat16 (as the record measured them) against float32 (as the ONNX graph computes them);
               every test image resized from its 1024 px PNG by Pillow's six filters and by bilinear sampling without a
               low-pass, against the Pillow bilinear the set was built with
  calibration  each seed's sigmoid of the logit, on the hospitals' validation images and on the official test split:
               ECE over 15 equal-count bins, Brier score, log-loss, CORP's decomposition, the recalibration slope and
               intercept; a temperature and a Platt map fitted on validation and scored on test
  features     the published seed's pooled 512 features: units that never fire, a logistic head refitted on the frozen
               features (pooled, each hospital alone, and the four averaged), heads for two findings the network was not
               trained on, and how far the two leading principal components separate view and label

The published model is the first run's: seed 1, fixed before training, the seed the record's intervals use. Aggregate
figures only: no image, no file name, no patient ID. The resampled test images (about 9 GB) are held in memory and
written nowhere.

Usage: python -I scripts/data/check_cxr_preupload.py --prep <nih-cxr-224> --raw <nih-chestxray14>
       --runs <seed 1 run> <seed 2 run> <seed 3 run> [--workers 6]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "docs" / "measurements" / "cxr-preupload-2026-11-14.json"
SIZE = 224
#: Pillow's resampling filters, in its own order; the set was built with BILINEAR (prepare_nih_cxr.py).
PIL_FILTERS = ("nearest", "box", "bilinear", "hamming", "bicubic", "lanczos")
REFERENCE = "bilinear"
#: Bilinear sampling at the target's pixel centres with no low-pass first: what a GPU texture lookup without mipmaps, or
#: PyTorch's interpolate without antialias, computes.
NO_ANTIALIAS = "bilinear, no low-pass"
FILTERS = (*PIL_FILTERS, NO_ANTIALIAS)
AGE_BANDS = ((0, 18), (18, 40), (40, 60), (60, 80), (80, 200))
ECE_BINS = 15
#: Findings other than effusion whose heads are refitted on the frozen features.
OTHER_FINDINGS = ("Cardiomegaly", "Atelectasis")
_CHUNK = 8 << 20


# ------------------------------------------------------------------------------------------------------- collection

class HashingReader(io.RawIOBase):
    """A raw reader that hashes and counts every byte read through it, so an archive is hashed in the same pass that
    decodes it."""

    def __init__(self, f):
        self.f, self.sha, self.n = f, hashlib.sha256(), 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        k = self.f.readinto(b)
        if k:
            self.sha.update(memoryview(b)[:k])
            self.n += k
        return k


def sha256_of(path: Path) -> tuple[int, str]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def read_manifest(path: Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {r["file"]: {"bytes": int(r["bytes"]), "sha256": r["sha256"]} for r in rows}


def compare_manifest(manifest: dict[str, dict], found: dict[str, tuple[int, str]]) -> dict:
    """Each manifest row against what was read: the counts, and the names of any file that differs or is missing."""
    bad = sorted(f for f, m in manifest.items() if f not in found or found[f] != (m["bytes"], m["sha256"]))
    return {"files": len(manifest), "bytes": int(sum(m["bytes"] for m in manifest.values())),
            "matching": len(manifest) - len(bad), "differing_or_missing": bad}


def bilinear_no_lowpass(a: np.ndarray, size: int) -> np.ndarray:
    """Bilinear at the target's pixel centres (half-pixel convention), every target pixel from its four nearest source
    pixels and no low-pass first; rounded to uint8."""
    h, w = a.shape
    ys = np.clip((np.arange(size) + 0.5) * h / size - 0.5, 0, h - 1)
    xs = np.clip((np.arange(size) + 0.5) * w / size - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    f = a.astype(np.float64)
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return np.clip(np.rint(top * (1 - wy) + bot * wy), 0, 255).astype(np.uint8)


def resize_all(g, size: int = SIZE) -> list[np.ndarray]:
    """One grey PIL image resized by every filter, in FILTERS order."""
    from PIL import Image

    out = [np.asarray(g.resize((size, size), getattr(Image.Resampling, name.upper())), dtype=np.uint8) for name in PIL_FILTERS]
    out.append(bilinear_no_lowpass(np.asarray(g, dtype=np.uint8), size))
    return out


def one_archive(job: tuple) -> dict:
    """Decode every PNG of one archive, hashing the archive as it streams; count the colour modes and sizes; resize each
    test image by every filter, returned with its row; check the reference filter against the prepared set."""
    from PIL import Image

    tgz, prep_npy, prep_pos, test_row = job
    prep = np.load(prep_npy, mmap_mode="r")
    rows, arrays = [], []
    modes, sizes = Counter(), Counter()
    members = mismatched = 0
    ref = FILTERS.index(REFERENCE)
    with open(tgz, "rb") as raw:
        reader = HashingReader(raw)
        buf = io.BufferedReader(reader, _CHUNK)
        with tarfile.open(fileobj=buf, mode="r|gz") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".png"):
                    continue
                name = m.name.rsplit("/", 1)[-1]
                f = tf.extractfile(m)
                if f is None:
                    raise ValueError(f"a member of {Path(tgz).name} could not be read")
                im = Image.open(io.BytesIO(f.read()))
                im.load()
                members += 1
                modes[im.mode] += 1
                sizes[im.size] += 1
                if name in test_row:
                    outs = resize_all(im.convert("L"))
                    rows.append(test_row[name])
                    arrays.append(np.stack(outs))
                    mismatched += int(not np.array_equal(outs[ref], prep[prep_pos[name]]))
        while buf.read(_CHUNK):
            pass
    return {"archive": Path(tgz).name, "bytes": reader.n, "sha256": reader.sha.hexdigest(), "members": members,
            "modes": dict(modes), "sizes": {f"{w}x{h}": c for (w, h), c in sizes.items()}, "test_written": len(rows),
            "reference_mismatched": mismatched, "rows": rows,
            "arrays": np.stack(arrays) if arrays else np.zeros((0, len(FILTERS), SIZE, SIZE), np.uint8)}


def read_entries(raw: Path) -> list[dict]:
    with open(raw / "Data_Entry_2017_v2020.csv", newline="") as f:
        rows = list(csv.reader(f))
    return [{"image": r[0], "labels": r[1].split("|"), "follow_up": int(r[2]), "patient": int(r[3]), "age": int(r[4]),
             "sex": r[5], "view": r[6], "orig_w": int(r[7]), "orig_h": int(r[8]), "spacing": float(r[9])} for r in rows[1:]]


def entries_check(entries: list[dict], train_val: set[str], test: set[str]) -> dict:
    names = [e["image"] for e in entries]
    pat_of = {e["image"]: e["patient"] for e in entries}
    pats_test = {pat_of[n] for n in test}
    pats_tv = {pat_of[n] for n in train_val}
    w = np.array([e["orig_w"] for e in entries], float)
    h = np.array([e["orig_h"] for e in entries], float)
    aspect = w / h
    sp = np.array([e["spacing"] for e in entries])
    return {"images": len(names), "unique_names": len(set(names)), "patients": len({e["patient"] for e in entries}),
            "views": dict(Counter(e["view"] for e in entries)), "train_val_list": len(train_val), "test_list": len(test),
            "lists_overlap": len(train_val & test), "in_no_list": len(set(names) - train_val - test),
            "listed_not_indexed": len((train_val | test) - set(names)),
            "patients_in_both_lists": len(pats_test & pats_tv),
            "original_size": {"square": int((w == h).sum()), "aspect_quantiles_w_over_h": _q(aspect),
                              "aspect_off_by_more_than_10pct": int((np.abs(aspect - 1) > 0.10).sum()),
                              "smaller_than_1024_either_side": int(((w < 1024) | (h < 1024)).sum()),
                              "pixel_spacing_mm_quantiles": _q(sp)}}


def _q(a: np.ndarray, qs=(0.0, 0.01, 0.5, 0.99, 1.0)) -> dict:
    return {f"q{int(round(q * 100)):02d}": float(np.quantile(a, q)) for q in qs}


# -------------------------------------------------------------------------------------------------------- cleansing

def pixel_flags(mean: np.ndarray, sd: np.ndarray, corr: np.ndarray) -> dict:
    """Counts of images a reader should look at: near uniform (SD under 10 of 255), very dark or bright (mean under 15
    or over 240), or unlike their view's mean image (correlation under 0.2, negative, or under -0.5, which an inverted
    film with the usual layout gives)."""
    return {"near_uniform": int((sd < 10).sum()), "very_dark": int((mean < 15).sum()), "very_bright": int((mean > 240).sum()),
            "correlation_below_0.2": int((corr < 0.2).sum()), "correlation_negative": int((corr < 0).sum()),
            "correlation_below_minus_0.5": int((corr < -0.5).sum()),
            "correlation_quantiles": _q(corr), "mean_quantiles": _q(mean), "sd_quantiles": _q(sd)}


def age_band(age: int) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age < hi:
            return f"{lo}-{hi - 1}" if hi < 200 else f"{lo}+"
    return "other"


def group_stats(rows: list[dict], y: np.ndarray) -> dict:
    """Images, patients, AP share, prevalence overall and by view, sex and age band; follow-ups and images per patient."""
    if not rows:
        return {"images": 0}
    ap = np.array([r["view"] == "AP" for r in rows])
    female = np.array([r["sex"] == "F" for r in rows])
    age = np.array([r["age"] for r in rows])
    per_pat = Counter(r["patient"] for r in rows)

    def prev(sel):
        return {"images": int(sel.sum()), "effusion": float(y[sel].mean()) if sel.any() else None}
    bands = np.array([age_band(a) for a in age])
    return {"images": len(rows), "patients": len(per_pat), "ap_share": float(ap.mean()), "effusion": float(y.mean()),
            "by_view": {"AP": prev(ap), "PA": prev(~ap)}, "female_share": float(female.mean()),
            "by_sex": {"F": prev(female), "M": prev(~female)},
            "age_quantiles": {k: float(v) for k, v in zip(("q25", "q50", "q75"), np.quantile(age, [0.25, 0.5, 0.75]), strict=True)},
            "by_age_band": {b: prev(bands == b) for b in sorted(set(bands))},
            "follow_up_median": float(np.median([r["follow_up"] for r in rows])),
            "images_per_patient": {"median": float(np.median(list(per_pat.values()))), "max": int(max(per_pat.values()))}}


def cooccurring(entries: list[dict], label: str = "Effusion", top: int = 6) -> dict:
    with_label = [e["labels"] for e in entries if label in e["labels"]]
    c = Counter(x for ls in with_label for x in ls if x != label)
    return {"images": len(with_label), "alone": int(sum(ls == [label] for ls in with_label)),
            "top": [{"finding": k, "share": v / len(with_label)} for k, v in c.most_common(top)]}


# ------------------------------------------------------------------------------------------------------ calibration

def expit(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def ece_equal_count(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error over `bins` groups of equal count (ties broken by position, stably)."""
    order = np.argsort(p, kind="mergesort")
    return float(sum(len(g) / len(p) * abs(p[g].mean() - y[g].mean()) for g in np.array_split(order, bins)))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> list[dict]:
    order = np.argsort(p, kind="mergesort")
    return [{"n": int(len(g)), "mean_p": float(p[g].mean()), "frequency": float(y[g].mean())} for g in np.array_split(order, bins)]


def fit_temperature(y: np.ndarray, z: np.ndarray) -> float:
    """The T minimising the log-loss of sigmoid(z / T), by golden-section search over log T in [-3, 3]."""
    from qsh.models import calibration as cal

    def nll(log_t):
        return cal.log_loss(expit(z / math.exp(log_t)), y)
    lo, hi = -3.0, 3.0
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(a), nll(b)
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(b)
    return math.exp((lo + hi) / 2)


def recalibration_line(y: np.ndarray, z: np.ndarray) -> dict:
    """Logistic regression of the label on the logit, unpenalised: slope 1 and intercept 0 for a calibrated model; a
    slope under 1 means overconfident scores."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(z.reshape(-1, 1), y)
    return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}


def calibration_block(y: np.ndarray, p: np.ndarray) -> dict:
    from qsh.models import calibration as cal

    c = cal.corp_decomposition(p, y)
    return {"n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()), "mean_p": float(p.mean()),
            "ece": ece_equal_count(y, p), "brier": cal.brier(p, y), "log_loss": cal.log_loss(p, y),
            "corp": {k: c[k] for k in ("mcb", "dsc", "unc")}}


def calibration_of(yv: np.ndarray, zv: np.ndarray, yt: np.ndarray, zt: np.ndarray, ap_t: np.ndarray) -> dict:
    """Validation and test as they are, then a temperature and a Platt map (on the probability's logit, the finance
    cards' convention) fitted on validation and scored on test."""
    from qsh.models import calibration as cal

    pv, pt = expit(zv), expit(zt)
    t = fit_temperature(yv, zv)
    platt = cal.fit_platt(pv, yv, on_logit=True)
    pt_t, pt_p = expit(zt / t), platt.apply(pt)
    return {"validation": calibration_block(yv, pv), "test": calibration_block(yt, pt),
            "test_by_view": {"AP": calibration_block(yt[ap_t], pt[ap_t]), "PA": calibration_block(yt[~ap_t], pt[~ap_t])},
            "recalibration_line_validation": recalibration_line(yv, zv), "recalibration_line_test": recalibration_line(yt, zt),
            "temperature": t, "test_after_temperature": calibration_block(yt, pt_t),
            "platt": platt.as_dict(), "test_after_platt": calibration_block(yt, pt_p),
            "reliability_test": reliability(yt, pt)}


# ----------------------------------------------------------------------------------------------------------- features

def separation(score: np.ndarray, y: np.ndarray) -> float:
    """How well a direction separates a binary label, whichever way it points: max(AUROC, 1 - AUROC)."""
    from sklearn.metrics import roc_auc_score

    a = float(roc_auc_score(y, score))
    return max(a, 1 - a)


def frozen_head(ftr: np.ndarray, ytr: np.ndarray, fte: np.ndarray, yte: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    """A logistic head on standardised frozen features: its test AUROC, and its weights for averaging."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    clf = LogisticRegression(max_iter=3000, C=1.0).fit((ftr - mu) / sd, ytr)
    return float(roc_auc_score(yte, clf.decision_function((fte - mu) / sd))), clf.coef_[0].copy(), float(clf.intercept_[0])


# ----------------------------------------------------------------------------------------------------------- the run

def _load_cxr():
    spec = importlib.util.spec_from_file_location("medical_cxr_fl", REPO / "scripts" / "train" / "medical_cxr_fl.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("medical_cxr_fl", mod)
    spec.loader.exec_module(mod)
    return mod


def rounded(x, digits: int = 6):
    if isinstance(x, float):
        return float(f"{x:.{digits}g}")
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v, digits) for v in x]
    return x


def main() -> None:  # noqa: C901 - one pass over the data, in the order of the record
    import torch
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    sys.path.insert(0, str(REPO / "src"))
    cxr = _load_cxr()
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = cxr.DEV

    # ---- collection: the files, the entries, every PNG decoded again
    manifest = read_manifest(a.raw / "MANIFEST.tsv")
    entries = read_entries(a.raw)
    train_val = set((a.raw / "train_val_list.txt").read_text().split())
    test_names = set((a.raw / "test_list.txt").read_text().split())
    with open(a.prep / "index.csv", newline="") as f:
        index = list(csv.DictReader(f))
    prep_pos = {r["image"]: int(r["position"]) for r in index}
    test_sorted = sorted(test_names)
    test_row = {n: i for i, n in enumerate(test_sorted)}
    work = np.zeros((len(test_sorted), len(FILTERS), SIZE, SIZE), np.uint8)
    archives = sorted(a.raw.glob("images_*.tar.gz"))
    jobs = [(str(t), str(a.prep / "images_224.npy"), prep_pos, test_row) for t in archives]
    found: dict[str, tuple[int, str]] = {}
    decoded = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(one_archive, jobs):
            work[np.array(res.pop("rows"), dtype=np.int64)] = res.pop("arrays")
            decoded.append(res)
            found[res["archive"]] = (res["bytes"], res["sha256"])
            print(json.dumps({k: res[k] for k in ("archive", "members", "test_written", "reference_mismatched")}), flush=True)
    for name in manifest:
        if name not in found and (a.raw / name).exists():
            found[name] = sha256_of(a.raw / name)
    modes, sizes = Counter(), Counter()
    for r in decoded:
        modes.update(r["modes"])
        sizes.update(r["sizes"])
    collection = {
        "manifest": compare_manifest(manifest, found), "entries": entries_check(entries, train_val, test_names),
        "decoded": {"pngs": int(sum(r["members"] for r in decoded)), "colour_modes": dict(modes), "sizes": dict(sizes),
                    "test_images_resampled": int(sum(r["test_written"] for r in decoded)),
                    "reference_filter_differs_from_prepared_set": int(sum(r["reference_mismatched"] for r in decoded))},
        "labels": {"file": "Data_Entry_2017_v2020.csv",
                   "findings": dict(Counter(x for e in entries for x in e["labels"]).most_common())}}
    print(json.dumps(collection["manifest"]), flush=True)

    # ---- the data as trained: the same hospitals, the same drops
    d = cxr.load(a.prep, a.raw)
    imgs = torch.from_numpy(np.load(a.prep / "images_224.npy")).to(dev)
    tr_sites, va_sites, sharp = cxr.hospitals(d, 1)
    tr_all, va_all = np.concatenate(tr_sites), np.concatenate(va_sites)
    test_idx = np.flatnonzero(d["test"] & ~d["drop"])
    reports = [json.loads((r / "report.json").read_text()) for r in a.runs]
    rebuilt = [{"hospital": h + 1, "patients": int(len(np.unique(d["patient"][t]))), "train_images": int(len(t)), "val_images": int(len(v))}
               for h, (t, v) in enumerate(zip(tr_sites, va_sites, strict=True))]
    recorded = [{k: s[k] for k in ("hospital", "patients", "train_images", "val_images")} for s in reports[0]["hospitals"]]
    site_patients = [set(np.unique(d["patient"][np.concatenate([t, v])]).tolist()) for t, v in zip(tr_sites, va_sites, strict=True)]
    shared = sum(len(site_patients[i] & site_patients[j]) for i in range(4) for j in range(i + 1, 4))

    # ---- cleansing: pixel statistics and each image's likeness to its view's mean image
    n = imgs.shape[0]
    means, sds = torch.empty(n, device=dev), torch.empty(n, device=dev)
    view_mean = {}
    ap_all = torch.from_numpy(d["ap"]).to(dev)
    tr_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    tr_mask[torch.from_numpy(tr_all).to(dev)] = True
    acc = {True: torch.zeros(SIZE, SIZE, dtype=torch.float64, device=dev), False: torch.zeros(SIZE, SIZE, dtype=torch.float64, device=dev)}
    cnt = {True: 0, False: 0}
    for s in range(0, n, 4096):
        x = imgs[s:s + 4096].float()
        means[s:s + 4096], sds[s:s + 4096] = x.mean((1, 2)), x.std((1, 2))
        for v in (True, False):
            sel = tr_mask[s:s + 4096] & (ap_all[s:s + 4096] == v)
            acc[v] += x[sel].double().sum(0)
            cnt[v] += int(sel.sum())
    for v in (True, False):
        m = (acc[v] / cnt[v]).float()
        view_mean[v] = (m - m.mean()) / (m - m.mean()).norm()
    corr = torch.empty(n, device=dev)
    for s in range(0, n, 4096):
        x = imgs[s:s + 4096].float()
        xc = x - x.mean((1, 2), keepdim=True)
        mv = torch.where(ap_all[s:s + 4096].view(-1, 1, 1), view_mean[True], view_mean[False])
        corr[s:s + 4096] = (xc * mv).sum((1, 2)) / xc.flatten(1).norm(dim=1).clamp_min(1e-6)
    mean_np, sd_np, corr_np = means.cpu().numpy(), sds.cpu().numpy(), corr.cpu().numpy()
    with open(a.prep / "duplicates.json") as f:
        dup = json.load(f)
    flags_all = pixel_flags(mean_np, sd_np, corr_np)
    flagged = (sd_np < 10) | (mean_np < 15) | (mean_np > 240) | (corr_np < 0.2)
    ages = np.array([int(r["age"]) for r in index])
    cleansing = {
        "duplicates": {"groups": len(dup["groups"]), "across_split": len(dup["cross_split"]), "dropped_by_training": int(d["drop"].sum())},
        "pixels_224": {**flags_all, "flagged_in_test": int(flagged[test_idx].sum()), "flagged_in_training": int(flagged[tr_all].sum())},
        "ages": {"min": int(ages.min()), "max": int(ages.max()), "over_100": int((ages > 100).sum()), "under_1": int((ages < 1).sum())},
        "sexes": dict(Counter(r["sex"] for r in index)),
        "hospitals_rebuilt": {"split_seed": 1, "sharpness": sharp, "equal_to_record": rebuilt == recorded,
                              "patients_in_two_hospitals": shared,
                              "patients_in_training_and_test": len(set(d["patient"][tr_all].tolist()) & set(d["patient"][test_idx].tolist()))}}
    print(json.dumps(cleansing["hospitals_rebuilt"]), flush=True)

    # ---- EDA
    rows = [{"view": r["view"], "sex": r["sex"], "age": int(r["age"]), "patient": int(r["patient"]), "follow_up": int(r["follow_up"])} for r in index]

    def stats(idx):
        return group_stats([rows[i] for i in idx], d["y"][idx])
    eda = {"train_val_kept": stats(np.flatnonzero(~d["test"] & ~d["drop"])), "test": stats(test_idx),
           "hospitals": [{"hospital": h + 1, **stats(np.concatenate([t, v]))} for h, (t, v) in enumerate(zip(tr_sites, va_sites, strict=True))],
           "effusion_recorded_with": cooccurring(entries)}

    # ---- the models
    def model_of(run: Path):
        m = cxr.resnet18_1ch().to(dev).to(memory_format=torch.channels_last)
        m.load_state_dict(torch.load(run / "fedavg_resnet18.pt", map_location=dev))
        return m.eval()

    @torch.no_grad()
    def scores(m, src, idx, mean, std, bf16=False, feats=False):
        out, fs = [], []
        for s in range(0, len(idx), 512):
            ii = torch.from_numpy(np.asarray(idx[s:s + 512])).to(dev)
            x = src[ii].unsqueeze(1).float().div_(255)
            x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                z, _, pooled = cxr.forward_feats(m, x)
            out.append(z.float().cpu())
            if feats:
                fs.append(pooled.float().cpu())
        zz = torch.cat(out).numpy().astype(np.float64)
        return (zz, torch.cat(fs).numpy()) if feats else zz

    norm = reports[0]["normalisation"]
    mean_s, std_s = norm["mean"], norm["std"]
    yt, apt, pat_t = d["y"][test_idx].astype(int), d["ap"][test_idx], d["patient"][test_idx]
    yv = d["y"][va_all].astype(int)

    # ---- scaling: the constants, the precision, the resampler
    s1 = s2 = 0.0
    for s in range(0, len(tr_all), 4096):
        x = imgs[torch.from_numpy(tr_all[s:s + 4096]).to(dev)].double().div(255)
        s1 += float(x.sum())
        s2 += float((x * x).sum())
    npx = len(tr_all) * SIZE * SIZE
    full_mean = s1 / npx
    full_std = math.sqrt(s2 / npx - full_mean ** 2)
    m1 = model_of(a.runs[0])
    z32 = scores(m1, imgs, test_idx, mean_s, std_s)
    flag_t = flagged[test_idx]
    cleansing["pixels_224"]["test_without_flagged"] = {
        "images": int((~flag_t).sum()), "flagged_effusion_share": float(yt[flag_t].mean()) if flag_t.any() else None,
        "auroc_all": float(roc_auc_score(yt, z32)), "auroc_without_flagged": float(roc_auc_score(yt[~flag_t], z32[~flag_t]))}
    z16 = scores(m1, imgs, test_idx, mean_s, std_s, bf16=True)
    zfull = scores(m1, imgs, test_idx, full_mean, full_std)
    zround = scores(m1, imgs, test_idx, round(mean_s, 2), round(std_s, 2))
    zv1 = scores(m1, imgs, va_all, mean_s, std_s)
    thr = cxr.operating_point(yv, zv1)
    scaling = {
        "normalisation": {"sample_of_2000": {"mean": mean_s, "std": std_s}, "all_training_images": {"mean": full_mean, "std": full_std},
                          "test_auroc": {"sample_constants": float(roc_auc_score(yt, z32)), "all_training_constants": float(roc_auc_score(yt, zfull)),
                                         "rounded_to_2_decimals": float(roc_auc_score(yt, zround))},
                          "max_abs_logit_change": {"all_training_constants": float(np.abs(zfull - z32).max()),
                                                   "rounded_to_2_decimals": float(np.abs(zround - z32).max())}},
        "precision": {"test_auroc_bfloat16": float(roc_auc_score(yt, z16)), "test_auroc_float32": float(roc_auc_score(yt, z32)),
                      "max_abs_logit_difference": float(np.abs(z16 - z32).max()), "mean_abs_logit_difference": float(np.abs(z16 - z32).mean()),
                      "decisions_differing_at_the_operating_point": int(((z16 >= thr) != (z32 >= thr)).sum()),
                      "tf32": "off in this check (cuDNN and matmul), so float32 here is float32"}}
    order = np.array([test_row[d["names"][i]] for i in test_idx])
    ref_k = FILTERS.index(REFERENCE)
    resampler = {}
    zref = None
    for k, name in enumerate(FILTERS):
        src = torch.from_numpy(np.ascontiguousarray(work[order, k])).to(dev)
        zk = scores(m1, src, np.arange(len(order)), mean_s, std_s)
        if k == ref_k:
            zref = zk
            same_pixels = bool(torch.equal(src, imgs[torch.from_numpy(test_idx).to(dev)]))
        resampler[name] = {"test_auroc": float(roc_auc_score(yt, zk)), "z": zk,
                           "mean_abs_pixel_vs_reference": float((src.float() - torch.from_numpy(np.ascontiguousarray(work[order, ref_k])).to(dev).float()).abs().mean())}
        del src
    for r in resampler.values():
        zk = r.pop("z")
        r["max_abs_logit_vs_reference"] = float(np.abs(zk - zref).max())
        r["mean_abs_logit_vs_reference"] = float(np.abs(zk - zref).mean())
        r["decisions_differing_at_the_operating_point"] = int(((zk >= thr) != (zref >= thr)).sum())
    scaling["resampler"] = {"reference": REFERENCE, "reference_equals_prepared_pixels": same_pixels, "images": len(order),
                            "operating_point_logit": thr, "filters": resampler}
    print(json.dumps({k: v["test_auroc"] for k, v in resampler.items()}), flush=True)

    # ---- calibration, every seed
    calib = []
    for run, rep in zip(a.runs, reports, strict=True):
        m = model_of(run)
        zt, zv = scores(m, imgs, test_idx, mean_s, std_s), scores(m, imgs, va_all, mean_s, std_s)
        calib.append({"seed": rep["seed"], "test_auroc_float32": float(roc_auc_score(yt, zt)),
                      "test_auroc_interval": cxr.auc_boot(yt, zt, pat_t), **calibration_of(yv, zv, yt, zt, apt)})
        print(json.dumps({"seed": rep["seed"], "auroc": calib[-1]["test_auroc_float32"], "ece_test": calib[-1]["test"]["ece"]}), flush=True)

    # ---- features of the published seed
    _, ftr = scores(m1, imgs, tr_all, mean_s, std_s, feats=True)
    _, fva = scores(m1, imgs, va_all, mean_s, std_s, feats=True)
    _, fte = scores(m1, imgs, test_idx, mean_s, std_s, feats=True)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ytr = d["y"][tr_all].astype(int)
    pooled_auc, _, _ = frozen_head(ftr, ytr, fte, yt, mu, sd)
    per_site, coefs, ints, sizes_ = [], [], [], []
    off = 0
    for h, t in enumerate(tr_sites):
        sl = slice(off, off + len(t))
        off += len(t)
        auc_h, w_h, b_h = frozen_head(ftr[sl], ytr[sl], fte, yt, mu, sd)
        per_site.append({"hospital": h + 1, "test_auroc": auc_h})
        coefs.append(w_h)
        ints.append(b_h)
        sizes_.append(len(t))
    wts = np.array(sizes_, float) / sum(sizes_)
    avg_w, avg_b = (wts[:, None] * np.array(coefs)).sum(0), float((wts * np.array(ints)).sum())
    avg_auc = float(roc_auc_score(yt, ((fte - mu) / sd) @ avg_w + avg_b))
    label_of = {e["image"]: e["labels"] for e in entries}
    others = {}
    for finding in OTHER_FINDINGS:
        y_tr = np.array([finding in label_of[d["names"][i]] for i in tr_all], int)
        y_te = np.array([finding in label_of[d["names"][i]] for i in test_idx], int)
        auc_o, _, _ = frozen_head(ftr, y_tr, fte, y_te, mu, sd)
        others[finding] = {"test_auroc": auc_o, "test_prevalence": float(y_te.mean())}
    pca = PCA(2, random_state=0).fit((ftr - mu) / sd)
    pc = pca.transform((fte - mu) / sd)
    features = {
        "dimension": int(ftr.shape[1]), "units_never_firing_on_test": int((fte.max(0) <= 0).sum()),
        "units_never_firing_on_training": int((ftr.max(0) <= 0).sum()),
        "network_test_auroc_float32": float(roc_auc_score(yt, z32)),
        "frozen_head": {"pooled_training": pooled_auc, "each_hospital_alone": per_site, "four_heads_averaged_by_size": avg_auc,
                        "other_findings": others},
        "principal_components": {"explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
                                 "separation_of_view": [separation(pc[:, j], apt.astype(int)) for j in range(2)],
                                 "separation_of_effusion": [separation(pc[:, j], yt) for j in range(2)]},
        "view_probe_recorded": {"published_seed": next(r for r in reports[0]["arms"] if r["arm"] == "FedAvg ResNet-18")["view_probe_auroc"],
                                "untrained_network": reports[0].get("untrained", {}).get("view_probe_auroc")}}
    del fva

    import PIL
    import sklearn

    env = cxr.environment()
    # The libraries' names, not where this machine keeps them: a record is published and a path names its owner.
    env["cublas_loaded"] = [Path(p).name for p in env.get("cublas_loaded", [])]
    env.update({"pillow": PIL.__version__, "scikit_learn": sklearn.__version__, "numpy": np.__version__})
    record = {"_about": ("The chest X-ray FedAvg ResNet-18's checks before a Hub upload (Round 86): collection, cleansing, EDA, "
                         "scaling, calibration and features. Written by scripts/data/check_cxr_preupload.py; aggregate "
                         "figures only."),
              "published_seed": reports[0]["seed"], "seeds": [r["seed"] for r in reports], "environment": env,
              "collection": collection, "cleansing": cleansing, "eda": eda, "scaling": scaling, "calibration": calib,
              "features": features, "wall_clock_s": time.time() - t0}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(rounded(record), indent=1) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
