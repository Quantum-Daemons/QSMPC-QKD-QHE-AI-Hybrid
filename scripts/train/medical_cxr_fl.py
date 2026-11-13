#!/usr/bin/env python3
"""Chest X-ray federation on NIH ChestX-ray14 (Round 86, the owner's item (3)).

Effusion, from scratch at 224 px, on the GPU with the whole set resident. Four simulated hospitals take the official
train_val patients with different AP/PA mixes (patient-level, so no patient is in two places); the official test split
is the common test set. Arms: centralised, FedAvg, each hospital alone, PA-only (all ResNet-18), and FedAvg once with a
ConvNeXt-atto built from torchvision's ConvNeXt class. Analyses: AUROC with patient-bootstrap intervals overall and by
view, the operating point chosen on validation, the view-only baseline, a linear view probe on the features, and CAMs
against NIH's Effusion boxes (all 153 are in the test split).

Usage: python -I scripts/train/medical_cxr_fl.py --prep <prepared dir> --raw <raw dir> --out <dir> [--arms ...] [--epochs 12]
                 [--seed 1] [--split-seed 1]

The prepared dir is what scripts/data/prepare_nih_cxr.py writes. The run needs a CUDA GPU (the whole set, 5.6 GB at 224 px,
is held on it); the functions the tests exercise (the hospitals, the duplicates, the AUROC and its interval, the operating
point, the CAM scoring) need neither the data nor a GPU. Data: NIH Clinical Center, ChestX-ray14 (Wang et al., CVPR 2017),
https://nihcc.app.box.com/v/ChestXray-NIHCC.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18
from torchvision.models.convnext import CNBlockConfig, ConvNeXt

DEV = torch.device("cuda")
# AP share each simulated hospital draws its patients towards: an outpatient clinic to an intensive-care unit.
HOSPITAL_AP_TARGETS = (0.05, 0.25, 0.50, 0.85)
VAL_FRACTION = 0.10


# ---------------------------------------------------------------------------------------------------------------- data

def load(prep: Path, raw: Path):
    with open(prep / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    patient = np.array([int(r["patient"]) for r in rows])
    ap = np.array([r["view"] == "AP" for r in rows])
    assert set(r["view"] for r in rows) <= {"AP", "PA"}
    test = np.array([r["split"] == "test" for r in rows])
    y = np.array([int(r["effusion"]) for r in rows], dtype=np.float32)
    names = [r["image"] for r in rows]
    # Pixel-identical duplicates (prepare_nih_cxr.py): across the split, the train_val copies go; within a split, all
    # but the first by name go.
    with open(prep / "duplicates.json") as f:
        dup = json.load(f)
    pos = {name: i for i, name in enumerate(names)}
    drop = np.zeros(n, bool)
    for g in dup["groups"]:
        idx = sorted(pos[x] for x in g)
        if any(test[i] for i in idx) and not all(test[i] for i in idx):
            for i in idx:
                if not test[i]:
                    drop[i] = True
            kept_test = [i for i in idx if test[i]]
            for i in kept_test[1:]:
                drop[i] = True
        else:
            for i in idx[1:]:
                drop[i] = True
    boxes = {}
    with open(raw / "BBox_List_2017.csv", newline="") as f:
        box_rows = list(csv.reader(f))[1:]
    for r in box_rows:
        if r[1] == "Effusion":
            boxes[pos[r[0]]] = tuple(float(v) for v in r[2:6])   # x, y, w, h on the 1024 x 1024 PNG
    return dict(n=n, patient=patient, ap=ap, test=test, y=y, names=names, drop=drop, boxes=boxes)


SHARPNESS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def _assign(d, pats, by_pat, k, seed):
    rng = np.random.default_rng(seed)
    s = np.array(HOSPITAL_AP_TARGETS)
    train = [[] for _ in s]
    val = [[] for _ in s]
    for p in pats:
        idx = by_pat[p]
        a = d["ap"][idx].mean()
        w = (a * s + (1 - a) * (1 - s)) ** k
        h = rng.choice(len(s), p=w / w.sum())
        (val if rng.random() < VAL_FRACTION else train)[h].extend(idx)
    return [np.array(sorted(t)) for t in train], [np.array(sorted(v)) for v in val]


def hospitals(d, seed: int):
    """Patient-level hospitals: a patient goes to hospital h with probability proportional to
    (a_p s_h + (1 - a_p)(1 - s_h))^k, how well its own AP share a_p matches the hospital's target s_h, sharpened by the k
    (of SHARPNESS) whose realised AP shares come closest to the targets; then 10% of each hospital's patients are its
    validation. Deterministic in the seed."""
    keep = ~d["drop"] & ~d["test"]
    pats = np.unique(d["patient"][keep])
    by_pat = {p: [] for p in pats}
    for i in np.flatnonzero(keep):
        by_pat[d["patient"][i]].append(i)
    best = None
    for k in SHARPNESS:
        tr, va = _assign(d, pats, by_pat, k, seed)
        err = sum((d["ap"][t].mean() - s) ** 2 for t, s in zip(tr, HOSPITAL_AP_TARGETS, strict=True))
        if best is None or err < best[0]:
            best = (err, k, tr, va)
    return best[2], best[3], best[1]


# -------------------------------------------------------------------------------------------------------------- models

def resnet18_1ch() -> nn.Module:
    m = resnet18(weights=None, num_classes=1)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return m


def convnext_atto_1ch() -> nn.Module:
    """timm's convnext_atto shape (depths 2, 2, 6, 2; dims 40, 80, 160, 320) on torchvision's ConvNeXt class."""
    cfg = [CNBlockConfig(40, 80, 2), CNBlockConfig(80, 160, 2), CNBlockConfig(160, 320, 6), CNBlockConfig(320, None, 2)]
    m = ConvNeXt(cfg, stochastic_depth_prob=0.0, num_classes=1)
    m.features[0][0] = nn.Conv2d(1, 40, kernel_size=4, stride=4)
    return m


def forward_feats(m: nn.Module, x: torch.Tensor):
    """(logit, last feature map, pooled features)."""
    if hasattr(m, "layer4"):
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        pooled = torch.flatten(m.avgpool(f), 1)
        return m.fc(pooled).squeeze(1), f, pooled
    f = m.features(x)
    pooled = m.avgpool(f)
    return m.classifier(pooled).squeeze(1), f, torch.flatten(pooled, 1)


# ----------------------------------------------------------------------------------------------------- augmentation

class Batcher:
    def __init__(self, imgs: torch.Tensor, y: np.ndarray, mean: float, std: float):
        self.imgs, self.y, self.mean, self.std = imgs, torch.from_numpy(y).to(DEV), mean, std

    def get(self, idx: np.ndarray, train: bool, g: torch.Generator | None = None):
        ii = torch.from_numpy(idx).to(DEV)
        x = self.imgs[ii].unsqueeze(1).float().div_(255)
        if train:
            b = x.shape[0]
            def u(lo, hi):
                return torch.empty(b, device=DEV).uniform_(lo, hi, generator=g)
            ang = u(-7, 7) * math.pi / 180
            sc = u(0.9, 1.1)
            tx, ty = u(-0.05, 0.05), u(-0.05, 0.05)
            cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
            theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            c = u(0.9, 1.1).view(-1, 1, 1, 1)
            br = u(-0.05, 0.05).view(-1, 1, 1, 1)
            m = x.mean(dim=(2, 3), keepdim=True)
            x = (x - m) * c + m + br
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last), self.y[ii]


# ------------------------------------------------------------------------------------------------------------ training

def train(m: nn.Module, bt: Batcher, idx: np.ndarray, epochs: int, lr_at, bs: int, seed: int, step0: int = 0):
    """Plain loop; lr_at(step) gives the learning rate. Returns the last step."""
    opt = torch.optim.AdamW(m.parameters(), lr=lr_at(step0), weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    m.train()
    step = step0
    for _ in range(epochs):
        perm = rng.permutation(idx)
        for s in range(0, len(perm) - bs + 1, bs):
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            x, y = bt.get(perm[s:s + bs], True, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, _, _ = forward_feats(m, x)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
    return step


def cosine(total: int, peak: float, warm: int):
    return lambda s: peak * min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))


@torch.no_grad()
def predict(m: nn.Module, bt: Batcher, idx: np.ndarray, feats: bool = False):
    m.eval()
    out, fs = [], []
    for s in range(0, len(idx), 512):
        x, _ = bt.get(idx[s:s + 512], False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, _, pooled = forward_feats(m, x)
        out.append(logit.float().cpu())
        if feats:
            fs.append(pooled.float().cpu())
    p = torch.cat(out).numpy()
    return (p, torch.cat(fs).numpy()) if feats else p


def fedavg(make, bt, sites, rounds, local_epochs, peak, bs, seed):
    m = make().to(DEV).to(memory_format=torch.channels_last)
    sizes = np.array([len(s) for s in sites], float)
    w = sizes / sizes.sum()
    glob = {k: v.detach().clone() for k, v in m.state_dict().items()}
    for r in range(rounds):
        lr = peak * 0.5 * (1 + math.cos(math.pi * r / rounds))   # cosine over the rounds, constant within one
        acc = None
        for h, idx in enumerate(sites):
            m.load_state_dict(glob)
            train(m, bt, idx, local_epochs, lambda s, lr=lr: lr, bs, seed * 1000 + r * 10 + h)
            sd = m.state_dict()
            if acc is None:
                acc = {k: (v.float() * w[h] if v.is_floating_point() else v.clone()) for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    if v.is_floating_point():
                        acc[k] += v.float() * w[h]
                    else:
                        acc[k] = torch.maximum(acc[k], v)
        glob = {k: (acc[k].to(glob[k].dtype) if glob[k].is_floating_point() else acc[k]) for k in glob}
    m.load_state_dict(glob)
    return m


# ---------------------------------------------------------------------------------------------------------- analysis

def auc_boot(y, p, patients, n_boot=1000, seed=0):
    """AUROC and a 95% interval from resampling patients with replacement (weights per image)."""
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    auc = roc_auc_score(y, p)
    order = np.argsort(p, kind="mergesort")
    ys, ps = y[order], p[order]
    upat, inv = np.unique(patients[order], return_inverse=True)
    # ties: rank groups
    _, grp = np.unique(ps, return_inverse=True)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(upat), len(upat)), minlength=len(upat)).astype(float)
        wt = cnt[inv]
        wn = wt * (ys == 0)
        wp = wt * (ys == 1)
        # negatives strictly below each position, plus half of the tied ones
        neg_by_grp = np.bincount(grp, weights=wn)
        below = np.concatenate([[0.0], np.cumsum(neg_by_grp)[:-1]])
        num = (wp * (below[grp] + 0.5 * neg_by_grp[grp])).sum()
        den = wp.sum() * wn.sum()
        if den > 0:
            vals.append(num / den)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auroc": float(auc), "lo": float(lo), "hi": float(hi)}


def operating_point(yv, pv, sens=0.80):
    """The threshold at which the validation sensitivity first reaches `sens`."""
    pos = np.sort(pv[yv == 1])[::-1]
    k = int(math.ceil(sens * len(pos))) - 1
    return float(pos[k])


def rates(y, p, thr):
    tp = ((p >= thr) & (y == 1)).sum()
    fn = ((p < thr) & (y == 1)).sum()
    tn = ((p < thr) & (y == 0)).sum()
    fp = ((p >= thr) & (y == 0)).sum()
    return {"sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)), "n": int(len(y)), "positives": int((y == 1).sum())}


def evaluate(name, m, bt, d, test_idx, val_idx, extra=None):
    p = predict(m, bt, test_idx)
    pv = predict(m, bt, val_idx)
    y, ap, pat = d["y"][test_idx], d["ap"][test_idx], d["patient"][test_idx]
    thr = operating_point(d["y"][val_idx], pv)
    # How well the effusion score alone tells AP from PA among the true negatives, and among the positives: 0.5 when the
    # score carries no view, above it when AP images score higher whatever their label.
    neg, pos = y == 0, y == 1
    res = {"arm": name, "overall": auc_boot(y, p, pat), "PA": auc_boot(y[~ap], p[~ap], pat[~ap]), "AP": auc_boot(y[ap], p[ap], pat[ap]),
           "score_view_auroc_negatives": float(roc_auc_score(ap[neg], p[neg])), "score_view_auroc_positives": float(roc_auc_score(ap[pos], p[pos])),
           "threshold_from_validation_sens_0.80": thr,
           "at_threshold": {"overall": rates(y, p, thr), "PA": rates(y[~ap], p[~ap], thr), "AP": rates(y[ap], p[ap], thr)}}
    if extra:
        res.update(extra)
    return res, p


def view_probe(m, bt, d, train_pool, test_idx, seed):
    rng = np.random.default_rng(seed)
    ap_tr = train_pool[d["ap"][train_pool]]
    pa_tr = train_pool[~d["ap"][train_pool]]
    k = min(5000, len(ap_tr), len(pa_tr))
    sub = np.concatenate([rng.choice(ap_tr, k, replace=False), rng.choice(pa_tr, k, replace=False)])
    _, ftr = predict(m, bt, sub, feats=True)
    _, fte = predict(m, bt, test_idx, feats=True)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=1.0).fit((ftr - mu) / sd, d["ap"][sub])
    s = clf.decision_function((fte - mu) / sd)
    return float(roc_auc_score(d["ap"][test_idx], s))


def cams(m, bt, d, size=224):
    """CAM by Grad-CAM on the last feature map (exactly CAM for a pooled linear head), upsampled to 224 and scored
    against each Effusion box: the hit (the map's maximum inside the box), the share of the map's positive mass inside
    it, and the share of the image the box covers, which is what a uniform map would score."""
    idx = np.array(sorted(d["boxes"]))
    x, _ = bt.get(idx, False)
    m.eval()
    out = []
    for s in range(0, len(idx), 32):
        xb = x[s:s + 32].clone().requires_grad_(False)
        with torch.enable_grad():
            logit, f, _ = forward_feats(m, xb)
            f.retain_grad()
            logit.sum().backward()
            a = f.grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((a * f).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False).squeeze(1).detach().cpu().numpy()
        m.zero_grad(set_to_none=True)
        out.append(cam)
    return score_cams(np.concatenate(out), idx, d["boxes"], d["ap"], size)


#: Zhang et al.'s pointing game widens each box by 15 pixels of the original image (ECCV 2016, section 4.1).
POINTING_TOLERANCE_PX = 15


def score_cams(cam: np.ndarray, idx: np.ndarray, boxes: dict, ap: np.ndarray, size: int = 224) -> dict:
    """Each map (size x size) against its image's box (x, y, w, h on the 1024 x 1024 PNG): the hit (the map's maximum
    inside the box, Zhang et al.'s pointing game), with no tolerance and with theirs of 15 pixels of the 1024 px image;
    the share of the map's mass inside the box; and the share of the image the box covers, which is what a uniform map
    would score on both."""
    sc = size / 1024.0
    tol = POINTING_TOLERANCE_PX * sc
    hits, hits_tol, mass, area, views = [], [], [], [], []
    for c, i in zip(cam, idx, strict=True):
        bx, by, bw, bh = boxes[i]
        x0, y0 = int(math.floor(bx * sc)), int(math.floor(by * sc))
        x1, y1 = int(math.ceil((bx + bw) * sc)), int(math.ceil((by + bh) * sc))
        r, cc = np.unravel_index(np.argmax(c), c.shape)
        hits.append(x0 <= cc < x1 and y0 <= r < y1)
        hits_tol.append(x0 - tol <= cc < x1 + tol and y0 - tol <= r < y1 + tol)
        tot = c.sum()
        mass.append(float(c[y0:y1, x0:x1].sum() / tot) if tot > 0 else 0.0)
        area.append((x1 - x0) * (y1 - y0) / size ** 2)
        views.append("AP" if ap[i] else "PA")
    hits, hits_tol, mass, area, views = np.array(hits), np.array(hits_tol), np.array(mass), np.array(area), np.array(views)

    def summ(sel):
        if not sel.any():
            return {"n": 0}
        return {"n": int(sel.sum()), "pointing_hit_rate": float(hits[sel].mean()), "pointing_hit_rate_15px": float(hits_tol[sel].mean()),
                "mass_in_box": float(mass[sel].mean()), "box_area_share": float(area[sel].mean())}
    return {"all": summ(np.ones(len(idx), bool)), "PA": summ(views == "PA"), "AP": summ(views == "AP")}


def centre_baseline(boxes: dict, idx: np.ndarray, ap: np.ndarray, size: int = 224) -> dict:
    """Zhang et al.'s centre baseline: a map that always points at the image's centre, scored as score_cams scores."""
    cam = np.zeros((len(idx), size, size))
    cam[:, size // 2, size // 2] = 1.0
    return score_cams(cam, idx, boxes, ap, size)


# ------------------------------------------------------------------------------------------------------- environment

def environment():
    env = {"kernel": platform.release(), "python": platform.python_version(), "torch": torch.__version__,
           "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0)}
    try:
        env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        env["driver"] = None
    torch.ones(64, 64, device=DEV) @ torch.ones(64, 64, device=DEV)
    torch.cuda.synchronize()
    with open("/proc/self/maps") as f:
        libs = sorted({line.split()[-1] for line in f if "libcublas.so" in line and "Lt" not in line})
    env["cublas_loaded"] = libs
    if libs:
        try:
            lib = ctypes.CDLL(libs[0])
            v = []
            for prop in range(3):
                val = ctypes.c_int()
                lib.cublasGetProperty(prop, ctypes.byref(val))
                v.append(val.value)
            env["cublas_version"] = ".".join(map(str, v))
        except Exception as e:  # noqa: BLE001
            env["cublas_version"] = f"unread: {e}"
    return env


# --------------------------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="central,fedavg,untrained,local,pa_only,fedavg_convnext")
    ap.add_argument("--update", action="store_true", help="add these arms to an existing report.json in --out, replacing any of the same name")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1, help="training seed")
    ap.add_argument("--split-seed", type=int, default=1, help="the hospitals' assignment, fixed across training seeds")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    t0 = time.time()
    d = load(a.prep, a.raw)
    imgs = torch.from_numpy(np.load(a.prep / "images_224.npy", mmap_mode="r")[:]).to(DEV)
    tr_sites, va_sites, sharp = hospitals(d, a.split_seed)
    tr_all = np.concatenate(tr_sites)
    va_all = np.concatenate(va_sites)
    test_idx = np.flatnonzero(d["test"] & ~d["drop"])
    sample = torch.from_numpy(np.random.default_rng(0).choice(tr_all, 2000, replace=False)).to(DEV)
    mean = float(imgs[sample].float().div(255).mean())
    std = float(imgs[sample].float().div(255).std())
    bt = Batcher(imgs, d["y"], mean, std)
    site_stats = [{"hospital": h + 1, "ap_target": HOSPITAL_AP_TARGETS[h], "patients": int(len(np.unique(d["patient"][t]))),
                   "train_images": int(len(t)), "val_images": int(len(v)), "ap_share": float(d["ap"][t].mean()),
                   "effusion_prevalence": float(d["y"][t].mean())} for h, (t, v) in enumerate(zip(tr_sites, va_sites, strict=True))]
    report = {"environment": environment(), "seed": a.seed, "split_seed": a.split_seed, "sharpness": sharp, "epochs": a.epochs, "batch": a.bs, "lr": a.lr,
              "normalisation": {"mean": mean, "std": std}, "dropped_duplicates": int(d["drop"].sum()),
              "test": {"images": int(len(test_idx)), "patients": int(len(np.unique(d["patient"][test_idx]))),
                       "ap_share": float(d["ap"][test_idx].mean()), "prevalence_PA": float(d["y"][test_idx][~d["ap"][test_idx]].mean()),
                       "prevalence_AP": float(d["y"][test_idx][d["ap"][test_idx]].mean())},
              "hospitals": site_stats, "arms": []}
    if a.update:
        with open(a.out / "report.json") as f:
            old = json.load(f)
        if old["hospitals"] != site_stats or old["seed"] != a.seed or old["split_seed"] != a.split_seed:
            raise SystemExit("--update needs the same seed, split and hospitals as the report it extends")
        report = old
    yt, apt = d["y"][test_idx], d["ap"][test_idx]
    report["view_only_auroc"] = float(roc_auc_score(yt, apt.astype(float)))
    boxed = np.array(sorted(d["boxes"]))
    report["boxes"] = {"effusion_boxes": int(len(boxed)), "all_in_test_split": bool(d["test"][boxed].all()),
                       "centre_baseline": centre_baseline(d["boxes"], boxed, d["ap"])}
    print(json.dumps({k: report[k] for k in ("test", "hospitals", "view_only_auroc", "dropped_duplicates")}, indent=1), flush=True)
    arms = a.arms.split(",")
    steps_per_epoch = len(tr_all) // a.bs

    def save():
        with open(a.out / "report.json", "w") as f:
            json.dump(report, f, indent=1)

    if "central" in arms:
        t = time.time()
        m = resnet18_1ch().to(DEV).to(memory_format=torch.channels_last)
        total = a.epochs * steps_per_epoch
        train(m, bt, tr_all, a.epochs, cosine(total, a.lr, steps_per_epoch), a.bs, a.seed)
        res, _ = evaluate("centralised ResNet-18", m, bt, d, test_idx, va_all, {"train_s": time.time() - t})
        res["view_probe_auroc"] = view_probe(m, bt, d, tr_all, test_idx, a.seed)
        res["cam"] = cams(m, bt, d)
        report["arms"].append(res)
        save()
        print(json.dumps(res), flush=True)
        torch.save(m.state_dict(), a.out / "central_resnet18.pt")
    if "fedavg" in arms:
        t = time.time()
        m = fedavg(resnet18_1ch, bt, tr_sites, a.epochs, 1, a.lr, a.bs, a.seed)
        res, _ = evaluate("FedAvg ResNet-18", m, bt, d, test_idx, va_all, {"train_s": time.time() - t, "rounds": a.epochs, "local_epochs": 1})
        res["view_probe_auroc"] = view_probe(m, bt, d, tr_all, test_idx, a.seed)
        res["cam"] = cams(m, bt, d)
        report["arms"].append(res)
        save()
        print(json.dumps(res), flush=True)
        torch.save(m.state_dict(), a.out / "fedavg_resnet18.pt")
    if "untrained" in arms:
        # The baselines for the probe and the maps: a ResNet-18 at its random initialisation, seeded on its own.
        torch.manual_seed(a.seed + 100)
        untrained = resnet18_1ch().to(DEV).to(memory_format=torch.channels_last)
        report["untrained"] = {"view_probe_auroc": view_probe(untrained, bt, d, tr_all, test_idx, a.seed), "cam": cams(untrained, bt, d)}
        report.pop("untrained_view_probe_auroc", None)
        save()
        print(json.dumps(report["untrained"]), flush=True)
    if "local" in arms:
        for h, (tr, va) in enumerate(zip(tr_sites, va_sites, strict=True)):
            t = time.time()
            m = resnet18_1ch().to(DEV).to(memory_format=torch.channels_last)
            spe = len(tr) // a.bs
            train(m, bt, tr, a.epochs, cosine(a.epochs * spe, a.lr, spe), a.bs, a.seed + h + 1)
            res, _ = evaluate(f"hospital {h + 1} alone, ResNet-18", m, bt, d, test_idx, va, {"train_s": time.time() - t})
            report["arms"].append(res)
            save()
            print(json.dumps(res), flush=True)
    if "pa_only" in arms:
        t = time.time()
        pa_tr = tr_all[~d["ap"][tr_all]]
        pa_va = va_all[~d["ap"][va_all]]
        m = resnet18_1ch().to(DEV).to(memory_format=torch.channels_last)
        spe = len(pa_tr) // a.bs
        train(m, bt, pa_tr, a.epochs, cosine(a.epochs * spe, a.lr, spe), a.bs, a.seed + 10)
        res, _ = evaluate("PA images only, ResNet-18", m, bt, d, test_idx, pa_va, {"train_s": time.time() - t})
        report["arms"].append(res)
        save()
        print(json.dumps(res), flush=True)
    if "fedavg_convnext" in arms:
        t = time.time()
        m = fedavg(convnext_atto_1ch, bt, tr_sites, a.epochs, 1, a.lr, a.bs, a.seed)
        res, _ = evaluate("FedAvg ConvNeXt-atto", m, bt, d, test_idx, va_all, {"train_s": time.time() - t, "rounds": a.epochs, "local_epochs": 1,
                                                                               "params": int(sum(p.numel() for p in m.parameters()))})
        res["view_probe_auroc"] = view_probe(m, bt, d, tr_all, test_idx, a.seed)
        res["cam"] = cams(m, bt, d)
        report["arms"].append(res)
        save()
        print(json.dumps(res), flush=True)
        torch.save(m.state_dict(), a.out / "fedavg_convnext_atto.pt")
    if "rescore" in arms:
        # The saved models' maps scored again by this version of score_cams, in place of what each arm recorded.
        for name, make, fname in (("centralised ResNet-18", resnet18_1ch, "central_resnet18.pt"), ("FedAvg ResNet-18", resnet18_1ch, "fedavg_resnet18.pt"),
                                  ("FedAvg ConvNeXt-atto", convnext_atto_1ch, "fedavg_convnext_atto.pt")):
            if not (a.out / fname).exists():
                continue
            m = make().to(DEV).to(memory_format=torch.channels_last)
            m.load_state_dict(torch.load(a.out / fname, map_location=DEV))
            for arm in report["arms"]:
                if arm["arm"] == name:
                    arm["cam"] = cams(m, bt, d)
        save()
    if not a.update:
        report["total_s"] = time.time() - t0
    save()


if __name__ == "__main__":
    main()
