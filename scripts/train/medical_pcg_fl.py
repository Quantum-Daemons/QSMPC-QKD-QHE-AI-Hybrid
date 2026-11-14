#!/usr/bin/env python3
"""A heart-sound federation on the PhysioNet/CinC Challenge 2016 data, each source held out in turn (Round 86).

The six training databases, a to f, were collected by different teams with different stethoscopes and populations, and
their share of abnormal recordings runs from 7% to 77%. Each is held out once: a small CNN on log-mel windows is trained
on the other five, pooled and under FedAvg with each source as a site, and scored on the held-out source, recording by
recording (the mean of its windows' logits). Three training seeds. The Challenge's own example algorithm, scored the same
way by database, ranged from 0.47 to 0.86 in its mean of sensitivity and specificity (Clifford et al., CinC 2016).

Usage: python -I scripts/train/medical_pcg_fl.py --prep <prepared dir> --out <dir> [--seeds 1,2,3] [--epochs 12]

The prepared dir is what scripts/data/prepare_physionet2016.py writes. The run needs a CUDA GPU; the functions the tests
exercise (the mel filterbank, the windows, the per-recording scores, the AUROC and its interval, the balanced accuracy)
need neither the data nor a GPU.
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
from sklearn.metrics import roc_auc_score

DEV = torch.device("cuda")
RATE = 2000
SOURCES = "abcdef"
WINDOW = 4 * RATE          # 4 s, about four cardiac cycles at rest
TRAIN_HOP = 2 * RATE       # training windows start every 2 s, each jittered within it
EVAL_HOP = 1 * RATE        # scored windows start every 1 s, the last one ending at the recording's end
N_FFT, STFT_HOP, N_MELS, F_MIN, F_MAX = 256, 20, 40, 25.0, 500.0


# ---------------------------------------------------------------------------------------------------------------- data

def load(prep: Path) -> dict:
    with open(prep / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {"source": np.array([r["source"] for r in rows]), "record": [r["record"] for r in rows],
            "y": np.array([int(r["abnormal"]) for r in rows], dtype=np.float32), "good": np.array([r["good_quality"] == "1" for r in rows]),
            "offset": np.array([int(r["offset"]) for r in rows]), "length": np.array([int(r["length"]) for r in rows])}


def window_starts(length: int, window: int, hop: int) -> np.ndarray:
    """Starts every `hop` samples, and one more so that the last window ends at the end; a recording shorter than a
    window gets one start at 0 (the caller pads it)."""
    if length <= window:
        return np.array([0])
    starts = np.arange(0, length - window + 1, hop)
    if starts[-1] != length - window:
        starts = np.append(starts, length - window)
    return starts


def training_windows(d: dict, recs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(recording, start) for one epoch: a window every TRAIN_HOP samples, each start jittered by up to the hop."""
    out_r, out_s = [], []
    for i in recs:
        n = d["length"][i]
        base = window_starts(n, WINDOW, TRAIN_HOP)
        jit = rng.integers(0, TRAIN_HOP, len(base))
        s = np.minimum(base + jit, max(n - WINDOW, 0))
        out_r.append(np.full(len(s), i))
        out_s.append(s)
    return np.concatenate(out_r), np.concatenate(out_s)


# ------------------------------------------------------------------------------------------------------------ features

def mel_filterbank(n_fft: int = N_FFT, n_mels: int = N_MELS, rate: int = RATE, f_min: float = F_MIN, f_max: float = F_MAX) -> np.ndarray:
    """Triangular filters equally spaced on the HTK mel scale, m = 2595 log10(1 + f / 700), over the rfft bins."""
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # noqa: E731
    edges = hz(np.linspace(mel(f_min), mel(f_max), n_mels + 2))
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for k in range(n_mels):
        lo, mid, hi = edges[k], edges[k + 1], edges[k + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[k] = np.maximum(0.0, np.minimum(up, down))
    return fb


class Features(nn.Module):
    """Log-mel of each window, standardised within the window, so the sources' different gains do not reach the model."""

    def __init__(self):
        super().__init__()
        self.register_buffer("fb", torch.tensor(mel_filterbank(), dtype=torch.float32))
        self.register_buffer("win", torch.hann_window(N_FFT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(x, N_FFT, STFT_HOP, window=self.win, return_complex=True).abs().pow(2)
        m = torch.log(torch.einsum("mf,bft->bmt", self.fb, spec) + 1e-6)
        m = (m - m.mean(dim=(1, 2), keepdim=True)) / (m.std(dim=(1, 2), keepdim=True) + 1e-5)
        return m.unsqueeze(1)


class SmallCnn(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        layers, c = [], 1
        for k in range(4):
            out = width * 2 ** k
            layers += [nn.Conv2d(c, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = out
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c, 1))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(m).mean(dim=(2, 3))).squeeze(1)


def batch(sig: torch.Tensor, d: dict, recs: np.ndarray, starts: np.ndarray) -> torch.Tensor:
    off = torch.from_numpy(d["offset"][recs] + starts).to(DEV)
    idx = off.unsqueeze(1) + torch.arange(WINDOW, device=DEV)
    lim = torch.from_numpy(d["offset"][recs] + d["length"][recs]).to(DEV).unsqueeze(1)
    x = sig[idx.clamp(max=sig.numel() - 1)].float()
    return torch.where(idx < lim, x, torch.zeros_like(x))     # zero past a short recording's end


# ------------------------------------------------------------------------------------------------------------ training

def augment(m: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """One time mask (up to 0.4 s) and one frequency mask (up to 4 bands) per window."""
    b, _, nm, nt = m.shape
    t0 = torch.randint(0, nt, (b,), device=DEV, generator=g)
    tw = torch.randint(0, 21, (b,), device=DEV, generator=g)
    f0 = torch.randint(0, nm, (b,), device=DEV, generator=g)
    fw = torch.randint(0, 5, (b,), device=DEV, generator=g)
    tt = torch.arange(nt, device=DEV).view(1, -1)
    ff = torch.arange(nm, device=DEV).view(1, -1)
    tmask = (tt >= t0.view(-1, 1)) & (tt < (t0 + tw).view(-1, 1))
    fmask = (ff >= f0.view(-1, 1)) & (ff < (f0 + fw).view(-1, 1))
    keep = ~(tmask.view(b, 1, 1, nt) | fmask.view(b, 1, nm, 1))
    return m * keep


def train(model, feats, sig, d, recs, epochs, lr_of_epoch, bs, seed):
    y = torch.from_numpy(d["y"]).to(DEV)
    pos = float(d["y"][recs].sum())
    pos_weight = torch.tensor((len(recs) - pos) / max(pos, 1.0), device=DEV)
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_of_epoch(0), weight_decay=1e-4)
    model.train()
    for e in range(epochs):
        for pg in opt.param_groups:
            pg["lr"] = lr_of_epoch(e)
        wr, ws = training_windows(d, recs, rng)
        perm = rng.permutation(len(wr))
        for s in range(0, len(perm), bs):
            k = perm[s:s + bs]
            with torch.no_grad():
                m = augment(feats(batch(sig, d, wr[k], ws[k])), g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(m)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y[torch.from_numpy(wr[k]).to(DEV)], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


@torch.no_grad()
def record_logits(model, feats, sig, d, recs) -> np.ndarray:
    model.eval()
    rr, ss = [], []
    for i in recs:
        s = window_starts(int(d["length"][i]), WINDOW, EVAL_HOP)
        rr.append(np.full(len(s), i))
        ss.append(s)
    rr, ss = np.concatenate(rr), np.concatenate(ss)
    out = []
    for s in range(0, len(rr), 512):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(feats(batch(sig, d, rr[s:s + 512], ss[s:s + 512]))).float().cpu())
    return per_record_mean(np.concatenate([o.numpy() for o in out]), rr, recs)


def per_record_mean(window_logits: np.ndarray, window_rec: np.ndarray, recs: np.ndarray) -> np.ndarray:
    """Each recording's score: the mean of its windows' logits, in the order of `recs`."""
    pos = {r: k for k, r in enumerate(recs)}
    k = np.array([pos[r] for r in window_rec])
    return np.bincount(k, weights=window_logits, minlength=len(recs)) / np.bincount(k, minlength=len(recs))


def fedavg(make, feats, sig, d, sites, rounds, peak, bs, seed):
    model = make().to(DEV)
    w = np.array([len(s) for s in sites], float)
    w /= w.sum()
    glob = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for r in range(rounds):
        lr = peak * 0.5 * (1 + math.cos(math.pi * r / rounds))
        acc = None
        for h, recs in enumerate(sites):
            model.load_state_dict(glob)
            train(model, feats, sig, d, recs, 1, lambda e, lr=lr: lr, bs, seed * 1000 + r * 10 + h)
            sd = model.state_dict()
            if acc is None:
                acc = {k: (v.float() * w[h] if v.is_floating_point() else v.clone()) for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    if v.is_floating_point():
                        acc[k] += v.float() * w[h]
                    else:
                        acc[k] = torch.maximum(acc[k], v)
        glob = {k: (acc[k].to(glob[k].dtype) if glob[k].is_floating_point() else acc[k]) for k in glob}
    model.load_state_dict(glob)
    return model


# ---------------------------------------------------------------------------------------------------------- analysis

def auc_boot(y, p, n_boot=1000, seed=0):
    """AUROC and a 95% interval from resampling recordings (the data name no patients, so this is narrower than a
    patient-level interval where a patient has several recordings)."""
    y, p = np.asarray(y), np.asarray(p)
    if y.min() == y.max():
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        if y[k].min() != y[k].max():
            vals.append(roc_auc_score(y[k], p[k]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auroc": float(roc_auc_score(y, p)), "lo": float(lo), "hi": float(hi)}


def macc(y, logit, thr: float = 0.0) -> dict:
    """The Challenge's mean of sensitivity and specificity, at a logit threshold (0 with a class-balanced loss)."""
    y = np.asarray(y)
    se = float(((logit >= thr) & (y == 1)).sum() / max((y == 1).sum(), 1))
    sp = float(((logit < thr) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return {"sensitivity": se, "specificity": sp, "macc": (se + sp) / 2}


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
    env["cublas_loaded"] = [Path(p).name for p in libs]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    d = load(a.prep)
    sig = torch.from_numpy(np.load(a.prep / "signals.npy")).to(DEV)
    feats = Features().to(DEV)
    sources = {s: np.flatnonzero(d["source"] == s) for s in SOURCES}
    report = {"environment": environment(), "epochs": a.epochs, "batch": a.bs, "lr": a.lr,
              "window_s": WINDOW / RATE, "features": {"n_fft": N_FFT, "hop": STFT_HOP, "n_mels": N_MELS, "f_min": F_MIN, "f_max": F_MAX},
              "sources": {s: {"recordings": int(len(r)), "abnormal": int(d["y"][r].sum()), "good_quality": int(d["good"][r].sum()),
                              "hours": float(d["length"][r].sum() / RATE / 3600)} for s, r in sources.items()},
              "params": int(sum(p.numel() for p in SmallCnn().parameters())), "runs": []}
    for seed in (int(x) for x in a.seeds.split(",")):
        # The optimistic reference: recordings split at random, 80/20, so one patient's recordings can sit on both sides.
        perm = np.random.default_rng(seed).permutation(len(d["y"]))
        test, train_recs = np.sort(perm[: len(perm) // 5]), np.sort(perm[len(perm) // 5:])
        torch.manual_seed(seed)
        t = time.time()
        model = SmallCnn().to(DEV)
        train(model, feats, sig, d, train_recs, a.epochs, lambda e: a.lr * 0.5 * (1 + math.cos(math.pi * e / a.epochs)), a.bs, seed)
        logit = record_logits(model, feats, sig, d, test)
        run = {"seed": seed, "held_out": "random 20% of recordings", "arm": "pooled", "train_s": time.time() - t,
               "auroc": auc_boot(d["y"][test], logit), "auroc_good_quality": auc_boot(d["y"][test][d["good"][test]], logit[d["good"][test]]),
               **macc(d["y"][test], logit)}
        report["runs"].append(run)
        print(json.dumps(run), flush=True)
        for held in SOURCES:
            test = sources[held]
            sites = [sources[s] for s in SOURCES if s != held]
            pooled = np.concatenate(sites)
            for arm in ("pooled", "fedavg"):
                torch.manual_seed(seed)
                t = time.time()
                if arm == "pooled":
                    model = SmallCnn().to(DEV)
                    train(model, feats, sig, d, pooled, a.epochs,
                          lambda e: a.lr * 0.5 * (1 + math.cos(math.pi * e / a.epochs)), a.bs, seed)
                else:
                    model = fedavg(SmallCnn, feats, sig, d, sites, a.epochs, a.lr, a.bs, seed)
                logit = record_logits(model, feats, sig, d, test)
                y, good = d["y"][test], d["good"][test]
                run = {"seed": seed, "held_out": held, "arm": arm, "train_s": time.time() - t, "auroc": auc_boot(y, logit),
                       "auroc_good_quality": auc_boot(y[good], logit[good]), **macc(y, logit)}
                report["runs"].append(run)
                print(json.dumps(run), flush=True)
                with open(a.out / "report.json", "w") as f:
                    json.dump(report, f, indent=1)
    report["total_s"] = time.time() - t0
    with open(a.out / "report.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
