#!/usr/bin/env python3
"""What one hospital's gradient would show the aggregator: gradient inversion against the chest X-ray federation.

On the owner's approval of 2026-11-23 (a GPU job; references/pets-evaluation-2026-11.md). FedAvg sends each hospital's
updated weights to the aggregator, and in this offline study (scripts/train/medical_cxr_fl.py) they travel in the clear;
the medical page's heart federation, by contrast, sums its updates under pairwise masks. Inverting Gradients (Geiping,
Bauermeister, Droege and Moeller, NeurIPS 2020, arXiv:2003.14053) recovers training images from a gradient: start from
noise and move the image until its gradient matches the observed one in cosine distance, with total variation as the
prior. Its authors' code is not used; this is a reimplementation.

This is the attack's worst case for the defender: ONE gradient step, on B films, at the published weights, the network
in training mode as FedAvg's local step runs it (batch statistics), with no augmentation and the labels known. A real
FedAvg update is the end of several local epochs over thousands of films with augmentation, which the paper finds far
harder to invert; that is stated beside the figures, not measured here.

The weights are `medical_fl.cxr_fedavg`'s safetensors export, tied to the published graph before anything else: on the
16 films of the published `eval_sample.json` they must give every row's class and its margin within 0.03 of the
published int8 graph's (the int8 graph's own tolerance is 0.001 against itself). The victims are those 16 films, from
NIH ChestX-ray14's official test split (public; "no restrictions on use"). Only aggregate similarity figures are
written; no image is.

    python scripts/audit_gradient_inversion_cxr.py --weights <model.safetensors> \
        -o docs/measurements/gradient-inversion-cxr-<date>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
MODEL_DIR = REPO / "models" / "medical_cxr14_fedavg_resnet18"


def _published(main_models: Path) -> tuple[dict, dict]:
    sample = json.loads((main_models / "eval_sample.json").read_text(encoding="utf-8"))
    spec = json.loads((main_models / "feature_spec.json").read_text(encoding="utf-8"))
    return sample, spec


def load_model(weights: Path, spec: dict):
    import torch
    from safetensors.torch import load_file
    from torchvision.models import resnet18

    net = resnet18(weights=None, num_classes=1)
    net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.load_state_dict(load_file(str(weights)))
    norm = spec["input"]["normalisation_in_graph"]
    return net, float(norm["mean"]), float(norm["std"])


def tie_to_published(net, mean: float, std: float, sample: dict) -> dict:
    """Every row's class, and its margin close to the published int8 graph's."""
    import torch

    x = torch.tensor(np.asarray(sample["samples"], dtype=np.float32).reshape(-1, 1, 224, 224) / 255.0)
    net.eval()
    with torch.no_grad():
        z = net((x - mean) / std).squeeze(1).double()
    p = torch.sigmoid(z).numpy()
    thr = float(sample["expected"]["decision_threshold"])
    rows = sample["expected"]["per_row"]
    margin = p - thr
    predicted = (margin >= 0).astype(int).tolist()
    diff = float(np.max(np.abs(margin - np.asarray(rows["margin"], dtype=float))))
    ok = predicted == list(rows["predicted"]) and diff <= 0.03
    return {"rows": len(predicted), "classes_equal": predicted == list(rows["predicted"]),
            "max_abs_margin_difference_vs_int8_graph": round(diff, 5), "tied": bool(ok)}


def invert(net, mean, std, x_true, y_true, iters: int, tv: float, seed: int, dev, lr: float = 0.1, signed: bool = False):
    """Geiping et al.'s attack: Adam on the image, 1 - cosine(grad', grad) over all parameters, plus tv * TV(image)."""
    import torch
    import torch.nn.functional as F

    params = [p for p in net.parameters() if p.requires_grad]

    def grads(x, create_graph):
        net.zero_grad(set_to_none=True)
        z = net((x - mean) / std).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(z, y_true)
        return torch.autograd.grad(loss, params, create_graph=create_graph)

    net.train()
    target = [g.detach() for g in grads(x_true, False)]
    tnorm = torch.sqrt(sum((g * g).sum() for g in target))
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.rand(x_true.shape, generator=gen, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(iters * f) for f in (3 / 8, 5 / 8, 7 / 8)], gamma=0.1)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        g = grads(x, True)
        dot = sum((a * b).sum() for a, b in zip(g, target, strict=True))
        gnorm = torch.sqrt(sum((a * a).sum() for a in g))
        tvl = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        loss = 1 - dot / (gnorm * tnorm + 1e-12) + tv * tvl
        loss.backward()
        if signed:
            x.grad.sign_()
        opt.step()
        sched.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return x.detach(), float(loss.detach())


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    return (float(peak_signal_noise_ratio(a, b, data_range=1.0)), float(structural_similarity(a, b, data_range=1.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--published", type=Path, default=MODEL_DIR, help="the folder holding eval_sample.json and feature_spec.json")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--tv", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--signed", action="store_true", help="step on the sign of the image gradient, as the authors do")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--films", type=int, default=None, help="use only the first N films (a smoke run)")
    ap.add_argument("--preview", type=Path, default=None, help="optional: a local PNG grid for a human to look at; never committed")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sample, spec = _published(a.published)
    net, mean, std = load_model(a.weights, spec)
    tie = tie_to_published(net, mean, std, sample)
    if not tie["tied"]:
        raise SystemExit(f"these weights are not the published model's: {tie}")
    net = net.to(dev)
    films = np.asarray(sample["samples"], dtype=np.float32).reshape(-1, 1, 224, 224) / 255.0
    labels = np.asarray(sample["labels"], dtype=np.float32)
    if a.films is not None:
        films, labels = films[:a.films], labels[:a.films]
    n = len(films)
    flat_mean = np.full((224, 224), float(films.mean()), dtype=np.float32)
    t0 = time.time()
    results, previews = {}, []
    for b in a.batches:
        rows = []
        for start in range(0, n - b + 1, b):
            idx = list(range(start, start + b))
            xt = torch.tensor(films[idx], device=dev)
            yt = torch.tensor(labels[idx], device=dev)
            xr, final = invert(net, mean, std, xt, yt, a.iters, a.tv, seed=1000 * b + start, dev=dev, lr=a.lr, signed=a.signed)
            rec = xr.cpu().numpy()
            # Batch order is not observable from a summed gradient: match each reconstruction to its best true film.
            for j, i in enumerate(idx):
                sims = [similarity(rec[k, 0], films[i, 0]) for k in range(b)]
                psnr, ssim = max(sims, key=lambda s: s[1])
                others = [similarity(films[o, 0], films[i, 0])[1] for o in range(n) if o != i]
                rows.append({"psnr": psnr, "ssim": ssim, "final_loss": final,
                             "baseline_flat_mean_psnr": similarity(flat_mean, films[i, 0])[0],
                             "baseline_flat_mean_ssim": similarity(flat_mean, films[i, 0])[1],
                             "baseline_best_other_film_ssim": max(others) if others else float("nan")})
                if a.preview is not None and j < 2 and len(previews) < 8:
                    previews.append((films[i, 0], rec[min(j, b - 1), 0]))
        ps, ss = np.array([r["psnr"] for r in rows]), np.array([r["ssim"] for r in rows])
        results[f"batch_{b}"] = {
            "films": len(rows), "gradients": len(rows) // b,
            "psnr_median": round(float(np.median(ps)), 2), "psnr_min": round(float(ps.min()), 2),
            "psnr_max": round(float(ps.max()), 2),
            "ssim_median": round(float(np.median(ss)), 4), "ssim_max": round(float(ss.max()), 4),
            "baseline_flat_mean_psnr_median": round(float(np.median([r["baseline_flat_mean_psnr"] for r in rows])), 2),
            "baseline_flat_mean_ssim_median": round(float(np.median([r["baseline_flat_mean_ssim"] for r in rows])), 4),
            "baseline_best_other_film_ssim_median": round(float(np.median([r["baseline_best_other_film_ssim"] for r in rows])), 4),
            "films_above_best_other_film_ssim": int(sum(r["ssim"] > r["baseline_best_other_film_ssim"] for r in rows)),
        }
        print(f"batch {b}:", json.dumps(results[f"batch_{b}"]), flush=True)
    if a.preview is not None and previews:
        from PIL import Image

        grid = np.concatenate([np.concatenate(p, axis=1) for p in previews], axis=0)
        Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8)).save(a.preview)
    record = {
        "what": "gradient inversion of the chest X-ray federation's model from one gradient step, the attack's worst case",
        "attack": "Geiping, Bauermeister, Droege and Moeller, NeurIPS 2020 (arXiv:2003.14053); reimplemented here",
        "model": {"key": "medical_fl.cxr_fedavg", "weights_sha256": hashlib.sha256(a.weights.read_bytes()).hexdigest(),
                  "tied_to_the_published_graph": tie},
        "victims": f"the {n} films of the published eval_sample.json, NIH ChestX-ray14's official test split (public)",
        "settings": {"iterations": a.iters, "tv_weight": a.tv, "signed_gradients": a.signed,
                     "optimiser": f"Adam, lr {a.lr} on pixels in [0, 1], stepped down 10x at 3/8, 5/8 and 7/8",
                     "loss": "1 - cosine over all parameters' gradients + tv x total variation",
                     "network_mode": "training (batch statistics), as FedAvg's local step", "labels": "known",
                     "augmentation": "none", "device": str(dev)},
        "results": results,
        "reading": "SSIM above the best other film's means the reconstruction resembles THIS film more than any other film in "
                   "the set does; PSNR and SSIM of a flat image at the mean grey are what knowing nothing gives. One step on "
                   "few films is the worst case; FedAvg's update after local epochs over thousands of augmented films is "
                   "much harder to invert, and secure aggregation leaves the aggregator only the sum of the hospitals' updates.",
        "seconds": round(time.time() - t0, 1),
        "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                              text=True).stdout.strip(),
                        "python": platform.python_version(), "torch": torch.__version__, "date": time.strftime("%Y-%m-%d")},
    }
    Path(a.output).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
