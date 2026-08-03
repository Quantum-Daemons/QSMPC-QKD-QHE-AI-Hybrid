"""A real training loop with artefacts that outlive the process.

What was here before (``qsh.ai.train.train_model``) was 86 lines with no scheduler, no validation
loop, no early stopping, no checkpointing and no logging, and it was imported by exactly one test.
Every model in the project was retrained from scratch on every pipeline run and discarded.

This module is the replacement. It is deliberately small but it does the things that make a training
run reproducible and a resulting artefact trustworthy:

  * seeded end to end (python / numpy / torch / cudnn) and the seed is recorded;
  * AMP on CUDA (bf16 on the Blackwell box), cosine schedule with warmup, gradient clipping;
  * a real validation split with early stopping on the *task* metric, not on loss;
  * best-epoch checkpointing to safetensors;
  * a manifest recording the git sha, the dataset digest, the resolved hyperparameters, the achieved
    metric and the wall-clock — so an artefact can always be traced back to what produced it.

The manifest is what lets the UI print a number and say where it came from. A model with no manifest
is a model whose metrics we do not print.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .registry import METRICS_PATH


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    patience: int = 8              # epochs without task-metric improvement before stopping
    seed: int = 42
    amp: bool = True
    num_workers: int = 4
    min_delta: float = 1e-4


@dataclass
class TrainResult:
    key: str
    best_metric: float
    best_epoch: int
    epochs_run: int
    history: list = field(default_factory=list)
    checkpoint: str = ""
    manifest: dict = field(default_factory=dict)


def seed_everything(seed: int) -> None:
    """Seed every RNG that can affect a run, and make cuDNN deterministic."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_sha() -> str:
    """Short git sha of the working tree, suffixed ``-dirty`` when uncommitted changes exist."""
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "-C", str(root), "diff", "--quiet"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def digest_array(*arrays: np.ndarray) -> str:
    """A stable digest of the training data, so a metric can be tied to the exact inputs."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes()[: 8 << 20])   # first 8 MiB is plenty to detect a different corpus
    return h.hexdigest()[:16]


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_dtype():
    """bf16 where supported (Blackwell), else fp16. bf16 avoids loss scaling entirely."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train_torch(
    key: str,
    model,
    train_loader,
    val_loader,
    *,
    metric_fn: Callable,
    metric_name: str,
    cfg: TrainConfig | None = None,
    out_dir: Path | None = None,
    loss_fn=None,
    extra_manifest: dict | None = None,
) -> TrainResult:
    """Train ``model``, keep the best epoch by ``metric_fn``, and write checkpoint + manifest.

    ``metric_fn(y_true, y_score) -> float`` is the *task* metric from the registry, not accuracy.
    Selecting on the task metric matters here: four of the five industries in this project have a
    documented failure mode where loss and accuracy both look healthy while the class that matters
    is not being learned at all.
    """
    import torch
    from torch import nn

    cfg = cfg or TrainConfig()
    seed_everything(cfg.seed)
    dev = _device()
    model = model.to(dev)
    loss_fn = loss_fn or nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = bool(cfg.amp and dev.type == "cuda")
    amp_dtype = _amp_dtype() if use_amp else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    out_dir = Path(out_dir or (Path.cwd() / "models" / key.replace(".", "_")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.safetensors"

    best, best_epoch, since_improve, history = -np.inf, -1, 0, []
    started = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            sched.step()
            running += float(loss.detach()) * xb.size(0)
            seen += xb.size(0)

        y_true, y_score = predict_loader(model, val_loader, dev)
        metric = float(metric_fn(y_true, y_score))
        history.append({"epoch": epoch, "train_loss": running / max(1, seen), metric_name: metric})

        if metric > best + cfg.min_delta:
            best, best_epoch, since_improve = metric, epoch, 0
            _save_safetensors(model, ckpt)
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                break

    elapsed = time.time() - started
    manifest = {
        "key": key,
        "metric_name": metric_name,
        "best_metric": best,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "wall_clock_s": round(elapsed, 1),
        "git_sha": git_sha(),
        "seed": cfg.seed,
        "hyperparameters": asdict(cfg),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "device": str(dev),
        "torch": __import__("torch").__version__,
        "python": platform.python_version(),
        "checkpoint": str(ckpt.name),
        **(extra_manifest or {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return TrainResult(key=key, best_metric=best, best_epoch=best_epoch, epochs_run=len(history),
                       history=history, checkpoint=str(ckpt), manifest=manifest)


def predict_loader(model, loader, dev=None):
    """Run ``model`` over ``loader`` and return ``(y_true, y_score)`` as numpy arrays."""
    import torch

    dev = dev or _device()
    model.eval().to(dev)
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(dev))
            ps.append(torch.softmax(out.float(), dim=1).cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _save_safetensors(model, path: Path) -> None:
    from safetensors.torch import save_file

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path))


def record_metrics(key: str, payload: dict) -> None:
    """Merge one model's achieved metrics into ``models/metrics.json``.

    This file is machine-written and read back by the registry. Nothing else may write it: a
    hand-edited metric is indistinguishable from a measured one once it is in the UI.
    """
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = json.loads(METRICS_PATH.read_text())
    except FileNotFoundError:
        current = {}
    current[key] = payload
    METRICS_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    from .registry import _achieved

    _achieved.cache_clear()
