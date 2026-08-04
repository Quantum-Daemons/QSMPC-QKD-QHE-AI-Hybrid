#!/usr/bin/env python
"""Fine-tune a ResNet-18 on MedMNIST v2 BreastMNIST, then distil it to the FHE-feasible student.

    T2 teacher   ImageNet-pretrained ResNet-18, transfer-learned onto 1-channel ultrasound
    T1 student   1 conv block (1->4ch), avg-pool, dense head at 14x14 — what runs under real CKKS

Published baseline (MedMNIST v2, Nature Scientific Data 2022): ResNet-18 @28px, AUC 0.901 / ACC 0.863.
Google AutoML Vision reaches AUC 0.919 on the same split.

BreastMNIST is 780 breast-ultrasound images, which is small enough that training from scratch is
the wrong move and transfer learning is the right one. The first convolution is adapted from 3
channels to 1 by summing the pretrained RGB kernels rather than reinitialising: the summed kernel
has the same response to a grey image that the original had to the same image in colour, so the
pretrained edge and texture filters survive the adaptation.

Usage:
    python scripts/train/medical_breastmnist.py [--size 28] [--epochs 60]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qsh.models import datasets, distil, export, metrics, train  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402


def build_resnet18(n_classes: int = 2, *, pretrained: bool = True):
    """ResNet-18 with a 1-channel stem and a fresh head, keeping the pretrained body."""
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # 3->1 channel stem. Summing across the input-channel axis preserves the pretrained filter's
    # response to a greyscale image; re-initialising here would discard the transfer entirely.
    old = model.conv1
    stem = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                     stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        stem.weight.copy_(old.weight.sum(dim=1, keepdim=True))
    model.conv1 = stem
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def loaders(size: int, batch_size: int, seed: int):
    """BreastMNIST train/val/test loaders with light augmentation on train only."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def tensors(split):
        imgs, labels = datasets.breastmnist(split, size=size)
        x = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255.0
        x = (x - 0.5) / 0.5
        return x, torch.tensor(labels, dtype=torch.long)

    xtr, ytr = tensors("train")
    xva, yva = tensors("val")
    xte, yte = tensors("test")

    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, generator=g),
        DataLoader(TensorDataset(xva, yva), batch_size=256),
        DataLoader(TensorDataset(xte, yte), batch_size=256),
        (xtr, ytr, xte, yte),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224],
                    help="MedMNIST resolution; 28 matches the published baseline")
    ap.add_argument("--from-scratch", action="store_true",
                    help="train without ImageNet initialisation. Costs accuracy and removes a "
                         "licensing question: ImageNet's terms bind the Database to non-commercial "
                         "research and say nothing about redistributing a model trained on it, so "
                         "ImageNet-derived weights cannot be published here (DATA_LICENSES.md).")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch

    started = time.time()
    train.seed_everything(args.seed)

    print(f"loading BreastMNIST @{args.size}px ...")
    tr_dl, va_dl, te_dl, (xtr, ytr, xte, yte) = loaders(args.size, 32, args.seed)
    print(f"  train {len(tr_dl.dataset)} | val {len(va_dl.dataset)} | test {len(te_dl.dataset)}")
    print(f"  class balance (train): {np.bincount(ytr.numpy()).tolist()}")

    model = build_resnet18(pretrained=not args.from_scratch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ResNet-18 (ImageNet-pretrained, 1-channel stem): {n_params:,} parameters")

    # Class weighting: BreastMNIST is roughly 3:1 malignant:benign, and AUC is the reported metric,
    # so the loss should not be allowed to solve the task by predicting the majority class.
    counts = np.bincount(ytr.numpy(), minlength=2).astype(np.float64)
    weight = torch.tensor((counts.sum() / (2.0 * np.maximum(counts, 1))), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight.to(train._device()))

    cfg = train.TrainConfig(epochs=args.epochs, batch_size=32, lr=args.lr,
                            weight_decay=1e-4, patience=15, seed=args.seed)
    result = train.train_torch(
        "medical_fl.resnet18", model, tr_dl, va_dl,
        metric_fn=lambda y, p: metrics.auroc(y, p), metric_name="auc",
        cfg=cfg, loss_fn=loss_fn,
        out_dir=args.out or (Path.cwd() / "models" / "medical_breastmnist_resnet18"),
        extra_manifest={"dataset": "MedMNIST v2 BreastMNIST", "resolution": args.size,
                        "pretrained": "ImageNet1K_V1", "adaptation": "conv1 RGB kernels summed to 1 channel"},
    )
    print(f"  best val AUC {result.best_metric:.4f} at epoch {result.best_epoch} "
          f"({result.epochs_run} epochs run)")

    # Reload the best epoch before reporting test numbers — the in-memory model is the last epoch.
    from safetensors.torch import load_file

    model.load_state_dict(load_file(result.checkpoint))
    y_true, y_score = train.predict_loader(model, te_dl)
    y_pred = y_score.argmax(1)
    test_metrics = metrics.with_companion(y_true, y_pred, y_score, headline="auc")

    spec = get_spec("medical_fl.resnet18")
    print("\n  teacher (ResNet-18, test split)")
    for k, v in test_metrics.items():
        print(f"    {k:22s} {v:.6f}")
    print(f"    published baseline     {spec.metric.baseline}  ({spec.metric.baseline_source})")

    # ---- T1 student: the FHE-feasible model -----------------------------------
    # The student has to be the network the encrypted path actually evaluates — one conv block at
    # 14x14, run as a genuine CKKS convolution by qsh.fhe.conv. A flattened-pixel logistic would be
    # a different model, and its fidelity number would not describe the encrypted run.
    print("\n  distilling to the FHE-feasible student (14x14, 1 conv block) ...")
    from torch.utils.data import DataLoader, TensorDataset

    from qsh.ai.cnn_layers import downsample, small_medmnist_cnn

    def to14(x):
        return torch.tensor(downsample(x.numpy(), 14), dtype=torch.float32)

    teacher_train = torch.tensor(train.predict_loader(model, tr_dl)[1].argmax(1), dtype=torch.long)
    teacher_val = torch.tensor(train.predict_loader(model, va_dl)[1].argmax(1), dtype=torch.long)
    xva14 = to14(va_dl.dataset.tensors[0])

    student = small_medmnist_cnn(in_ch=1, n_classes=2, img_size=14, channels=(4,), hidden=16)
    student_params = sum(p.numel() for p in student.parameters())
    student_res = train.train_torch(
        "medical_fl.student", student,
        DataLoader(TensorDataset(to14(xtr), teacher_train), batch_size=32, shuffle=True),
        DataLoader(TensorDataset(xva14, teacher_val), batch_size=256),
        metric_fn=lambda y, p: metrics.accuracy(y, p.argmax(1)), metric_name="agreement",
        cfg=train.TrainConfig(epochs=120, batch_size=32, lr=2e-3, patience=25, seed=args.seed),
        out_dir=Path.cwd() / "models" / "medical_breastmnist_student",
        extra_manifest={"role": "FHE-feasible student distilled from medical_fl.resnet18"},
    )
    from safetensors.torch import load_file as _load

    student.load_state_dict(_load(student_res.checkpoint))
    student_pred = train.predict_loader(
        student, DataLoader(TensorDataset(to14(xte), torch.tensor(y_true)), batch_size=256))[1].argmax(1)

    fidelity = distil.measure_fidelity(
        teacher_key="medical_fl.resnet18", student_key="medical_fl.student",
        teacher_pred=y_pred, student_pred=student_pred, y_true=y_true,
        metric_fn=metrics.accuracy, metric_name="accuracy",
        student_params=student_params, teacher_params=n_params,
    )
    print(f"    agreement with teacher      {fidelity.agreement:.4f}")
    print(f"    accuracy teacher/student    {fidelity.teacher_metric:.4f} / {fidelity.student_metric:.4f}"
          f"  (delta {fidelity.metric_delta_pp:+.2f} pp)")
    print(f"    parameters                  {n_params:,} -> {fidelity.student_params}"
          f"   ({n_params/max(fidelity.student_params,1):,.0f}x smaller)")

    # ---- artefacts ------------------------------------------------------------
    out = Path(result.manifest and result.checkpoint).parent
    report = export.verify_and_quantize(model, xte.numpy(), out, kind="torch")
    if report.int8_skipped_reason:
        print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB "
              f"(int8 skipped: {report.int8_skipped_reason})")
    else:
        print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB -> int8 {report.int8_bytes/1e6:.2f} MB "
              f"({report.compression:.1f}x), int8 agreement {report.int8_agreement:.4f}")

    (out / "feature_spec.json").write_text(json.dumps({
        "input": {"shape": [1, 1, args.size, args.size], "dtype": "float32",
                  "preprocessing": "pixel/255 then (x-0.5)/0.5"},
        "classes": ["benign/normal", "malignant"],
    }, indent=2) + "\n")

    train.record_metrics("medical_fl.resnet18", {
        **test_metrics,
        "dataset": spec.dataset.name,
        "resolution": args.size,
        "n_parameters": n_params,
        "best_val_auc": round(result.best_metric, 6),
        # Recorded from what actually ran. This was a fixed string, so a --from-scratch run still
        # claimed ImageNet initialisation -- the exact provenance the licence analysis turns on.
        "transfer_learning": (
            "none — trained from random initialisation, so the weights are not ImageNet-derived "
            "and carry no ImageNet terms (see DATA_LICENSES.md)"
            if args.from_scratch
            else "ImageNet1K_V1 ResNet-18, conv1 summed to 1 channel, fresh fc head"
        ),
        "git_sha": train.git_sha(),
        "export": {"fp32_bytes": report.fp32_bytes, "int8_bytes": report.int8_bytes,
                   "int8_agreement": report.int8_agreement, "opset": report.opset},
        "wall_clock_s": round(time.time() - started, 1),
    })
    train.record_metrics("medical_fl.student", fidelity.as_metrics())
    print(f"\n  wrote {out} and models/metrics.json in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
