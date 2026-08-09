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

#: Architectures this script can train, and the registry key each one records into.
#: Both are MedMNIST v2 baselines at 28px, which is the whole point of offering the second one:
#: the published table gives ResNet-18 AUC 0.901 and ResNet-50 AUC 0.857 under the same 100-epoch
#: Adam regime with no ImageNet initialisation, so the pair is comparable to each other AND to the
#: baseline. Architectures without a published from-scratch number on this corpus are deliberately
#: absent -- MedMNIST+ benchmarks AlexNet, VGG16, DenseNet-121 and EfficientNet-B4 on BreastMNIST but
#: only ImageNet1k-pretrained, which is the dependency this project dropped for licence reasons.
ARCHS = {"resnet18": "medical_fl.resnet18", "resnet50": "medical_fl.resnet50"}


def build_resnet(arch: str, n_classes: int = 2, *, pretrained: bool = True):
    """A torchvision ResNet with a 1-channel stem and a fresh head.

    The stem rewrite is the only architecture-specific step and it is identical for both depths:
    torchvision names the first convolution ``conv1`` and the classifier ``fc`` in every ResNet.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

    ctor, wts = {"resnet18": (resnet18, ResNet18_Weights),
                 "resnet50": (resnet50, ResNet50_Weights)}[arch]
    model = ctor(weights=wts.IMAGENET1K_V1 if pretrained else None)

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


#: MedMNIST BreastMNIST label order, from the package's own INFO: {0: malignant, 1: "normal,
#: benign"}. Named because this project had them inverted in four places at once - the published
#: weights.json, the browser fixture, the Space, and the use-case sample selection - so the demo
#: rendered the OPPOSITE diagnosis to what the model meant.
MALIGNANT, BENIGN = 0, 1


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
    ap.add_argument("--arch", choices=sorted(ARCHS), default="resnet18",
                    help="MedMNIST v2 publishes a 28px baseline for both: ResNet-18 AUC 0.901, "
                         "ResNet-50 AUC 0.857. Training the deeper one is expected to score LOWER "
                         "on this 780-image corpus, which is the published finding, not a failure.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    key = ARCHS[args.arch]
    artifact_dir = args.out or (Path.cwd() / "models" / f"medical_breastmnist_{args.arch}")

    import torch

    started = time.time()
    train.seed_everything(args.seed)

    print(f"loading BreastMNIST @{args.size}px ...")
    tr_dl, va_dl, te_dl, (xtr, ytr, xte, yte) = loaders(args.size, 32, args.seed)
    print(f"  train {len(tr_dl.dataset)} | val {len(va_dl.dataset)} | test {len(te_dl.dataset)}")
    print(f"  class balance (train): {np.bincount(ytr.numpy()).tolist()}")

    model = build_resnet(args.arch, pretrained=not args.from_scratch)
    n_params = sum(p.numel() for p in model.parameters())
    init = "random init" if args.from_scratch else "ImageNet1K_V1"
    print(f"  {args.arch} ({init}, 1-channel stem): {n_params:,} parameters")

    # Class weighting: BreastMNIST is roughly 2.71:1 normal/benign:malignant (MedMNIST labels the
    # classes {0: malignant, 1: "normal, benign"}), and AUC is the reported metric,
    # so the loss should not be allowed to solve the task by predicting the majority class.
    counts = np.bincount(ytr.numpy(), minlength=2).astype(np.float64)
    weight = torch.tensor((counts.sum() / (2.0 * np.maximum(counts, 1))), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight.to(train._device()))

    cfg = train.TrainConfig(epochs=args.epochs, batch_size=32, lr=args.lr,
                            weight_decay=1e-4, patience=15, seed=args.seed)
    result = train.train_torch(
        key, model, tr_dl, va_dl,
        metric_fn=lambda y, p: metrics.auroc(y, p), metric_name="auc",
        cfg=cfg, loss_fn=loss_fn,
        out_dir=artifact_dir,
        extra_manifest={"dataset": "MedMNIST v2 BreastMNIST", "resolution": args.size,
                        "architecture": args.arch,
                        "pretrained": "none" if args.from_scratch else "ImageNet1K_V1",
                        "adaptation": "conv1 RGB kernels summed to 1 channel"},
    )
    print(f"  best val AUC {result.best_metric:.4f} at epoch {result.best_epoch} "
          f"({result.epochs_run} epochs run)")

    # Reload the best epoch before reporting test numbers — the in-memory model is the last epoch.
    from safetensors.torch import load_file

    model.load_state_dict(load_file(result.checkpoint))
    y_true, y_score = train.predict_loader(model, te_dl)
    y_pred = y_score.argmax(1)
    test_metrics = metrics.with_companion(y_true, y_pred, y_score, headline="auc")

    spec = get_spec(key)
    print(f"\n  teacher ({args.arch}, test split)")
    for k, v in test_metrics.items():
        print(f"    {k:22s} {v:.6f}")
    print(f"    published baseline     {spec.metric.baseline}  ({spec.metric.baseline_source})")

    # Only the browser-tier teacher owns a student. `medical_fl.student` is published and pinned and
    # its recorded fidelity names resnet18 as its teacher, so distilling a second architecture into
    # the same key would replace a published artefact with a different lineage while the metrics
    # still described the old one.
    distils_student = key == "medical_fl.resnet18"

    # ---- artefacts ------------------------------------------------------------
    out = Path(result.manifest and result.checkpoint).parent
    # int8 is required for the browser-tier teacher and optional for a comparison teacher. The
    # gate itself is unchanged in both cases: int8 is built and measured either way, and a graph
    # that misses the bar is withheld rather than published.
    report = export.verify_and_quantize(model, xte.numpy(), out, kind="torch",
                                        int8_required=distils_student)
    if report.int8_skipped_reason:
        print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB "
              f"(int8 skipped: {report.int8_skipped_reason})")
    else:
        print(f"\n  ONNX fp32 {report.fp32_bytes/1e6:.2f} MB -> int8 {report.int8_bytes/1e6:.2f} MB "
              f"({report.compression:.1f}x), int8 agreement {report.int8_agreement:.4f}")

    (out / "feature_spec.json").write_text(json.dumps({
        "input": {"shape": [1, 1, args.size, args.size], "dtype": "float32",
                  "preprocessing": "pixel/255 then (x-0.5)/0.5"},
        # MedMNIST BreastMNIST: {0: malignant, 1: "normal, benign"}. Index order matters - the
        # browser renders classes[argmax], so an inverted list reports the OPPOSITE diagnosis.
        "classes": ["malignant", "normal/benign"],
    }, indent=2) + "\n")

    train.record_metrics(key, {
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
    if not distils_student:
        print(f"\n  no student distilled: medical_fl.student belongs to medical_fl.resnet18, and "
              f"{key} is a published-comparison teacher. Nothing overwrites the student.")
        print(f"\n  wrote {out} and models/metrics.json in {time.time()-started:.0f}s")
        return 0

    # ---- T1 student: the FHE-feasible model -----------------------------------
    # The student has to be the network the encrypted path actually evaluates — one conv block at
    # 14x14, run as a genuine CKKS convolution by qsh.fhe.conv. A flattened-pixel logistic would be
    # a different model, and its fidelity number would not describe the encrypted run.
    #
    # Only the browser-tier teacher gets a student. `medical_fl.student` is published, pinned, and
    # its recorded fidelity names resnet18 as its teacher; distilling from a second architecture
    # into the same key would overwrite a published artefact with a different lineage while the
    # metrics still claimed the old one.

    from torch.utils.data import DataLoader, TensorDataset

    from qsh.ai.cnn_layers import downsample, small_medmnist_cnn

    def to14(x):
        return torch.tensor(downsample(x.numpy(), 14), dtype=torch.float32)

    teacher_train = torch.tensor(train.predict_loader(model, tr_dl)[1].argmax(1), dtype=torch.long)
    teacher_val = torch.tensor(train.predict_loader(model, va_dl)[1].argmax(1), dtype=torch.long)
    xva14 = to14(va_dl.dataset.tensors[0])

    student = small_medmnist_cnn(in_ch=1, n_classes=2, img_size=14, channels=(4,), hidden=16)
    student_params = sum(p.numel() for p in student.parameters())
    # The student gets the SAME two protections the teacher above already has, and had neither.
    # Without them it collapsed to a constant predictor: it emitted class 1 for all 156 test images,
    # scoring accuracy 0.730769, which is exactly the 114/156 majority-class rate, and AUC 0.354 -
    # below random. That is a model which has learned nothing, and it is the model the CKKS path
    # runs, so the encrypted medical demo was evaluating a constant.
    #
    # (1) Class-weighted loss. A 2.71:1 split lets plain cross-entropy solve the task by answering
    #     with the majority class, and that is the optimum it found.
    # (2) Selection on AUC, not accuracy. train_torch's own docstring warns that selecting on
    #     accuracy hides "the class that matters is not being learned at all" - a constant predictor
    #     scores 0.73 on accuracy here and 0.5 on AUC, so only one of the two can reject it.
    tcounts = np.bincount(teacher_train.numpy(), minlength=2).astype(np.float64)
    tweight = torch.tensor(tcounts.sum() / (2.0 * np.maximum(tcounts, 1)), dtype=torch.float32)
    student_res = train.train_torch(
        "medical_fl.student", student,
        DataLoader(TensorDataset(to14(xtr), teacher_train), batch_size=32, shuffle=True),
        DataLoader(TensorDataset(xva14, teacher_val), batch_size=256),
        metric_fn=lambda y, p: metrics.auroc(y, p), metric_name="auc",
        loss_fn=torch.nn.CrossEntropyLoss(weight=tweight.to(train._device())),
        cfg=train.TrainConfig(epochs=120, batch_size=32, lr=2e-3, patience=25, seed=args.seed),
        out_dir=Path.cwd() / "models" / "medical_breastmnist_student",
        extra_manifest={"role": f"FHE-feasible student distilled from {key}"},
    )
    from safetensors.torch import load_file as _load

    student.load_state_dict(_load(student_res.checkpoint))

    # Tune the malignant cut on VALIDATION and fold it into the head, before anything is exported.
    # Argmax over two logits is a 0.5 threshold and the CKKS path has nowhere else to apply a cut,
    # so without this the model cannot express the precision/recall trade a screening task needs -
    # and with a 2.71:1 split, accuracy is maximised by never calling anything malignant.
    y_val_true = np.asarray(va_dl.dataset.tensors[1]).ravel().astype(int)
    val_scores = train.predict_loader(student, DataLoader(
        TensorDataset(xva14, torch.tensor(y_val_true)), batch_size=256))[1]
    val_p_mal = torch.softmax(torch.tensor(val_scores), 1).numpy()[:, MALIGNANT]
    # Criterion: Youden's J (TPR - FPR), NOT minority-class F1. F1 was tried first and degenerated -
    # it picked the lowest grid point, because when a model discriminates weakly, F1 on the minority
    # class is maximised by calling EVERYTHING positive: recall goes to 1 and precision falls only to
    # the base rate. On validation that scored 0.4242, which is exactly the all-positive value
    # 2*0.269/1.269, and the folded student then predicted malignant for all 156 test images. J
    # cannot be gamed that way: all-positive gives TPR=1, FPR=1, J=0.
    grid = np.unique(np.round(np.linspace(0.05, 0.95, 91), 4))
    pos, neg = y_val_true == MALIGNANT, y_val_true == BENIGN
    j_at = [float((val_p_mal[pos] >= q).mean() - (val_p_mal[neg] >= q).mean()) for q in grid]
    q_star = float(grid[int(np.argmax(j_at))])
    folded_val = np.where(val_p_mal >= q_star, MALIGNANT, BENIGN)
    if len(np.unique(folded_val)) < 2:
        # Belt and braces: any criterion that collapses to one class is rejected outright rather
        # than shipped, whatever it scored.
        print(f"  malignant cut q={q_star:.3f} collapses to a single class on validation - "
              f"NOT folding; the student keeps its argmax operating point")
    else:
        print(f"  malignant cut tuned on validation: q={q_star:.3f} "
              f"(Youden's J {max(j_at):.4f}); folding it into the dense head")
        distil.fold_threshold_into_torch_head(student, q_star, positive_class=MALIGNANT)

    # Keep the score matrix, not just the argmax. The registry declares AUC as this model's headline
    # and AUC needs scores; discarding them here is why the student recorded `accuracy` while the
    # panel showed an AUC baseline of 0.901 beside it.
    student_scores = train.predict_loader(
        student, DataLoader(TensorDataset(to14(xte), torch.tensor(y_true)), batch_size=256))[1]
    student_pred = student_scores.argmax(1)

    # Export the student to ONNX so it can be published and verified in a browser like the teacher.
    # It already had a safetensors checkpoint; what it lacked was a graph anyone could actually run.
    student_dir = Path.cwd() / "models" / "medical_breastmnist_student"
    student_report = export.verify_and_quantize(
        student, to14(xte).numpy(), student_dir, kind="torch")
    # Readable weights, so the browser can evaluate the PUBLISHED student under CKKS rather than a
    # synthetic kernel pattern of the same shape. ONNX alone is not enough: extracting conv kernels
    # from it in a browser needs a parser this project does not ship.
    w = distil.save_cnn_student_weights(
        student, student_dir, teacher_key=key, student_key="medical_fl.student",
        img_size=14, classes=["malignant", "normal/benign"])
    print(f"  student weights.json: {w['n_parameters']} params")
    print(f"  student ONNX: {student_report.fp32_bytes/1e6:.2f} MB, "
          f"domains={student_report.onnx_domains}, browser_runnable={student_report.browser_runnable}")

    fidelity = distil.measure_fidelity(
        teacher_key=key, student_key="medical_fl.student",
        teacher_pred=y_pred, student_pred=student_pred, y_true=y_true,
        metric_fn=metrics.accuracy, metric_name="accuracy",
        student_params=student_params, teacher_params=n_params,
    )
    print(f"    agreement with teacher      {fidelity.agreement:.4f}")
    print(f"    accuracy teacher/student    {fidelity.teacher_metric:.4f} / {fidelity.student_metric:.4f}"
          f"  (delta {fidelity.metric_delta_pp:+.2f} pp)")
    print(f"    parameters                  {n_params:,} -> {fidelity.student_params}"
          f"   ({n_params/max(fidelity.student_params,1):,.0f}x smaller)")

    # Record the metric the REGISTRY declares, beside the decision metric fidelity measures.
    # measure_fidelity compares decisions, so its metric must be decision-based (accuracy); the
    # registry headline is AUC. Recording only the first left the panel printing a 0.901 AUC
    # baseline under a 0.7308 accuracy - two different quantities, rendered as if comparable.
    student_auc = metrics.auroc(y_true, student_scores)
    teacher_auc = float(test_metrics["auc"])
    # A constant predictor is the failure this model actually had, and it is invisible in accuracy
    # (0.7308 = the majority-class rate) and in agreement. Name it in the record so it cannot ship
    # as a respectable-looking number again.
    student_labels = np.asarray(student_pred).ravel()
    n_classes_predicted = int(len(np.unique(student_labels)))
    if n_classes_predicted < 2:
        print(f"  !! student predicts a SINGLE class ({student_labels[0]}) for every test input - "
              f"it has not learned the task; accuracy {fidelity.student_metric:.4f} is just the "
              f"majority-class rate")
    print(f"    auc teacher/student         {teacher_auc:.4f} / {student_auc:.4f}"
          f"   <- the metric the registry declares, so this is the one the baseline compares to")
    train.record_metrics("medical_fl.student", {
        **fidelity.as_metrics(),
        "auc": round(float(student_auc), 6),
        "n_classes_predicted": n_classes_predicted,
        "majority_class_rate": round(float(max(np.mean(y_true == 0), np.mean(y_true == 1))), 6),
        "metric_note": (
            "Two metrics are recorded and they answer different questions. `auc` is the registry's "
            "declared headline and is what the published baseline compares against. "
            "`student_metric`/`teacher_metric` are accuracy, because fidelity compares DECISIONS "
            "and a decision has no score to rank - so agreement and metric_delta are necessarily "
            "computed on a decision metric. Do not read the accuracy against the AUC baseline."
        ),
    })
    print(f"\n  wrote {out} and models/metrics.json in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
