"""The pre-registered square-activation medical student (`references/medical-square-student-preregistration.md`).

Trains two arms on the published `medical_fl.resnet18` teacher's labels, seeds 1 to 5: the control (the published
student's architecture: conv, ReLU, avg-pool, dense 196->16, ReLU, dense 16->2) and the candidate (conv, batch norm,
x^2, avg-pool, dense 196->2). Then runs the candidate's exported seed (best validation AUC) end to end under CKKS with
TenSEAL at both pre-registered parameter sets, on all 156 test images, and writes one JSON record with the criteria
evaluated as the pre-registration states them.

    python scripts/train/medical_square_student.py --work ../qsh-square-work
    python scripts/train/medical_square_student.py --work ../qsh-square-work --ckks-only   # re-time on a quiet machine

The encrypted pass is the one the browser module will run: the client encrypts the image's nine convolution taps,
each replicated across the four channels (784 slots); the evaluator multiplies by the kernel taps, adds the bias,
squares, and takes one inner product per class with the pooled dense weights; the client decrypts two logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PREREG = REPO / "references" / "medical-square-student-preregistration.md"
TEACHER_ONNX = REPO / "models" / "medical_breastmnist_resnet18" / "onnx" / "model.onnx"
RECORD = REPO / "docs" / "measurements" / "medical-square-student-2026-11-19.json"
SEEDS = (1, 2, 3, 4, 5)
#: The pre-registered parameter sets: ring dimension, coefficient modulus (the last prime is the special prime) and
#: scale, with the HomomorphicEncryption.org standard's 128-bit post-quantum bound for a ternary secret at that N.
PARAMS = {
    "A": {"N": 8192, "bits": [50, 30, 30, 30, 50], "scale_bits": 30, "pq_bound_bits": 202},
    "B": {"N": 16384, "bits": [60, 40, 40, 40, 60], "scale_bits": 40, "pq_bound_bits": 411},
    # The addendum (references/medical-square-student-preregistration-addendum-a2.md), after A failed criterion 3.
    "A2": {"N": 8192, "bits": [48, 34, 34, 34, 52], "scale_bits": 34, "pq_bound_bits": 202},
}
MARGIN, FLOOR, MAX_LOGIT_ERR = 0.05, 0.60, 1e-3      # criteria 1, 2 and 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def square_student():
    import torch.nn as nn

    class SquareStudent(nn.Module):
        """conv 1->4 (3x3, pad 1), batch norm, x^2, avg-pool 2, dense 196->2: CKKS depth 3."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.pool = nn.AvgPool2d(2)
            self.fc = nn.Linear(4 * 7 * 7, 2)

        def forward(self, x):
            y = self.bn(self.conv(x))
            return self.fc(self.pool(y * y).flatten(1))

    return SquareStudent()


def folded(model) -> dict:
    """The candidate's inference weights in float64, batch norm folded into the convolution (exact in eval mode)."""
    import torch

    with torch.no_grad():
        bn = model.bn
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = (model.conv.weight * s[:, None, None, None]).double().cpu().numpy()     # (4, 1, 3, 3)
        b = ((model.conv.bias - bn.running_mean) * s + bn.bias).double().cpu().numpy()
        return {"conv_w": w.reshape(4, 9), "conv_b": b,
                "fc_w": model.fc.weight.double().cpu().numpy(), "fc_b": model.fc.bias.double().cpu().numpy()}


def taps(x14: np.ndarray) -> np.ndarray:
    """(14, 14) -> (9, 196): tap (di, dj) is the zero-padded image shifted by (di, dj), as Conv2d correlates."""
    xp = np.pad(x14, 1)
    return np.stack([xp[di:di + 14, dj:dj + 14].reshape(-1) for di in range(3) for dj in range(3)])


def pooled_dense(fc_w: np.ndarray) -> np.ndarray:
    """(2, 196) dense weights over the pooled (4, 7, 7) map -> (2, 784) weights over the slot layout c * 196 + p."""
    v = np.zeros((2, 784))
    for c in range(4):
        for pi in range(14):
            for pj in range(14):
                v[:, c * 196 + pi * 14 + pj] = fc_w[:, c * 49 + (pi // 2) * 7 + (pj // 2)] / 4.0
    return v


def plain_logits(x14: np.ndarray, p: dict) -> np.ndarray:
    conv = p["conv_w"] @ taps(x14) + p["conv_b"][:, None]                       # (4, 196)
    pooled = (conv ** 2).reshape(4, 7, 2, 7, 2).mean(axis=(2, 4)).reshape(-1)
    return p["fc_w"] @ pooled + p["fc_b"]


def ckks_run(params: dict, p: dict, images: np.ndarray) -> dict:
    """Every image end to end under CKKS: encrypt the taps, convolve, square, pooled dense, decrypt two logits."""
    import tenseal as ts

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=params["N"], coeff_mod_bit_sizes=params["bits"])
    ctx.global_scale = 2 ** params["scale_bits"]
    ctx.generate_galois_keys()
    w_rep = [np.repeat(p["conv_w"][:, k], 196).tolist() for k in range(9)]
    b_rep = np.repeat(p["conv_b"], 196).tolist()
    v = pooled_dense(p["fc_w"])
    errs, differ, times, logits_all = [], 0, [], []
    sizes = {}
    load_before = os.getloadavg()
    for x14 in images:
        t0 = time.perf_counter()
        enc = [ts.ckks_vector(ctx, np.tile(row, 4).tolist()) for row in taps(x14)]       # the client
        acc = None
        for k in range(9):                                                                 # the evaluator
            term = enc[k] * w_rep[k]
            acc = term if acc is None else acc + term
        acc = acc + b_rep
        acc = acc.square()
        out = [acc.dot(v[j].tolist()) + float(p["fc_b"][j]) for j in range(2)]
        got = np.array([o.decrypt()[0] for o in out])                                     # the client
        times.append(time.perf_counter() - t0)
        want = plain_logits(x14, p)
        errs.append(float(np.max(np.abs(got - want))))
        differ += int(np.argmax(got) != np.argmax(want))
        logits_all.append(got.tolist())
        if not sizes:
            sizes = {"input_bytes": int(sum(len(e.serialize()) for e in enc)),
                     "output_bytes": int(sum(len(o.serialize()) for o in out))}
    return {
        **params, "modulus_bits": int(sum(params["bits"])),
        "within_pq_bound": bool(sum(params["bits"]) <= params["pq_bound_bits"]),
        "images": int(len(images)), "max_abs_logit_error": max(errs), "median_abs_logit_error": float(np.median(errs)),
        "decisions_differ": differ, "median_s": float(np.median(times)), "max_s": float(np.max(times)), **sizes,
        "load_average_before": [round(x, 2) for x in load_before],
        "load_average_after": [round(x, 2) for x in os.getloadavg()],
        "decrypted_logits": logits_all,
    }


def environment() -> dict:
    import tenseal
    import torch

    env = {"python": platform.python_version(), "torch": torch.__version__, "tenseal": tenseal.__version__,
           "kernel": platform.release(), "cpu": platform.processor() or platform.machine()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        try:
            env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:                                                     # pragma: no cover - reported, not needed
            env["driver"] = "unread"
    return env


OUT_DIR = REPO / "models" / "medical_breastmnist_square_student"
KEY = "medical_fl.square_student"


def record_square_metrics(model, record: dict, x14te: np.ndarray, yte: np.ndarray, teacher_te: np.ndarray, *,
                          n_parameters: int, keep_revision: bool = False) -> None:
    """The registry's record of the exported seed. Only the parameter sets that met criterion 3 are listed under `ckks`;
    the one that did not is named under `ckks_not_offered`, so a flattened table cannot show it beside them as if offered.
    `keep_revision` re-records an unchanged model without unpinning it: `record_metrics` reads a record with no
    `hf_revision` as a retrain, which would invalidate the published pin."""
    import torch

    from qsh.models import distil, metrics, registry, train

    seed = int(record["criteria"]["export_seed"])
    scores = train.predict_loader(model, torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x14te), torch.tensor(yte)), batch_size=256))[1]
    pred = np.asarray(scores).argmax(1)
    fidelity = distil.measure_fidelity(
        teacher_key="medical_fl.resnet18", student_key=KEY, teacher_pred=teacher_te, student_pred=pred, y_true=yte,
        metric_fn=metrics.accuracy, metric_name="accuracy",
        student_params=n_parameters, teacher_params=11171266)
    sq = record["arms"]["square"]
    revision = (registry.achieved(KEY) or {}).get("hf_revision") if keep_revision else None
    train.record_metrics(KEY, {
        **({"hf_revision": revision} if revision else {}),
        **fidelity.as_metrics(),
        "auc": round(float(metrics.auroc(yte, scores)), 6),
        "n_classes_predicted": int(len(np.unique(pred))),
        "majority_class_rate": round(float(max(np.mean(yte == 0), np.mean(yte == 1))), 6),
        "five_seeds_test_auc": {"seeds": {s: r["test_auc"] for s, r in sq.items()},
                                "mean": record["criteria"]["mean_test_auc"]["square"],
                                "sd": record["criteria"]["sd_test_auc"]["square"],
                                "control_mean": record["criteria"]["mean_test_auc"]["control"],
                                "control_sd": record["criteria"]["sd_test_auc"]["control"]},
        "export_seed": seed,
        "ckks": {k: {kk: v[kk] for kk in ("N", "scale_bits", "max_abs_logit_error", "decisions_differ", "median_s",
                                           "input_bytes", "output_bytes")}
                 for k, v in record["ckks"].items() if v["criterion_3"]},
        "ckks_not_offered": {k: (f"N = {v['N']}, {'+'.join(map(str, v['bits']))} bits at scale 2^{v['scale_bits']}: largest "
                                 f"logit error {v['max_abs_logit_error']:.1e}, over the pre-registered 1e-3; not offered")
                             for k, v in record["ckks"].items() if not v["criterion_3"]},
        "preregistration": record["preregistration"],
        "metric_note": ("`auc` is the exported seed's test AUC on all 156 images (the registry's headline); "
                        "`five_seeds_test_auc` is the pre-registered comparison with the published architecture "
                        "trained by the same recipe. `student_metric`/`teacher_metric` are accuracy, because "
                        "fidelity compares decisions."),
    })


def export_artifacts(work: Path, record: dict, x14te: np.ndarray, yte: np.ndarray, teacher_te: np.ndarray) -> None:
    """The folder the Hub repository is published from, written as the published student's script writes its own."""
    import shutil

    import torch
    from safetensors.torch import load_file

    from qsh.models import distil, export, metrics, train

    seed = int(record["criteria"]["export_seed"])
    ckpt = work / f"square_seed{seed}" / "model.safetensors"
    model = square_student()
    model.load_state_dict(load_file(ckpt))
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # The same gate every published graph passes: export, verify fp32 against the module, quantise, verify int8 (kept
    # only if it passes), strip torch.export's node metadata. The browser fetches the fp32 graph to re-score the sample.
    report = export.verify_and_quantize(model, x14te, OUT_DIR, kind="torch", int8_required=False)
    shutil.copy(ckpt, OUT_DIR / "model.safetensors")
    manifest = json.loads((work / f"square_seed{seed}" / "manifest.json").read_text())
    manifest["key"] = KEY                                   # the registry's key, not the experiment run's
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    w = folded(model)
    weights = {
        "kind": "cnn-1conv-square-1dense",
        "note": ("The fully encrypted medical student, layer by layer, with its batch normalisation folded into the "
                 "convolution (exact in evaluation mode). Evaluate strictly in this order: conv (valid convolution over "
                 "the input zero-padded by `conv.padding`), the square, average pool by `pool.stride`, flatten "
                 "channel-major, then the dense layer. The browser runs it end to end under CKKS: the client encrypts "
                 "the image's nine convolution taps, each repeated for the four channels; the evaluator multiplies by "
                 "the kernel taps, adds the bias, squares, and takes one inner product per class with the dense weights "
                 "averaged over each pooling window; the client decrypts two logits. No layer is decrypted in between."),
        "teacher_key": "medical_fl.resnet18", "student_key": KEY,
        "input": {"shape": [1, 14, 14], "preprocessing": ["pixel / 255", "(x - 0.5) / 0.5", "area resize to 14x14"]},
        "conv": {"in_ch": 1, "out_ch": 4, "kernel": 3, "padding": 1, "activation": "square",
                 "batch_norm": "folded", "weight": w["conv_w"].reshape(4, 1, 3, 3).tolist(), "bias": w["conv_b"].tolist()},
        "pool": {"type": "avg", "stride": 2},
        "dense": [{"in": 196, "out": 2, "weight": w["fc_w"].tolist(), "bias": w["fc_b"].tolist(), "activation": "none"}],
        "classes": ["malignant", "normal/benign"],
        "n_parameters": int(w["conv_w"].size + w["conv_b"].size + w["fc_w"].size + w["fc_b"].size),
        "ckks": {k: {"N": v["N"], "coeff_mod_bits": v["bits"], "scale_bits": v["scale_bits"]}
                 for k, v in PARAMS.items() if record.get("criteria", {}).get("3_fidelity", {}).get(k)},
    }
    (OUT_DIR / "weights.json").write_text(json.dumps(weights) + "\n")

    record_square_metrics(model, record, x14te, yte, teacher_te, n_parameters=weights["n_parameters"])
    print(f"  exported seed {seed} to {OUT_DIR.relative_to(REPO)}: fp32 graph {report.fp32_bytes} bytes, "
          f"{weights['n_parameters']} parameters after folding")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, required=True, help="checkpoints and logs, outside the repository")
    ap.add_argument("--ckks-only", action="store_true", help="re-run the CKKS phase on the recorded export")
    ap.add_argument("--export", action="store_true",
                    help="write models/medical_breastmnist_square_student/ from the recorded export seed")
    ap.add_argument("--record-only", action="store_true",
                    help="re-record the exported seed's metrics, keeping its pin; the published files are not rewritten")
    ap.add_argument("--sets", nargs="+", choices=sorted(PARAMS), default=None,
                    help="with --ckks-only: run these parameter sets only and keep the others' recorded results")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, TensorDataset

    from qsh.ai.cnn_layers import downsample, small_medmnist_cnn
    from qsh.models import datasets, metrics, train

    def split(name):
        imgs, labels = datasets.breastmnist(name, size=28)
        x = (np.asarray(imgs, dtype=np.float32)[:, None] / 255.0 - 0.5) / 0.5
        return x, np.asarray(labels).ravel().astype(np.int64)

    (xtr, ytr), (xva, yva), (xte, yte) = split("train"), split("val"), split("test")
    to14 = lambda x: np.asarray(downsample(x, 14), dtype=np.float32)            # noqa: E731
    x14tr, x14va, x14te = to14(xtr), to14(xva), to14(xte)

    if args.record_only:
        from safetensors.torch import load_file as _load

        record = json.loads(RECORD.read_text())
        model = square_student()
        model.load_state_dict(_load(args.work / f"square_seed{record['criteria']['export_seed']}" / "model.safetensors"))
        model.eval()
        sess = ort.InferenceSession(str(TEACHER_ONNX), providers=["CPUExecutionProvider"])
        record_square_metrics(model, record, x14te, yte, sess.run(None, {"input": xte})[0].argmax(1),
                              n_parameters=json.loads((OUT_DIR / "weights.json").read_text())["n_parameters"],
                              keep_revision=True)
        return 0

    if args.export:
        sess = ort.InferenceSession(str(TEACHER_ONNX), providers=["CPUExecutionProvider"])
        export_artifacts(args.work, json.loads(RECORD.read_text()), x14te, yte,
                         sess.run(None, {"input": xte})[0].argmax(1))
        return 0

    record = json.loads(RECORD.read_text()) if args.ckks_only else {}
    if not args.ckks_only:
        sess = ort.InferenceSession(str(TEACHER_ONNX), providers=["CPUExecutionProvider"])
        teach = lambda x: sess.run(None, {"input": x})[0].argmax(1)              # noqa: E731
        t_tr, t_va = teach(xtr), teach(xva)
        record = {
            "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": sha256(PREREG)},
            "teacher": {"onnx": str(TEACHER_ONNX.relative_to(REPO)), "sha256": sha256(TEACHER_ONNX),
                        "agrees_with_truth_train": float(np.mean(t_tr == ytr)),
                        "agrees_with_truth_val": float(np.mean(t_va == yva))},
            "environment": environment(), "arms": {"control": {}, "square": {}},
        }
        counts = np.bincount(t_tr, minlength=2).astype(np.float64)
        weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)
        for arm in ("control", "square"):
            for seed in SEEDS:
                train.seed_everything(seed)
                model = (small_medmnist_cnn(in_ch=1, n_classes=2, img_size=14, channels=(4,), hidden=16)
                         if arm == "control" else square_student())
                res = train.train_torch(
                    f"medical_square_student.{arm}", model,
                    DataLoader(TensorDataset(torch.tensor(x14tr), torch.tensor(t_tr)), batch_size=32, shuffle=True),
                    DataLoader(TensorDataset(torch.tensor(x14va), torch.tensor(t_va)), batch_size=256),
                    metric_fn=lambda y, p: metrics.auroc(y, p), metric_name="auc",
                    loss_fn=torch.nn.CrossEntropyLoss(weight=weight.to(train._device())),
                    cfg=train.TrainConfig(epochs=120, batch_size=32, lr=2e-3, patience=25, seed=seed),
                    out_dir=args.work / f"{arm}_seed{seed}",
                    extra_manifest={"role": f"pre-registered {arm} arm, seed {seed}"},
                )
                model.load_state_dict(load_file(res.checkpoint))
                model.eval()
                scores = train.predict_loader(model, DataLoader(
                    TensorDataset(torch.tensor(x14te), torch.tensor(yte)), batch_size=256))[1]
                pred = np.asarray(scores).argmax(1)
                record["arms"][arm][str(seed)] = {
                    "val_auc": round(float(res.best_metric), 6), "test_auc": round(metrics.auroc(yte, scores), 6),
                    "best_epoch": int(res.best_epoch), "epochs_run": int(res.epochs_run),
                    "predicts_both_classes": bool(len(set(pred.tolist())) == 2),
                    "parameters": int(sum(q.numel() for q in model.parameters())),
                    "checkpoint_sha256": sha256(Path(res.checkpoint)),
                }
                if arm == "square":
                    np.savez(args.work / f"square_seed{seed}_folded.npz", **folded(model))
                print(f"  {arm} seed {seed}: val {record['arms'][arm][str(seed)]['val_auc']:.4f} "
                      f"test {record['arms'][arm][str(seed)]['test_auc']:.4f}", flush=True)

        means = {a: float(np.mean([r["test_auc"] for r in record["arms"][a].values()])) for a in ("control", "square")}
        sds = {a: float(np.std([r["test_auc"] for r in record["arms"][a].values()], ddof=1)) for a in ("control", "square")}
        sq = record["arms"]["square"]
        export = max(sq, key=lambda s: sq[s]["val_auc"])
        record["criteria"] = {
            "mean_test_auc": {k: round(v, 6) for k, v in means.items()},
            "sd_test_auc": {k: round(v, 6) for k, v in sds.items()},
            "1_accuracy": bool(means["square"] >= means["control"] - MARGIN),
            "2_no_collapse": bool(all(r["test_auc"] >= FLOOR and r["predicts_both_classes"] for r in sq.values())),
            "export_seed": int(export),
        }

    export = record["criteria"]["export_seed"]
    p = dict(np.load(args.work / f"square_seed{export}_folded.npz"))
    plain = np.array([plain_logits(x, p) for x in x14te[:, 0]])
    # The AUC reads column 1 of a PROBABILITY matrix (metrics._positive_score), as predict_loader returns; raw
    # logits there rank by the class-1 logit alone. The first run of 2026-11-19 recorded that (0.77193).
    probs = np.exp(plain - plain.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    record["export"] = {**record.get("export", {}), "seed": export,
                        "plain_test_auc_float64": round(metrics.auroc(yte, probs), 6),
                        "folded_weights_sha256": sha256(args.work / f"square_seed{export}_folded.npz")}
    if not (args.ckks_only and args.sets):
        record["ckks"] = {}
    for name, params in PARAMS.items():
        if args.sets and name not in args.sets:
            continue
        run = ckks_run(params, p, x14te[:, 0])
        run["criterion_3"] = bool(run["max_abs_logit_error"] <= MAX_LOGIT_ERR and run["decisions_differ"] == 0)
        record["ckks"][name] = run
        print(f"  CKKS {name}: N={params['N']} max |err| {run['max_abs_logit_error']:.3e}, decisions differ "
              f"{run['decisions_differ']}, median {run['median_s']*1e3:.0f} ms, load {run['load_average_before']}",
              flush=True)
    record["criteria"]["3_fidelity"] = {k: v["criterion_3"] for k, v in record["ckks"].items()}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record["criteria"][k] for k in record["criteria"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
