#!/usr/bin/env python3
"""The chest X-ray federation's FedAvg ResNet-18, exported for the Hub (Round 86, the owner's choice of 2026-11-14).

The model is the first FedAvg run of `scripts/train/medical_cxr_fl.py`: seed 1, fixed before training, the run whose
patient-bootstrap intervals the record reports; seeds 2 and 3 give the spread. It was trained from random
initialisation on NIH ChestX-ray14's train_val patients, placed in four simulated hospitals by view (12 rounds of one
local epoch), and is scored here on the official test split.

The exported graph takes grey pixels scaled to [0, 1] at 224 x 224 and applies the training normalisation itself, so a
client cannot apply another, and returns two logits, [0, z], with z the effusion logit: softmax gives the probability,
and argmax is z > 0, which the shared export gate (`qsh.models.export.verify_and_quantize`) reads as two classes. The
gate checks the float32 graph against PyTorch on 128 held-out test images, quantises the weights to int8 and checks
agreement; this script then runs both graphs with ONNX Runtime over the whole test split, fits a Platt map on the
hospitals' validation images, and writes:

  models/medical_cxr14_fedavg_resnet18/   onnx/model.onnx, onnx/model_int8.onnx (if it passes), model.safetensors,
                                          feature_spec.json, manifest.json, export_report.json, fixtures.npy
  .../onnx/features.onnx, features_int8   the same network up to its pooled 512 features, normalisation inside, for a
                                          browser that trains a head on the frozen features or encrypts them for a
                                          head under CKKS; features_report.json holds its checks: float32 against
                                          PyTorch, int8 against float32, and the network's own head over the int8
                                          features on the whole test split
  .../eval_sample.json                    16 test films, four for each class and view, drawn once from distinct
                                          patients among the well-exposed ones, at 224 px as uint8, with NIH's credit
                                          and each film's expected decision from the int8 graph
                                          (`qsh.models.sample_per_row`)
  models/metrics.json                     the record the model card is generated from

fixtures.npy holds test images and never leaves this machine (`scripts/publish_hf.py` denies *.npy); the 16 films of
the evaluation sample are published with the model, which NIH's terms allow with its credit.

Usage: python -I scripts/train/export_cxr_fedavg.py --prep <nih-cxr-224> --raw <nih-chestxray14> --runs <seed 1 run>
       <seed 2 run> <seed 3 run>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

KEY = "medical_fl.cxr_fedavg"
OUT = REPO / "models" / "medical_cxr14_fedavg_resnet18"
N_FIXTURES = 128
CLASSES = ["no effusion", "effusion"]
#: Films per (class, view) in the published evaluation sample: 2 x 2 x 4 = 16.
EVAL_PER_CELL = 4
EVAL_SEED = 2026


def _load_cxr():
    spec = importlib.util.spec_from_file_location("medical_cxr_fl", REPO / "scripts" / "train" / "medical_cxr_fl.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("medical_cxr_fl", mod)
    spec.loader.exec_module(mod)
    return mod


def effusion_graph(net, mean: float, std: float):
    """The network behind the normalisation, returning [0, z]."""
    import torch

    class EffusionGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        def forward(self, x):
            z = self.net((x - self.mean) / self.std)
            return torch.cat([torch.zeros_like(z), z], dim=1)
    return EffusionGraph()


def features_graph(net, mean: float, std: float):
    """The network up to its pooled features (512), behind the same normalisation."""
    import torch

    class FeaturesGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        def forward(self, x):
            n = self.net
            x = (x - self.mean) / self.std
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
            return torch.flatten(n.avgpool(x), 1)
    return FeaturesGraph()


def eval_rows(y: np.ndarray, ap: np.ndarray, patient: np.ndarray, imgs: np.ndarray, idx: np.ndarray,
              per_cell: int = EVAL_PER_CELL, seed: int = EVAL_SEED) -> np.ndarray:
    """`per_cell` films for each (label, view), drawn once with `seed` from the given (test) positions, one film per
    patient across all cells, among films the pre-upload checks would not flag for exposure (pixel SD at least 10, mean
    between 15 and 240 of 255). The draw reads no model output."""
    rng = np.random.default_rng(seed)
    x = imgs[idx].reshape(len(idx), -1).astype(np.float64)
    ok = (x.std(1) >= 10) & (x.mean(1) >= 15) & (x.mean(1) <= 240)
    chosen, used = [], set()
    for label in (0, 1):
        for view in (False, True):
            cand = idx[ok & (y[idx] == label) & (ap[idx] == view)]
            got = 0
            for i in rng.permutation(cand):
                if patient[i] in used:
                    continue
                chosen.append(int(i))
                used.add(patient[i])
                got += 1
                if got == per_cell:
                    break
    return np.array(chosen)


def fixtures_from(y: np.ndarray, idx: np.ndarray, n: int = N_FIXTURES, seed: int = 0) -> np.ndarray:
    """Half positives and half negatives of the given (test) positions, drawn once and sorted."""
    rng = np.random.default_rng(seed)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    return np.sort(np.concatenate([rng.choice(pos, n // 2, replace=False), rng.choice(neg, n - n // 2, replace=False)]))


def ort_logits(path: Path, imgs: np.ndarray, idx: np.ndarray, batch: int = 64) -> np.ndarray:
    """The graph's z over the given images, ONNX Runtime on the CPU."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for s in range(0, len(idx), batch):
        x = imgs[idx[s:s + batch]][:, None].astype(np.float32) / 255.0
        out.append(sess.run(None, {name: x})[0][:, 1])
    return np.concatenate(out).astype(np.float64)


def calibration_record(yv: np.ndarray, zv: np.ndarray, yt: np.ndarray, zt: np.ndarray) -> dict:
    """A Platt map on the probability's logit, fitted on the hospitals' validation images and scored on the official test
    split, in the shape the finance cards print (`scripts/publish_hf.py`, `_calibration_section`)."""
    from qsh.models import calibration as cal

    pv, pt = 1 / (1 + np.exp(-zv)), 1 / (1 + np.exp(-zt))
    platt = cal.fit_platt(pv, yv, on_logit=True)
    pc = platt.apply(pt)

    def _r(d: dict) -> dict:
        return {k: (round(v, 8) if isinstance(v, float) else v) for k, v in d.items()}
    return {
        **platt.as_dict(),
        "argument": "the graph's softmax probability of effusion, the sigmoid of z",
        "fitted_on": "four hospitals' validation images; scored on the official test split",
        "fitted_by": "scripts/train/export_cxr_fedavg.py",
        "page_reads_this_map": False,
        "raw_is": (f"The sigmoid of z, from a network trained with an unweighted cross-entropy on hospitals where 8 to 14% of "
                   f"images show effusion; on the test split, where {yt.mean():.1%} do, its mean is {pt.mean():.1%}."),
        "brier_raw": round(cal.brier(pt, yt), 8), "brier_calibrated": round(cal.brier(pc, yt), 8),
        "log_loss_raw": round(cal.log_loss(pt, yt), 8), "log_loss_calibrated": round(cal.log_loss(pc, yt), 8),
        "corp_raw": _r(cal.corp_decomposition(pt, yt)), "corp_calibrated": _r(cal.corp_decomposition(pc, yt)),
    }


def main() -> int:
    import torch
    from safetensors.torch import save_file
    from sklearn.metrics import roc_auc_score

    from qsh.models import export, train

    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="the FedAvg runs, the published seed first")
    a = ap.parse_args()
    t0 = time.time()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cxr = _load_cxr()
    dev = cxr.DEV
    reports = [json.loads((r / "report.json").read_text()) for r in a.runs]
    rep = reports[0]
    d = cxr.load(a.prep, a.raw)
    imgs_np = np.load(a.prep / "images_224.npy", mmap_mode="r")[:]
    imgs = torch.from_numpy(imgs_np).to(dev)
    tr_sites, va_sites, _ = cxr.hospitals(d, rep["split_seed"])
    va_all = np.concatenate(va_sites)
    test_idx = np.flatnonzero(d["test"] & ~d["drop"])
    yt, yv = d["y"][test_idx].astype(int), d["y"][va_all].astype(int)
    apt, pat_t = d["ap"][test_idx], d["patient"][test_idx]
    mean, std = rep["normalisation"]["mean"], rep["normalisation"]["std"]

    def net_of(run: Path):
        m = cxr.resnet18_1ch()
        m.load_state_dict(torch.load(run / "fedavg_resnet18.pt", map_location="cpu"))
        return m.eval()

    @torch.no_grad()
    def torch_z(m, idx):
        g = effusion_graph(m, mean, std).to(dev).eval()
        out = []
        for s in range(0, len(idx), 512):
            x = imgs[torch.from_numpy(idx[s:s + 512]).to(dev)].unsqueeze(1).float().div_(255)
            out.append(g(x)[:, 1].double().cpu())
        return torch.cat(out).numpy()

    seeds_auc = []
    for run in a.runs:
        seeds_auc.append(float(roc_auc_score(yt, torch_z(net_of(run), test_idx))))
    net = net_of(a.runs[0])
    zt, zv = torch_z(net, test_idx), torch_z(net, va_all)

    # ---- the graph, through the shared gate
    fix_idx = fixtures_from(d["y"].astype(int), test_idx)
    fixtures = imgs_np[fix_idx][:, None].astype(np.float32) / 255.0
    graph = effusion_graph(net, mean, std).eval().cpu()
    report = export.verify_and_quantize(graph, fixtures, OUT, kind="torch", int8_required=False)
    zt_onnx = ort_logits(OUT / "onnx" / "model.onnx", imgs_np, test_idx)
    int8_path = OUT / report.int8_path if report.int8_path else None
    zt_int8 = ort_logits(int8_path, imgs_np, test_idx) if int8_path else None

    # ---- the features graph: float32 against PyTorch, int8 against float32, the network's head over int8 features
    import onnx
    import onnxruntime as ort

    fgraph = features_graph(net, mean, std).eval().cpu()
    f32, f8 = OUT / "onnx" / "features.onnx", OUT / "onnx" / "features_int8.onnx"
    export.export_torch(fgraph, fixtures[:1], f32, output_name="features")
    export.quantize_int8(f32, f8)
    with torch.no_grad():
        ref = fgraph(torch.from_numpy(fixtures)).numpy().astype(np.float64)

    def feats(path: Path, x: np.ndarray) -> np.ndarray:
        s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return np.concatenate([s.run(None, {s.get_inputs()[0].name: x[i:i + 64]})[0] for i in range(0, len(x), 64)]).astype(np.float64)
    got32, got8 = feats(f32, fixtures), feats(f8, fixtures)
    cos = (got8 * got32).sum(1) / (np.linalg.norm(got8, axis=1) * np.linalg.norm(got32, axis=1))
    w_fc, b_fc = net.fc.weight.detach().double().numpy()[0], float(net.fc.bias.detach().double().numpy()[0])
    s8 = ort.InferenceSession(str(f8), providers=["CPUExecutionProvider"])
    z_feat8 = np.concatenate([s8.run(None, {s8.get_inputs()[0].name: imgs_np[test_idx[i:i + 64]][:, None].astype(np.float32) / 255.0})[0]
                              for i in range(0, len(test_idx), 64)]).astype(np.float64) @ w_fc + b_fc
    features_report = {
        "fp32_path": "onnx/features.onnx", "int8_path": "onnx/features_int8.onnx", "output": "features", "dimension": int(ref.shape[1]),
        "fp32_bytes": f32.stat().st_size, "int8_bytes": f8.stat().st_size,
        "opset": int(next(o.version for o in onnx.load(str(f32), load_external_data=False).opset_import if o.domain in ("", "ai.onnx"))),
        "n_fixtures": int(len(fixtures)), "fp32_max_abs_error_vs_torch": float(np.abs(got32 - ref).max()),
        "int8_max_abs_error_vs_fp32": float(np.abs(got8 - got32).max()), "int8_min_cosine_vs_fp32": float(cos.min()),
        "head_auroc_on_int8_features_test": float(roc_auc_score(yt, z_feat8)),
        "head": "the network's own last layer (fc), applied to the graph's features in float64",
        "use": ("features for a head trained in the browser, or encrypted for a head under CKKS; the network's own head "
                "on these features is the classifier graph's z"),
    }
    (OUT / "features_report.json").write_text(json.dumps(features_report, indent=2) + "\n")

    # ---- the evaluation sample: 16 films and each one's expected decision from the int8 graph
    from qsh.models import registry, sample_per_row

    rows = eval_rows(d["y"].astype(int), d["ap"], d["patient"], imgs_np, test_idx)
    ds = registry.get(KEY).dataset
    p_thr = float(1 / (1 + np.exp(-cxr.operating_point(yv, zv))))
    sample = {
        "provenance": ("Real NIH ChestX-ray14 radiographs from the official test split, which this model was scored on and "
                       "never trained on: 16 films, four for each class (Effusion or not, as NIH's labels mined from the "
                       "reports say) and each view (AP or PA), drawn once from distinct patients among well-exposed films; "
                       "the draw reads no model output. " + ds.attribution),
        "split": "the official ChestX-ray14 test split, 16 films, four for each class and view",
        "licence": ds.licence, "licence_url": ds.url, "attribution": ds.attribution,
        "modifications": ("each 1024 px PNG converted to grey and resized to 224 x 224 with Pillow's bilinear filter, "
                          "stored as uint8 values"),
        "task": "classification",
        "input": {"shape": [1, 1, 224, 224], "dtype": "float32", "preprocessing": ["divide by 255"],
                  "layout": "samples[i] is a flat row-major 224*224 uint8 image, values 0-255"},
        "classes": {"0": CLASSES[0], "1": CLASSES[1]},
        "samples": [imgs_np[i].ravel().astype(int).tolist() for i in rows],
        "labels": [int(d["y"][i]) for i in rows],
        "views": ["AP" if d["ap"][i] else "PA" for i in rows],
        "expected": {"graph": "model_int8.onnx", "decision_threshold": p_thr,
                     "note": ("Measured here with onnxruntime on exactly these 16 films at the published operating point "
                              "(validation sensitivity 0.80). Sixteen films show the model running, not how well it "
                              "ranks: the card's AUROC is over 25,589.")},
    }
    rec = sample_per_row.per_row_record(sample, OUT / "onnx" / "model_int8.onnx")
    sample["expected"]["accuracy"] = float(np.mean(np.array(rec["predicted"]) == np.array(sample["labels"])))
    sample["expected"]["per_row"] = rec
    (OUT / "eval_sample.json").write_text(json.dumps(sample) + "\n")

    # ---- what the card states
    thr = cxr.operating_point(yv, zv)
    at = cxr.rates(yt, zt_onnx, thr)
    ci = cxr.auc_boot(yt, zt_onnx, pat_t)
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(OUT / "model.safetensors"),
              metadata={"format": "pt", "architecture": "torchvision resnet18, 1-channel conv1, 1 logit"})
    ckpt_sha = hashlib.sha256((a.runs[0] / "fedavg_resnet18.pt").read_bytes()).hexdigest()
    (OUT / "feature_spec.json").write_text(json.dumps({
        "input": {"shape": [1, 1, 224, 224], "dtype": "float32",
                  "preprocessing": ("the 1024 px PNG converted to grey and resized to 224 x 224 with Pillow's bilinear "
                                    "filter, then pixel/255; the graph subtracts the training mean and divides by the "
                                    "training standard deviation itself"),
                  "normalisation_in_graph": {"mean": mean, "std": std}},
        "output": "two logits [0, z]; z is the effusion logit, softmax gives its probability",
        "classes": CLASSES,
        "operating_point": {"logit": thr, "chosen_on": "the four hospitals' validation images, sensitivity 0.80"},
    }, indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps({
        "key": KEY, "seed": rep["seed"], "split_seed": rep["split_seed"], "dataset": "NIH ChestX-ray14",
        "label": "Effusion", "architecture": "resnet18", "resolution": 224, "pretrained": "none",
        "adaptation": "1-channel 7x7 stem and one logit, built so; no weights from elsewhere",
        "federation": {"hospitals": 4, "rounds": rep["epochs"], "local_epochs": 1, "aggregation": "FedAvg, weighted by images",
                       "hospital_ap_targets": list(cxr.HOSPITAL_AP_TARGETS)},
        "hyperparameters": {"batch_size": rep["batch"], "lr": rep["lr"], "optimiser": "AdamW, weight decay 1e-4",
                            "schedule": "cosine over the rounds, constant within one", "precision": "bfloat16 autocast"},
        "normalisation": {"mean": mean, "std": std, "measured_on": "2,000 training images"},
        "trained_with": {k: rep["environment"].get(k) for k in ("torch", "torch_cuda", "gpu", "driver", "cublas_version", "kernel")},
        "source_checkpoint_sha256": ckpt_sha, "checkpoint": "model.safetensors",
        "training_script": "scripts/train/medical_cxr_fl.py", "export_script": "scripts/train/export_cxr_fedavg.py",
    }, indent=2) + "\n")

    sd = float(np.std(seeds_auc, ddof=1))
    payload = {
        "auroc": float(roc_auc_score(yt, zt_onnx)),
        "auroc_ci_95": {"lo": round(ci["lo"], 4), "hi": round(ci["hi"], 4), "method": "patient bootstrap, 1,000 resamples",
                        "source": "scripts/train/export_cxr_fedavg.py"},
        "auroc_pa": float(roc_auc_score(yt[~apt], zt_onnx[~apt])),
        "auroc_ap": float(roc_auc_score(yt[apt], zt_onnx[apt])),
        "sensitivity_at_operating_point": at["sensitivity"], "specificity_at_operating_point": at["specificity"],
        "operating_point_logit": thr,
        "n_test_images": int(len(test_idx)), "n_test_patients": int(len(np.unique(pat_t))), "test_positives": int(yt.sum()),
        "n_parameters": int(sum(p.numel() for p in net.parameters())),
        "seed_spread": {"mean": round(float(np.mean(seeds_auc)), 4), "sd": round(sd, 4), "lo": round(min(seeds_auc), 4),
                        "hi": round(max(seeds_auc), 4), "n_seeds": len(seeds_auc),
                        "source": "PyTorch float32 over the official test split, scripts/train/export_cxr_fedavg.py"},
        "onnx_vs_torch_max_abs_logit": float(np.abs(zt_onnx - zt).max()),
        "calibration": calibration_record(yv, zv, yt, zt_onnx),
        "baseline_gap_note": (
            f"{roc_auc_score(yt, zt_onnx):.3f} on the official split, above Wang et al.'s 0.7585 (a ResNet-50 from ImageNet "
            "with only its last layers trained, for all 14 findings at once) and below Baltruschat et al.'s 0.818 to 0.822 "
            "(ResNet-38 to -101 at 448 px with age, sex and view as inputs, Sci Rep 2019, Table 5). This model reads 224 px, "
            "one finding, from random initialisation, under FedAvg over four simulated hospitals: protocols differ, so the "
            "comparison places it, and does not rank methods."),
        "dataset": "NIH ChestX-ray14",
        "transfer_learning": ("none: trained from random initialisation, so the weights carry no ImageNet terms "
                              "(DATA_LICENSES.md)"),
        "export": {"fp32_bytes": report.fp32_bytes, "int8_bytes": report.int8_bytes, "int8_agreement": report.int8_agreement,
                   "opset": report.opset},
        "git_sha": train.git_sha(),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    payload["features_int8_head_auroc"] = features_report["head_auroc_on_int8_features_test"]
    if zt_int8 is not None:
        payload["auroc_int8"] = float(roc_auc_score(yt, zt_int8))
        payload["int8_decisions_differing_at_operating_point"] = int(((zt_int8 >= thr) != (zt_onnx >= thr)).sum())
    train.record_metrics(KEY, payload)
    print(json.dumps({k: v for k, v in payload.items() if isinstance(v, int | float)}, indent=1))
    print(f"wrote {OUT.relative_to(REPO)} and models/metrics.json in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
