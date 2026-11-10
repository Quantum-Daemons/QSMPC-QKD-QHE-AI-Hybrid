#!/usr/bin/env python
"""The published finance MLP: one network on AMLworld HI-Small, the model the page runs under CKKS and garbled circuits (Round 86).

WHY THIS AND NOT THE 32 FEATURES. The 32 per-account aggregates give F1 about 0.2, the level of the "basic features" row of the Graph Feature Preprocessor paper (Blanusa et al., ICAIF 2024, arXiv:2402.08593, Table 5, 19.75 for XGBoost). This adds the
paper's kind of feature, computed in the open (`finance_aml_fan.py`): per-transaction fan-in and fan-out in time windows, vertex statistics and scatter-gather, per batch of 2,048 as the paper processes them. On the same windows an XGBoost reaches F1 0.40 and this network 0.38.

WHY THIS SIZE. Two hidden layers of 128 and 64, a shape the literature evaluates for tabular private inference (QUOTIENT's 2 x 124; a pyramid [h, h/2] in the November 2026 clinical benchmark, arXiv:2611.02856), and one model rather than a
choice of sizes: the owner removed model-size selectors because they invited comparisons of shapes nothing had trained. `references/REFERENCES.md` has the sources read at origin, including the search negative: no paper found runs a trained
MLP on finance tabular data under a CKKS and MPC hybrid.

WHAT WAS CHOSEN ON THE VALIDATION WINDOW, and nothing here looks at the test window to choose: labels against the teacher's scores (labels won), the scaling (standardisation against the robust median and 5-95 percentile range, by each
scaling's best validation AUPRC over its seeds), and the seed: five are trained for each scaling and the best validation AUPRC ships, with all recorded. Values are clipped to +/- 10 after scaling, which bounds what a secret share holds.

THE THRESHOLD. Tuned on validation and folded into the last bias, because the encrypted path decides by argmax over two logits and has nowhere downstream to apply a cut.

    python scripts/train/finance_aml_mlp.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts" / "train")]

from finance_aml import (  # noqa: E402
    FEATURES,
    _best_threshold,
    causal_account_features,
    graph_features,
    load_sorted,
    temporal_windows,
)
from finance_aml_fan import BATCH_WINDOW, transaction_features  # noqa: E402

from qsh.models import distil, export, metrics, registry, train  # noqa: E402

KEY = "finance_aml.mlp_large"
HIDDEN = [128, 64]
SCALINGS = ("standard", "robust")
SEEDS = (0, 1, 2, 3, 4)
CLIP = 10.0
EPOCHS, PATIENCE, BATCH, LR, WEIGHT_DECAY = 40, 6, 16_384, 1e-3, 1e-5
N_FIXTURES = 16
CDLA_SHARING_1 = "https://cdla.dev/sharing-1-0/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def scaling_of(kind: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and scale from the fit window alone. A zero scale would be a constant column, of which this corpus has none."""
    if kind == "standard":
        centre, scale = x.mean(0), x.std(0)
    elif kind == "robust":
        centre = np.median(x[::3], axis=0)
        low, high = np.percentile(x[::3], [5, 95], axis=0)
        scale = high - low
    else:
        raise ValueError(kind)
    return centre.astype(np.float32), np.where(scale == 0, 1.0, scale).astype(np.float32)


def network(hidden: list[int], n_in: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = n_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, 2))
    return nn.Sequential(*layers)


class Scored(nn.Module):
    """Raw features in, two logits out: the scaling travels in the graph, so a caller cannot scale twice or not at all."""

    def __init__(self, centre: np.ndarray, scale: np.ndarray, net: nn.Sequential) -> None:
        super().__init__()
        self.register_buffer("centre", torch.tensor(centre, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.clamp((x - self.centre) / self.scale, -CLIP, CLIP))


def _scaled(x: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.clip((x - centre) / scale, -CLIP, CLIP), dtype=torch.float32, device=DEVICE)


def _positive_scores(model: nn.Module, z: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = [torch.softmax(model(z[i:i + 262_144]), 1)[:, 1] for i in range(0, len(z), 262_144)]
    return torch.cat(out).cpu().numpy()


def fit_one(hidden: list[int], zf: torch.Tensor, yf: torch.Tensor, zv: torch.Tensor, yv: np.ndarray, seed: int) -> tuple[nn.Sequential, float, int]:
    """Class-weighted cross-entropy with early stopping on validation AUPRC."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = network(hidden, zf.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n, pos = len(zf), int(yf.sum())
    w = torch.tensor([n / (2 * (n - pos)), n / (2 * pos)], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    best, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(zf[b]), yf[b]).backward()
            opt.step()
        model.eval()
        score = float(average_precision_score(yv, _positive_scores(model, zv)))
        if score > best + 1e-5:
            best, best_state, best_epoch, stale = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, best_epoch


def teacher_decisions(x: np.ndarray) -> np.ndarray:
    """The published teacher's decisions at its recorded threshold, for agreement. Its own graph, not a refit."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(REPO / "models/finance_aml_gbm/onnx/model.onnx"), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    threshold = float((registry.achieved("finance_aml.gbm") or {})["decision_threshold"])
    out: list[np.ndarray] = []
    for i in range(0, len(x), 200_000):
        p = session.run(None, {name: x[i:i + 200_000].astype(np.float32)})[1]
        out.append(np.asarray([d[1] for d in p]) if isinstance(p, list) else np.asarray(p)[:, 1])
    return (np.concatenate(out) >= threshold).astype(int)


def layers_of(model: nn.Sequential) -> list[dict]:
    """Each dense layer as the encrypted paths read it: floats exactly as the float32 weights hold them."""
    linear = [m for m in model if isinstance(m, nn.Linear)]
    return [{"in": m.in_features, "out": m.out_features,
             "weight": [[float(v) for v in row] for row in m.weight.detach().cpu().numpy()],
             "bias": [float(v) for v in m.bias.detach().cpu().numpy()],
             "activation": "relu" if k < len(linear) - 1 else "none"} for k, m in enumerate(linear)]


def fixture_notices(key: str) -> dict:
    """The fixture rows are AMLworld data, so the file that carries them carries the notices CDLA-Sharing-1.0 asks for (sections 3.1 and 3.3)."""
    dataset = registry.get(key).dataset
    return {
        "licence": dataset.licence, "licence_url": CDLA_SHARING_1, "attribution": dataset.attribution,
        "modifications": (f"Not the published transaction rows: `fixtures.rows` is {N_FIXTURES} rows drawn from the held-out temporal test window, half of them laundering, "
                          "each one 111 features computed from the transactions by the training script, stored as float32. `fixtures.logits` is what the exported graph gives them."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    started = time.time()
    print("loading AMLworld HI-Small ...", flush=True)
    df = load_sorted()
    fit, val, te, tr = temporal_windows(df)
    causal = causal_account_features(df)
    base_fit, base_val, base_te = graph_features(fit, causal=causal), graph_features(val, fit_on=fit, causal=causal), graph_features(te, fit_on=tr, causal=causal)
    y_fit, y_val, y_te = (d["label"].to_numpy(dtype=np.int8) for d in (fit, val, te))
    print(f"  computing per-transaction features in batches of {BATCH_WINDOW} ...", flush=True)
    extra, extra_names = transaction_features(df)
    n_fit, n_val = len(y_fit), len(y_val)
    x_fit = np.hstack([base_fit, extra[:n_fit]])
    x_val = np.hstack([base_val, extra[n_fit:n_fit + n_val]])
    x_te = np.hstack([base_te, extra[n_fit + n_val:]])
    features = list(FEATURES) + extra_names
    if x_fit.shape[1] != 111:
        raise SystemExit(f"expected 111 features, got {x_fit.shape[1]}")
    teacher = teacher_decisions(base_te)
    print(f"  windows: fit {len(y_fit):,} ({int(y_fit.sum())} pos), val {len(y_val):,} ({int(y_val.sum())}), test {len(y_te):,} ({int(y_te.sum())})", flush=True)
    labels = torch.tensor(y_fit, dtype=torch.long, device=DEVICE)

    best: tuple | None = None
    all_runs: list[dict] = []
    for kind in SCALINGS:
        centre, scale = scaling_of(kind, x_fit)
        zf, zv, zt = _scaled(x_fit, centre, scale), _scaled(x_val, centre, scale), _scaled(x_te, centre, scale)
        clipped = float((np.abs((x_fit - centre) / scale) > CLIP).mean())
        runs, kept = [], None
        for seed in SEEDS:
            model, val_auprc, epoch = fit_one(HIDDEN, zf, labels, zv, y_val, seed)
            s_val, s_te = _positive_scores(model, zv), _positive_scores(model, zt)
            thr = _best_threshold(y_val, s_val)
            pred = (s_te >= thr).astype(int)
            row = {"scaling": kind, "seed": seed, "best_epoch": epoch, "val_auprc": round(val_auprc, 6), "test_auprc": round(float(average_precision_score(y_te, s_te)), 6),
                   "test_minority_f1": round(float(metrics.minority_class_f1(y_te, pred)), 6)}
            runs.append(row)
            print(f"  {row}", flush=True)
            if kept is None or val_auprc > kept[1]:
                kept = (model, val_auprc, seed, thr)
        all_runs += runs
        if best is None or kept[1] > best[0][1]:
            best = (kept, kind, centre, scale, zt, clipped, runs)
    (model, val_auprc, seed, thr), kind, centre, scale, zt, clipped, runs = best
    distil.fold_threshold_into_torch_head(model, thr)

    # Scored at argmax, which is what the encrypted path decides by.
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(zt[i:i + 262_144]).argmax(1) for i in range(0, len(zt), 262_144)]).cpu().numpy()
    s_te = _positive_scores(model, zt)
    f1s = sorted(r["test_minority_f1"] for r in runs)
    n_params = sum(p.numel() for p in model.parameters())
    tp = int(((pred == 1) & (y_te == 1)).sum())

    # The same features through the booster the teacher uses: the classical comparator on this model's own inputs.
    import xgboost as xgb

    spw = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    booster = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, min_child_weight=4, reg_lambda=2.0, scale_pos_weight=spw,
                                eval_metric="aucpr", tree_method="hist", device="cuda" if DEVICE == "cuda" else "cpu", n_jobs=16, random_state=0)
    booster.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
    booster.set_params(device="cpu")
    b_thr = _best_threshold(y_val, booster.predict_proba(x_val)[:, 1])
    b_score = booster.predict_proba(x_te)[:, 1]
    b_pred = (b_score >= b_thr).astype(int)

    record = {
        "dataset": registry.get(KEY).dataset.name, "split": "temporal, inductive", "n_train": int(len(y_fit)), "n_val": int(len(y_val)), "n_test": int(len(y_te)),
        "minority_class_f1": round(float(metrics.minority_class_f1(y_te, pred)), 6),
        "auprc": round(float(average_precision_score(y_te, s_te)), 6), "auprc_baseline": round(float(y_te.mean()), 8),
        "auroc": round(float(roc_auc_score(y_te, s_te)), 6), "accuracy": round(float((pred == y_te).mean()), 6),
        "precision": round(tp / max(int(pred.sum()), 1), 6), "recall": round(tp / max(int(y_te.sum()), 1), 6),
        "teacher_agreement": round(float((pred == teacher).mean()), 6), "n_parameters": int(n_params), "n_features": len(features), "batch_window": BATCH_WINDOW,
        "seed": int(seed), "selected_on": "validation AUPRC, over two scalings and five seeds each", "seeds": all_runs,
        "median_of_seeds_minority_f1": f1s[len(f1s) // 2], "min_of_seeds_minority_f1": f1s[0], "max_of_seeds_minority_f1": f1s[-1],
        "threshold_on_validation": round(float(thr), 6), "scaling": kind, "share_clipped_on_fit_window": round(clipped, 8),
        "xgboost_same_features": {"minority_class_f1": round(float(metrics.minority_class_f1(y_te, b_pred)), 6), "auprc": round(float(average_precision_score(y_te, b_score)), 6)},
        "git_sha": train.git_sha(), "data_digest": train.digest_array(x_fit[:10000], y_fit[:10000]), "wall_clock_s": round(time.time() - started, 1),
    }
    teacher_record = registry.achieved("finance_aml.gbm") or {}
    record["baseline_gap_note"] = (
        f"On HI-Small the published figures are 63.23 for GFP + XGBoost, 62.86 for GFP + LightGBM and 56.77 for PNA (Altman et al., NeurIPS 2023, Table 2), 68.16 for Multi-PNA+EU (Egressy et al., AAAI 2024), and 63.72 for a GNN "
        f"also evaluated under CKKS (Bilgi et al., arXiv:2610.18054). The Graph Feature Preprocessor paper's own decomposition (Blanusa et al., ICAIF 2024, Table 5) puts basic features at 19.75 and fan-in and fan-out at 56.88 to 59.71 for XGBoost. "
        f"These 111 features are the 32 per-account aggregates plus per-transaction fan-in and fan-out, vertex statistics and scatter-gather in time windows, in batches of {BATCH_WINDOW}, with no cycle enumeration, on a 56/14/30 temporal split rather than the papers' 60/20/20. "
        f"The comparison that means most here is with the booster on the same features: AUPRC {record['auprc']:.4f} against {record['xgboost_same_features']['auprc']:.4f}. This project's published teacher, on the 32 aggregates, has AUPRC {float(teacher_record['auprc']):.4f}. AMLworld is synthetic by IBM's own statement.")

    out = REPO / registry.ARTIFACT_DIRS[KEY]
    wrapped = Scored(centre, scale, model.cpu()).eval()
    rng = np.random.default_rng(seed)
    rows = x_te[rng.choice(len(x_te), size=512, replace=False)]
    report = export.verify_and_quantize(wrapped, rows, out, kind="torch", n_features=len(features), int8_required=False)
    # A graph this small gains nothing from int8, and the registry's student artifacts do not list one.
    (out / "onnx" / "model_int8.onnx").unlink(missing_ok=True)
    record["export"] = {"fp32_bytes": report.fp32_bytes, "opset": report.opset, "int8": "not shipped: the fp32 graph is already small"}

    # Held-out rows with their expected logits: what the encrypted paths run on and are checked against, in any language.
    fixture_index = np.concatenate([rng.choice(np.flatnonzero(y_te == 1), N_FIXTURES // 2, replace=False), rng.choice(np.flatnonzero(y_te == 0), N_FIXTURES // 2, replace=False)])
    rows_f = x_te[fixture_index].astype(np.float32)
    with torch.no_grad():
        logits_f = wrapped(torch.tensor(rows_f)).numpy()
    (out / "weights.json").write_text(json.dumps({
        "kind": "mlp", "key": KEY, "n_features": len(features), "features": features,
        "input": {"mean": None, "preprocessing": [], "note": "The ONNX graph scales its own input. The layers below take scaled input, as `scaling` says."},
        "scaling": {"kind": kind, "center": [float(v) for v in centre], "scale": [float(v) for v in scale], "clip": CLIP},
        "layers": layers_of(model), "n_parameters": int(n_params), "classes": {"0": "clean", "1": "laundering"},
        "fixtures": {"rows": [[float(v) for v in r] for r in rows_f], "labels": [int(v) for v in y_te[fixture_index]], "logits": [[float(v) for v in r] for r in logits_f], "row_index_in_test_window": [int(v) for v in fixture_index]},
        **fixture_notices(KEY),
        "note": (f"Trained on labels with class weights, on the fit window; threshold tuned on the validation window and folded into the last bias, so argmax over the two logits is the decision. "
                 f"Fixture rows are from the held-out test window, half of them laundering by construction. Features see transactions up to the end of their batch of {BATCH_WINDOW}, as the Graph Feature Preprocessor paper processes them."),
    }, indent=None, separators=(",", ":")) + "\n")
    (out / "feature_spec.json").write_text(json.dumps({"features": features, "batch_window": BATCH_WINDOW, "notes": "The first 32 are the per-account aggregates of finance_aml.py, and the rest are per-transaction window features from finance_aml_fan.py, log-compressed as that file says. The graph scales them itself."}, indent=2) + "\n")
    train.record_metrics(KEY, record)
    print(f"  {KEY}: F1 {record['minority_class_f1']} (median of seeds {record['median_of_seeds_minority_f1']}), AUPRC {record['auprc']}, booster on the same features F1 {record['xgboost_same_features']['minority_class_f1']}, {n_params} parameters, scaling {kind}", flush=True)
    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
