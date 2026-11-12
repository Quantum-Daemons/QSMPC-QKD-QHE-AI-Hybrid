#!/usr/bin/env python
"""The compact transaction GNN: message passing on a capped, padded neighbourhood, the model the finance page runs under CKKS and garbled circuits (Round 86).

WHAT IT IS AND WHY THIS SHAPE. For one transaction u -> v, 4 neighbour slots per direction at each endpoint (`finance_aml_neighbours.py`) form a tree of fixed, public shape, so the
server learns nothing from it. Each slot and each endpoint is embedded by a shared linear map, a ReLU per dimension (`A_slot` for slots, `A_end` for endpoints, no bias so an empty slot
stays zero), the slot embeddings are summed per direction, and a second layer per endpoint mixes the endpoint with its two sums. Sums over additive shares cost no AND gates, which is why
the ReLU count is the width times the accounts that carry an embedding and not the number of edges: 608 ReLUs. The published MLP's 114-feature MLP's channel runs beside it and joins at the head.

WHAT THE LITERATURE HAS AND THIS DOES NOT CLAIM. The nearest published private GNNs for financial data run on servers: Bilgi et al. (arXiv:2610.18054) under CKKS with bootstrapping on a GPU
and TGHE (arXiv:2606.26664) under CKKS on capped ego trees with quadratic activations. A search of the literature to November 2026 found no private GNN in a browser and none under garbled
circuits, which is recorded as a search result and not as a claim of being first. The official Multi-GNN models reach minority-class F1 of about 0.65 to 0.68 on HI-Small with 100 by 100 sampled
neighbours and edges from the whole test window; this model reads 16 edges and nothing later than its batch.

WHAT WAS CHOSEN ON THE VALIDATION WINDOW, and nothing here looks at the test window to choose: the shape, the learning rate and the weight decay in a sweep recorded in `references/REFERENCES.md`;
then here the endpoint scaling and the seed, five seeds each, the best validation AUPRC shipping with every run recorded. Threshold on validation, folded into the last bias.

DISTILLED FROM IBM'S MULTI-PNA+EU (Round 86). With `--teacher-probs`, the target on each fit-window edge is (1 - alpha) * label + alpha * the
probability IBM's official Multi-PNA+EU gives it, run on this project's windows (`scripts/train/ibm_multignn/`); an edge it never scored keeps its
label. This is generalised distillation in the sense of Lopez-Paz, Bottou, Schoelkopf and Vapnik (ICLR 2016, arXiv:1511.03643): PNA's
neighbourhoods on the fit window include transactions later than the one scored, information this model never sees, and it reaches this model only
through the training targets, so nothing changes at inference. alpha was chosen on the validation window (0, 0.3, 0.5 and 0.7, three seeds each:
mean validation AUPRC 0.5440, 0.5817, 0.5809 and 0.5802).

    python scripts/train/finance_aml_gnn.py [--teacher-probs <dir>/train.npz --alpha 0.3]
"""

from __future__ import annotations

import argparse
import hashlib
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
    _best_threshold,
    causal_account_features,
    load_sorted,
    model_inputs,
    network_calibration,
    temporal_windows,
)
from finance_aml_fan import BATCH_WINDOW  # noqa: E402
from finance_aml_neighbours import D, neighbourhoods, statistic_names  # noqa: E402

from qsh.models import distil, metrics, registry, train  # noqa: E402

KEY = "finance_aml.gnn_compact"
H, HQ, HQ2, H2, HEAD = 16, 128, 64, 32, 64
SEEDS = (0, 1, 2, 3, 4)
ENDPOINT_SCALINGS = ("none", "standard")
CLIP = 10.0
EPOCHS, PATIENCE, BATCH, LR, WEIGHT_DECAY = 60, 8, 16_384, 3e-4, 1e-4
N_FIXTURES = 16
CDLA_SHARING_1 = "https://cdla.dev/sharing-1-0/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_QUERY, N_STATS = 114, 16


class CompactGnn(nn.Module):
    """Two endpoints, 4D slots: see the module docstring. `relu_count` is what the garbled circuits pay."""

    def __init__(self, n_slot: int) -> None:
        super().__init__()
        self.q1, self.q2 = nn.Linear(N_QUERY, HQ), nn.Linear(HQ, HQ2)
        self.a_end, self.a_slot = nn.Linear(N_STATS, H, bias=False), nn.Linear(n_slot, H, bias=False)
        self.me = nn.ModuleList([nn.Linear(H, H2, bias=False) for _ in range(2)])
        self.incoming = nn.ModuleList([nn.Linear(H, H2, bias=False) for _ in range(2)])
        self.outgoing = nn.ModuleList([nn.Linear(H, H2, bias=False) for _ in range(2)])
        self.bias2 = nn.Parameter(torch.zeros(2, H2))
        self.c = nn.Linear(HQ2 + 2 * H2, HEAD)
        self.d = nn.Linear(HEAD, 2)

    relu_count = HQ + (2 + 4 * D) * H + HQ2 + 2 * H2 + HEAD

    def forward(self, x: torch.Tensor, ends: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        relu = torch.relu
        query = relu(self.q2(relu(self.q1(x))))
        he, hs = relu(self.a_end(ends)), relu(self.a_slot(slots))
        parts = []
        for r in range(2):
            first = 2 * r * D
            into, outof = hs[:, first:first + D].sum(1), hs[:, first + D:first + 2 * D].sum(1)
            parts.append(relu(self.me[r](he[:, r]) + self.incoming[r](into) + self.outgoing[r](outof) + self.bias2[r]))
        return self.d(relu(self.c(torch.cat([query, *parts], 1))))


def from_weights(w: dict) -> CompactGnn:
    """The network a `weights.json` describes, on the CPU and in eval mode: the inverse of what `main()` writes."""
    L = w["layers"]
    model = CompactGnn(len(w["slots"]["features"]))

    def put(lin: nn.Linear, weight, bias=None) -> None:
        lin.weight.data = torch.tensor(weight, dtype=torch.float32)
        if bias is not None:
            lin.bias.data = torch.tensor(bias, dtype=torch.float32)

    put(model.q1, L["q1"]["weight"], L["q1"]["bias"])
    put(model.q2, L["q2"]["weight"], L["q2"]["bias"])
    put(model.a_end, L["a_end"])
    put(model.a_slot, L["a_slot"])
    for r in range(2):
        put(model.me[r], L["second"][r]["me"])
        put(model.incoming[r], L["second"][r]["incoming"])
        put(model.outgoing[r], L["second"][r]["outgoing"])
        model.bias2.data[r] = torch.tensor(L["second"][r]["bias"], dtype=torch.float32)
    put(model.c, L["c"]["weight"], L["c"]["bias"])
    put(model.d, L["d"]["weight"], L["d"]["bias"])
    return model.eval()


def _scores(model: nn.Module, data: tuple[torch.Tensor, ...]) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([torch.softmax(model(*(t[i:i + 262_144] for t in data)), 1)[:, 1] for i in range(0, len(data[0]), 262_144)]).cpu().numpy()


def fit_one(n_slot: int, train_data: tuple[torch.Tensor, ...], labels: torch.Tensor, val_data: tuple[torch.Tensor, ...], y_val: np.ndarray, seed: int,
            target: torch.Tensor | None = None) -> tuple[CompactGnn, float, int]:
    """Class-weighted cross-entropy with early stopping on validation AUPRC, as the published MLP's trainer does.

    With `target`, the probability of laundering each fit-window edge is trained towards (the distillation target), the same
    class weights, by the edge's label, multiply a cross-entropy against that soft target."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CompactGnn(n_slot).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n, pos = len(labels), int(labels.sum())
    w = torch.tensor([n / (2 * (n - pos)), n / (2 * pos)], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    best, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            logits = model(*(t[b] for t in train_data))
            if target is None:
                loss = loss_fn(logits, labels[b])
            else:
                logp, cw = torch.log_softmax(logits, 1), w[labels[b]]
                loss = -(cw * (target[b] * logp[:, 1] + (1 - target[b]) * logp[:, 0])).sum() / cw.sum()
            loss.backward()
            opt.step()
        model.eval()
        score = float(average_precision_score(y_val, _scores(model, val_data)))
        if score > best + 1e-5:
            best, best_state, best_epoch, stale = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, best_epoch


def _dense(m: nn.Linear) -> dict:
    return {"weight": [[float(v) for v in row] for row in m.weight.detach().cpu().numpy()], "bias": [float(v) for v in m.bias.detach().cpu().numpy()]}


def _matrix(m: nn.Linear) -> list[list[float]]:
    return [[float(v) for v in row] for row in m.weight.detach().cpu().numpy()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-probs", type=Path, default=None,
                    help="train.npz from scripts/train/ibm_multignn/dump_logits.py: edge ids and Multi-PNA+EU probabilities on the fit window")
    ap.add_argument("--alpha", type=float, default=0.3, help="weight of the teacher's probability in the target; chosen on validation")
    args = ap.parse_args()
    started = time.time()
    print("loading AMLworld HI-Small ...", flush=True)
    df = load_sorted()
    fit, val, te, tr = temporal_windows(df)
    print(f"  the shared inputs, in batches of {BATCH_WINDOW} ...", flush=True)
    matrices, names, reach = model_inputs(df, fit, val, te, tr, causal_account_features(df))
    x = np.vstack(matrices)
    if x.shape[1] != N_QUERY:
        raise SystemExit(f"expected {N_QUERY} features, got {x.shape[1]}")
    print("  neighbourhoods ...", flush=True)
    nb = neighbourhoods(df, DEVICE, horizon_minutes=reach)
    n_fit, n_val = len(fit), len(val)
    n_all = n_fit + n_val
    y = df["label"].to_numpy(dtype=np.int8)
    y_val, y_te = y[n_fit:n_all], y[n_all:]
    print(f"  windows: fit {n_fit:,}, val {n_val:,}, test {len(y_te):,} ({int(y_te.sum())} laundering)", flush=True)

    centre, scale = x[:n_fit].mean(0), np.where(x[:n_fit].std(0) == 0, 1.0, x[:n_fit].std(0))
    X = torch.tensor(np.clip((x - centre) / scale, -CLIP, CLIP), dtype=torch.float32, device=DEVICE)
    # Amount and time gap are scaled by their spread on the fit window's filled slots and not centred, so an empty slot stays zero; the flags and the one-hot are already 0 or 1.
    slot_scale = torch.ones(nb["slots"].shape[-1])
    slot_scale[:2] = nb["slots"][:n_fit][nb["filled"][:n_fit]][:, :2].std(0)
    S = (nb["slots"] / slot_scale).clamp(-CLIP, CLIP).to(DEVICE)
    labels = torch.tensor(y[:n_fit].astype(np.int64), device=DEVICE)
    ends_raw = nb["endpoints"]
    target, distillation = None, None
    if args.teacher_probs is not None:
        blob = args.teacher_probs.read_bytes()
        teacher = np.load(args.teacher_probs)
        ids, probs, tlab = teacher["ids"], teacher["probs"], teacher["labels"]
        if ids.min() < 0 or ids.max() >= n_fit or not np.array_equal(tlab, y[ids]):
            raise SystemExit(f"{args.teacher_probs}: its edge ids or labels are not this fit window's rows")
        soft = y[:n_fit].astype(np.float32)
        soft[ids] = (1 - args.alpha) * soft[ids] + args.alpha * probs.astype(np.float32)
        target = torch.tensor(soft, device=DEVICE)
        distillation = {
            "teacher": ("IBM Multi-PNA+EU, the official code IBM/Multi-GNN at 252b0252afca109d1d216c411c59ff70753b25fc with "
                        "scripts/train/ibm_multignn/project_split.patch, on this project's windows, seed 1, the checkpoint of its best validation F1"),
            "alpha": args.alpha, "teacher_probs_sha256": hashlib.sha256(blob).hexdigest(),
            "fit_edges_scored_by_the_teacher": int(len(ids)), "fit_edges": int(n_fit),
            "chosen_on": "validation AUPRC over alpha 0, 0.3, 0.5 and 0.7, three seeds each (0.5440, 0.5817, 0.5809, 0.5802)",
            "method": "generalised distillation, Lopez-Paz, Bottou, Schoelkopf and Vapnik, ICLR 2016, arXiv:1511.03643",
        }
        print(f"  distilling from {args.teacher_probs} at alpha {args.alpha}: {len(ids):,} of {n_fit:,} fit edges carry a teacher probability", flush=True)

    all_runs: list[dict] = []
    best: tuple | None = None
    for kind in ENDPOINT_SCALINGS:
        if kind == "none":
            e_centre, e_scale = torch.zeros(N_STATS), torch.ones(N_STATS)
        else:
            flat = ends_raw[:n_fit].reshape(-1, N_STATS)
            e_centre, e_scale = flat.mean(0), flat.std(0).clamp(min=1e-6)
        E = ((ends_raw - e_centre) / e_scale).clamp(-CLIP, CLIP).to(DEVICE)
        fit_data, val_data, te_data = ((X[a:b], E[a:b], S[a:b]) for a, b in ((0, n_fit), (n_fit, n_all), (n_all, len(y))))
        kept = None
        for seed in SEEDS:
            model, val_auprc, epoch = fit_one(S.shape[-1], fit_data, labels, val_data, y_val, seed, target)
            s_val, s_te = _scores(model, val_data), _scores(model, te_data)
            thr = _best_threshold(y_val, s_val)
            row = {"endpoint_scaling": kind, "seed": seed, "best_epoch": epoch, "val_auprc": round(val_auprc, 6), "test_auprc": round(float(average_precision_score(y_te, s_te)), 6),
                   "test_minority_f1": round(float(metrics.minority_class_f1(y_te, (s_te >= thr).astype(int))), 6)}
            all_runs.append(row)
            print(f"  {row}", flush=True)
            if kept is None or val_auprc > kept[1]:
                kept = (model, val_auprc, seed, thr)
        if best is None or kept[1] > best[0][1]:
            best = (kept, kind, e_centre, e_scale, te_data)
    (model, val_auprc, seed, thr), kind, e_centre, e_scale, te_data = best
    distil.fold_threshold_into_torch_head(model, thr)
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(*(t[i:i + 262_144] for t in te_data)) for i in range(0, len(y_te), 262_144)])
    pred = logits.argmax(1).cpu().numpy()
    s_te = torch.softmax(logits, 1)[:, 1].cpu().numpy()
    # The folded network's Platt map, fitted on validation and scored on test, as the teacher's and the student's records carry one.
    E = ((ends_raw - e_centre) / e_scale).clamp(-CLIP, CLIP).to(DEVICE)
    val_data = (X[n_fit:n_all], E[n_fit:n_all], S[n_fit:n_all])
    with torch.no_grad():
        logits_val = torch.cat([model(*(t[i:i + 262_144] for t in val_data)) for i in range(0, n_val, 262_144)]).cpu().numpy()
    calibration = network_calibration(logits_val, y_val, logits.cpu().numpy(), y_te, fitted_by=f"scripts/train/finance_aml_gnn.py at git {train.git_sha()}, in the run that trained these weights", model="GNN")
    runs = [r for r in all_runs if r["endpoint_scaling"] == kind]
    f1s = sorted(r["test_minority_f1"] for r in runs)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    n_params = sum(p.numel() for p in model.parameters())
    record = {
        "dataset": registry.get(KEY).dataset.name, "split": "temporal, inductive, the published MLP's windows", "n_train": n_fit, "n_val": n_val, "n_test": len(y_te),
        "minority_class_f1": round(float(metrics.minority_class_f1(y_te, pred)), 6), "auprc": round(float(average_precision_score(y_te, s_te)), 6), "auprc_baseline": round(float(y_te.mean()), 8),
        "auroc": round(float(roc_auc_score(y_te, s_te)), 6), "accuracy": round(float((pred == y_te).mean()), 6), "precision": round(tp / max(int(pred.sum()), 1), 6), "recall": round(tp / max(int(y_te.sum()), 1), 6),
        "n_parameters": int(n_params), "n_features": N_QUERY, "n_slots_per_direction": D, "relu_count": CompactGnn.relu_count, "batch_window": BATCH_WINDOW, "batch_reach_minutes": reach, "seed": int(seed),
        "selected_on": "validation AUPRC, over two endpoint scalings and five seeds each", "seeds": all_runs, "endpoint_scaling": kind,
        "median_of_seeds_minority_f1": f1s[len(f1s) // 2], "min_of_seeds_minority_f1": f1s[0], "max_of_seeds_minority_f1": f1s[-1], "threshold_on_validation": round(float(thr), 6),
        "calibration": calibration,
        **({"distillation": distillation} if distillation else {}),
        "git_sha": train.git_sha(), "data_digest": train.digest_array(x[:10000], y[:10000]), "wall_clock_s": round(time.time() - started, 1),
    }
    mlp = json.loads((REPO / registry.ARTIFACT_DIRS["finance_aml.mlp_large"] / "weights.json").read_text())
    fixture_index = np.array(mlp["fixtures"]["row_index_in_test_window"], dtype=np.int64)
    index_all = fixture_index + n_all
    with torch.no_grad():
        recorded = model(X[index_all], ((ends_raw[index_all] - e_centre) / e_scale).clamp(-CLIP, CLIP).to(DEVICE), S[index_all]).cpu().numpy()

    def describe(i: int) -> list[dict]:
        """What the page's figure names: each filled slot's direction, partner, amount and format. Accounts are letters in the order they appear, per row."""
        row = df.iloc[int(i)]
        letters = {f"{row.from_bank}{row.from_account}": "A", f"{row.to_bank}{row.to_account}": "B"}
        def letter(acct: str) -> str:
            return letters.setdefault(acct, chr(ord("A") + len(letters)))
        out = []
        for s, e in enumerate(nb["edges"][i].tolist()):
            if not bool(nb["filled"][i, s]):
                continue
            ed = df.iloc[e]
            out.append({"slot": s, "from": letter(f"{ed.from_bank}{ed.from_account}"), "to": letter(f"{ed.to_bank}{ed.to_account}"), "amount": round(float(ed.amount_paid), 2), "currency": str(ed.pay_currency), "format": str(ed.payment_format), "minutes_before_scored": int((row.timestamp - ed.timestamp).total_seconds() / 60)})
        return out

    out = REPO / registry.ARTIFACT_DIRS[KEY]
    out.mkdir(parents=True, exist_ok=True)
    dataset = registry.get(KEY).dataset
    (out / "weights.json").write_text(json.dumps({
        "kind": "gnn", "key": KEY, "classes": {"0": "clean", "1": "laundering"}, "n_parameters": int(n_params), "relu_count": CompactGnn.relu_count,
        "query": {"features": names, "scaling": {"kind": "standard", "center": [float(v) for v in centre], "scale": [float(v) for v in scale], "clip": CLIP}},
        "endpoints": {"features": statistic_names(), "scaling": {"kind": kind, "center": [float(v) for v in e_centre], "scale": [float(v) for v in e_scale], "clip": CLIP}},
        "slots": {"per_direction": D, "order": ["into u", "out of u", "into v", "out of v"], "formats": nb["formats"], "features": ["log_amount", "log_minutes_before_batch_end", "cross_currency", "same_bank", "partner_is_other_endpoint", "partner_also_at_other_endpoint"] + [f"format_{f}" for f in nb["formats"]],
                  "scaling": {"kind": "scale only, so an empty slot stays zero", "scale": [float(v) for v in slot_scale], "clip": CLIP}},
        "layers": {"q1": _dense(model.q1), "q2": _dense(model.q2), "a_end": _matrix(model.a_end), "a_slot": _matrix(model.a_slot),
                   "second": [{"me": _matrix(model.me[r]), "incoming": _matrix(model.incoming[r]), "outgoing": _matrix(model.outgoing[r]), "bias": [float(v) for v in model.bias2[r].detach().cpu().numpy()]} for r in range(2)],
                   "c": _dense(model.c), "d": _dense(model.d)},
        "fixtures": {"labels": [int(v) for v in y_te[fixture_index]], "logits": [[float(v) for v in r] for r in recorded], "row_index_in_test_window": [int(v) for v in fixture_index],
                     "x": [[float(v) for v in x[i]] for i in index_all], "endpoints": [[[float(v) for v in r] for r in ends_raw[i]] for i in index_all],
                     "slots": [[[float(v) for v in r] for r in nb["slots"][i]] for i in index_all], "neighbourhoods": [describe(int(i)) for i in index_all]},
        "licence": dataset.licence, "licence_url": CDLA_SHARING_1, "attribution": dataset.attribution,
        "modifications": (f"Not the published transaction rows: `fixtures` holds {N_FIXTURES} rows of the held-out temporal test window, the published MLP's fixture rows, half of them laundering. Each is {N_QUERY} features, 2 endpoint statistics vectors and "
                          f"{4 * D} neighbour-slot vectors computed from the transactions by the training scripts, stored as float32, and `neighbourhoods` names each row's accounts by a letter, in order of appearance, with the amount, currency and format of each edge. `logits` is what the network gives them."),
        "note": (f"Trained on labels with class weights on the fit window; threshold tuned on the validation window and folded into the last bias, so argmax over the two logits is the decision. Neighbours and features see transactions up to the end of their batch of {BATCH_WINDOW}, at most {reach} minutes past the transaction, as the Graph Feature Preprocessor paper processes them."),
    }, indent=None, separators=(",", ":")) + "\n")
    train.record_metrics(KEY, record)
    print(f"  {KEY}: F1 {record['minority_class_f1']} (median of seeds {record['median_of_seeds_minority_f1']}), AUPRC {record['auprc']}, AUROC {record['auroc']}, {CompactGnn.relu_count} ReLUs, {n_params} parameters, endpoint scaling {kind}")
    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
