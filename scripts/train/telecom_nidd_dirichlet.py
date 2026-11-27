#!/usr/bin/env python3
"""5G-NIDD federated over eight Dirichlet-split clients, with a 1D-CNN baseline, as pre-registered
(`references/telecom-nidd-dirichlet-preregistration.md`; Round 86, batch 8).

The two-station federation's data, split, standardisation, local step and masked aggregation are imported from
`telecom_nidd_federated.py` and its student, not typed again; only the number of clients and the partition differ. The
linear arm runs on the CPU; the 1D-CNN arms on `--device` (the owner approved the GPU on 2026-11-27).
Writes docs/measurements/telecom-nidd-dirichlet-2026-11-27.json and web/public/data/telecom_nidd/dirichlet.json.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/train/telecom_nidd_dirichlet.py [--device cuda] [--seeds 1 2 3 4 5]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.ai.federated import SecureAggregator  # noqa: E402
from qsh.models import datasets as D  # noqa: E402
from qsh.models.registry import DS_5G_NIDD  # noqa: E402

PREREG = REPO / "references" / "telecom-nidd-dirichlet-preregistration.md"
PREREG_SHA256 = "0551ea0242bd6a698991c6d555761fd78a147dfb0b29479f6d015c16ec3ebb19"  # committed in 16abef9
RECORD = REPO / "docs" / "measurements" / "telecom-nidd-dirichlet-2026-11-27.json"
WEB = REPO / "web" / "public" / "data" / "telecom_nidd" / "dirichlet.json"
CLIENTS, ALPHA, MIN_FIT_ROWS = 8, 0.3, 100
CNN_ROUNDS, CNN_BATCH, CNN_LR = 30, 256, 1e-3


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _module("telecom_nidd_federated", REPO / "scripts" / "train" / "telecom_nidd_federated.py")
S = F.S


def partition(y: np.ndarray, yv: np.ndarray, k: int, seed: int) -> tuple[list[np.ndarray], list[np.ndarray], int, list[int]]:
    """Yurochkin et al.'s heterogeneous partition: for each class, its fit rows shuffled and divided among the clients by
    proportions drawn from Dir(alpha), its validation rows in the same proportions. Redrawn from 100 seed + j while a
    client holds fewer than MIN_FIT_ROWS fit rows; returns the clients' fit and validation indices, the seed used and
    every seed tried."""
    tried = []
    for j in range(0, 1000):
        s = seed if j == 0 else 100 * seed + j
        tried.append(s)
        rng = np.random.default_rng(s)
        fit = [[] for _ in range(CLIENTS)]
        val = [[] for _ in range(CLIENTS)]
        for c in range(k):
            p = rng.dirichlet(ALPHA * np.ones(CLIENTS))
            for rows, out in ((np.flatnonzero(y == c), fit), (np.flatnonzero(yv == c), val)):
                rows = rows[rng.permutation(len(rows))]
                cuts = np.floor(np.cumsum(p)[:-1] * len(rows)).astype(int)
                for client, part in enumerate(np.split(rows, cuts)):
                    out[client].extend(part.tolist())
        fit_a = [np.sort(np.asarray(f, dtype=np.int64)) for f in fit]
        if min(len(f) for f in fit_a) >= MIN_FIT_ROWS:
            return fit_a, [np.sort(np.asarray(v, dtype=np.int64)) for v in val], s, tried
    raise SystemExit(f"seed {seed}: no partition with every client at {MIN_FIT_ROWS} fit rows in 1,000 draws")


class Channel(F.Channel):
    """The two-station channel's aggregation and its exactness check, for eight clients."""

    def __init__(self) -> None:
        self.agg = SecureAggregator(n_clients=CLIENTS)
        self.messages = 0


def weighted_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    _, per = S.macro_f1(y, pred, k)
    support = np.bincount(y, minlength=k).astype(np.float64)
    return float(np.dot(per, support) / support.sum())


def cnn(k: int):
    """This project's reading of Jeong et al.'s network: 36 -> 128 linear, read as a length-128 one-channel sequence;
    1D convolutions of 128 and 64 filters, kernel 3, each with ReLU; max pool of 2; flatten; 128 with ReLU; k."""
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(36, 128)
            self.conv = nn.Sequential(nn.Conv1d(1, 128, 3), nn.ReLU(), nn.Conv1d(128, 64, 3), nn.ReLU(), nn.MaxPool1d(2))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 62, 128), nn.ReLU(), nn.Linear(128, k))

        def forward(self, x):
            return self.head(self.conv(self.project(x).unsqueeze(1)))

    return Net()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    started = time.time()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    import torch
    from scipy.optimize import minimize_scalar
    from sklearn.model_selection import train_test_split

    # ---- the federation's data, split and standardisation, exactly
    x, y, _names, classes = D.load_5g_nidd()
    k = len(classes)
    cut = D.NIDD_SECOND_FILE_STARTS_AT_CLEANED
    onehot = np.eye(k)
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42, stratify=y)
    xtr, xte, ytr, yte = x[itr], x[ite], y[itr], y[ite]
    fi, vi = train_test_split(np.arange(len(ytr)), test_size=0.10, random_state=0, stratify=ytr)
    numeric = [i for i, n in enumerate(D.NIDD_FEATURES) if n not in D.NIDD_CATEGORICAL]
    st_te = (ite >= cut).astype(int)
    centre = xtr[:, numeric].mean(axis=0).astype(np.float64)
    scale = xtr[:, numeric].std(axis=0).astype(np.float64)
    scale[scale == 0] = 1.0
    xf, yf, xv, yv = xtr[fi], ytr[fi], xtr[vi], ytr[vi]
    record: dict = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
                    "of_federation": {"file": str(F.RECORD.relative_to(REPO)), "sha256": hashlib.sha256(F.RECORD.read_bytes()).hexdigest()},
                    "rows": {"fit": int(len(yf)), "validation": int(len(yv)), "test": int(len(yte))},
                    "clients": CLIENTS, "alpha": ALPHA, "device_cnn": args.device,
                    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                    "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None},
                    "seeds": {}}

    def test_scores(logits: np.ndarray) -> dict:
        pred = logits.argmax(1)
        per_station = {F.KEYS[s]: round(S.macro_f1(yte[st_te == s], pred[st_te == s], k)[0], 6) for s in (0, 1)}
        return {"station_macro_f1": per_station, "flamby_metric": round(float(np.mean(list(per_station.values()))), 6),
                "union_macro_f1": round(S.macro_f1(yte, pred, k)[0], 6), "union_weighted_f1": round(weighted_f1(yte, pred, k), 6),
                "union_accuracy": round(float(np.mean(pred == yte)), 6)}

    for seed in args.seeds:
        fit_c, val_c, used, tried = partition(yf, yv, k, seed)
        # The clients' standardisation, combined by the parallel formula, checked against the federation's.
        parts = []
        for c in range(CLIENTS):
            rows = np.concatenate([xf[fit_c[c]], xv[val_c[c]]])
            parts.append((len(rows), rows[:, numeric].mean(axis=0).astype(np.float64), rows[:, numeric].std(axis=0).astype(np.float64)))
        n_all = sum(n for n, _, _ in parts)
        c_f = sum(n * c_ for n, c_, _ in parts) / n_all
        s_f = np.sqrt(sum(n * (s_ * s_ + (c_ - c_f) ** 2) for n, c_, s_ in parts) / n_all)
        s_f[s_f == 0] = 1.0
        dev = float(max(np.max(np.abs(c_f - centre) / scale), np.max(np.abs(s_f - scale) / scale)))
        if dev > F.STANDARDISATION_TOLERANCE:
            raise SystemExit(f"seed {seed}: the clients' combined standardisation is {dev:.2e} from the federation's")

        def z(a: np.ndarray, c_f: np.ndarray = c_f, s_f: np.ndarray = s_f) -> np.ndarray:
            return np.clip((a[:, numeric] - c_f) / s_f, -S.CLIP, S.CLIP)

        Zf = [z(xf[i]) for i in fit_c]
        Yf = [yf[i] for i in fit_c]
        Zv = [z(xv[i]) for i in val_c]
        Yv = [yv[i] for i in val_c]
        Zt = z(xte)
        n_fit = np.array([len(v) for v in Yf], dtype=np.float64)
        n_val = np.array([len(v) for v in Yv], dtype=np.float64)
        w_fit, w_val = n_fit / n_fit.sum(), n_val / n_val.sum()
        info = {"partition_seed_used": used, "seeds_tried": tried,
                "fit_rows_by_client": n_fit.astype(int).tolist(), "validation_rows_by_client": n_val.astype(int).tolist(),
                "class_counts_by_client": [np.bincount(v, minlength=k).tolist() for v in Yf],
                "standardisation_max_relative_difference": dev}
        print(f"seed {seed}: partition {used}, fit rows by client {info['fit_rows_by_client']}", flush=True)

        # ---- arm 1: linear FedAvg under pairwise masks, the federation's procedure with eight clients
        channel = Channel()
        d = Zf[0].shape[1]

        def unpack(theta: np.ndarray, d: int = d) -> tuple[np.ndarray, np.ndarray]:
            return theta[: k * d].reshape(k, d), theta[k * d:]

        def pooled_val_loss(W: np.ndarray, b: np.ndarray, T: float = 1.0, channel: Channel = channel,
                            w_val: np.ndarray = w_val, Zv: list = Zv, Yv: list = Yv) -> float:
            return float(channel.sum([np.array([w_val[c] * S.log_loss(S.softmax((Zv[c] @ W.T + b) / T), onehot[Yv[c]])])
                                      for c in range(CLIENTS)])[0])

        runs = {}
        for l2 in S.L2_GRID:
            theta = np.zeros(k * d + k)
            for _ in range(F.ROUNDS):
                local = [F.fit_from(Zf[c], onehot[Yf[c]], l2, theta, F.LOCAL_ITERATIONS) for c in range(CLIENTS)]
                theta = channel.sum([w_fit[c] * local[c] for c in range(CLIENTS)])
            runs[l2] = {"theta": theta, "validation_log_loss": round(pooled_val_loss(*unpack(theta)), 8)}
            print(f"  linear l2 {l2:g}: pooled validation log loss {runs[l2]['validation_log_loss']:.6f} "
                  f"({time.time() - started:.0f} s)", flush=True)
        table = [{"l2": l2, "validation_log_loss": runs[l2]["validation_log_loss"]} for l2 in S.L2_GRID]
        best = min(table, key=lambda r: (r["validation_log_loss"], r["l2"]))["l2"]
        W, b = unpack(runs[best]["theta"])
        T = float(np.exp(minimize_scalar(lambda t, W=W, b=b: pooled_val_loss(W, b, float(np.exp(t))), bounds=(-4, 4),
                                         method="bounded").x))
        linear = {"penalty_table": table, "l2": best, "temperature": T, "masked_messages": channel.messages,
                  "test": test_scores((Zt @ W.T + b) / T)}
        print(f"  linear: l2 {best:g}, T {T:.4f}, FLamby {linear['test']['flamby_metric']:.4f}", flush=True)

        # ---- arms 2 and 3: the 1D-CNN, federated over the same clients and pooled, in the clear
        dev_t = torch.device(args.device)
        Zv_all, Yv_all = np.vstack(Zv), np.concatenate(Yv)

        def logits_of(net, Z: np.ndarray, dev_t: torch.device = dev_t) -> np.ndarray:
            net.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(Z), 8192):
                    out.append(net(torch.tensor(Z[i:i + 8192], dtype=torch.float32, device=dev_t)).double().cpu().numpy())
            return np.concatenate(out)

        def epoch(net, Z: np.ndarray, Y: np.ndarray, g: torch.Generator, opt=None, dev_t: torch.device = dev_t) -> None:
            """One pass over Z in a shuffled order; a federated client starts a fresh Adam each round, the pooled
            reference keeps one across its epochs, as centralised training does."""
            net.train()
            opt = opt or torch.optim.Adam(net.parameters(), lr=CNN_LR)
            loss_fn = torch.nn.CrossEntropyLoss()
            Zt_, Yt_ = torch.tensor(Z, dtype=torch.float32, device=dev_t), torch.tensor(Y, dtype=torch.long, device=dev_t)
            order = torch.randperm(len(Y), generator=g).to(dev_t)
            for i in range(0, len(Y), CNN_BATCH):
                idx = order[i:i + CNN_BATCH]
                opt.zero_grad()
                loss_fn(net(Zt_[idx]), Yt_[idx]).backward()
                opt.step()

        def averaged(states: list[dict], weights: np.ndarray) -> dict:
            return {key: sum(float(w) * s[key].float() for s, w in zip(states, weights, strict=True)) for key in states[0]}

        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        net = cnn(k).to(dev_t)
        best_cnn = (-1.0, None, None)
        for r in range(CNN_ROUNDS):
            start = {key: v.clone() for key, v in net.state_dict().items()}
            states = []
            for c in range(CLIENTS):
                net.load_state_dict(start)
                epoch(net, Zf[c], Yf[c], g)
                states.append({key: v.detach().clone() for key, v in net.state_dict().items()})
            net.load_state_dict(averaged(states, w_fit))
            v = S.macro_f1(Yv_all, logits_of(net, Zv_all).argmax(1), k)[0]
            if v > best_cnn[0]:
                best_cnn = (v, r + 1, {key: t.clone() for key, t in net.state_dict().items()})
        net.load_state_dict(best_cnn[2])
        fed_cnn = {"round": best_cnn[1], "validation_macro_f1": round(best_cnn[0], 6), "test": test_scores(logits_of(net, Zt))}
        print(f"  1D-CNN federated: round {best_cnn[1]}, FLamby {fed_cnn['test']['flamby_metric']:.4f} "
              f"({time.time() - started:.0f} s)", flush=True)

        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        net = cnn(k).to(dev_t)
        Zf_all, Yf_all = np.vstack(Zf), np.concatenate(Yf)
        best_pool = (-1.0, None, None)
        opt = torch.optim.Adam(net.parameters(), lr=CNN_LR)
        for e in range(CNN_ROUNDS):
            epoch(net, Zf_all, Yf_all, g, opt)
            v = S.macro_f1(Yv_all, logits_of(net, Zv_all).argmax(1), k)[0]
            if v > best_pool[0]:
                best_pool = (v, e + 1, {key: t.clone() for key, t in net.state_dict().items()})
        net.load_state_dict(best_pool[2])
        pooled_cnn = {"epoch": best_pool[1], "validation_macro_f1": round(best_pool[0], 6), "test": test_scores(logits_of(net, Zt))}
        print(f"  1D-CNN pooled: epoch {best_pool[1]}, FLamby {pooled_cnn['test']['flamby_metric']:.4f}", flush=True)
        record["seeds"][str(seed)] = {"partition": info, "linear_fedavg_masked": linear, "cnn_fedavg": fed_cnn,
                                      "cnn_pooled": pooled_cnn}
        RECORD.write_text(json.dumps(record, indent=1) + "\n")

    # ---- summaries and the two stated comparisons
    fed = json.loads(F.RECORD.read_text())

    def summary(arm: str) -> dict:
        vals = {m: [record["seeds"][s][arm]["test"][m] for s in record["seeds"]]
                for m in ("flamby_metric", "union_macro_f1", "union_weighted_f1", "union_accuracy")}
        return {m: {"mean": round(float(np.mean(v)), 6), "sd": round(float(np.std(v, ddof=1)), 6) if len(v) > 1 else None}
                for m, v in vals.items()}

    record["summary"] = {arm: summary(arm) for arm in ("linear_fedavg_masked", "cnn_fedavg", "cnn_pooled")}
    pooled_flamby = fed["comparison"]["pooled"]["flamby_metric"]
    lin = record["summary"]["linear_fedavg_masked"]["flamby_metric"]["mean"]
    record["C1"] = {"linear_eight_clients_mean_flamby": lin, "pooled_flamby": pooled_flamby,
                    "gap": round(pooled_flamby - lin, 6), "within_0_01": bool(pooled_flamby - lin <= 0.01)}
    from scipy import stats as st

    diffs = np.array([record["seeds"][s]["cnn_fedavg"]["test"]["flamby_metric"]
                      - record["seeds"][s]["linear_fedavg_masked"]["test"]["flamby_metric"] for s in record["seeds"]])
    half = float(st.t.ppf(0.975, len(diffs) - 1) * np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else None
    record["C2"] = {"cnn_minus_linear_mean_flamby": round(float(diffs.mean()), 6),
                    "ci95": [round(float(diffs.mean() - half), 6), round(float(diffs.mean() + half), 6)] if half is not None else None,
                    "seeds": len(diffs)}
    record["references"] = {"two_station_federated_flamby": fed["comparison"]["federated"]["flamby_metric"],
                            "pooled_flamby": pooled_flamby,
                            "alone_flamby": [a["flamby_metric"] for a in fed["comparison"]["alone"]]}
    record["attribution"] = {"dataset": "5G-NIDD", "licence": DS_5G_NIDD.licence, "citation": DS_5G_NIDD.attribution}
    record["wall_clock_s"] = round(time.time() - started, 1)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    WEB.write_text(json.dumps({"_generated_by": "scripts/train/telecom_nidd_dirichlet.py; do not edit by hand",
                               "preregistration": record["preregistration"], "record": str(RECORD.relative_to(REPO)),
                               "clients": CLIENTS, "alpha": ALPHA, "summary": record["summary"], "C1": record["C1"],
                               "C2": record["C2"], "references": record["references"], "device_cnn": args.device,
                               "per_seed": {s: {arm: v[arm]["test"] for arm in ("linear_fedavg_masked", "cnn_fedavg", "cnn_pooled")}
                                            for s, v in record["seeds"].items()},
                               "licence": DS_5G_NIDD.licence, "attribution": {"citation": DS_5G_NIDD.attribution}},
                              indent=1) + "\n")
    print(json.dumps({"C1": record["C1"], "C2": record["C2"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
