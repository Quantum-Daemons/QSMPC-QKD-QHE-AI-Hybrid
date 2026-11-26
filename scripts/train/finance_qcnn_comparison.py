#!/usr/bin/env python3
"""The finance QCNN's fair comparison on AMLworld, as pre-registered (Round 86, batch 6).

references/finance-qcnn-amlworld-comparison-preregistration.md fixed the arms, their grids, the seeds and the metrics on
2026-11-26 before anything ran (the owner's choices of that day). This runs them and writes the record:

  --arm qcnn1 | qcnn2 | noent   the QCNN arms 1, 2 and 5, one trainer run a seed (scripts/train/qcnn_variational.py), each
                                into WORK/<arm>/seed<k>/ with its validation and test logits
  --comparators                 arms 3 and 4 on the CPU: the quantum kernels, logistic regression, the booster, the RBF
                                support vector machine and the matrix-product-state classifier
  --schmidt                     arm 6: Schmidt ranks of every arm-1 seed's state, by layer, on 1,000 test rows
  --record                      every arm's figures, intervals and paired differences into
                                docs/measurements/finance-qcnn-amlworld-comparison-2026-11-26.json

Arms 1 and 2 run on the GPU (the owner's approval of 2026-11-26 for their 40 runs); arm 5 on the CPU.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/train/finance_qcnn_comparison.py --arm qcnn1 --seeds 0-19
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PREREG = REPO / "references" / "finance-qcnn-amlworld-comparison-preregistration.md"
PREREG_SHA256 = "dc56325772e004894575ba59e71faf7e50d40f8250ac46c81f0d27a6b8322d2c"
WORK = REPO.parent / "tools_external" / "finance_qcnn_comparison_work"
RECORD = REPO / "docs" / "measurements" / "finance-qcnn-amlworld-comparison-2026-11-26.json"
TRAINER = REPO / "scripts" / "train" / "qcnn_variational.py"
COMMON = ["--usecase", "finance_amlworld", "--arch", "tree", "--layers", "3", "--epochs", "120", "--qubits", "8",
          "--threads", "4", "--select-on", "ap", "--save-scores"]
ARMS = {
    "qcnn1": {"flags": ["--encoding-scale", "0.5"], "device": "gpu"},
    "qcnn2": {"flags": ["--encoding-scale", "1", "--encoding", "quantile", "--reupload", "2"], "device": "gpu"},
    "noent": {"flags": ["--encoding-scale", "0.5", "--no-entangle"], "device": "cpu"},
}


def check_prereg() -> str:
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    return got


def seeds_of(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def run_arm(arm: str, seeds: list[int]) -> int:
    spec = ARMS[arm]
    env = dict(os.environ)
    if spec["device"] == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""                 # arm 5 is pre-registered on the CPU
    for s in seeds:
        out = WORK / arm / f"seed{s}"
        out.mkdir(parents=True, exist_ok=True)
        if (out / "weights.json").exists() and (out / "test_logits.npy").exists():
            print(f"{arm} seed {s}: done already, kept")
            continue
        t0 = time.time()
        cmd = [sys.executable, "-I", str(TRAINER), *COMMON, *spec["flags"], "--seed", str(s), "--out", str(out)]
        with open(out / "train.log", "w") as log:
            code = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{arm} seed {s}: exit {code} in {time.time() - t0:.0f} s", flush=True)
        if code:
            return code
    return 0


# --------------------------------------------------------------------------- #
#  Data: the first pre-registration's eight inputs, and the quantile map of arm 2
# --------------------------------------------------------------------------- #
def load_inputs() -> dict:
    from qsh.models import datasets

    z = np.load(datasets.artifacts("aml_qcnn_inputs.npz"))
    X, y, part = z["X"].astype(np.float64), z["y"].astype(np.int64), z["part"]
    fit, val, te = (np.flatnonzero(part == k) for k in (0, 1, 2))
    ref = np.sort(X[fit], axis=0)
    q = np.stack([np.searchsorted(ref[:, j], X[:, j], side="right") / len(ref) for j in range(X.shape[1])], axis=1)
    return {"X": X, "q": q, "y": y, "fit": fit, "val": val, "te": te}


def fit_window_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Every fit-window transaction's eight inputs, built as the first pre-registration built its fit sample, cached."""
    cache = WORK / "fit_window_inputs.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]
    from qsh.models import datasets

    sys.path.insert(0, str(REPO / "scripts" / "train"))
    from finance_aml import causal_account_features, load_sorted, model_inputs, temporal_windows
    from finance_aml_mlp import CLIP, scaling_of

    meta = json.loads(datasets.artifacts("aml_qcnn_inputs.json").read_text())
    df = load_sorted()
    fit_w, val_w, te_w, tr_w = temporal_windows(df)
    (x_fit, _, _), names, _ = model_inputs(df, fit_w, val_w, te_w, tr_w, causal_account_features(df))
    centre, scale = scaling_of("standard", x_fit)
    cols = [names.index(n) for n in meta["features"]]
    X = np.clip((x_fit - centre) / scale, -CLIP, CLIP)[:, cols].astype(np.float32)
    y = fit_w["label"].to_numpy(dtype=np.int8)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y)
    return X, y


# --------------------------------------------------------------------------- #
#  Arm 3: quantum kernels
# --------------------------------------------------------------------------- #
def zz_states(phi: np.ndarray, reps: int = 2) -> np.ndarray:
    """Havlicek et al.'s ZZ feature map on 8 wires, each wire coupled to the next: (H, phase)^reps on |0>.

    The phase layer is exp(i (sum_j phi_j Z_j + sum_j (pi - phi_j)(pi - phi_{j+1}) Z_j Z_{j+1})), diagonal in the
    computational basis; qubit q is bit q of the index, as the trainer's convention."""
    n, w = phi.shape
    dim = 1 << w
    bits = ((np.arange(dim)[:, None] >> np.arange(w)[None, :]) & 1).astype(np.float64)   # dim x w
    zs = 1.0 - 2.0 * bits                                                                    # Z eigenvalues
    pair = (np.pi - phi[:, :-1]) * (np.pi - phi[:, 1:])                                       # n x (w-1)
    phase = phi @ zs.T + pair @ (zs[:, :-1] * zs[:, 1:]).T                                     # n x dim
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    hw = h
    for _ in range(w - 1):
        hw = np.kron(h, hw)
    state = np.zeros((n, dim), dtype=np.complex128)
    state[:, 0] = 1.0
    for _ in range(reps):
        state = state @ hw.T
        state = state * np.exp(1j * phase)
    return state.astype(np.complex64)


def bloch(states: np.ndarray, w: int = 8) -> np.ndarray:
    """Each wire's reduced state as its Bloch vector (<X>, <Y>, <Z>): rows x (3 w)."""
    n, dim = states.shape
    out = np.zeros((n, 3 * w), dtype=np.float64)
    idx = np.arange(dim)
    for q in range(w):
        flip = idx ^ (1 << q)
        zq = 1.0 - 2.0 * ((idx >> q) & 1)
        amp, amp_f = states, states[:, flip]
        out[:, 3 * q] = np.real(np.sum(np.conj(amp) * amp_f, axis=1))
        out[:, 3 * q + 1] = np.real(np.sum(np.conj(amp) * amp_f * (1j * zq)[None, :] * -1, axis=1))
        out[:, 3 * q + 2] = np.real(np.sum(np.abs(amp) ** 2 * zq[None, :], axis=1))
    return out


def kernel_blocks(kind: str, A: np.ndarray, B: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """k(A_i, B_j) for one kind: 'product' and 'zz' take angles, 'projected' takes Bloch vectors."""
    if kind == "product":
        k = np.ones((len(A), len(B)))
        for j in range(A.shape[1]):
            k *= np.cos((A[:, j][:, None] - B[:, j][None, :]) / 2) ** 2
        return k
    if kind == "zz":
        return np.abs(A @ np.conj(B).T) ** 2
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d, 0) / 2)          # ||rho - rho'||_F^2 = |r - r'|^2 / 2


def rep(kind: str, phi: np.ndarray) -> np.ndarray:
    if kind == "product":
        return phi
    s = zz_states(phi)
    return s if kind == "zz" else bloch(s)


def score_svm(svm, kind: str, train_rep: np.ndarray, rows_rep: np.ndarray, gamma, chunk: int = 20000) -> np.ndarray:
    sv = train_rep[svm.support_]
    coef = svm.dual_coef_[0]
    out = np.empty(len(rows_rep))
    for i in range(0, len(rows_rep), chunk):
        out[i:i + chunk] = kernel_blocks(kind, rows_rep[i:i + chunk], sv, gamma) @ coef + svm.intercept_[0]
    return out


def quantum_kernels(d: dict) -> dict:
    from sklearn.metrics import average_precision_score as ap
    from sklearn.svm import SVC

    y, fit, val, te = d["y"], d["fit"], d["val"], d["te"]
    res: dict = {}
    for kind in ("product", "zz", "projected"):
        best = None
        table = []
        for b in (0.25, 0.5, 1.0):
            phi = b * np.pi * d["q"]
            R_fit = rep(kind, phi[fit])
            gamma = None
            if kind == "projected":
                dd = (R_fit ** 2).sum(1)[:, None] + (R_fit ** 2).sum(1)[None, :] - 2 * R_fit @ R_fit.T
                off = dd[~np.eye(len(dd), dtype=bool)] / 2
                gamma = float(1.0 / np.median(off))
            K = kernel_blocks(kind, R_fit, R_fit, gamma)
            off = K[~np.eye(len(K), dtype=bool)]
            concentration = float(np.mean(np.abs(off - off.mean()) < 0.01))
            R_val = rep(kind, phi[val])
            for C in (0.1, 1.0, 10.0):
                svm = SVC(C=C, kernel="precomputed").fit(K, y[fit])
                v = float(ap(y[val], score_svm(svm, kind, R_fit, R_val, gamma)))
                table.append({"bandwidth": b, "C": C, "validation_ap": round(v, 6), "support_vectors": int(len(svm.support_)),
                              "concentration_within_0_01": round(concentration, 6)})
                print(f"  kernel {kind} b {b} C {C}: val AP {v:.4f}", flush=True)
                if best is None or v > best[0]:
                    best = (v, b, C, svm, gamma)
        v, b, C, svm, gamma = best
        phi = b * np.pi * d["q"]
        R_fit, R_te, R_val = rep(kind, phi[fit]), rep(kind, phi[te]), rep(kind, phi[val])
        res[kind] = {"table": table, "chosen": {"bandwidth": b, "C": C}, "validation_ap": round(v, 6),
                     "test_scores": score_svm(svm, kind, R_fit, R_te, gamma),
                     "val_scores": score_svm(svm, kind, R_fit, R_val, gamma)}
    return res


# --------------------------------------------------------------------------- #
#  Arm 4: classical comparators
# --------------------------------------------------------------------------- #
def mps_classifier(q_fit, y_fit, q_rows: list[np.ndarray], chi: int, seed: int = 0, epochs: int = 50):
    """Stoudenmire and Schwab's MPS on [cos(pi x / 2), sin(pi x / 2)] per input, contracted to one logit."""
    import torch

    torch.manual_seed(seed)
    w = q_fit.shape[1]
    dims = [1] + [chi] * (w - 1) + [1]
    cores = torch.nn.ParameterList()
    for k in range(w):
        eye = torch.zeros(dims[k], 2, dims[k + 1])
        for s in range(2):
            m = torch.eye(dims[k], dims[k + 1])
            eye[:, s, :] = m / np.sqrt(2)
        cores.append(torch.nn.Parameter(eye + 0.01 * torch.randn(dims[k], 2, dims[k + 1])))
    bias = torch.nn.Parameter(torch.zeros(()))
    scale = torch.nn.Parameter(torch.ones(()))

    def forward(x):
        phi = torch.stack([torch.cos(np.pi * x / 2), torch.sin(np.pi * x / 2)], dim=2)     # n x w x 2
        v = torch.ones(len(x), 1)
        for k in range(w):
            m = torch.einsum("ns,asb->nab", phi[:, k, :], cores[k])
            v = torch.einsum("na,nab->nb", v, m)
        return scale * v[:, 0] * (2 ** (w / 2)) + bias

    xt = torch.tensor(q_fit, dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.float32)
    opt = torch.optim.Adam([*cores, bias, scale], lr=0.01)
    lossf = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), 256):
            j = order[i:i + 256]
            opt.zero_grad()
            loss = lossf(forward(xt[j]), yt[j])
            loss.backward()
            opt.step()
    with torch.no_grad():
        return [forward(torch.tensor(r, dtype=torch.float32)).numpy().astype(np.float64) for r in q_rows]


def classical(d: dict) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score as ap
    from sklearn.svm import SVC

    X, q, y, fit, val, te = d["X"], d["q"], d["y"], d["fit"], d["val"], d["te"]
    res: dict = {}
    lr = LogisticRegression(max_iter=2000).fit(X[fit], y[fit])
    res["logistic_regression"] = {"validation_ap": round(float(ap(y[val], lr.decision_function(X[val]))), 6),
                                  "test_scores": lr.decision_function(X[te]), "val_scores": lr.decision_function(X[val])}
    best = None
    for depth in (3, 5):
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=depth, early_stopping=False,
                                            random_state=0).fit(X[fit], y[fit])
        v = float(ap(y[val], gb.decision_function(X[val])))
        print(f"  booster depth {depth}: val AP {v:.4f}", flush=True)
        if best is None or v > best[0]:
            best = (v, depth, gb)
    v, depth, gb = best
    res["gradient_boosting"] = {"chosen": {"max_depth": depth}, "validation_ap": round(v, 6),
                                "test_scores": gb.decision_function(X[te]), "val_scores": gb.decision_function(X[val])}
    best = None
    for gm in (0.1, 1.0, 10.0):
        for C in (0.1, 1.0, 10.0):
            svm = SVC(C=C, kernel="rbf", gamma=gm / X.shape[1]).fit(X[fit], y[fit])
            v = float(ap(y[val], svm.decision_function(X[val])))
            print(f"  rbf svm gamma {gm}/8 C {C}: val AP {v:.4f}", flush=True)
            if best is None or v > best[0]:
                best = (v, gm, C, svm)
    v, gm, C, svm = best
    res["rbf_svm"] = {"chosen": {"gamma_times_inputs": gm, "C": C}, "validation_ap": round(v, 6),
                      "test_scores": svm.decision_function(X[te]), "val_scores": svm.decision_function(X[val])}
    best = None
    for chi in (2, 4, 8):
        sv, st = mps_classifier(q[fit], y[fit], [q[val], q[te]], chi)
        v = float(ap(y[val], sv))
        print(f"  mps chi {chi}: val AP {v:.4f}", flush=True)
        if best is None or v > best[0]:
            best = (v, chi, sv, st)
    v, chi, sv, st = best
    res["mps"] = {"chosen": {"bond_dimension": chi}, "validation_ap": round(v, 6), "test_scores": st, "val_scores": sv}
    # The two that also run on the whole fit window (every one of its transactions), scored on the same windows.
    Xw, yw = fit_window_inputs()
    lrw = LogisticRegression(max_iter=2000).fit(Xw, yw)
    res["logistic_regression_whole_fit_window"] = {"test_scores": lrw.decision_function(X[te]),
                                                   "val_scores": lrw.decision_function(X[val])}
    gbw = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=depth, early_stopping=False,
                                         random_state=0).fit(Xw, yw)
    res["gradient_boosting_whole_fit_window"] = {"test_scores": gbw.decision_function(X[te]),
                                                 "val_scores": gbw.decision_function(X[val])}
    return res


def run_comparators() -> int:
    d = load_inputs()
    t0 = time.time()
    out = WORK / "comparators"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, block in {**classical(d), **quantum_kernels(d)}.items():
        np.save(out / f"{name}_test.npy", np.asarray(block.pop("test_scores"), dtype=np.float32))
        np.save(out / f"{name}_val.npy", np.asarray(block.pop("val_scores"), dtype=np.float32))
        summary[name] = block
    summary["wall_clock_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
#  Arm 6: Schmidt ranks
# --------------------------------------------------------------------------- #
def schmidt_ranks(states: np.ndarray, nq: int = 8, tol: float = 1e-10) -> np.ndarray:
    """Rows x cuts: the Schmidt rank of each row's state across the cut after wire k (wires 0..k | k+1..nq-1).

    Wire q is bit q of the index, so the low k+1 bits are the first side: reshape to (2**(nq-k-1), 2**(k+1))."""
    out = np.zeros((len(states), nq - 1), dtype=np.int64)
    for k in range(nq - 1):
        m = states.reshape(len(states), 1 << (nq - k - 1), 1 << (k + 1))
        s = np.linalg.svd(m, compute_uv=False)
        out[:, k] = (s > tol * s[:, :1]).sum(axis=1)
    return out


def run_schmidt() -> int:
    import torch

    sys.path.insert(0, str(REPO / "scripts" / "train"))
    import qcnn_variational as Q

    d = load_inputs()
    rows = np.random.default_rng(0).permutation(d["te"])[:1000]
    x = torch.tensor(0.5 * np.tanh(d["X"][rows]), dtype=torch.float32)      # arm 1's encoding, c = 0.5
    result = {}
    for s in range(20):
        w = json.loads((WORK / "qcnn1" / f"seed{s}" / "weights.json").read_text())
        trace: list = []
        with torch.no_grad():
            Q.circuit(x, torch.tensor(w["theta"], dtype=torch.float32), 8, readout_qubits=0, entangle=True,
                      layers=w["layers"], arch="tree", layer_trace=trace)
        result[f"seed{s}"] = {label: int(schmidt_ranks(st.numpy().astype(np.complex128)).max()) for label, st in trace}
    largest = max(max(v.values()) for v in result.values())
    out = {"rows": 1000, "bound": 16, "largest_rank_over_seeds_layers_cuts_and_rows": largest, "by_seed": result}
    (WORK / "schmidt.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"largest Schmidt rank {largest} of the bound 16; wrote {WORK / 'schmidt.json'}")
    return 0


# --------------------------------------------------------------------------- #
#  The record: intervals, ensembles, paired differences
# --------------------------------------------------------------------------- #
def ap_weighted(order_y: np.ndarray, w: np.ndarray) -> float:
    """Average precision of rows already sorted by descending score, each row counted w times."""
    tp = np.cumsum(w * order_y)
    total = np.cumsum(w)
    pos = tp[-1]
    if pos == 0:
        return float("nan")
    prec = np.divide(tp, total, out=np.zeros_like(tp, dtype=np.float64), where=total > 0)
    return float(np.sum(prec * w * order_y) / pos)


def bootstrap_weights(y: np.ndarray, n: int, seed: int = 0):
    """Stratified resamples by label, as counts per row (positives and negatives each resampled to their own size)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n):
        w = np.bincount(rng.choice(pos, size=len(pos)), minlength=len(y)) + \
            np.bincount(rng.choice(neg, size=len(neg)), minlength=len(y))
        yield w.astype(np.float64)


def intervals(y: np.ndarray, scores: dict[str, np.ndarray], ship: str, n: int = 1000) -> dict:
    """Each score's AP with a 95 percent stratified bootstrap interval, and its paired difference from `ship`."""
    orders = {k: np.argsort(-v, kind="stable") for k, v in scores.items()}
    draws = {k: [] for k in scores}
    for w in bootstrap_weights(y, n):
        for k, o in orders.items():
            draws[k].append(ap_weighted(y[o], w[o]))
    out = {}
    for k in scores:
        a = np.asarray(draws[k])
        diff = np.asarray(draws[ship]) - a
        out[k] = {"ap_ci95": [round(float(np.percentile(a, 2.5)), 6), round(float(np.percentile(a, 97.5)), 6)],
                  "ship_minus_this_ci95": None if k == ship else
                  [round(float(np.percentile(diff, 2.5)), 6), round(float(np.percentile(diff, 97.5)), 6)]}
    return out


def f1_at_val_best(yv, sv, yt, st) -> float:
    from sklearn.metrics import f1_score, precision_recall_curve

    p, r, thr = precision_recall_curve(yv, sv)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    cut = thr[int(np.argmax(f))]
    return float(f1_score(yt, (st >= cut).astype(int), zero_division=0))


def run_record() -> int:
    from sklearn.metrics import average_precision_score as ap

    got = check_prereg()
    d = load_inputs()
    yv, yt = d["y"][d["val"]], d["y"][d["te"]]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    arms: dict = {}
    scores: dict = {}
    for arm, seeds in (("qcnn1", range(20)), ("qcnn2", range(20)), ("noent", range(5))):
        per = []
        val_p, test_p = [], []
        for s in seeds:
            o = WORK / arm / f"seed{s}"
            w = json.loads((o / "weights.json").read_text())
            vl, tl = np.load(o / "val_logits.npy").astype(np.float64), np.load(o / "test_logits.npy").astype(np.float64)
            per.append({"seed": s, "val_ap": round(float(ap(yv, vl)), 6), "test_ap": round(float(ap(yt, tl)), 6),
                        "selected": w.get("selection")})
            val_p.append(sig(vl)), test_p.append(sig(tl))
        ens_v, ens_t = np.mean(val_p, axis=0), np.mean(test_p, axis=0)
        best = max(per, key=lambda r: (r["val_ap"], -r["seed"]))
        arms[arm] = {"seeds": per, "mean_test_ap": round(float(np.mean([r["test_ap"] for r in per])), 6),
                     "sd_test_ap": round(float(np.std([r["test_ap"] for r in per], ddof=1)), 6),
                     "ensemble_test_ap": round(float(ap(yt, ens_t)), 6),
                     "ensemble_minority_f1": round(f1_at_val_best(yv, ens_v, yt, ens_t), 6),
                     "best_on_validation": best["seed"]}
        scores[f"{arm}_ensemble"] = ens_t
        if arm == "qcnn1":
            o = WORK / arm / f"seed{best['seed']}"
            vl, tl = np.load(o / "val_logits.npy").astype(np.float64), np.load(o / "test_logits.npy").astype(np.float64)
            arms[arm]["shipped_seed"] = {"seed": best["seed"], "test_ap": best["test_ap"],
                                         "minority_f1": round(f1_at_val_best(yv, vl, yt, tl), 6)}
            scores["qcnn1_shipped"] = tl
    comp = json.loads((WORK / "comparators" / "summary.json").read_text())
    for name in [k for k in comp if k != "wall_clock_s"] + ["logistic_regression_whole_fit_window",
                                                            "gradient_boosting_whole_fit_window"]:
        st = np.load(WORK / "comparators" / f"{name}_test.npy").astype(np.float64)
        sv = np.load(WORK / "comparators" / f"{name}_val.npy").astype(np.float64)
        arms[name] = {**comp.get(name, {}), "test_ap": round(float(ap(yt, st)), 6),
                      "minority_f1": round(f1_at_val_best(yv, sv, yt, st), 6)}
        scores[name] = st
    ci = intervals(yt, scores, "qcnn1_shipped")
    prevalence = float(yt.mean())
    ship = arms["qcnn1"]["shipped_seed"]
    record = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
        "test_prevalence": round(prevalence, 8), "twice_prevalence": round(2 * prevalence, 8),
        "arms": arms, "intervals": ci,
        "schmidt": json.loads((WORK / "schmidt.json").read_text()) if (WORK / "schmidt.json").exists() else None,
        "criteria": {"1_shipped_seed_at_least_twice_prevalence": bool(ship["test_ap"] >= 2 * prevalence),
                     "1_export_check": "run by tests/test_r86_qcnn_under_ckks.py and the page's clear contraction on the shipped seed",
                     "comparators_above_the_shipped_seed": sorted(k for k, v in arms.items()
                                                                  if k not in ("qcnn1",) and
                                                                  (v.get("test_ap", v.get("ensemble_test_ap", 0)) or 0) > ship["test_ap"])},
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--comparators", action="store_true")
    ap.add_argument("--schmidt", action="store_true")
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()
    check_prereg()
    if args.arm:
        return run_arm(args.arm, seeds_of(args.seeds))
    if args.comparators:
        return run_comparators()
    if args.schmidt:
        return run_schmidt()
    if args.record:
        return run_record()
    ap.error("choose --arm, --comparators, --schmidt or --record")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
