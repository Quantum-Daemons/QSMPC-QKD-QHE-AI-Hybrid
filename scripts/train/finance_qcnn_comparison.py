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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--comparators", action="store_true")
    args = ap.parse_args()
    check_prereg()
    if args.arm:
        return run_arm(args.arm, seeds_of(args.seeds))
    if args.comparators:
        return run_comparators()
    ap.error("choose --arm or --comparators (the Schmidt check and the record follow in this file)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
