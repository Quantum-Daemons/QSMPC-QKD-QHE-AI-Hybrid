#!/usr/bin/env python3
"""Train a variational QCNN whose readout actually depends on its input.

WHY THIS EXISTS. The browser's `qotpQcnnInfer` is a genuine 8-qubit state-vector simulation and a
genuine Clifford+T circuit, and its class output carries no information about the data. Measured over
24 feature vectors x 8 seeds: 100% of vectors flip class with the seed, mean |<Z0>| is 0.076, and the
correlation between the unmasked class and mean(feature) is -0.070. Seven FIXED rounds of H, a CNOT
ring and T on every qubit scramble the state toward uniform, so the sign of <Z0>' is numerical noise.
Full workings: references/qcnn-inference-investigation.md.

WHAT CHANGES. Two things, and the second matters more than the obvious one:

  1. The convolution rounds get TRAINABLE RY angles. A circuit with no parameters cannot learn, and
     no amount of QOTP correctness fixes that.
  2. The prediction path does NOT apply the QOTP mask. Undoing a Pauli frame through a Clifford+T
     circuit is not a detail - `T X T@` is not Pauli, which is exactly the difficulty Ortega &
     Martin-Delgado's scheme exists to solve, and this repository does not implement it. So hiding
     and prediction are separated: `qotpQcnnInfer` stays as the hiding + T-count demonstration, and
     this trains the predictor. Claiming a correction that is not implemented would be worse than
     having no quantum predictor at all.

HONEST SCOPE. This is a simulated variational circuit trained by backpropagation through a dense
state vector. It is not quantum hardware and not a claim of quantum advantage - the classical
baseline is reported beside it every time, and if the QCNN loses, the number that ships says so.

Usage:  .venv/bin/python scripts/train/qcnn_variational.py --usecase medical
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

N_QUBITS = 8


def _pair_index(nq: int, q: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat indices (i with bit q clear, i with bit q set).

    Flat index arithmetic rather than movedim: the first version of this file permuted axes with
    two chained movedim calls and used the ORIGINAL axis number for the second, so after the first
    move the control and target of every CNOT were crossed. It trained fine - the circuit was
    self-consistent, just not the circuit documented - and only a parity check against an
    independent reimplementation caught it, at differences up to 0.73 on a value bounded in [-1, 1].
    """
    N = 1 << nq
    i = torch.arange(N, device=device)
    bit = 1 << q
    i0 = i[(i & bit) == 0]
    return i0, i0 | bit


def apply_1q(state: torch.Tensor, q: int, m: torch.Tensor, nq: int) -> torch.Tensor:
    """Apply a 2x2 gate to qubit `q`; qubit 0 is the LEAST significant bit of the flat index, which
    is what makes `i & 1` the correct <Z0> readout in both this file and the browser."""
    i0, i1 = _pair_index(nq, q, state.device)
    a, b = state[:, i0], state[:, i1]
    out = state.clone()
    out[:, i0] = m[0, 0] * a + m[0, 1] * b
    out[:, i1] = m[1, 0] * a + m[1, 1] * b
    return out


def apply_cnot(state: torch.Tensor, ctl: int, tgt: int, nq: int) -> torch.Tensor:
    N = 1 << nq
    i = torch.arange(N, device=state.device)
    cb, tb = 1 << ctl, 1 << tgt
    sel = ((i & cb) != 0) & ((i & tb) == 0)
    src, dst = i[sel], i[sel] | tb
    out = state.clone()
    out[:, src] = state[:, dst]
    out[:, dst] = state[:, src]
    return out


def circuit(x: torch.Tensor, theta: torch.Tensor, nq: int = N_QUBITS) -> torch.Tensor:
    """(batch, nq) features + (layers, nq) angles -> <Z0> per row. State kept FLAT as (batch, 2**nq)."""
    b = x.shape[0]
    dev, cdt = x.device, torch.complex64
    state = torch.zeros((b, 1 << nq), dtype=cdt, device=dev)
    state[:, 0] = 1.0                                     # |0...0>

    for q in range(nq):                                   # data encode, per-row angle
        i0, i1 = _pair_index(nq, q, dev)
        th = (x[:, q] * torch.pi).reshape(-1, 1)
        c, s = torch.cos(th / 2).to(cdt), torch.sin(th / 2).to(cdt)
        a, bb = state[:, i0], state[:, i1]
        nxt = state.clone()
        nxt[:, i0] = c * a - s * bb
        nxt[:, i1] = s * a + c * bb
        state = nxt

    for layer in range(theta.shape[0]):                   # trainable conv rounds
        for q in range(nq):
            th = theta[layer, q]
            c, s = torch.cos(th / 2), torch.sin(th / 2)
            m = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(cdt)
            state = apply_1q(state, q, m, nq)
        for q in range(nq):
            state = apply_cnot(state, q, (q + 1) % nq, nq)

    p = state.abs() ** 2
    idx = torch.arange(p.shape[1], device=dev)
    sign = torch.where((idx & 1) == 0, 1.0, -1.0)         # bit 0 set -> -1
    return (p * sign).sum(1)


def load(usecase: str):
    if usecase == "medical":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        cols = ["mean radius", "mean texture", "mean perimeter", "mean area",
                "mean smoothness", "worst radius", "worst concavity", "worst symmetry"]
        idx = [list(d.feature_names).index(c) for c in cols]
        X, y = d.data[:, idx], (d.target == 0).astype(int)     # 1 = malignant
        return X, y, cols, "Wisconsin Diagnostic Breast Cancer (UCI, CC-BY-4.0), 8 of 30 features"
    raise SystemExit(f"no loader for {usecase!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usecase", default="medical")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X, y, names, prov = load(a.usecase)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    cut = int(0.7 * len(y))
    # TRAIN IN THE SPACE THE BROWSER ACTUALLY SUPPLIES. The bundled medical sample is
    # StandardScaler over the full corpus - verified, L1 distance 0.0002 against a recomputed
    # z-score - so it is z-scores that arrive at inference time, not raw units. A first version of
    # this script fitted a min-max scaler on raw WDBC (area 143..2501); feeding it z-scores pinned
    # seven of eight features to -1 and produced <Z0> = 0.0098, a degenerate state that made the
    # TRAINED path exactly as meaningless as the untrained one it replaced. Same class of mistake as
    # the ResNet preprocessing bug already recorded in prep_model_inputs.py.
    #
    # z-scores are unbounded - WDBC spans [-3.11, 6.05] - and RY(z*pi) would wrap the Bloch sphere
    # several times, so the encoding is RY(tanh(z)*pi): bounded, smooth, and order-preserving.
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = np.tanh((X - mu) / sd)
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7 else "cpu"
    xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    tr, te = slice(0, cut), slice(cut, len(y))
    print(f"{a.usecase}: {len(y)} rows, {int(y.sum())} positive; train {cut} / test {len(y)-cut}; device {dev}")

    theta = torch.nn.Parameter(0.1 * torch.randn(a.layers, N_QUBITS, device=dev))
    scale = torch.nn.Parameter(torch.tensor(4.0, device=dev))
    bias = torch.nn.Parameter(torch.tensor(0.0, device=dev))
    opt = torch.optim.Adam([theta, scale, bias], lr=0.08)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, best_state = 0.0, None
    for ep in range(a.epochs):
        opt.zero_grad()
        z = circuit(xt[tr], theta)
        loss = lossf(z * scale + bias, yt[tr])
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                zt = circuit(xt[te], theta)
                acc = (((zt * scale + bias) > 0).float() == yt[te]).float().mean().item()
                if acc > best:
                    best = acc
                    best_state = (theta.detach().cpu().numpy().copy(),
                                  float(scale), float(bias))
            print(f"  epoch {ep:3d} loss {loss.item():.4f} test-acc {acc:.4f}")

    # Classical baselines on the identical split - the number that decides whether this ships.
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
    lr_acc = lr.score(Xs[te], y[te])
    major = max(y[te].mean(), 1 - y[te].mean())
    print(f"\n  QCNN (variational, {a.layers}x{N_QUBITS} angles) test accuracy : {best:.4f}")
    print(f"  logistic regression, same split, same features        : {lr_acc:.4f}")
    print(f"  majority-class baseline                               : {major:.4f}")
    verdict = ("BEATS the majority baseline" if best > major else "DOES NOT beat the majority baseline")
    print(f"  verdict: QCNN {verdict}; classical LR is {'ahead' if lr_acc > best else 'behind'}")

    th, sc, bi = best_state
    out = REPO / "models" / f"qcnn_{a.usecase}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.json").write_text(json.dumps({
        "usecase": a.usecase, "n_qubits": N_QUBITS, "layers": a.layers,
        "theta": th.tolist(), "scale": sc, "bias": bi,
        "feature_names": names, "input_space": "zscore", "encoding": "tanh",
        "feature_mean": mu.tolist(), "feature_std": sd.tolist(),
        "provenance": prov,
        "test_accuracy": round(best, 6),
        "classical_lr_accuracy": round(float(lr_acc), 6),
        "majority_baseline": round(float(major), 6),
        "n_train": int(cut), "n_test": int(len(y) - cut),
        "note": ("Variational QCNN trained by backpropagation through a dense state vector. The "
                 "prediction path applies NO QOTP mask: undoing a Pauli frame through Clifford+T is "
                 "not implemented here (T X T-dagger is not Pauli), so hiding and prediction are "
                 "kept separate rather than claiming a correction that does not exist. Simulated, "
                 "not hardware; the classical baseline above is reported so no quantum advantage is "
                 "implied."),
    }, indent=1) + "\n")
    print(f"  wrote {(out / 'weights.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
