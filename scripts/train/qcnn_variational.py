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


def circuit(x: torch.Tensor, theta: torch.Tensor, nq: int = N_QUBITS,
            readout_qubits: int = 0) -> torch.Tensor:
    """(batch, nq) features + (layers, nq) angles -> a readout. State kept FLAT as (batch, 2**nq).

    ``readout_qubits = 0`` gives the original single ``<Z0>`` expectation, shape (batch,), which is
    BINARY BY CONSTRUCTION: one number through one scalar scale and bias is one logit. That is what
    this circuit produced until Round 34, and it is why an 8-class macro F1 could not honestly be
    reported for it - six of eight classes had a floor of zero.

    ``readout_qubits = m`` gives the marginal probabilities of the first m qubits, shape
    (batch, 2**m), which is a genuine K-class readout for K = 2**m. Qiskit's VQC standardises on
    exactly this (a computational-basis readout mapped to classes), and at m=3 for 8 classes the map
    is the identity: nothing is discarded and the probabilities sum to 1 by construction.

    ENDIANNESS. Qubit q is bit q of the flat index, qubit 0 the LEAST significant - the same
    convention the ``idx & 1`` below relies on for ``<Z0>``, and the one ``quantumSim.ts`` uses. So
    class ``c = i & (2**m - 1)``. PennyLane puts the FIRST LISTED wire in the MOST significant bit, so
    an LSB-first read of bits 0,1,2 equals ``qml.probs(wires=[7,6,5])`` and NOT ``wires=[0,1,2]``.
    Getting that backwards permutes the class map by ``[0,4,2,6,1,5,3,7]`` while leaving accuracy,
    macro F1 and every loss curve identical - and four of the eight classes are FIXED POINTS of that
    permutation, so a spot check on the obvious ones passes. ``tests/test_qcnn_multiclass_readout.py``
    covers all eight individually for that reason.
    """
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
    if readout_qubits:
        # Marginal over the first `readout_qubits` wires: sum |amplitude|^2 into 2**m bins keyed by
        # the low m bits of the flat index. One pass over the state vector, so the cost against the
        # <Z0> readout is nothing next to the 2**nq state preparation that precedes it.
        k = 1 << readout_qubits
        return torch.zeros(p.shape[0], k, device=dev, dtype=p.dtype).index_add_(
            1, (idx & (k - 1)).to(torch.long), p)
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
    if usecase == "finance":
        import pandas as pd
        df = pd.read_csv(REPO / "data" / "finance" / "paysim_sample.tsv", sep="\t", comment="#")
        cols = [c for c in df.columns if c != "isFraud"]
        X, y = df[cols].to_numpy(float), df["isFraud"].to_numpy(int)
        return X, y, cols, ("PaySim simulator output (Kaggle ealaxi/paysim1, CC-BY-SA-4.0; "
                            "Lopez-Rojas et al. 2016) - SYNTHETIC transactions from aggregated real "
                            "mobile-money logs, 8 AML features, the same 8 the browser pipeline feeds")
    raise SystemExit(f"no loader for {usecase!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usecase", default="medical")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X, y, names, prov = load(a.usecase)
    # SEED TORCH TOO. `--seed` drove only the numpy split permutation, while the parameter init came
    # from an unseeded `torch.randn`, so the same command produced different weights every run -
    # medical moved across 0.9181 / 0.9064 / 0.8947 / 0.8480 accuracy on identical inputs. Published
    # weights that cannot be regenerated are exactly what this project refuses to ship, and the swing
    # was large enough to be mistaken for a real effect of the changes being made alongside it.
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    # THREE-WAY SPLIT. An earlier version selected the best epoch on the same rows it then reported,
    # which is test-set leakage and makes every published figure optimistic by construction. Epoch
    # selection now happens on a validation split the reported number never sees.
    n = len(y)
    cut, vcut = int(0.7 * n), int(0.85 * n)
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
    # WHICH SPACE DOES THE BROWSER SUPPLY? It differs per use case, and getting it wrong is how the
    # first trained model became as meaningless as the untrained one it replaced.
    #
    #   medical  the bundled sample is StandardScaler output (verified, L1 0.0002), so z-scores
    #            arrive at inference time and the browser must NOT rescale.
    #   finance  the bundled sample is RAW and mixed-scale - `amount` is log 0..17.7, `amount_zscore`
    #            is already a z-score, five columns are binary 0/1 - and all 16 bundled rows are found
    #            in this tsv within 1e-3, so the browser supplies exactly these units. It therefore
    #            MUST standardise, with the moments fitted here and shipped alongside.
    #
    # Either way the encoding is RY(tanh(z)*pi): bounded, smooth, order-preserving. tanh on a raw
    # log-amount of 14 would saturate at 1.0 and throw the column away, which is why standardising
    # first is not optional for finance.
    INPUT_SPACE = {"medical": "zscore", "finance": "raw"}[a.usecase]
    mu, sd = X[:cut].mean(0), X[:cut].std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = np.tanh((X - mu) / sd)
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7 else "cpu"
    xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    tr, va, te = slice(0, cut), slice(cut, vcut), slice(vcut, n)
    print(f"{a.usecase}: {n} rows, {int(y.sum())} positive; train {cut} / val {vcut-cut} / "
          f"test {n-vcut}; device {dev}")

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
                zt = circuit(xt[va], theta)
                pred = ((zt * scale + bias) > 0).float()
                acc = (pred == yt[va]).float().mean().item()
                # SELECT ON THE METRIC THAT WILL BE REPORTED. Selecting the best epoch by accuracy on
                # an imbalanced problem picks a checkpoint that can afford to ignore the minority
                # class: on finance it cost 0.14 of minority-class F1 (0.6992 at the F1-best epoch
                # against 0.5641 at the accuracy-best one) while accuracy moved by 0.01. The
                # selection criterion has to match the headline or it optimises the wrong thing.
                tp = float((pred * yt[va]).sum())
                fp = float((pred * (1 - yt[va])).sum())
                fn = float(((1 - pred) * yt[va]).sum())
                f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
                prev = float(yt[va].mean())
                score = f1 if min(prev, 1 - prev) < 0.25 else acc
                if score > best:
                    best = score
                    best_state = (theta.detach().cpu().numpy().copy(),
                                  float(scale), float(bias))
            print(f"  epoch {ep:3d} loss {loss.item():.4f} VAL acc {acc:.4f} VAL minority-F1 {f1:.4f}")

    # Classical baselines on the identical split - the number that decides whether this ships.
    #
    # AND THE RIGHT METRIC, which is not always accuracy. This project treats accuracy on an
    # imbalanced problem as a defect: at finance's 6.33% fraud rate the QCNN scores 0.9692 accuracy
    # against a 0.9367 majority baseline, which reads as a good model and says almost nothing. Its
    # minority-class F1 is 0.6992 against an all-positive baseline of 0.1191 - still a real 5.9x
    # lift, but a very different sentence. Both are recorded; the imbalanced case leads with F1.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    lr = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr])
    lr_acc = lr.score(Xs[te], y[te])
    major = max(y[te].mean(), 1 - y[te].mean())
    prevalence = float(y[te].mean())
    imbalanced = min(prevalence, 1 - prevalence) < 0.25

    with torch.no_grad():
        th_t = torch.tensor(best_state[0], dtype=torch.float32, device=dev)
        z_all = circuit(xt[te], th_t)
        qcnn_pred = ((z_all * best_state[1] + best_state[2]) > 0).int().cpu().numpy()
    qcnn_f1 = float(f1_score(y[te], qcnn_pred, zero_division=0))
    # `best` is the SELECTION score (F1 when imbalanced), not accuracy. Recompute accuracy from the
    # selected checkpoint's own predictions - an earlier version printed `best` under an "accuracy"
    # label, so finance reported accuracy 0.8027 when that was its F1.
    qcnn_acc = float((qcnn_pred == y[te]).mean())
    lr_f1 = float(f1_score(y[te], lr.predict(Xs[te]), zero_division=0))
    all_pos_f1 = float(2 * prevalence / (1 + prevalence)) if prevalence else 0.0
    print(f"\n  test prevalence {prevalence:.4f} -> {'IMBALANCED, F1 leads' if imbalanced else 'balanced, accuracy leads'}")
    print(f"  QCNN  minority-class F1 {qcnn_f1:.4f}   accuracy {qcnn_acc:.4f}")
    print(f"  LR    minority-class F1 {lr_f1:.4f}   accuracy {lr_acc:.4f}")
    print(f"  baselines: all-positive F1 {all_pos_f1:.4f}   majority-class accuracy {major:.4f}")
    if imbalanced:
        print(f"  headline: F1 {qcnn_f1:.4f} vs all-positive {all_pos_f1:.4f} "
              f"({qcnn_f1/max(all_pos_f1,1e-9):.1f}x). Accuracy {qcnn_acc:.4f} against a {major:.4f} "
              f"majority baseline would flatter it and is NOT the headline.")
    verdict = ("BEATS" if (qcnn_f1 if imbalanced else qcnn_acc) > (all_pos_f1 if imbalanced else major)
               else "DOES NOT beat")
    print(f"  verdict: QCNN {verdict} its baseline; classical LR is "
          f"{'ahead' if (lr_f1 if imbalanced else lr_acc) > (qcnn_f1 if imbalanced else qcnn_acc) else 'behind'}")

    th, sc, bi = best_state
    out = REPO / "models" / f"qcnn_{a.usecase}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.json").write_text(json.dumps({
        "usecase": a.usecase, "n_qubits": N_QUBITS, "layers": a.layers,
        "theta": th.tolist(), "scale": sc, "bias": bi,
        "feature_names": names, "input_space": INPUT_SPACE, "encoding": "tanh",
        "feature_mean": mu.tolist(), "feature_std": sd.tolist(),
        "provenance": prov,
        "test_accuracy": round(qcnn_acc, 6),
        "classical_lr_accuracy": round(float(lr_acc), 6),
        "majority_baseline": round(float(major), 6),
        "test_minority_f1": round(qcnn_f1, 6),
        "classical_lr_minority_f1": round(lr_f1, 6),
        "all_positive_f1": round(all_pos_f1, 6),
        "test_prevalence": round(prevalence, 6),
        "headline_metric": "minority_class_f1" if imbalanced else "accuracy",
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
