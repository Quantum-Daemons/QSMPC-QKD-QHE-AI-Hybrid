#!/usr/bin/env python3
"""The murmur study's learning-rate search, privacy-accounted (Round 86, the owner's choice of 2026-11-14).

Random search in the sense of Papernot and Steinke, "Hyperparameter Tuning with Renyi Differential Privacy" (ICLR 2022,
arXiv:2110.03620 v2): each candidate Q draws a learning rate uniformly from a grid fixed in advance (here three rates the
PhysioNet 2016 stand-in ranked highest, so CirCor did not choose them) and a seed, trains with DP on the FIT people, and
scores the model on the VALIDATION people with Gaussian noise on the AUROC. The number of candidates K is drawn from a
truncated negative binomial (eta 0, the logarithmic distribution, E[K] 5); only the best-scoring candidate is released.
Each person goes to validation by a hash of the person, independently of everyone else, so a person added or removed
changes one side only and Q's Renyi DP at each order is the larger of the training's and the score's; the score's noise
is set so that it never exceeds the training's at any order.

Theorem 2 of the paper: if Q is (lam, eps)-RDP and (lam_hat, eps_hat)-RDP, the best of K ~ TNB(eta, gamma) runs is
(lam, eps')-RDP with eps' = eps + (1 + eta)(1 - 1/lam_hat) eps_hat + (1 + eta) log(1/gamma) / lam_hat + log(E[K]) / (lam - 1),
lam_hat = 1 allowed. The per-candidate noise is calibrated so that the whole search meets each target epsilon at delta
1e-5, converted as the study's DP-SGD arm is. K, the losing candidates and their scores are not released: they stay in
the local report and the builder leaves them out of the record.

Usage: python -I scripts/train/medical_murmur_dp_search.py --prep <CirCor> --public-prep <PhysioNet 2016> --out <dir>
       --models head-panns --grid 0.003,0.01,0.03 [--panns Cnn14_mAP=0.431.pth] [--arms central,federated]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("medical_murmur_dp", HERE / "medical_murmur_dp.py")
study = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("medical_murmur_dp", study)
_spec.loader.exec_module(study)

ORDERS = tuple(range(2, 257))


def tnb_gamma_for(eta: float, expected_k: float) -> float:
    """The gamma giving this E[K] for the truncated negative binomial (Definition 1): eta = 0 is the logarithmic
    distribution, E[K] = (1/gamma - 1) / log(1/gamma); otherwise E[K] = eta (1 - gamma) / (gamma (1 - gamma^eta))."""
    def ek(g):
        if eta == 0:
            return (1 / g - 1) / math.log(1 / g)
        return eta * (1 - g) / (g * (1 - g ** eta))
    lo, hi = 1e-9, 1 - 1e-9
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ek(mid) > expected_k else (lo, mid)
    return (lo + hi) / 2


def sample_k(eta: float, gamma: float, rng: np.random.Generator) -> int:
    ks = np.arange(1, 2000)
    if eta == 0:
        logp = ks * math.log(1 - gamma) - np.log(ks)
    else:
        logp = ks * math.log(1 - gamma) + np.cumsum(np.log((ks - 1 + eta) / ks))
    p = np.exp(logp - logp.max())
    return int(rng.choice(ks, p=p / p.sum()))


def search_rdp(eps_q, lam: int, lam_hat: int, eta: float, gamma: float, expected_k: float) -> float:
    hat = 0.0 if lam_hat == 1 else (1 + eta) * (1 - 1 / lam_hat) * eps_q(lam_hat)
    return eps_q(lam) + hat + (1 + eta) * math.log(1 / gamma) / lam_hat + math.log(expected_k) / (lam - 1)


def search_epsilon(eps_q, eta: float, expected_k: float, delta: float = study.DELTA) -> float:
    """The whole search's epsilon: Theorem 2 over lam and lam_hat, then the conversion of Canonne, Kamath and Steinke."""
    gamma = tnb_gamma_for(eta, expected_k)
    table = {lam: eps_q(lam) for lam in ORDERS}
    best = math.inf
    for lam_hat in (1, *ORDERS):
        for lam in ORDERS:
            r = search_rdp(table.__getitem__, lam, lam_hat, eta, gamma, expected_k)
            best = min(best, r + math.log1p(-1 / lam) - (math.log(delta) + math.log(lam)) / (lam - 1))
    return best


def validation_people(d: dict, people: np.ndarray, salt: bytes = b"ps22-validation") -> np.ndarray:
    """A person goes to validation when a hash of the person's ID is 0 mod 5: independent of everyone else."""
    return np.array([int(hashlib.sha256(salt + str(int(d["people"][k])).encode()).hexdigest(), 16) % 5 == 0 for k in people])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--public-prep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="head-panns")
    ap.add_argument("--arms", default="central,federated")
    ap.add_argument("--searches", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--expected-k", type=float, default=5.0)
    ap.add_argument("--grid", required=True, help="the rates, chosen on public data before any CirCor run")
    ap.add_argument("--panns", type=Path, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    grid = tuple(float(x) for x in a.grid.split(","))
    dp = study._qsh_dp()
    d = study.load(a.prep)
    sig = torch.from_numpy(np.load(a.prep / "signals.npy")).to(study.DEV)
    feats = study.pcg.Features().to(study.DEV)
    train_people, test_people = study.split(d)
    in_val = validation_people(d, train_people)
    fit_people, val_people = train_people[~in_val], train_people[in_val]
    ctx = {"d": d, "sig": sig, "feats": feats}
    heads, info = study.public_heads(d, sig, feats, a.public_prep, a.out, a.panns)
    ctx.update(heads)
    val_recs, test_recs = study.records_of(d, val_people), study.records_of(d, test_people)
    y_val, y_test = d["py"][val_recs], d["py"][test_recs]
    n_pos, n_neg = int(y_val.sum()), int((1 - y_val).sum())
    # A person adds or removes at most two records: AUROC moves by at most 1/(P-1) a positive record and 1/(N-1) a
    # negative one, P and N the validation counts.
    score_sens = max(2 / (n_pos - 2), 1 / (n_pos - 1) + 1 / (n_neg - 1), 2 / (n_neg - 2))
    steps = int(study.EPOCHS * len(fit_people) / study.EXPECTED_BATCH)
    q = study.EXPECTED_BATCH / len(fit_people)
    report = {"environment": study.pcg.environment(), "public_cnn": info, "fit": len(fit_people), "val": len(val_people),
              "val_records": [n_pos, n_neg], "score_sensitivity": score_sens, "steps": steps, "eta": a.eta,
              "expected_k": a.expected_k, "grid": grid, "runs": []}

    def eps_train(model_name: str, arm: str, noise: float):
        dim = ctx[model_name].dim
        if arm == "central":
            return lambda lam: steps * study.rdp_sampled_gaussian(q, noise, lam)
        d2 = 2.0 ** study.SCALE_BITS + math.sqrt(dim)
        rho = dp.rho_of(int(noise), steps, d2, study.SITES, dim)
        return lambda lam: lam * rho

    def calibrate(model_name: str, arm: str, target: float):
        def eps_for(noise):
            return search_epsilon(eps_train(model_name, arm, noise), a.eta, a.expected_k)
        if arm == "federated":
            lo, hi = 1, 2
            while eps_for(hi) > target:
                hi *= 2
            while hi - lo > max(1, hi // 1000):
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if eps_for(mid) > target else (lo, mid)
            noise = hi
        else:
            noise = study._bisect(eps_for, target)
        et = eps_train(model_name, arm, noise)
        sigma_s = score_sens / math.sqrt(2 * min(et(lam) / lam for lam in ORDERS))
        return noise, sigma_s, eps_for(noise)

    rng_master = np.random.default_rng(2026)
    gamma = tnb_gamma_for(a.eta, a.expected_k)
    for model_name in a.models.split(","):
        for arm in a.arms.split(","):
            for target in study.EPSILONS:
                noise, sigma_s, eps_check = calibrate(model_name, arm, target)
                sites = study.sites_of(d, fit_people)
                for search in range(a.searches):
                    t0 = time.time()
                    rng = np.random.default_rng(rng_master.integers(1 << 32))
                    k = sample_k(a.eta, gamma, rng)
                    best, tried = None, []
                    for _ in range(k):
                        lr = float(rng.choice(grid))
                        seed = int(rng.integers(1, 1 << 30))
                        score, _ = study.train(model_name, arm, ctx, fit_people, sites, steps, noise, True, seed, lr)
                        raw = float(study.pcg.roc_auc_score(y_val, score(val_recs)))
                        noisy = raw + float(rng.normal(0, sigma_s))
                        tried.append({"lr": lr, "val_auroc": raw, "noisy": noisy})
                        if best is None or noisy > best[0]:
                            best = (noisy, lr, score)
                    s_test = best[2](test_recs)
                    pred = s_test > 0
                    report["runs"].append({
                        "model": model_name, "arm": arm, "epsilon": target, "search": search, "noise": noise,
                        "score_sigma": sigma_s, "epsilon_accounted": eps_check, "chosen_lr": best[1], "noisy_score": best[0],
                        "auroc": study.auc_people(y_test, s_test, d["person_of"][test_recs]),
                        "balanced_accuracy": float(0.5 * (pred[y_test == 1].mean() + (~pred[y_test == 0]).mean())),
                        "share_predicted_positive": float(pred.mean()), "train_s": time.time() - t0,
                        "local_only": {"k": k, "tried": tried}})
                    print(json.dumps({"model": model_name, "arm": arm, "epsilon": target, "search": search,
                                      "auroc": round(report["runs"][-1]["auroc"]["auroc"], 3)}), flush=True)
                    with open(a.out / "report.json", "w") as f:
                        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
