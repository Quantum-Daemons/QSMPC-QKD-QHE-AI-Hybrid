#!/usr/bin/env python3
"""The murmur heads by objective perturbation, which needs no learning rate (Round 86, the owner's choice of 2026-11-14).

Redberg, Koskela and Wang, "Improving the Privacy and Practicality of Objective Perturbation for Differentially Private
Linear Learners" (NeurIPS 2023, arXiv:2401.00583), Algorithm 1 (approximate minima perturbation with gradient clipping)
and Theorem 4.1, as read on arxiv.org/html. Each PERSON is one example: the mean of its labelled records' window features,
mapped to x = [z / (sqrt2 |z|), 1/sqrt2] (unit norm, the intercept last), label 1 if any of its records has a murmur.
The unweighted logistic loss f(t, y) = log(1 + e^t) - y t is convex and beta = 1/4 smooth with |f'| < 1, so the gradient
of one person's loss is at most |x| = 1 = C and the clip never binds. The objective sum_i l(theta, x_i) + (lambda/2)
|theta|^2 + b^T theta, b ~ N(0, sigma^2 I), is minimised by L-BFGS until its gradient norm is at most tau, then
N(0, sigma_out^2 I) is added. Theorem 4.1: for every alpha > 1 it is (alpha, eps)-RDP with

  eps(alpha) <= -log(1 - beta/lambda) + C^2 / (2 sigma^2) + log E_{X ~ N(0, C^2/sigma^2)}[e^{(alpha - 1)|X|}] / (alpha - 1)
                + 2 tau^2 alpha / (sigma_out^2 lambda^2),

and E[e^{t|X|}] = 2 e^{t^2 s^2 / 2} Phi(t s) for X ~ N(0, s^2). lambda = 10, tau = 0.01 and sigma_out = 0.15 are fixed
before any data is seen (the paper's tau and sigma_out); the optimiser does not enter the guarantee, so there is nothing
to tune on CirCor. The decision threshold comes from the PhysioNet 2016 stand-in, fitted the same way at each epsilon.

Usage: python -I scripts/train/medical_murmur_dp_objpert.py --prep <CirCor> --public-prep <PhysioNet 2016> --out <dir>
       [--features head-panns,head-public] [--panns Cnn14_mAP=0.431.pth]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("medical_murmur_dp", HERE / "medical_murmur_dp.py")
study = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("medical_murmur_dp", study)
_spec.loader.exec_module(study)

BETA, C, LAMBDA, TAU, SIGMA_OUT = 0.25, 1.0, 10.0, 0.01, 0.15
ORDERS = tuple([1 + k / 20 for k in range(1, 40)] + list(range(3, 257)))


def rdp(alpha: float, sigma: float, lam: float = LAMBDA, tau: float = TAU, s_out: float = SIGMA_OUT) -> float:
    s, t = C / sigma, alpha - 1
    log_e = math.log(2.0) + t * t * s * s / 2 + norm.logcdf(t * s)
    return -math.log1p(-BETA / lam) + s * s / 2 + log_e / t + 2 * tau * tau * alpha / (s_out * s_out * lam * lam)


def epsilon(sigma: float, delta: float = study.DELTA) -> float:
    return min(rdp(o, sigma) + math.log1p(-1 / o) - (math.log(delta) + math.log(o)) / (o - 1) for o in ORDERS)


def sigma_for(target: float) -> float:
    return study._bisect(epsilon, target)


def unit_rows(z: torch.Tensor) -> torch.Tensor:
    z = z / (math.sqrt(2) * z.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return torch.cat([z, torch.full((len(z), 1), 1 / math.sqrt(2), device=z.device)], 1).double()


def solve(x: torch.Tensor, y: torch.Tensor, b: torch.Tensor, lam: float = LAMBDA, tau: float = TAU) -> tuple[torch.Tensor, float]:
    theta = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def objective():
        opt.zero_grad()
        t = x @ theta
        loss = torch.nn.functional.softplus(t).sum() - (y * t).sum() + lam / 2 * theta.dot(theta) + b.dot(theta)
        loss.backward()
        return loss
    g = math.inf
    for _ in range(20):
        opt.step(objective)
        objective()
        g = float(theta.grad.norm())
        if g <= tau:
            break
    if g > tau:
        raise RuntimeError(f"the minimiser stopped at gradient norm {g} > tau {tau}: the guarantee needs tau")
    return theta.detach(), g


def person_rows(h, d: dict, people: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Each person's mean window feature over its labelled records, and each record's mean (for scoring)."""
    rs = torch.zeros(len(d["pats"]), h.x.shape[1], device=study.DEV).index_add_(0, h.win_patient, h.x)
    per_person = torch.stack([rs[torch.from_numpy(d["members"][k]).to(study.DEV)].sum(0)
                              / h.counts[torch.from_numpy(d["members"][k]).to(study.DEV)].sum() for k in people])
    return unit_rows(per_person), unit_rows(rs / h.counts.unsqueeze(1))


def release(x: torch.Tensor, y: torch.Tensor, sigma: float, seed: int) -> tuple[torch.Tensor, float]:
    g = torch.Generator(device=study.DEV).manual_seed(seed)
    b = torch.randn(x.shape[1], generator=g, device=study.DEV, dtype=torch.float64) * sigma
    theta, gnorm = solve(x, y, b)
    if sigma > 0:
        theta = theta + torch.randn(x.shape[1], generator=g, device=study.DEV, dtype=torch.float64) * SIGMA_OUT
    return theta, gnorm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--public-prep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--features", default="head-panns,head-public")
    ap.add_argument("--panns", type=Path, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = study.load(a.prep)
    sig = torch.from_numpy(np.load(a.prep / "signals.npy")).to(study.DEV)
    feats = study.pcg.Features().to(study.DEV)
    heads, info = study.public_heads(d, sig, feats, a.public_prep, a.out, a.panns)
    train_people, test_people = study.split(d)
    share = float(d["person_y"][train_people].mean())
    proxy, proxy_info = study.public_proxy(a.public_prep, feats, len(train_people), share, a.out, "abf", True)
    pd = proxy["d"]
    pfit, pval = study.holdout(pd, np.arange(len(pd["pats"])), 7)
    test_recs = study.records_of(d, test_people)
    y_test, owner = d["py"][test_recs], d["person_of"][test_recs]
    report = {"environment": study.pcg.environment(), "public_cnn": info, "stand_in": proxy_info, "lambda": LAMBDA,
              "tau": TAU, "sigma_out": SIGMA_OUT, "beta": BETA, "C": C, "delta": study.DELTA, "runs": []}
    for fname in a.features.split(","):
        x_train, rec_rows = person_rows(heads[fname], d, train_people)
        y_train = torch.from_numpy(d["person_y"][train_people]).double().to(study.DEV)
        x_test = rec_rows[torch.from_numpy(test_recs).to(study.DEV)]
        px, prec = person_rows(proxy[fname], pd, pfit)
        py_fit = torch.from_numpy(pd["person_y"][pfit]).double().to(study.DEV)
        pval_recs = study.records_of(pd, pval)
        px_val, y_pval = prec[torch.from_numpy(pval_recs).to(study.DEV)], pd["py"][pval_recs]
        for target in (None, *study.EPSILONS):
            sigma = sigma_for(target) if target else 0.0
            for seed in range(1, a.seeds + 1):
                th_pub, _ = release(px, py_fit, sigma, 1000 + seed)
                t_bal = study.decision_thresholds([(px_val @ th_pub).cpu().numpy()], y_pval)
                theta, gnorm = release(x_train, y_train, sigma, seed)
                s_test = (x_test @ theta).cpu().numpy()
                report["runs"].append({
                    "features": fname, "epsilon": target, "seed": seed, "sigma": sigma,
                    "epsilon_accounted": epsilon(sigma) if sigma else None, "grad_norm": gnorm,
                    "auroc": study.auc_people(y_test, s_test, owner),
                    "at_thresholds": {name: study.balanced_at(s_test, y_test, t) for name, t in t_bal.items()}})
                print(json.dumps({"features": fname, "epsilon": target, "seed": seed,
                                  "auroc": round(report["runs"][-1]["auroc"]["auroc"], 3)}), flush=True)
            with open(a.out / "report.json", "w") as f:
                json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
