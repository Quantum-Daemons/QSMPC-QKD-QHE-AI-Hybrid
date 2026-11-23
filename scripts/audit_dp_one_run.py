#!/usr/bin/env python3
"""A one-run audit of the two DP training procedures here (Steinke, Nasr and Jagielski, NeurIPS 2023).

On the owner's choice of 2026-11-22 (references/pets-evaluation-2026-11.md). *Privacy Auditing with One (1) Training
Run* (arXiv:2305.08846) audits a DP training procedure by inserting m auditing examples ("canaries"), each included
independently with probability 1/2, guessing from the output which were included, and bounding the number of correct
guesses an (epsilon, delta)-DP procedure allows: with r guesses and v right, Corollary 5.4 bounds
P[at least v right] by f(v) + 2 m delta max_i (f(v - i) - f(v)) / i, where f(v) = P[Binomial(r, e^eps / (1 + e^eps))
>= v]. The largest epsilon this bound rejects at 95 percent is the audit's empirical lower bound (their Appendix D's
p-value and bisection, implemented here and checked against the values that appendix prints). A lower bound above the
accountant's epsilon would mean the accounting or the implementation is wrong; one below it shows nothing more.

The canaries are the paper's white-box kind: gradients, one per coordinate (a "Dirac" canary, a vector of length the
clipping norm along one axis), seen by an adversary who observes every release, the noisy sum of each round or step,
and knows every other contribution. Each other contribution is therefore left out (the murmur study's 511 people stay,
with zero gradients, so its Poisson sampling is as it was). A canary's score is the sum over releases of its coordinate.
Runs are independent and each canary is in one run only, so the runs pooled are one audit in the corollary's sense:
every bit of S changes one run, which is (epsilon, delta)-DP. Guesses are the top and bottom r/2 scores, for four r
fixed in advance, each tested at 0.05 / 4 (Bonferroni), and the largest bound is reported.

- **fedheart**: the page's record-level DP federation (`qsh.ai.fed_heart`, `clientFedHeart.ts`): 50 rounds, each of
  the four hospitals adding discrete Gaussian noise of variance V to each coordinate of its fixed-point sum (2^-16),
  V from `dp.variance_for` at epsilon 8, 4, 2 and 1, delta 1e-6, epsilon from `dp.epsilon_of`. Noise from
  `dp.seeded_rand_below` (not private; the CSPRNG's distribution), drawn by `dp.discrete_gaussian`, the page's sampler.
- **murmur**: the murmur study's central DP-SGD (`scripts/train/medical_murmur_dp.py`): people Poisson-sampled at
  q = 64/511, 239 steps, each sampled person's gradient clipped to 1 and the sum noised by `noisy_sum` itself, at the
  two noise multipliers its search route ran (1.89138 and 3.45294), epsilon from its `dpsgd_epsilon`, delta 1e-5.
- **Seeded checks**: the heart federation with a sixteenth of the variance its accountant assumes, and the DP-SGD with
  every person in every step under an accountant that assumes sampling. Each must be caught: a lower bound above the
  epsilon its accountant states.

On the CPU only. Only aggregate figures are written.

    python scripts/audit_dp_one_run.py -o docs/measurements/dp-one-run-audit-<date>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CONFIDENCE_P = 0.05
GUESS_GRID = (100, 300, 1000, 3000)          # r, guesses: the top and bottom r/2 scores; fixed before any run


# ------------------------------------------------------------------------------------------- the paper's bound


def p_value(m: int, r: int, v: int, eps: float, delta: float) -> float:
    """Corollary 5.4, as Appendix D computes it: the bound on P[v or more of r guesses right] under (eps, delta)-DP."""
    from scipy.stats import binom

    q = 1.0 / (1.0 + math.exp(-eps))
    beta = float(binom.sf(v - 1, r, q))
    if v <= 0:
        return 1.0
    s = np.cumsum(binom.pmf(np.arange(v - 1, -1, -1), r, q))         # s_i = sum of P[Bin = k] for k in v-i .. v-1
    alpha = max(0.0, float(np.max(s / np.arange(1, v + 1))))
    return min(1.0, beta + alpha * delta * 2 * m)


def epsilon_lower_bound(m: int, r: int, v: int, delta: float, p: float = CONFIDENCE_P) -> float:
    """Appendix D's search: the largest epsilon whose bound is still below p, by doubling then 30 bisections."""
    lo, hi = 0.0, 1.0
    while p_value(m, r, v, hi, delta) < p:
        hi += 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if p_value(m, r, v, mid, delta) < p:
            lo = mid
        else:
            hi = mid
    return lo


def audit(scores: np.ndarray, included: np.ndarray, delta: float, accounted: float) -> dict:
    """Guesses at each r of the grid, each tested at p / len(grid); the largest bound, and whether it is under the
    accountant's epsilon."""
    m = len(scores)
    order = np.argsort(scores, kind="stable")
    rows = []
    for r in GUESS_GRID:
        if r > m:
            continue
        top, bottom = order[-(r // 2):], order[: r // 2]
        v = int(included[top].sum() + (~included[bottom]).sum())
        lb = epsilon_lower_bound(m, r, v, delta, CONFIDENCE_P / len(GUESS_GRID))
        rows.append({"guesses": r, "right": v, "epsilon_lower_bound": round(lb, 4)})
    best = max(row["epsilon_lower_bound"] for row in rows)
    return {"canaries": int(m), "included": int(included.sum()), "rows": rows, "epsilon_lower_bound": best,
            "epsilon_accounted": round(accounted, 4), "under_the_accountant": best <= accounted}


# --------------------------------------------------------------------------------------------- the mechanisms


def fedheart_scores(runs: int, variance: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The page's DP federation's releases with one Dirac canary per coordinate: each round, a hospital's sum is its
    integer gradient sum (the canary's 2^16 in its coordinate when included; the rest known and left out) plus
    `discrete_gaussian(V)` from each of the four hospitals."""
    from qsh.ai import dp, fed_heart

    dim, unit, n_sites = 14, 2 ** fed_heart.DP_FIXED_POINT_BITS, 4
    rng = np.random.default_rng(seed)
    scores, included = np.zeros(runs * dim), np.zeros(runs * dim, dtype=bool)
    for run in range(runs):
        inc = rng.random(dim) < 0.5
        rb = dp.seeded_rand_below(f"one-run-fedheart-{seed}-{run}")
        total = [0] * dim
        for _ in range(fed_heart.DP_ROUNDS):
            for k in range(dim):
                s = unit if inc[k] else 0
                for _site in range(n_sites):
                    s += dp.discrete_gaussian(variance, rb)
                total[k] += s
        scores[run * dim:(run + 1) * dim] = total
        included[run * dim:(run + 1) * dim] = inc
    return scores, included


def _murmur():
    import torch

    spec = importlib.util.spec_from_file_location("murmur_dp", REPO / "scripts" / "train" / "medical_murmur_dp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DEV = torch.device("cpu")              # its noise is drawn on DEV, which the study sets to the GPU
    return mod


def murmur_scores(mod, runs: int, sigma: float, steps: int, q: float, fillers: int, seed: int,
                  subsample: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The study's central DP-SGD releases, `noisy_sum` of the Poisson-sampled people's clipped gradients, with one
    Dirac canary per coordinate of the 129-weight head and the study's 511 people as zero gradients."""
    import torch

    dim = 129
    rng = np.random.default_rng(seed)
    g = torch.Generator(device=mod.DEV).manual_seed(seed)
    rows = torch.zeros((fillers + dim, dim))
    rows[fillers + torch.arange(dim), torch.arange(dim)] = mod.CLIP
    scores, included = np.zeros(runs * dim), np.zeros(runs * dim, dtype=bool)
    for run in range(runs):
        inc = rng.random(dim) < 0.5
        people = np.concatenate([np.arange(fillers), fillers + np.flatnonzero(inc)])
        total = torch.zeros(dim, dtype=torch.float64)
        for _ in range(steps):
            chosen = people[rng.random(len(people)) < q] if subsample else people
            if len(chosen):                   # the study's loop takes no step on an empty sample
                total += mod.noisy_sum(rows[torch.from_numpy(chosen)], True, sigma, g).double()
        scores[run * dim:(run + 1) * dim] = total.numpy()
        included[run * dim:(run + 1) * dim] = inc
    return scores, included


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fedheart-runs", type=int, default=1000)
    ap.add_argument("--murmur-runs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    from qsh.ai import dp, fed_heart

    t0 = time.time()
    out = {"fedheart": {}, "murmur": {}, "seeded_checks": {}}
    dim, unit, n_sites = 14, 2 ** fed_heart.DP_FIXED_POINT_BITS, 4
    sens = fed_heart.DP_CLIP * unit + math.sqrt(dim)
    for eps in (8.0, 4.0, 2.0, 1.0):
        V = dp.variance_for(eps, fed_heart.DP_ROUNDS, sens, n_sites, dim, fed_heart.DP_DELTA)
        acc = dp.epsilon_of(V, fed_heart.DP_ROUNDS, sens, n_sites, dim, fed_heart.DP_DELTA)
        s, inc = fedheart_scores(args.fedheart_runs, V, args.seed)
        out["fedheart"][f"epsilon_{eps:g}"] = {"variance_per_hospital": V, **audit(s, inc, fed_heart.DP_DELTA, acc)}
        print("fedheart", eps, out["fedheart"][f"epsilon_{eps:g}"]["epsilon_lower_bound"], flush=True)
    V1 = dp.variance_for(1.0, fed_heart.DP_ROUNDS, sens, n_sites, dim, fed_heart.DP_DELTA)
    s, inc = fedheart_scores(args.fedheart_runs, V1 // 16, args.seed + 1)
    out["seeded_checks"]["fedheart_a_sixteenth_of_the_variance"] = {
        "variance_per_hospital": V1 // 16, "accountant_assumes": V1,
        **audit(s, inc, fed_heart.DP_DELTA, dp.epsilon_of(V1, fed_heart.DP_ROUNDS, sens, n_sites, dim,
                                                         fed_heart.DP_DELTA))}

    mod = _murmur()
    q, steps, fillers = mod.EXPECTED_BATCH / 511, 239, 511
    for sigma in (1.89138, 3.45294):
        s, inc = murmur_scores(mod, args.murmur_runs, sigma, steps, q, fillers, args.seed)
        out["murmur"][f"noise_{sigma:g}"] = {"noise_multiplier": sigma, "sampling_rate": q, "steps": steps,
                                             **audit(s, inc, mod.DELTA, mod.dpsgd_epsilon(q, sigma, steps, mod.DELTA))}
        print("murmur", sigma, out["murmur"][f"noise_{sigma:g}"]["epsilon_lower_bound"], flush=True)
    s, inc = murmur_scores(mod, args.murmur_runs, 1.89138, steps, q, fillers, args.seed + 1, subsample=False)
    out["seeded_checks"]["murmur_every_person_every_step"] = {
        "noise_multiplier": 1.89138, "accountant_assumes_sampling_rate": q, "steps": steps,
        **audit(s, inc, mod.DELTA, mod.dpsgd_epsilon(q, 1.89138, steps, mod.DELTA))}

    record = {
        "what": "a one-run DP audit (white-box Dirac canaries, every release observed) of the heart federation's and "
                "the murmur study's DP training",
        "method": "Steinke, Nasr and Jagielski, Privacy Auditing with One (1) Training Run, NeurIPS 2023 "
                  "(arXiv:2305.08846): Corollary 5.4 and Appendix D, implemented here",
        "design": {"guesses": list(GUESS_GRID), "each_tested_at": CONFIDENCE_P / len(GUESS_GRID),
                   "confidence": 1 - CONFIDENCE_P, "fedheart_runs": args.fedheart_runs, "murmur_runs": args.murmur_runs,
                   "canaries_per_run": {"fedheart": dim, "murmur": 129}, "seed": args.seed,
                   "noise": "fedheart: dp.seeded_rand_below (not private); murmur: torch's generator, as the study"},
        **out,
        "reading": "A lower bound under the accountant's epsilon is consistent with the accounting and proves nothing more; "
                   "the seeded checks show the audit catches a mechanism that adds less noise or samples no one.",
        "seconds": round(time.time() - t0, 1),
        "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                              text=True).stdout.strip(),
                        "python": platform.python_version(), "numpy": np.__version__, "date": time.strftime("%Y-%m-%d")},
    }
    Path(args.output).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("fedheart", "murmur", "seeded_checks")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
