#!/usr/bin/env python3
"""A one-run guessing-game audit of the two DP training procedures (Mahloujifar, Melis and Chaudhuri, ICML 2025).

On the owner's choices of 2026-11-24 (references/pets-evaluation-2026-11.md): the k-candidate canary game, run in one
training run on both studies. *Auditing f-Differential Privacy in One Run* (arXiv:2410.22235) inserts m canary slots,
each holding one of k candidates drawn uniformly; an adversary guesses the candidate of at most c' slots and abstains
on the rest, and Theorem 3.1 bounds, for an f-DP mechanism, the chance of c or more right guesses. Its Algorithm 3
decides whether that chance is under tau; the paper's appendix code (`audit_rh_with_cap`) is reimplemented here, and a
hypothesis it rejects at tau = 0.05 is one the run shows false. Hypotheses are Gaussian DP, mu-GDP for mu on a grid;
the audit's figure is the largest mu rejected, a lower bound on the mechanism's own mu.

- **k = 2, both studies**: the include-or-exclude canaries of `scripts/audit_dp_one_run.py` (white-box Dirac gradients,
  one per coordinate, every release observed, every other contribution known), so changing a slot is adding or
  removing one canary, the relation the accountants state. The guess is the top and bottom c'/2 scores.
- **k = 7, the heart federation**: each run's 14 coordinates are two slots of 7; the canary is one of 7 one-hot
  gradients of the clipping norm, so changing a slot swaps one record for another. The guess is the slot's largest
  total, on the c' slots whose top two totals are furthest apart.

Beside each audit: the accountant's mu for the same relation, against which the lower bound must stay (the heart
federation's zCDP rho gives mu = sqrt(2 rho) for adding or removing, and twice that for a swap, by group privacy; the
noise is a sum of discrete Gaussians, for which mu-GDP is the continuous Gaussian's, an approximation, which is why the
riskcal bound below is from rho itself); and Kulynych et al.'s bound on a reconstruction's success, 1 - f(1/k) (NeurIPS
2025, Theorem 3.1), which `scripts/audit_dp_kcandidate_bound.py` adds from riskcal in its own environment. Seeded
checks: a sixteenth of the variance, and DP-SGD with every person in every step, each of which the audit must reject.

    python scripts/audit_dp_kcandidate.py -o docs/measurements/dp-kcandidate-audit-<date>.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from statistics import NormalDist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

TAU = 0.05
GUESS_FRACTIONS = (0.01, 0.03, 0.1, 0.3)       # c' as a share of m, fixed before any run; each tested at TAU / 4
MU_STEP, MU_MAX = 0.001, 20.0


# ------------------------------------------------------------------------- the paper's Algorithm 3 (appendix code)


_N = NormalDist()


def gdp_blow_up_inverse(mu: float):
    """The inverse of 1 - f for mu-GDP: f(a) = Phi(Phi^-1(1 - a) - mu), so (1 - f)^-1 (x) = Phi(Phi^-1(x) - mu)."""
    def inv(x: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return _N.cdf(_N.inv_cdf(x) - mu)
    return inv


def not_rejected(inverse_blow_up, m: int, c: int, c_cap: int, threshold: float, k: int) -> bool:
    """`audit_rh_with_cap` of the paper's appendix: True when the hypothesis survives the observation (c right of c_cap
    guesses on m slots of k candidates), False when it is rejected at `threshold`."""
    threshold = threshold * c_cap / m
    alpha = threshold * c / c_cap
    beta = threshold * (c_cap - c) / c_cap
    h, r = [0.0] * (c + 1), [0.0] * (c + 1)
    h[c], r[c] = beta, alpha
    for i in range(c - 1, -1, -1):
        h[i] = max(h[i + 1], (k - 1) * inverse_blow_up(r[i + 1]))
        r[i] = r[i + 1] + (i / (c_cap - i)) * (h[i] - h[i + 1])
    return not (r[0] + h[0] > c_cap / m)


def mu_lower_bound(m: int, c: int, c_cap: int, k: int, threshold: float) -> float:
    """The largest mu (to MU_STEP, rounded down) whose mu-GDP hypothesis the observation rejects; 0 when it rejects none.
    A stronger hypothesis (a smaller mu) is rejected whenever a weaker one is, so the rejected mu form an interval from
    0 and a bisection finds its end."""
    if not_rejected(gdp_blow_up_inverse(MU_STEP), m, c, c_cap, threshold, k):
        return 0.0
    lo, hi = MU_STEP, MU_MAX                       # lo rejected; hi must not be
    if not not_rejected(gdp_blow_up_inverse(hi), m, c, c_cap, threshold, k):
        raise SystemExit(f"mu {MU_MAX} is rejected too: widen MU_MAX")
    while hi - lo > MU_STEP:
        mid = (lo + hi) / 2
        if not_rejected(gdp_blow_up_inverse(mid), m, c, c_cap, threshold, k):
            hi = mid
        else:
            lo = mid
    return math.floor(lo / MU_STEP) * MU_STEP


def gdp_epsilon(mu: float, delta: float) -> float:
    """The epsilon of mu-GDP at delta (Dong, Roth and Su, Corollary 2.13), by bisection."""
    if mu <= 0:
        return 0.0
    def d(eps: float) -> float:
        return _N.cdf(-eps / mu + mu / 2) - math.exp(eps) * _N.cdf(-eps / mu - mu / 2)
    lo, hi = 0.0, 1.0
    while d(hi) > delta:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if d(mid) > delta else (lo, mid)
    return hi


def audit(m: int, right_by_guesses: dict[int, int], k: int, delta: float, *, mu_accounted: float | None = None,
          epsilon_accounted: float | None = None) -> dict:
    """The largest mu rejected over the guess grid, and whether it is under the accountant: by mu where the accountant
    is Gaussian DP, else by the epsilon at the study's delta."""
    rows = []
    for c_cap, c in sorted(right_by_guesses.items()):
        mu = mu_lower_bound(m, c, c_cap, k, TAU / len(GUESS_FRACTIONS))
        rows.append({"guesses": c_cap, "right": c, "mu_lower_bound": mu})
    best = max(r["mu_lower_bound"] for r in rows)
    eps = gdp_epsilon(best, delta)
    under = best <= mu_accounted if mu_accounted is not None else eps <= float(epsilon_accounted)
    return {"slots": m, "k": k, "rows": rows, "mu_lower_bound": best,
            "mu_accounted": None if mu_accounted is None else round(mu_accounted, 4),
            "epsilon_lower_bound": round(eps, 4), "delta": delta, "under_the_accountant": bool(under)}


def guesses_of(m: int) -> list[int]:
    return sorted({max(2, int(round(f * m)) // 2 * 2) for f in GUESS_FRACTIONS})


# --------------------------------------------------------------------------------------------------- the games


def binary_game(scores: np.ndarray, included: np.ndarray) -> dict[int, int]:
    """k = 2, add or remove: the top c'/2 scores guessed in, the bottom c'/2 out."""
    order = np.argsort(scores, kind="stable")
    out = {}
    for c_cap in guesses_of(len(scores)):
        top, bottom = order[-(c_cap // 2):], order[: c_cap // 2]
        out[c_cap] = int(included[top].sum() + (~included[bottom]).sum())
    return out


def fedheart_swap_scores(runs: int, variance: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The heart federation's releases with 14 // k slots of k one-hot candidates a run: the chosen candidate's
    coordinate carries the clipping norm (2^16 in fixed point) each round; every hospital adds its discrete Gaussian
    noise to every coordinate. Returns the (slots, k) totals and the chosen candidates."""
    from qsh.ai import dp, fed_heart
    dim, unit, n_sites = 14, 2 ** fed_heart.DP_FIXED_POINT_BITS, 4
    per_run = dim // k
    rng = np.random.default_rng(seed)
    totals, chosen = np.zeros((runs * per_run, k)), np.zeros(runs * per_run, dtype=int)
    for run in range(runs):
        pick = rng.integers(0, k, per_run)
        rb = dp.seeded_rand_below(f"kcand-fedheart-{k}-{seed}-{run}")
        t = np.zeros(dim)
        for _ in range(fed_heart.DP_ROUNDS):
            for j in range(dim):
                s = unit if (j < per_run * k and j % k == pick[j // k]) else 0
                for _site in range(n_sites):
                    s += dp.discrete_gaussian(variance, rb)
                t[j] += s
        totals[run * per_run:(run + 1) * per_run] = t[:per_run * k].reshape(per_run, k)
        chosen[run * per_run:(run + 1) * per_run] = pick
    return totals, chosen


def swap_game(totals: np.ndarray, chosen: np.ndarray) -> dict[int, int]:
    """k > 2: guess each slot's largest total, on the c' slots whose two largest totals are furthest apart."""
    top2 = np.sort(totals, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    guess = totals.argmax(axis=1)
    order = np.argsort(-margin, kind="stable")
    return {c_cap: int((guess[order[:c_cap]] == chosen[order[:c_cap]]).sum()) for c_cap in guesses_of(len(chosen))}


def main() -> int:
    from audit_dp_one_run import _murmur, fedheart_scores, murmur_scores

    from qsh.ai import dp, fed_heart

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fedheart-runs", type=int, default=1000)
    ap.add_argument("--murmur-runs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    t0 = time.time()
    dim, unit, n_sites = 14, 2 ** fed_heart.DP_FIXED_POINT_BITS, 4
    sens = fed_heart.DP_CLIP * unit + math.sqrt(dim)
    delta_h = fed_heart.DP_DELTA
    out: dict = {"fedheart": {}, "murmur": {}, "seeded_checks": {}}

    def rho_of(V: float) -> float:
        """zCDP of the T-round sum of four discrete Gaussians of variance V per coordinate, sensitivity `sens`."""
        return fed_heart.DP_ROUNDS * sens ** 2 / (2 * n_sites * V)

    for eps in (8.0, 4.0, 2.0, 1.0):
        V = dp.variance_for(eps, fed_heart.DP_ROUNDS, sens, n_sites, dim, delta_h)
        rho = rho_of(V)
        s, inc = fedheart_scores(args.fedheart_runs, V, args.seed + 10)
        arm = {"variance_per_hospital": V, "rho": rho,
               "k2_add_remove": audit(len(s), binary_game(s, inc), 2, delta_h, mu_accounted=math.sqrt(2 * rho))}
        totals, chosen = fedheart_swap_scores(args.fedheart_runs, V, 7, args.seed + 20)
        arm["k7_swap"] = audit(len(chosen), swap_game(totals, chosen), 7, delta_h, mu_accounted=2 * math.sqrt(2 * rho))
        out["fedheart"][f"epsilon_{eps:g}"] = arm
        print("fedheart", eps, arm["k2_add_remove"]["mu_lower_bound"], arm["k7_swap"]["mu_lower_bound"], flush=True)
    V1 = dp.variance_for(1.0, fed_heart.DP_ROUNDS, sens, n_sites, dim, delta_h)
    s, inc = fedheart_scores(args.fedheart_runs, V1 // 16, args.seed + 11)
    out["seeded_checks"]["fedheart_a_sixteenth_of_the_variance"] = {
        "variance_per_hospital": V1 // 16, "accountant_assumes": V1, "rho_accounted": rho_of(V1),
        "k2_add_remove": audit(len(s), binary_game(s, inc), 2, delta_h, mu_accounted=math.sqrt(2 * rho_of(V1)))}

    mod = _murmur()
    q, steps, fillers = mod.EXPECTED_BATCH / 511, 239, 511
    for sigma in (1.89138, 3.45294):
        s, inc = murmur_scores(mod, args.murmur_runs, sigma, steps, q, fillers, args.seed + 10)
        acc = float(mod.dpsgd_epsilon(q, sigma, steps, mod.DELTA))
        out["murmur"][f"noise_{sigma:g}"] = {
            "noise_multiplier": sigma, "sampling_rate": q, "steps": steps, "epsilon_accounted": round(acc, 4),
            "k2_add_remove": audit(len(s), binary_game(s, inc), 2, mod.DELTA, epsilon_accounted=acc)}
        print("murmur", sigma, out["murmur"][f"noise_{sigma:g}"]["k2_add_remove"]["mu_lower_bound"], flush=True)
    s, inc = murmur_scores(mod, args.murmur_runs, 1.89138, steps, q, fillers, args.seed + 11, subsample=False)
    acc = float(mod.dpsgd_epsilon(q, 1.89138, steps, mod.DELTA))
    out["seeded_checks"]["murmur_every_person_every_step"] = {
        "noise_multiplier": 1.89138, "accountant_assumes_sampling_rate": q, "steps": steps, "epsilon_accounted": round(acc, 4),
        "k2_add_remove": audit(len(s), binary_game(s, inc), 2, mod.DELTA, epsilon_accounted=acc)}

    record = {
        "what": "a one-run k-candidate guessing-game audit (white-box canaries, every release observed) of the heart "
                "federation's and the murmur study's DP training",
        "method": "Mahloujifar, Melis and Chaudhuri, Auditing f-Differential Privacy in One Run, ICML 2025 "
                  "(arXiv:2410.22235): Theorem 3.1 and Algorithm 3 as the appendix's audit_rh_with_cap, reimplemented; "
                  "mu-GDP hypotheses",
        "design": {"tau": TAU, "guess_shares": list(GUESS_FRACTIONS), "each_tested_at": TAU / len(GUESS_FRACTIONS),
                   "mu_resolution": MU_STEP, "fedheart_runs": args.fedheart_runs,
                   "murmur_runs": args.murmur_runs, "seed": args.seed,
                   "noise": "fedheart: dp.seeded_rand_below (not private); murmur: torch's generator, as the study"},
        **out,
        "reading": "A mu lower bound under the accountant's mu is consistent with the accounting and proves nothing more; "
                   "the seeded checks must exceed theirs. The murmur arms' accountant is not Gaussian DP (Poisson-"
                   "subsampled Gaussian), so they are compared by epsilon at the study's delta, the lower bound's mu "
                   "converted as Gaussian DP; riskcal's bound from the privacy-loss distribution stands beside them "
                   "(scripts/audit_dp_kcandidate_bound.py).",
        "seconds": round(time.time() - t0, 1),
        "environment": {"git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                              text=True).stdout.strip(),
                        "python": platform.python_version(), "numpy": np.__version__, "date": time.strftime("%Y-%m-%d")},
    }
    Path(args.output).write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: record[k] for k in ("fedheart", "murmur", "seeded_checks")}, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
