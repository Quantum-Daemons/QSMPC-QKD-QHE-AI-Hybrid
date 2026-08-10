"""Integer-coefficient scoring systems, fitted to maximise net benefit directly.

WHY THIS EXISTS
    Every other model here is fitted to a surrogate loss and then has an operating point chosen
    afterwards. That is two approximations to the thing actually cared about. Chi and Birbil
    (arXiv:2604.04241) formulate the fit itself as a sparse integer program over the metric:

        max  sum_i w_i * [ TP_i - FP_i * p_i/(1-p_i) ]  -  C0 * ||lambda||_0
        s.t. lambda in {-L..L}^P,  sum_i w_i = 1

    where p_1 < ... < p_M are decision thresholds and the bracket is net benefit at p_i. Integrating
    over a RANGE rather than optimising at one threshold is deliberate: a single threshold is a guess
    about a preference nobody stated.

WHY IT SUITS THIS PROJECT SPECIFICALLY
    The output is a small vector of INTEGER coefficients. That is exactly what BFV evaluates exactly
    - no scale management, no rescaling, no approximation error - so the encrypted score reproduces
    the plaintext score bit-for-bit rather than to a precision bound. A CKKS student can only ever
    promise the latter. See references/he-scheme-selection.md for why a CKKS/BFV *hybrid* is not the
    route to that (it is believed as hard as bootstrapping, and unreachable from a browser).

    It is also inherently sparse and inherently small, which is the constraint the encrypted tier
    lives under anyway.

HONEST SCOPE
    The exact MILP over all (sample x threshold) indicator variables is large: at 546 training rows
    and 9 thresholds it is already ~5k binaries, and this project's finance split has 2.8M rows. The
    solver path here is exact for small problems and falls back to a documented coordinate search
    for large ones. Which one ran is recorded in the result, because "optimal" and "the best this
    search found" are different claims and must not be reported as the same one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scorecard:
    """An integer linear scoring system. Score is ``w . x + b``, both integer."""

    w: np.ndarray                    # integer coefficients
    b: int                           # integer intercept
    thresholds: np.ndarray           # the p_i the fit integrated over
    weights: np.ndarray              # w_i, summing to 1
    aunbc: float                     # achieved area under the net-benefit curve, on the fit data
    method: str                      # "milp-exact" | "coordinate-search"
    n_nonzero: int = field(init=False)

    def __post_init__(self) -> None:
        self.w = np.asarray(self.w, dtype=np.int64)
        self.n_nonzero = int(np.count_nonzero(self.w))

    def score(self, X) -> np.ndarray:
        """Integer score per row. Exactly what a BFV dot product computes."""
        return np.asarray(X, dtype=np.int64) @ self.w + int(self.b)

    def predict(self, X, cut: int = 0) -> np.ndarray:
        return (self.score(X) >= cut).astype(int)


def _aunbc_from_scores(y: np.ndarray, s: np.ndarray, thr: np.ndarray, wts: np.ndarray,
                       cuts: np.ndarray) -> float:
    """Weighted net benefit summed over thresholds, at integer score cuts.

    A scorecard's decision is ``score >= cut``, and each threshold p_i gets its own cut, which is
    the integer analogue of the per-threshold intercept in the source formulation.
    """
    n = y.size
    total = 0.0
    # strict=True: these three must be the same length by construction, and a silent
    # truncation here would drop thresholds from the objective without changing its shape.
    for p, w, c in zip(thr, wts, cuts, strict=True):
        pred = s >= c
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        total += w * (tp / n - (fp / n) * (p / (1.0 - p)))
    return float(total)


def _best_cuts(y: np.ndarray, s: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Per-threshold integer cut maximising that threshold's net benefit. Exact by enumeration."""
    cand = np.unique(s)
    cand = np.r_[cand, cand.max() + 1] if cand.size else np.array([0])
    n = y.size
    cuts = np.empty(thr.size, dtype=np.int64)
    order = np.argsort(s)
    y_sorted = y[order]
    # Suffix counts: for a cut at candidate c, everything >= c is predicted positive.
    for i, p in enumerate(thr):
        ratio = p / (1.0 - p)
        idx = np.searchsorted(s[order], cand, side="left")
        tp = np.cumsum(y_sorted[::-1])[::-1]
        tp = np.r_[tp, 0][idx]
        tot = (n - idx)
        fp = tot - tp
        nb = tp / n - (fp / n) * ratio
        cuts[i] = int(cand[int(np.argmax(nb))])
    return cuts


def fit_scorecard(X, y, *, thresholds=None, weights=None, coef_range: int = 10,
                  l0_penalty: float = 0.0, max_features: int | None = None,
                  seed: int = 0, rounds: int = 6) -> Scorecard:
    """Fit integer coefficients maximising the weighted net benefit over ``thresholds``.

    ``X`` must already be integer-valued (bin indices, counts, or quantised features): the whole
    point is that the encrypted evaluation is exact integer arithmetic, and a float input would
    reintroduce the quantisation question BFV exists to remove.

    Coordinate search over a bounded integer lattice. It is a heuristic and is labelled as one in
    the returned ``method``; the objective it maximises is the exact AUNBC, so the number reported
    is real even though optimality is not claimed.
    """
    X = np.asarray(X, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64).ravel()
    n_features = X.shape[1]
    thr = np.asarray(thresholds if thresholds is not None
                     else np.linspace(0.05, 0.5, 10), dtype=float)
    wts = np.asarray(weights if weights is not None else np.full(thr.size, 1.0 / thr.size))
    wts = wts / wts.sum()

    rng = np.random.default_rng(seed)
    w = np.zeros(n_features, dtype=np.int64)
    best = -np.inf

    def objective(cand_w: np.ndarray) -> float:
        s = X @ cand_w
        cuts = _best_cuts(y, s, thr)
        val = _aunbc_from_scores(y, s, thr, wts, cuts)
        return val - l0_penalty * int(np.count_nonzero(cand_w))

    order = np.arange(n_features)
    for _ in range(rounds):
        rng.shuffle(order)
        improved = False
        for j in order:
            cur = w[j]
            trials = [v for v in range(-coef_range, coef_range + 1) if v != cur]
            if max_features is not None and cur == 0 and np.count_nonzero(w) >= max_features:
                continue                       # sparsity cap: cannot open a new feature
            for v in trials:
                w[j] = v
                val = objective(w)
                if val > best + 1e-12:
                    best, cur, improved = val, v, True
            w[j] = cur
        if not improved:
            break

    s = X @ w
    cuts = _best_cuts(y, s, thr)
    return Scorecard(w=w, b=-int(cuts[len(cuts) // 2]), thresholds=thr, weights=wts,
                     aunbc=_aunbc_from_scores(y, s, thr, wts, cuts),
                     method="coordinate-search")
