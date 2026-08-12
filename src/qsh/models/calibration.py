"""Probability calibration, sized for the sample counts this project actually has.

WHY PLATT AND NOT ISOTONIC, and why not a reliability diagram

The measured positive-event counts here are medical 42, finance 164, satellite 113, and genomics has
no outcome label at all. Two separate published results bear on that, and a web summary conflates
them, so both are stated with the right attribution:

  * Niculescu-Mizil and Caruana, "Predicting Good Probabilities With Supervised Learning", ICML 2005:
    "When the calibration set is small (less than about 200-1000 cases), Platt Scaling outperforms
    Isotonic Regression with all nine learning methods."
  * Niculescu-Mizil and Caruana, "Obtaining Calibrated Probabilities from Boosting", UAI 2005: the
    same ordering "when the calibration set is small (less than about 2000 cases)", because isotonic
    is less constrained and so easier to overfit, while "Platt's method also has some overfitting
    control built in".

Every count above is inside both regimes, so Platt is the method and isotonic is not.

Separately: no page here may draw a reliability DIAGRAM. Riley et al. (Stat Med 2021,
doi:10.1002/sim.9025) put the floor for a flexible calibration curve at about 200 events; nothing
here clears it and medical does not clear 100. That rule is already recorded in the project and this
module does not relax it.

But "no diagram" is not "no calibration", and that is the gap this closes. The CORP decomposition
(Dimitriadis, Gneiting and Jordan, arXiv:2008.03033, PNAS 118 e2016191118) yields SCALAR miscalibration
and discrimination terms that are reportable exactly where a curve is not, because the pool-adjacent-
violators algorithm chooses no bin count. Bin count is the free parameter that makes a curve at n=156
a picture of noise, and PAV removes it rather than picking it well.

WHAT CALIBRATION IS NOT, stated because this sits next to an encrypted tier

Calibration is a monotone post-transform of a score. It cannot fix ranking, so it moves neither AUC
nor AUC-PR. And it must be applied AFTER the encrypted-vs-plaintext comparison, never before: the
CKKS approximation error lives in the raw logits, so measuring precision on calibrated probabilities
would measure a property of the sigmoid instead of a property of the ciphertext.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Platt:
    """A fitted two-parameter sigmoid, ``p = 1 / (1 + exp(A * s + B))``.

    Two floats, so the browser can apply the identical transform from the weights JSON without
    shipping a calibrator object or re-deriving anything.
    """

    a: float
    b: float
    n_calibration: int
    positives_calibration: int

    def apply(self, scores: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(self.a * np.asarray(scores, dtype=np.float64) + self.b))

    def as_dict(self) -> dict:
        return {
            "method": "platt",
            "a": float(self.a),
            "b": float(self.b),
            "formula": "p = 1 / (1 + exp(a * score + b))",
            "n_calibration": int(self.n_calibration),
            "positives_calibration": int(self.positives_calibration),
            "why_not_isotonic": (
                "Platt outperforms isotonic below about 200-1000 calibration cases (Niculescu-Mizil "
                "and Caruana, ICML 2005) and below about 2000 (same authors, UAI 2005); this "
                "calibration set is inside both regimes, and isotonic is easier to overfit there."),
        }


def fit_platt(scores: np.ndarray, labels: np.ndarray, *, max_iter: int = 200) -> Platt:
    """Fit Platt scaling by logistic regression of the label on the raw score.

    Uses the Platt/Lin-Weng-Lin target smoothing rather than the raw 0/1 labels: with few positives,
    regressing on hard labels drives the fitted sigmoid to saturate and produces probabilities of
    exactly 0 and 1, which a log-loss cannot score. The smoothing is the standard fix and is why
    Platt has "some overfitting control built in".
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(int)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must align")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"calibration needs both classes, got {n_pos} positive and {n_neg} negative - a fit on "
            f"one class would return a constant and look like a calibrated model")

    hi = 1.0 / (n_pos + 2.0)
    lo = (n_neg + 1.0) / (n_neg + 2.0)
    t = np.where(y == 1, 1.0 - hi, 1.0 - lo)

    from sklearn.linear_model import LogisticRegression

    # LogisticRegression takes no continuous target, so the smoothed targets are supplied as sample
    # weights on duplicated rows - the standard encoding, exact rather than approximate.
    x = np.concatenate([s, s]).reshape(-1, 1)
    yy = np.concatenate([np.ones_like(s), np.zeros_like(s)])
    w = np.concatenate([t, 1.0 - t])
    lr = LogisticRegression(C=1e10, max_iter=max_iter, solver="lbfgs")
    lr.fit(x, yy, sample_weight=w)
    # sklearn fits p = sigmoid(w*s + c); Platt's form is p = 1/(1+exp(A*s+B)), so A = -w, B = -c.
    return Platt(a=float(-lr.coef_[0][0]), b=float(-lr.intercept_[0]),
                 n_calibration=len(s), positives_calibration=n_pos)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs: np.ndarray, labels: np.ndarray, *, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), eps, 1.0 - eps)
    y = np.asarray(labels).ravel().astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def corp_decomposition(probs: np.ndarray, labels: np.ndarray) -> dict:
    """CORP decomposition of the Brier score into miscalibration, discrimination and uncertainty.

    Recalibrates with the pool-adjacent-violators algorithm (isotonic regression of the label on the
    forecast), then:

        MCB = S - S_recalibrated          how much the Brier score would improve if perfectly calibrated
        DSC = UNC - S_recalibrated        how much better than the base rate the ranking is
        UNC = base_rate * (1 - base_rate) the Brier score of always predicting the base rate

    with the identity ``S = MCB - DSC + UNC`` holding exactly. That identity is asserted in the tests,
    which is what makes this a decomposition rather than three separately-computed numbers.

    Reportable where a reliability DIAGRAM is not, because PAV selects no bin count. That is the whole
    reason this project can say something about calibration at n=156 without drawing a curve nobody
    should trust.
    """
    from sklearn.isotonic import IsotonicRegression

    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(float)
    if p.shape != y.shape:
        raise ValueError(f"probs {p.shape} and labels {y.shape} must align")

    s = float(np.mean((p - y) ** 2))
    base = float(np.mean(y))
    unc = base * (1.0 - base)
    recal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit_transform(p, y)
    s_rc = float(np.mean((recal - y) ** 2))
    return {
        "brier": s,
        "mcb": s - s_rc,
        "dsc": unc - s_rc,
        "unc": unc,
        "brier_recalibrated": s_rc,
        "base_rate": base,
        "n": int(len(p)),
        "positives": int(y.sum()),
        "method": "CORP (pool-adjacent-violators); no bin count is chosen",
        "source": "Dimitriadis, Gneiting and Jordan, arXiv:2008.03033, PNAS 118 e2016191118",
    }


def wilson_interval(successes: int, n: int, *, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a proportion. Defensible at the sample sizes here.

    Riley's >=200-events floor rules out a flexible calibration CURVE. It does not rule out an
    interval on a proportion, and this project prints accuracies to seven decimals against a measured
    seed standard deviation of about 0.02 - five of those digits are noise, and an interval is the
    honest replacement.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)
