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

ROUND 46 CORRECTED THIS, AND THE CORRECTION IS ABOUT THE DENOMINATOR.

Those thresholds are stated by both papers in CALIBRATION CASES - the size of the set the calibrator
is fitted on. The counts above are POSITIVE EVENTS. They are different quantities, and this file
compared one against the other: `models/metrics.json` records `n_calibration = 710969` for finance,
which is 355x the UAI figure and 711x the ICML one, not "inside both regimes". The finance figure was
also stale at 164 - `scripts/train/finance_aml.py` already records that as the 2D overlay
subsample's count and puts the test split at 2,321 events.

Read at primary source, ICML 2005 says the opposite of the conclusion drawn from it, verbatim:

    "When there are 1000 or more points in the calibration set, Isotonic Regression always yields
     performance as good as, or better than, Platt Scaling."

ROUND 49 MEASURED IT. THE ANSWER IS NOT A SWAP, AND IT IS NOT "PLATT WAS FINE" EITHER.

The comparison ran on the shipped split - fitted on the 710,969-case validation window, scored on
the held-out 1,523,504-case test window, teacher reproduced to every recorded digit. Full workings:
references/finance-calibration-isotonic-vs-platt.md.

  1. THE CALIBRATION SET CARRIES 801 EVENTS, NOT 2,321. 2,321 is the TEST split's count. The
     argument above named the wrong partition, one line below `positives_calibration = 801`.

  2. `fit_platt` WAS BEING HANDED A PROBABILITY, NOT A LOG-ODDS. FIXED - see the section below.
     `p = 1/(1+exp(a*s+b))` is linear in `s`; 90% of this teacher's scores are below 1e-4, so `a*s`
     vanished and the fitted map returned the single constant 0.00090148 for nine of ten deciles.
     Its whole output support was [0.00090148, 0.695715] - it could not express a risk below 1 in
     1,109 for any input. That was not Platt losing to isotonic; it was Platt applied to the wrong
     argument, and it is why the arm that shipped through Round 49 degraded log-loss (+8.7%) and ECE
     (2.83x the uncalibrated score's) while improving Brier. The unit tests below could not catch
     it: their fixture passed a logit, the trainer passed a probability, and each was internally
     consistent.

  3. FED `logit(s)`, THE SAME `fit_platt` MATCHES ISOTONIC. Brier 0.00142199 against 0.00142218,
     log-loss 0.00818964 against 0.00820025, CORP MCB 1.18e-05 against 9.34e-06. ICML's sentence
     holds in the exact form it is written - AS GOOD AS - at 710,969 cases.

  4. ISOTONIC MOVES THE RANKING, AND THE CLAIM BELOW THAT IT CANNOT IS TRUE ONLY OF PLATT. Isotonic
     is WEAKLY monotone: it collapsed 1,388,669 distinct scores to 138, and cost AUROC 0.00031674
     and AUPRC 0.01121590 - 9.09% of the metric this project reports beside its headline. Both Platt
     arms moved neither, to ten decimals. Isotonic also emitted exactly 0 on 464,176 test rows, one
     of them a true positive, and a step function is not two floats the browser can apply.

  5. CASES OR EVENTS: BOTH READINGS, BECAUSE THE SWEEPS DISAGREE AND THAT IS THE RESULT. Base rate
     is events over cases, so no sweep can pin two and vary the third. Against the SHIPPED arm the
     case count binds: pinning cases at 710,969 and cutting events 267-fold (801 -> 3) never flips
     the sign, while pinning events at 100 and cutting cases flips it between 10,000 and 30,000.
     Against the CORRECT arm the event count binds, with a crossover at 100-200 events - two orders
     of magnitude below the papers' CASE thresholds, and already passed by 801.

SO THE METHOD STAYS PLATT AND THE ARGUMENT IT IS RETAINED ON IS DIFFERENT. Not "the papers are about
balanced problems", which the sweep does not support as written, but: at this size the two are within
noise on both proper scoring rules, and isotonic breaks the ranking and the two-float publication
contract that Platt does not. The change worth making is to the ARGUMENT of `fit_platt`, not to the
choice of calibrator.

THE ARGUMENT FIX IS NOW APPLIED, AND IT LIVES HERE RATHER THAN AT THE CALL SITE.

`fit_platt(..., on_logit=True)` regresses on `logit(s)`, and the fitted `Platt` CARRIES that flag so
`apply` performs the same transform and `as_dict` publishes it. Putting it in the caller instead was
the other option and is the worse one: the published JSON would still read
`p = 1/(1+exp(a*score+b))` while the trainer applied it to `logit(score)`, so a browser following the
contract would reproduce a different function from the one that was measured. That is the same defect
class as the bug being fixed - a map and its argument agreeing nowhere - so the flag travels with the
two floats. `logit` clips at LOGIT_CLIP_EPS = 1e-12 first: nothing on this split is clipped (the raw
scores span 4.4e-11 to 0.99997), but a retrain that saturates a leaf must not produce an infinity.

What that bought, on the same split and seed the table above was measured on: Brier 0.00145433 ->
0.00142199, log-loss 0.01049835 -> 0.00818964 (the shipped arm was WORSE than not calibrating at all
on this one, at 0.00965987 uncalibrated), ECE(20) 0.00183223 -> 0.00078864, CORP MCB 4.4116e-05 ->
1.1782e-05. AUROC and AUPRC are unchanged to ten decimals, as they must be for any strictly monotone
map. The output support is [1.680e-07, 0.988628] rather than [0.00090148, 0.695715].

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

Calibration is a monotone post-transform of a score. Where that map is STRICTLY monotone - Platt is,
being a sigmoid - it cannot fix ranking and moves neither AUC nor AUC-PR; measured on finance, both
Platt arms leave AUROC 0.95255826 and AUPRC 0.12334102 unchanged to ten decimals. Isotonic is only
WEAKLY monotone and the same measurement shows it moving both, by -0.00031674 and -0.01121590. So
"calibration cannot move the ranking" is a statement about strict monotonicity, and stating it
without that qualifier is what let a 9.09% AUPRC cost look impossible.

And calibration must be applied AFTER the encrypted-vs-plaintext comparison, never before: the CKKS
approximation error lives in the raw logits, so measuring precision on calibrated probabilities
would measure a property of the sigmoid instead of a property of the ciphertext.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Clip applied before the log-odds transform. The finance teacher's raw scores are strictly inside
#: (0, 1) on the shipped split - min 4.4e-11, max 0.99997 - so nothing is actually clipped there
#: today. The guard exists so a retrain that saturates a leaf to exactly 0 or 1 cannot hand the
#: calibrator an infinity, which would be a silent NaN in `a` rather than a visible failure.
LOGIT_CLIP_EPS = 1e-12

#: The two published formula strings, one per input convention. A consumer gets one of these
#: verbatim in the weights JSON alongside `a` and `b`, and must be able to reproduce `Platt.apply`
#: from the JSON alone - that contract is the subject of
#: `tests/test_calibration.py::test_the_shipped_parameters_reproduce_the_transform`.
_FORMULA_ON_SCORE = "p = 1 / (1 + exp(a * score + b))"
_FORMULA_ON_LOGIT = (
    "p = 1 / (1 + exp(a * logit(clip(score, logit_clip_eps, 1 - logit_clip_eps)) + b)), "
    "logit(q) = log(q / (1 - q))")


def logit(p: np.ndarray, *, eps: float = LOGIT_CLIP_EPS) -> np.ndarray:
    """Log-odds of a probability, clipped so a saturated 0 or 1 cannot become an infinity."""
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


@dataclass(frozen=True)
class Platt:
    """A fitted two-parameter sigmoid, ``p = 1 / (1 + exp(A * s + B))``.

    Two floats, so the browser can apply the identical transform from the weights JSON without
    shipping a calibrator object or re-deriving anything.

    ``on_logit`` records WHICH ``s`` those two floats were fitted against, and it has to be part of
    the object rather than a convention at the call site. The map is linear in its argument, so
    feeding it a probability when it was fitted on a log-odds is not an approximation - it is a
    different function. Round 49 measured what that costs on finance: fitted on the probability
    directly, ``a * s`` vanishes across the nine deciles that sit below 1e-4 and the map returns one
    constant for all of them. `apply` therefore performs the transform itself, and `as_dict`
    publishes it, so no consumer has to remember.
    """

    a: float
    b: float
    n_calibration: int
    positives_calibration: int
    on_logit: bool = False
    logit_clip_eps: float = LOGIT_CLIP_EPS

    def _argument(self, scores: np.ndarray) -> np.ndarray:
        """The quantity the sigmoid is evaluated at - identical to what the fit regressed on."""
        if not self.on_logit:
            return np.asarray(scores, dtype=np.float64)
        return logit(scores, eps=self.logit_clip_eps)

    def apply(self, scores: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(self.a * self._argument(scores) + self.b))

    def as_dict(self) -> dict:
        return {
            "method": "platt",
            "a": float(self.a),
            "b": float(self.b),
            "on_logit": bool(self.on_logit),
            "logit_clip_eps": float(self.logit_clip_eps),
            "formula": _FORMULA_ON_LOGIT if self.on_logit else _FORMULA_ON_SCORE,
            "n_calibration": int(self.n_calibration),
            "positives_calibration": int(self.positives_calibration),
            "why_not_isotonic": (
                "MEASURED Round 49 on this split, not argued: isotonic Brier 0.00142218 / log-loss "
                "0.00820025 against 0.00142199 / 0.00818964 for the same fit_platt fed logit(s) - "
                "within noise on both proper scoring rules, so ICML 2005's 'as good as, or better "
                "than' holds here in the 'as good as' form. Platt is kept on the tie-break: "
                "isotonic is only WEAKLY monotone, collapsed 1,388,669 distinct scores to 138, and "
                "cost AUROC 0.00031674 and AUPRC 0.01121590 (-9.09%), while both Platt arms moved "
                "neither to ten decimals; it also emitted exactly 0 on 464,176 test rows, one a "
                "true positive, and 138 breakpoints are not two floats a browser can apply. ONE "
                "CORRECTION TO THE OLD REASONING SURVIVES HERE: the calibration set carries 801 "
                "events, not 2,321 (that is the test split). The second - that a/b were fitted on "
                "a PROBABILITY rather than a log-odds, making the map return 0.00090148 for nine "
                "of ten deciles - is FIXED, not outstanding: on_logit is true above and the map "
                "published here is the logit(s) arm those numbers describe. Workings: "
                "references/finance-calibration-isotonic-vs-platt.md."),
        }


def fit_platt(scores: np.ndarray, labels: np.ndarray, *, on_logit: bool = False,
              max_iter: int = 200) -> Platt:
    """Fit Platt scaling by logistic regression of the label on the raw score.

    Uses the Platt/Lin-Weng-Lin target smoothing rather than the raw 0/1 labels: with few positives,
    regressing on hard labels drives the fitted sigmoid to saturate and produces probabilities of
    exactly 0 and 1, which a log-loss cannot score. The smoothing is the standard fix and is why
    Platt has "some overfitting control built in".

    ``on_logit=True`` regresses on ``logit(scores)`` instead, and is what a caller holding
    PROBABILITIES wants. The literature's ``s`` is an SVM margin or a logit - a quantity whose
    log-odds are roughly linear - and the fitted map is linear in whatever it is handed, so a
    probability from a well-separated booster leaves ``a * s`` numerically zero across most of the
    range. The returned `Platt` carries the flag, so `apply` and `as_dict` cannot disagree with the
    fit about which argument it was.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if on_logit:
        s = logit(s)
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
                 n_calibration=len(s), positives_calibration=n_pos,
                 on_logit=bool(on_logit), logit_clip_eps=LOGIT_CLIP_EPS)


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
