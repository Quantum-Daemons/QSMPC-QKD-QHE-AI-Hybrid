"""The metrics each industry is actually judged on — and the misleading ones, printed beside them.

Four of the five verticals in this project have a documented, published failure mode in which the
obvious aggregate metric looks excellent while the thing that matters is broken. Those are not
hypotheticals; each has a citation:

  finance    IBM AMLworld HI-Small is 1 laundering transaction in 981. The NeurIPS 2023 D&B paper
             reports minority-class F1 precisely because "traditional metrics like accuracy do not
             provide a reliable measure". Saito & Rehmsmeier (PLOS ONE, 2015) show the PR baseline
             is P/(P+N) and moves with imbalance while ROC does not — at 1-in-981 the PR baseline is
             about 0.001, so an AUPRC of 0.30 is a 300x lift and an ROC-AUC of 0.95 says nothing.

  telecom    Engelen, Rimmer & Joosen (WTMC 2021 @ IEEE S&P Workshops) corrected CIC-IDS2017 and
             found aggregate random-forest accuracy sits at 0.99 both before and after, while Bot
             F1 moves 0.60 -> 0.99 and Web XSS 0.11 -> 0.79. The aggregate is blind to the fix.

  satellite  Kim et al. (AAAI-22, arXiv:2109.05257) show point-adjustment "has a great possibility
             of overestimating the detection performance", and that under it "even a random anomaly
             score can easily turn into a state-of-the-art TAD method". A second clause read "TSB-AD
             found a uniform random score ranks 26th of 32 detectors - i.e. it beats six real
             detectors" and was removed in Round 33: the rank could not be verified (TSB-AD describes
             40 algorithms and recommends VUS-PR), and "beats six real detectors" was an inference the
             source never made. Kim et al. was already cited on the line above for a different quote,
             so dropping the unverified clause loses nothing.
             Telemanom's own headline 0.71 is F-beta at beta=0.5 under an event-overlap rule, not F1.

  genomics   Breast-cancer PRS AUROC is 0.60-0.69 in European ancestry and drops to 0.57-0.65 in
             African ancestry. Reporting a single pooled AUROC hides the portability gap, which is
             the clinically important finding.

So each function here is named for exactly what it computes, and :func:`with_companion` returns the
honest metric together with the flattering one so the UI can show both in the same row.
"""

from __future__ import annotations

import numpy as np

#: Telemanom and ESA-ADB both report F-beta at beta=0.5 (precision-weighted), never F1.
ESA_BETA = 0.5


def minority_class_f1(y_true, y_pred, *, positive: int = 1) -> float:
    """F1 of the positive (rare) class alone. The AMLworld / Elliptic++ reporting convention."""
    from sklearn.metrics import f1_score

    return float(f1_score(np.asarray(y_true).ravel(), _labels(y_pred), pos_label=positive,
                          average="binary", zero_division=0))


def auprc(y_true, y_score) -> float:
    """Average precision. Compare against the prevalence baseline, not against 0.5."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.asarray(y_true).ravel(), _positive_score(y_score)))


def auprc_baseline(y_true) -> float:
    """P/(P+N): the AUPRC a random ranker achieves. Meaningless to omit at extreme imbalance."""
    y = np.asarray(y_true).ravel()
    return float(np.mean(y == 1)) if y.size else 0.0


def auroc(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")     # undefined, and must not be silently reported as 0.5
    return float(roc_auc_score(y, _positive_score(y_score)))


def accuracy(y_true, y_pred) -> float:
    """Present only as a companion, never as a headline for an imbalanced task."""
    return float(np.mean(np.asarray(y_true).ravel() == _labels(y_pred)))


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(np.asarray(y_true).ravel(), _labels(y_pred), average="macro", zero_division=0))


def per_class_f1(y_true, y_pred, class_names: list[str] | None = None) -> dict:
    """Per-class F1. The only defensible headline for multi-class intrusion detection."""
    from sklearn.metrics import f1_score

    y = np.asarray(y_true).ravel()
    p = _labels(y_pred)
    classes = sorted(set(y.tolist()) | set(p.tolist()))
    scores = f1_score(y, p, labels=classes, average=None, zero_division=0)
    names = class_names or [str(c) for c in classes]
    return {names[i] if i < len(names) else str(classes[i]): float(s) for i, s in enumerate(scores)}


def net_benefit(y_true, y_score, threshold: float) -> float:
    """Vickers-Elkin net benefit at one decision threshold.

        NB(p) = TP/N - (FP/N) * p/(1-p)

    The subtracted term is the exchange rate the threshold itself declares: a clinician willing to
    act at probability p is, by that choice, saying one false positive costs p/(1-p) of a true
    positive. So net benefit is on the scale of "true positives per patient, net of the false
    positives they cost", and treat-all / treat-none are directly comparable on the same axis.

    This is why it is preferred over accuracy, F1 and Youden's J for a screening operating point.
    Van Calster et al. (Lancet Digital Health 2025;7(12):100916) assess 32 measures and find all
    classification measures IMPROPER for clinically relevant thresholds other than 0.5 or the
    prevalence, with F1 the only one of the 32 possessing neither required characteristic. Youden's J
    is not improper but is the symmetric special case - it implicitly prices a missed cancer and a
    false alarm the same, which no screening programme does (arXiv:2605.07829).
    """
    y = np.asarray(y_true).ravel().astype(int)
    s_ = _positive_score(y_score)
    n = y.size
    if n == 0:
        return float("nan")
    pred = s_ >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    q = float(min(max(threshold, 1e-9), 1 - 1e-9))
    return float(tp / n - (fp / n) * (q / (1.0 - q)))


def area_under_net_benefit(y_true, y_score, *, lo: float = 0.05, hi: float = 0.5,
                           steps: int = 46) -> float:
    """AUNBC - net benefit integrated over a RANGE of thresholds, not evaluated at one.

        AUNBC = (1/N) * sum_i (p_{i+1} - p_i) * (TP_i - FP_i * p_i/(1-p_i))

    Chi and Birbil (arXiv:2604.04241) optimise exactly this, and the reason to integrate is that a
    single threshold is a guess about a clinical preference nobody stated. A range says "somewhere
    between these two exchange rates", which is the honest form of the same claim.

    ``lo``/``hi`` bound the plausible threshold range and MUST be chosen from the decision context,
    not from the data. The default 0.05-0.5 spans "act on a 1-in-20 suspicion" to "act at even
    odds", which brackets breast-screening recall practice; state it wherever a number is reported.
    """
    grid = np.linspace(lo, hi, steps)
    return float(np.sum([(grid[i + 1] - grid[i]) * net_benefit(y_true, y_score, grid[i])
                         for i in range(len(grid) - 1)]))


def best_threshold_by_net_benefit(y_true, y_score, *, lo: float = 0.05, hi: float = 0.5,
                                  steps: int = 46) -> tuple:
    """The threshold maximising net benefit, and its margin over treat-all and treat-none.

    Returns ``(threshold, nb, nb_treat_all, nb_treat_none)`` with **treat-all evaluated at the SAME
    threshold**, which is the whole point of a decision curve: all three lines are functions of p and
    only matched-p comparisons mean anything. Selection maximises the MARGIN over treat-all, not the
    raw net benefit.

    Getting this wrong is easy and quiet. A first version here scored the model on the grid and
    treat-all once at the grid midpoint, then reported "best +0.2308 at q=0.050 (treat-all -0.0150)"
    - a decisive-looking win. At q=0.050 the model flagged every validation case, so it WAS treat-all,
    and treat-all at that same threshold also scores +0.2308. The margin was zero and the comparison
    manufactured it by moving the goalposts to a different p.

    Treat-none is 0 at every threshold by construction (it never acts, so no TP and no FP), which is
    why any negative net benefit means the model is worse than not screening at all.
    """
    y = np.asarray(y_true).ravel().astype(int)
    grid = np.linspace(lo, hi, steps)
    all_pos = np.ones_like(y, dtype=float)
    nbs = np.array([net_benefit(y, y_score, q) for q in grid])
    ref = np.array([net_benefit(y, all_pos, q) for q in grid])
    margin = nbs - np.maximum(ref, 0.0)          # 0.0 is treat-none, flat across p
    k = int(np.argmax(margin))
    return float(grid[k]), float(nbs[k]), float(ref[k]), 0.0


def event_wise_counts(y_true, y_pred) -> dict:
    """The tp/fp/fn behind :func:`event_wise_fbeta`, and how many EVENTS it was computed over.

    The event count is the point of this function. Event-wise F-beta divides by a number of labelled
    events, not samples, and there are far fewer of those than a sample count suggests: this
    project's ESA-ADB test window is 441,600 points and holds exactly **3** labelled events. The
    headline recorded until the Round 71 retrain, 0.4545, was precisely ``tp=1, fp=1, fn=2`` on those
    three, and the retrain's 0 is ``tp=0, fp=2, fn=3``. One event either way moves it by more than 0.2.

    A score over three events renders as the same kind of decimal as a benchmark computed over
    hundreds, so the count has to travel with the score or the comparison misleads silently. Note
    also that ``n_anomaly_intervals`` recorded beside it counts the WHOLE timeline (320), not the
    handful inside the scored window - reading that as the evaluation's sample size is the specific
    mistake this function exists to prevent.
    """
    true_events = _runs(np.asarray(y_true).ravel().astype(bool))
    pred_events = _runs(_labels(y_pred).astype(bool))
    detected = sum(1 for (s, e) in true_events if any(ps < e and pe > s for ps, pe in pred_events))
    matched = sum(1 for (ps, pe) in pred_events if any(ps < e and pe > s for s, e in true_events))
    return {
        "n_true_events": len(true_events),
        "n_pred_runs": len(pred_events),
        "tp": detected,
        "fn": len(true_events) - detected,
        "fp": len(pred_events) - matched,
    }


def corrected_event_wise_fbeta(y_true, y_pred, *, beta: float = ESA_BETA) -> float:
    """ESA-ADB's *corrected* event-wise F-beta - the quantity its published table actually reports.

    ESA-ADB multiplies event-wise precision by the TIME-LEVEL true-negative rate:

        Prec_corr = TP_e / (TP_e + FP_e) * TNR_t,     TNR_t = TN_t / (TN_t + FP_t)

    and that correction is the whole reason most of its baselines read < 0.001 while an uncorrected
    score on the same predictions looks respectable. Reporting an UNCORRECTED event-wise F0.5 beside
    ESA-ADB's 0.786 compares two different quantities, and the uncorrected one is the flattering
    direction, so the corrected value is computed here and recorded next to it.

    Source: arXiv:2406.17826 (ESA-ADB), Supplementary Table 13 caption, which names it "the
    corrected event-wise F0.5-score".
    """
    c = event_wise_counts(y_true, y_pred)
    if c["n_true_events"] == 0:
        return float("nan")

    y = np.asarray(y_true).ravel().astype(bool)
    p = _labels(y_pred).astype(bool)
    tn_t = int(np.sum(~y & ~p))
    fp_t = int(np.sum(~y & p))
    tnr_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) else 0.0

    tp, fn, fp = c["tp"], c["fn"], c["fp"]
    precision = (tp / (tp + fp) if (tp + fp) else 0.0) * tnr_t
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


def event_wise_fbeta(y_true, y_pred, *, beta: float = ESA_BETA) -> float:
    """Event-wise F-beta under the overlap rule used by Telemanom and ESA-ADB.

    A labelled anomaly *sequence* counts as detected if any predicted anomaly point falls inside it
    (one true positive per labelled event, not per point). Predicted anomaly runs that overlap no
    labelled event count as one false positive each.

    This is the metric those papers actually report. It is neither point-wise F1 nor the
    point-adjusted F1 that TSB-AD showed a random score can beat, and conflating the three is how
    "Telemanom F1 = 0.71" ends up in a table.

    Report :func:`event_wise_counts` beside it. The denominator is events, and the sample count
    does not tell you how many there are.
    """
    c = event_wise_counts(y_true, y_pred)
    if c["n_true_events"] == 0:
        return float("nan")

    tp, fn, fp = c["tp"], c["fn"], c["fp"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return float((1 + b2) * precision * recall / (b2 * precision + recall))


def point_adjusted_f1(y_true, y_pred) -> float:
    """Point-adjusted F1 — computed ONLY so the demo can show how inflated it is.

    Under point adjustment, detecting a single point inside an anomaly segment marks the whole
    segment correct. Never report this as the headline; report it next to
    :func:`event_wise_fbeta` to make the inflation visible.
    """
    from sklearn.metrics import f1_score

    y = np.asarray(y_true).ravel().astype(bool)
    p = _labels(y_pred).astype(bool).copy()
    for s, e in _runs(y):
        if p[s:e].any():
            p[s:e] = True
    return float(f1_score(y.astype(int), p.astype(int), zero_division=0))


def stratified_auroc(y_true, y_score, groups) -> dict:
    """AUROC overall and per group. For PRS this is per genetic-ancestry group, and the gap is the result."""
    groups = np.asarray(groups).ravel()
    out = {"overall": auroc(y_true, y_score)}
    y = np.asarray(y_true).ravel()
    s = _positive_score(y_score)
    for g in sorted(set(groups.tolist())):
        m = groups == g
        out[str(g)] = auroc(y[m], s[m]) if m.sum() > 1 else float("nan")
    return out


def with_companion(y_true, y_pred, y_score, *, headline: str) -> dict:
    """The honest metric plus the flattering aggregate, so a UI row can show both."""
    if headline == "minority_class_f1":
        return {
            "minority_class_f1": minority_class_f1(y_true, y_pred),
            "auprc": auprc(y_true, y_score),
            "auprc_baseline": auprc_baseline(y_true),
            "accuracy": accuracy(y_true, y_pred),        # companion: expected to look great and mean little
            "auroc": auroc(y_true, y_score),
        }
    if headline == "event_wise_f0_5":
        return {
            "event_wise_f0_5": event_wise_fbeta(y_true, y_pred),
            "point_adjusted_f1": point_adjusted_f1(y_true, y_pred),   # companion: inflated by construction
        }
    if headline == "macro_f1":
        return {
            "macro_f1": macro_f1(y_true, y_pred),
            "per_class_f1": per_class_f1(y_true, y_pred),
            "aggregate_accuracy": accuracy(y_true, y_pred),           # companion: 0.99 either way
        }
    if headline == "auc_pr":
        # OPS-SAT-AD's own headline. The positive class is about a fifth of the data, so accuracy
        # reads ~0.80 for a detector that never fires; AUC PR cannot be satisfied by abstaining.
        # auprc_baseline is the positive rate, i.e. what a coin-flip detector scores - printed so
        # the headline is read against the floor rather than against zero.
        return {
            "auc_pr": auprc(y_true, y_score),
            "auc_pr_baseline": auprc_baseline(y_true),
            "f1": minority_class_f1(y_true, y_pred),
            "auroc": auroc(y_true, y_score),
            "accuracy": accuracy(y_true, y_pred),        # companion: flattered by the class balance
        }
    if headline in ("auc", "auroc", "ood_accuracy"):
        return {
            "auc": auroc(y_true, y_score),
            "accuracy": accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
        }
    raise ValueError(f"unknown headline metric: {headline!r}")


# --------------------------------------------------------------------------- #
def _labels(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim > 1 and a.shape[1] > 1:
        return a.argmax(1).astype(int)
    a = a.ravel()
    return (a > 0.5).astype(int) if a.dtype.kind == "f" else a.astype(int)


def _positive_score(a) -> np.ndarray:
    """Score for the positive class: column 1 of a probability matrix, else the raw vector."""
    a = np.asarray(a, dtype=float)
    return a[:, 1] if a.ndim > 1 and a.shape[1] == 2 else a.ravel()


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as half-open ``[start, end)`` intervals."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))
