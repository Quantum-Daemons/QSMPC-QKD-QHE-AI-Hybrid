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

  satellite  Kim et al. (AAAI-22) show point-adjustment "has a great possibility of overestimating
             the detection performance", and TSB-AD (NeurIPS 2024 D&B) found a uniform random score
             ranks 26th of 32 detectors under point-adjusted F1 — i.e. it beats six real detectors.
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


def event_wise_fbeta(y_true, y_pred, *, beta: float = ESA_BETA) -> float:
    """Event-wise F-beta under the overlap rule used by Telemanom and ESA-ADB.

    A labelled anomaly *sequence* counts as detected if any predicted anomaly point falls inside it
    (one true positive per labelled event, not per point). Predicted anomaly runs that overlap no
    labelled event count as one false positive each.

    This is the metric those papers actually report. It is neither point-wise F1 nor the
    point-adjusted F1 that TSB-AD showed a random score can beat, and conflating the three is how
    "Telemanom F1 = 0.71" ends up in a table.
    """
    true_events = _runs(np.asarray(y_true).ravel().astype(bool))
    pred_events = _runs(_labels(y_pred).astype(bool))
    if not true_events:
        return float("nan")

    detected = sum(1 for (s, e) in true_events if any(ps < e and pe > s for ps, pe in pred_events))
    matched_pred = sum(1 for (ps, pe) in pred_events if any(ps < e and pe > s for s, e in true_events))

    tp = detected
    fn = len(true_events) - detected
    fp = len(pred_events) - matched_pred
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
