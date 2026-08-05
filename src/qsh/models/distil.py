"""Teacher (plaintext, real) -> student (encrypted, small), and the measured cost of the gap.

Homomorphic evaluation is cost-bounded and will stay that way: the largest model demonstrated
end-to-end under CKKS is 8B parameters at roughly a 10^4x runtime gap (IACR ePrint 2026/935), and
EncFormer's own authors put their secure BERT pipeline at "roughly 3x10^4 slower" than plaintext.
So a project that wants both a real industry model *and* real encryption has exactly two honest
options: shrink the model until it fits the cipher, or state plainly that the crypto demo runs on
something smaller than the headline model.

This module takes the second option and makes it measurable. For each use case:

    teacher   the registry's T2 model - the real 2026 literature architecture, plaintext
    student   the registry's T1 model - linear or tiny-CNN, evaluated under real CKKS/MPC

and reports :class:`FidelityReport`, whose headline field ``agreement`` is how often the student
reproduces the teacher's decision on held-out data. ``metric_delta`` is what that costs in the
industry's own metric.

This generalises what ``qsh.usecases.finance_aml._fraud_model`` already did for one vertical
(XGBoost teacher -> logistic student) to all of them, and turns an incidental number into a
first-class run metric. It is also the direct, measured answer to the source report's PoC success
criterion of "private-inference accuracy degradation <= 1pp" — which is a claim until someone
measures it per model, on real data, and publishes the number even when it is bad.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class FidelityReport:
    """What the encrypted student costs, relative to the plaintext teacher it replaces."""

    teacher_key: str
    student_key: str
    n_eval: int
    agreement: float               # fraction of held-out samples where student == teacher decision
    teacher_metric: float          # the industry metric, teacher, on ground truth
    student_metric: float          # the industry metric, student, on ground truth
    metric_name: str
    disagreement_on_positives: float   # agreement is easy when the positive class is 1-in-981
    student_params: int
    teacher_params: int | None
    #: Set when teacher and student BOTH score 0 on the fidelity slice. metric_delta is then 0 by
    #: cancellation, not because encryption was free -- and it is not comparable to the teacher's
    #: headline metric, which is computed over the whole test window rather than this later slice.
    #: Without this the satellite card showed metric_delta 0.0 beside a teacher headline of 0.714,
    #: which reads as "the encrypted student matched a 0.714 teacher exactly".
    both_scored_zero: bool = False

    @property
    def metric_delta(self) -> float:
        """Teacher metric minus student metric. Positive means encryption cost accuracy.

        Read together with :attr:`both_scored_zero`: a delta of 0 there means neither model scored
        on the fidelity slice, which is not the same claim as "the student matched the teacher".
        """
        return self.teacher_metric - self.student_metric

    @property
    def metric_delta_pp(self) -> float:
        return 100.0 * self.metric_delta

    def as_metrics(self) -> dict:
        """Flat dict for the run's metric stream and for ``models/metrics.json``."""
        d = asdict(self)
        d["metric_delta"] = round(self.metric_delta, 6)
        d["metric_delta_pp"] = round(self.metric_delta_pp, 4)
        return d


def distil_linear(teacher_scores: np.ndarray, X: np.ndarray, *,
                  class_weight: str | None = "balanced", max_iter: int = 1000, seed: int = 0):
    """Fit a logistic student against the *teacher's own decisions*, not the ground-truth labels.

    Training on the teacher's output is what makes this distillation rather than just a second
    model: the student is trying to be the teacher, so ``agreement`` measures the substitution and
    not two independent fits that happen to correlate.
    """
    from sklearn.linear_model import LogisticRegression

    labels = np.asarray(teacher_scores)
    if labels.ndim > 1:
        labels = labels.argmax(1)
    student = LogisticRegression(max_iter=max_iter, class_weight=class_weight, random_state=seed)
    student.fit(np.asarray(X, dtype=float), labels.astype(int))
    return student


def measure_fidelity(*, teacher_key: str, student_key: str, teacher_pred: np.ndarray,
                     student_pred: np.ndarray, y_true: np.ndarray, metric_fn, metric_name: str,
                     student_params: int, teacher_params: int | None = None) -> FidelityReport:
    """Compare a student against its teacher on held-out data.

    ``disagreement_on_positives`` is reported separately and deliberately. With a 1-in-981 positive
    rate — the real ratio in IBM AMLworld HI-Small — a student that predicts the negative class for
    everything scores 99.9% agreement while being useless. Overall agreement alone would hide that.
    """
    teacher_pred = _labels(teacher_pred)
    student_pred = _labels(student_pred)
    y_true = np.asarray(y_true).astype(int).ravel()

    agreement = float(np.mean(teacher_pred == student_pred))
    pos = y_true == 1
    disagree_pos = float(np.mean(teacher_pred[pos] != student_pred[pos])) if pos.any() else 0.0

    return FidelityReport(
        teacher_key=teacher_key,
        student_key=student_key,
        n_eval=int(y_true.size),
        agreement=round(agreement, 6),
        teacher_metric=round(float(metric_fn(y_true, teacher_pred)), 6),
        student_metric=round(float(metric_fn(y_true, student_pred)), 6),
        metric_name=metric_name,
        disagreement_on_positives=round(disagree_pos, 6),
        student_params=int(student_params),
        teacher_params=int(teacher_params) if teacher_params is not None else None,
        both_scored_zero=bool(
            float(metric_fn(y_true, teacher_pred)) == 0.0
            and float(metric_fn(y_true, student_pred)) == 0.0
        ),
    )


def _labels(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim > 1 and a.shape[1] > 1:
        return a.argmax(1).astype(int)
    return (a.ravel() > 0.5).astype(int) if a.dtype.kind == "f" else a.ravel().astype(int)
