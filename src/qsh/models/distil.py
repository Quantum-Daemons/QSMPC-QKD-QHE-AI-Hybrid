"""Teacher (plaintext, real) -> student (encrypted, small), and the measured cost of the gap.

Homomorphic evaluation is cost-bounded AT THE OPERATING POINT THIS DEMO TARGETS - one browser tab,
SEAL compiled to single-threaded WASM - not in general, and not forever: Park et al.
(arXiv:2601.18511v2) run Llama-3-8B under CKKS on eight GPUs, and Cheddar (ASPLOS 2026) runs
ResNet-20 under CKKS in about 0.7-1.3 s per image on one GPU. EncFormer's own authors put their
secure BERT pipeline at "roughly 3x10^4 slower" than plaintext. So a project that wants both a real
industry model *and* encryption inside one tab has two honest options: shrink the model until it
fits the cipher at that operating point, or state plainly that the crypto demo runs on something
smaller than the headline model.

This module takes the second option and makes it measurable. For each use case:

    teacher   the registry's T2 model - the real 2026 literature architecture, plaintext
    student   the registry's T1 model - linear or tiny-CNN, designed to be evaluated under CKKS/MPC

and reports :class:`FidelityReport`, whose headline field ``agreement`` is how often the student
reproduces the teacher's decision on held-out data. ``metric_delta`` is what that costs in the
industry's own metric. Both predictions are computed IN PLAINTEXT here, so this measures the
distillation; what CKKS noise costs is measured separately, per run, in the browser
(``encrypted_agreement``), and only for the model the page actually encrypts.

This generalises what ``qsh.usecases.finance_aml._fraud_model`` already did for one vertical
(XGBoost teacher -> logistic student) to all of them, and turns an incidental number into a
first-class run metric. It is also the direct, measured answer to the source report's PoC success
criterion of "private-inference accuracy degradation <= 1pp" — which is a claim until someone
measures it per model, on real data, and publishes the number even when it is bad.
"""

from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class FidelityReport:
    """What the student (the encrypted tier's model) costs, relative to the plaintext teacher it
    replaces. Both sides are scored in plaintext; this is distillation fidelity, not CKKS fidelity."""

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
    #: Set when the TEACHER scores 0 on the fidelity slice, whatever the student scores. There is
    #: then no fidelity claim to make: the slice contains nothing the teacher got right, so the
    #: delta measures the student against a model that failed, not against the teacher's ability.
    #:
    #: The condition is deliberately the teacher alone, and the earlier version of this flag fired
    #: only when BOTH scored 0. That version was written for a real incident -- the satellite card
    #: showed metric_delta 0.0 beside a teacher headline of 0.714, reading as "the encrypted student
    #: matched a 0.714 teacher exactly". But requiring both to be zero left the worse shape live:
    #: teacher 0.0 against student 0.0032 gives a NEGATIVE delta, which renders as the 49-parameter
    #: encrypted student outperforming an 80,486-parameter LSTM. It is the same defect one step to
    #: the side, and it shipped to the public demo. A guard whose condition is narrower than the
    #: thing it guards against will eventually be walked around.
    #:
    #: Neither number is comparable to the teacher's headline metric either way, because the headline
    #: is computed over the whole test window and this is a later slice of it.
    teacher_scored_zero: bool = False
    #: How many things the metric was actually computed over, and what one of them is. ``n_eval``
    #: counts SAMPLES, which for an event-wise metric is the wrong denominator by orders of
    #: magnitude: the satellite fidelity slice is 307,601 samples and 2 labelled anomaly events.
    #: A score over 2 events looks identical to a score over 2,000 and means almost nothing, so the
    #: count is recorded beside it rather than left to be inferred from ``n_eval``.
    n_scoring_units: int | None = None
    scoring_unit: str = "samples"

    @property
    def both_scored_zero(self) -> bool:
        """The stricter, older condition. Kept because ``metrics.json`` records written by earlier
        rounds carry this key and the UI still reads it as a fallback."""
        return self.teacher_metric == 0.0 and self.student_metric == 0.0

    @property
    def metric_delta(self) -> float:
        """Teacher metric minus student metric. Positive means encryption cost accuracy.

        Read together with :attr:`teacher_scored_zero`. When that is set this delta is not a cost of
        encryption at all: at teacher 0 it is 0 by cancellation if the student also scored nothing,
        and negative if the student scored anything, neither of which is the claim "the student
        matched (or beat) the teacher".
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
        # Properties are not fields, so asdict() drops them. The UI reads both keys.
        d["both_scored_zero"] = self.both_scored_zero
        return d


def distil_linear(teacher_scores: np.ndarray, X: np.ndarray, *,
                  class_weight: str | None = "balanced", max_iter: int = 1000, seed: int = 0,
                  soft: bool = True, degree: int = 1):
    """Fit a logistic student against the *teacher's own output*, not the ground-truth labels.

    Training on the teacher's output is what makes this distillation rather than just a second
    model: the student is trying to be the teacher, so ``agreement`` measures the substitution and
    not two independent fits that happen to correlate.

    ``soft`` distils from the teacher's PROBABILITIES rather than its hard argmax. Hard labels throw
    away exactly the information distillation exists to transfer - how confident the teacher was -
    and at a 1-in-981 positive rate that is most of the signal. Implemented by presenting each row
    twice, once per class, weighted by the teacher's probability for that class, which is the
    standard reduction of a soft-target cross-entropy to a weighted hard-label fit.

    Measured on AMLworld HI-Small with tuned thresholds: hard 0.0503, soft 0.0587 minority-class F1.

    ``degree`` > 1 fits the same logistic model over a POLYNOMIAL EXPANSION of the features, returned
    as a Pipeline that also owns its scaler. This is the one axis the CKKS cost bound does not
    block, and the reason is worth stating: the expansion is a function of the plaintext input, so
    the client computes it BEFORE encrypting. The encrypted operation is still a single packed dot
    product, at the same multiplicative depth, over a longer vector - and the ring carries far
    more slots (``FHEConfig.poly_modulus_degree // 2``) than any expansion here reaches, so the
    expansion is free on the crypto side.

    It is not free statistically: d features become d(d+3)/2, so 19 -> 209 and 32 -> 560, and the
    sample-per-parameter ratio falls accordingly. Measured on OPS-SAT-AD, and the numbers an
    earlier revision of this docstring gave do NOT match the shipped record. It claimed:

    WITHDRAWN-CLAIM-BEGIN
    "minority-class F1 0.8485 -> 0.9391 and AUC-PR 0.9235 -> 0.9810, which beats both the teacher
    (0.9189) and the published baseline (0.979)"
    WITHDRAWN-CLAIM-END

    Against `models/metrics.json` for `satellite_comms.opssat_student`:

        auc_pr            0.968055     (claimed 0.9810)
        minority_class_f1 0.904348     (claimed 0.9391)
        teacher auc_pr    0.968107     student is 5.2e-05 BELOW it, not above
        teacher f1        0.918919     student F1 is BELOW it too
        published         0.979        0.968055 is BELOW it

    Three errors, and they compound. **The claim against the published baseline is false** - 0.968055
    is below 0.979. **The comparison with the teacher (0.9189) uses two different metrics**: 0.9189 is
    the teacher's F1 and was being read against the student's AUC-PR, which is the two-fields-one-word
    trap this project has a test for. And **the degree-1 "before" figures are reproducible from
    nothing**: no committed artefact records a pre-expansion score, so 0.8485 and 0.9235 cannot be
    checked at all.

    What the record supports: the quadratic expansion produces a student that TRACKS its teacher
    closely on the declared metric - auc_pr 0.968055 against 0.968107 - at 209 expanded features
    (210 parameters with the bias, the record's `student_params`), one
    encrypted dot product, unchanged multiplicative depth. That is the claim worth making, and it is
    about fidelity at low cost rather than about beating anyone. Check it per model rather than
    assuming.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    scores = np.asarray(teacher_scores, dtype=float)
    Xf = np.asarray(X, dtype=float)
    lr = LogisticRegression(max_iter=max_iter, class_weight=class_weight, random_state=seed)
    # At degree 1 this stays a bare estimator, so every existing caller and every persisted
    # `weights.json` keeps exactly the shape it had.
    student = lr if degree <= 1 else Pipeline([
        ("poly", PolynomialFeatures(degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("lr", lr),
    ])
    weight_kw = "lr__sample_weight" if degree > 1 else "sample_weight"

    if soft and scores.ndim > 1 and scores.shape[1] == 2:
        p1 = scores[:, 1]
        Xd = np.vstack([Xf, Xf])
        yd = np.concatenate([np.ones(len(p1), dtype=int), np.zeros(len(p1), dtype=int)])
        wd = np.concatenate([p1, 1.0 - p1])
        student.fit(Xd, yd, **{weight_kw: wd})
        return student

    labels = scores.argmax(1) if scores.ndim > 1 else scores
    student.fit(Xf, labels.astype(int))
    return student


def fold_threshold_into_intercept(student, threshold: float):
    """Bake a tuned decision threshold into the student's bias, in place.

    The encrypted path decides by argmax over the two logits, which is a 0.5 threshold and nothing
    else - there is no place downstream to apply a tuned cut. Folding it in is what lets the CKKS
    result use the same operating point the metrics report.

    For a binary logistic, thresholding p at q is thresholding the logit at log(q/(1-q)), so
    subtracting that constant from the positive class's intercept moves the argmax boundary to the
    tuned point. Measured effect on AMLworld: minority-class F1 0.0186 -> 0.0503 at the same
    weights, because a 0.5 cut on a 1-in-981 problem is nowhere near the optimum.
    """
    q = float(min(max(threshold, 1e-9), 1 - 1e-9))
    shift = float(np.log(q / (1.0 - q)))
    inter = np.asarray(student.intercept_, dtype=float)
    if inter.size == 1:                      # sklearn binary form: one logit, positive class
        student.intercept_ = inter - shift
    else:                                    # multinomial form: shift the positive class only
        inter = inter.copy()
        inter[1] -= shift
        student.intercept_ = inter
    return student


def fold_threshold_into_torch_head(model, threshold: float, *, positive_class: int = 1):
    """Same idea as :func:`fold_threshold_into_intercept`, for a 2-logit torch head, in place.

    Argmax over two logits IS a 0.5 threshold, and the encrypted path has nowhere downstream to
    apply a tuned cut - so a screening model that should trade precision for recall cannot express
    that unless the cut is baked into the weights. Predicting ``positive_class`` when
    ``sigmoid(z_pos - z_other) >= q`` means shifting that class's bias by ``-log(q/(1-q))``.

    Needed for the medical student because its positive class is MALIGNANT and it is the minority
    (2.71:1). At argmax it caught 1 of 42 malignant cases while scoring a respectable-looking
    accuracy, because on an imbalanced screening task accuracy is maximised by not screening.
    """
    import torch

    q = float(min(max(threshold, 1e-9), 1 - 1e-9))
    shift = float(np.log(q / (1.0 - q)))
    last = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.bias[positive_class] -= shift
    return model


def save_student(student, out_dir, *, n_features: int, teacher_key: str,
                 student_key: str, classes: list | None = None, scaler=None,
                 fixtures: np.ndarray | None = None) -> dict:
    """Persist the distilled student so it can be published and independently verified.

    The student is the model that actually runs under CKKS -- it is the whole point of the
    encrypted tier -- and until Round 19 it existed only as a fidelity number in metrics.json. It
    was fit in memory, measured, and discarded, which meant the one model a visitor most needs to
    check was the one model they could not obtain. Teachers were published from Round 18; this
    closes the asymmetry.

    Writes plain weights (a linear model is small enough to read by eye, which is a feature here)
    plus an ONNX graph via the same export path the teachers use, so the browser can load it
    exactly as it loads any other published model.
    """
    import json

    from . import export

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # A degree>1 student is a Pipeline that owns its own expansion and scaler. The coefficients then
    # live on the final estimator and describe the EXPANDED space, while the graph still takes raw
    # features - so the two have different widths and both facts have to be recorded.
    pipeline = getattr(student, "steps", None)
    estimator = student[-1] if pipeline else student
    poly = student.named_steps.get("poly") if pipeline else None
    fitted_scaler = student.named_steps.get("scale") if pipeline else scaler

    coef = np.asarray(estimator.coef_, dtype=float)
    intercept = np.asarray(estimator.intercept_, dtype=float)
    (out / "weights.json").write_text(json.dumps({
        "kind": "logistic-linear",
        "note": ("The CKKS-linear student: the model this use case's encrypted tier is designed "
                 "to evaluate under homomorphic encryption. Small because it has to close at this "
                 "demo's operating point (CKKS in one browser tab), not because encrypted "
                 "evaluation is cost-bounded in general; distilled against the teacher's own "
                 "decisions, so `agreement` measures substitution rather than two independent fits "
                 "that happen to correlate."),
        "teacher_key": teacher_key, "student_key": student_key,
        "n_features": int(n_features),
        # The standardisation the student was FIT on. Without it the published weights cannot be
        # evaluated correctly from raw features: the graph accepts them and returns a confident
        # wrong answer, which is the same failure the teachers avoid by shipping
        # `input.preprocessing` beside their samples. None means the student consumes raw features.
        "input": {
            # A pipeline student bakes expansion AND scaling into the graph, so the caller does
            # nothing and `preprocessing` is honestly empty. A bare student still needs the caller
            # to z-score, and the transform it was FIT on is published here for that.
            "preprocessing": ([] if (pipeline or fitted_scaler is None)
                              else ["subtract mean", "divide by scale"]),
            "mean": (None if (pipeline or fitted_scaler is None)
                     else [float(v) for v in fitted_scaler.mean_]),
            "scale": (None if (pipeline or fitted_scaler is None)
                      else [float(v) for v in fitted_scaler.scale_]),
            # Present only when the student is polynomial. `W` below is indexed in the EXPANDED
            # space, so anyone reproducing the encrypted dot product from raw features needs the
            # exact expansion and its term order - sklearn's, which is
            # [x_1..x_d, x_1^2, x_1x_2, ..., x_d^2] with include_bias=False.
            **({} if not poly else {"expansion": {
                "kind": "polynomial",
                "degree": int(poly.degree),
                "include_bias": bool(poly.include_bias),
                "interaction_only": bool(poly.interaction_only),
                "n_features_raw": int(n_features),
                "n_features_expanded": int(coef.shape[-1]),
                "term_order": "sklearn PolynomialFeatures, [x_i], then [x_i*x_j] for i<=j",
                "note": ("Computed on the PLAINTEXT input before encryption, so the encrypted step "
                         "is still one packed dot product at unchanged multiplicative depth. The "
                         "expansion and the z-scoring are both inside the published ONNX graph, "
                         "which takes raw features."),
            }}),
        },
        "W": coef.tolist(), "b": intercept.tolist(),
        "n_parameters": int(coef.size + intercept.size),
        "classes": classes if classes is not None else list(range(max(2, coef.shape[0] + 1))),
    }, indent=2) + "\n")

    # Same fixtures-and-verify path the teachers take, so the artefact set matches. `fixtures` is
    # passed in when the graph takes RAW features - a pipeline student scales internally, so random
    # standard normals would exercise it far outside the range it was fitted on and make the
    # fp32/int8 agreement check meaningless.
    if fixtures is None:
        rng = np.random.default_rng(0)
        fixtures = rng.normal(size=(64, n_features)).astype(np.float32)
    report = export.verify_and_quantize(student, np.asarray(fixtures, dtype=np.float32), out,
                                        kind="sklearn", n_features=int(n_features))
    return {"n_parameters": int(coef.size + intercept.size),
            "onnx_domains": report.onnx_domains,
            "browser_runnable": report.browser_runnable}


def measure_fidelity(*, teacher_key: str, student_key: str, teacher_pred: np.ndarray,
                     student_pred: np.ndarray, y_true: np.ndarray, metric_fn, metric_name: str,
                     student_params: int, teacher_params: int | None = None,
                     n_scoring_units: int | None = None,
                     scoring_unit: str = "samples") -> FidelityReport:
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
        teacher_scored_zero=bool(float(metric_fn(y_true, teacher_pred)) == 0.0),
        n_scoring_units=(int(n_scoring_units) if n_scoring_units is not None else None),
        scoring_unit=scoring_unit,
    )


def _labels(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim > 1 and a.shape[1] > 1:
        return a.argmax(1).astype(int)
    return (a.ravel() > 0.5).astype(int) if a.dtype.kind == "f" else a.ravel().astype(int)


def save_cnn_student_weights(model, out_dir, *, teacher_key: str, student_key: str,
                             img_size: int, classes: list | None = None) -> dict:
    """Write a torch CNN student's weights in a form a browser can evaluate under CKKS.

    ``save_student`` above handles the sklearn-linear students, whose coefficients drop straight
    into a ``weights.json``. The medical student is a small torch CNN, so it shipped ONNX and
    safetensors and no readable weights -- and the browser cannot get conv kernels out of either
    without an ONNX parser. The consequence was that the encrypted convolution ran on the trained
    student's SHAPE with a synthetic kernel pattern: a real homomorphic convolution, but not the
    published model's inference, so ``crypto_fidelity`` stayed an offline number.

    The layout below is the network, not a generic dump: one conv block, an average pool, then two
    dense layers with a ReLU between them. Emitting it structurally means the browser reconstructs
    the same computation rather than guessing from tensor shapes.
    """
    import json
    import pathlib

    import torch.nn as nn

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Walk the modules IN ORDER and tag each weight layer with the activation that follows it.
    # Inferring activations from layer types instead would have silently dropped the ReLU that sits
    # between the conv and the pool: a re-implementation without it still agreed with the model on
    # every argmax in the test split while its logits were off by 1.5e-01, which is exactly the
    # failure mode that looks correct until it is not.
    conv, conv_act, denses = None, "none", []
    pending: list = []                       # weight layers awaiting their activation tag
    acts: dict[int, str] = {}
    for m in list(model.features) + list(model.classifier):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            pending.append(m)
            acts[id(m)] = "none"
            if isinstance(m, nn.Conv2d):
                conv = m
            else:
                denses.append(m)
        elif isinstance(m, nn.ReLU) and pending:
            acts[id(pending[-1])] = "relu"
    conv_act = acts.get(id(conv), "none") if conv is not None else "none"
    if conv is None or len(denses) < 2:
        raise ValueError(
            f"expected one Conv2d and at least two Linear layers, found conv={conv is not None} "
            f"linear={len(denses)}. This writer encodes the student's actual topology; a different "
            f"architecture needs its own encoding rather than a reinterpretation of this one."
        )

    def arr(t):
        return t.detach().cpu().numpy().tolist()

    pad = int(conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding)
    payload = {
        "kind": "cnn-1conv-2dense",
        "note": ("The FHE-feasible medical student, layer by layer. Evaluate strictly in this "
                 "order: conv (valid convolution over the input zero-padded by `conv.padding`), "
                 "conv.activation, average pool by `pool.stride`, flatten channel-major, then each "
                 "dense with its own `activation`. Every activation is recorded from the trained "
                 "module order rather than assumed - the conv is followed by a ReLU, and omitting "
                 "it still reproduces every argmax on the test split while shifting the logits, "
                 "which is why it is stated here explicitly. The browser runs conv and both dense "
                 "layers under CKKS and the ReLU/argmax under MPC, the hybrid this project "
                 "describes."),
        "teacher_key": teacher_key, "student_key": student_key,
        "input": {"shape": [1, int(conv.in_channels), img_size, img_size],
                  "preprocessing": ["divide by 255", "subtract 0.5", "divide by 0.5",
                                    f"resize to {img_size}x{img_size}"]},
        "conv": {"in_ch": int(conv.in_channels), "out_ch": int(conv.out_channels),
                 "kernel": int(conv.kernel_size[0]), "padding": pad,
                 "activation": conv_act,
                 "weight": arr(conv.weight), "bias": arr(conv.bias) if conv.bias is not None else None},
        "pool": {"type": "avg", "stride": 2},
        "dense": [{"in": int(d.in_features), "out": int(d.out_features),
                   "weight": arr(d.weight), "bias": arr(d.bias) if d.bias is not None else None,
                   "activation": acts.get(id(d), "none")}
                  for d in denses[:2]],
        "classes": classes if classes is not None else [0, 1],
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    (out / "weights.json").write_text(json.dumps(payload, indent=1) + "\n")
    return {"n_parameters": payload["n_parameters"], "path": str(out / "weights.json")}
