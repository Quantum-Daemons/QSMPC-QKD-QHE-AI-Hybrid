"""Explainable AI (XAI) — exact, faithful feature attributions for the private-inference models.

The report flags XAI. **EU AI Act Article 13 transparency** is the instrument usually cited for it, but
it is NOT yet in application: the AI Omnibus (Regulation (EU) 2026/1744, in force 2026-07-27) deferred
Chapter III Sections 1-3 to 2027-12-02 (Annex III) and 2028-08-02 (Annex I). So treat this module as
preparation for a 2027/2028 obligation, not compliance with a current one.

When it does apply, Article 13 obliges high-risk AI to be interpretable. Of this project's use cases,
diagnostic and in-vitro diagnostic software would be high-risk through Annex I (MDR Rule 11, IVDR
Rule 3), while AML monitoring by a bank is not (Annex III 5(b) covers creditworthiness; profiling by
law enforcement would be, under 6(e)) and an intrusion detector used solely for cybersecurity is not
either on recital 55's reading; `web/src/lib/aiActScope.ts` states each, with its conditions. An earlier revision of this paragraph listed "credit/AML" as high-risk. Finance and healthcare
supervisors already expect **SHAP-style feature attributions** (+ counterfactuals)
under instruments that are in force today: US SR 11-7 model risk management and GDPR Art. 22
right-to-explanation. Those are the reasons to build it now.

(An earlier revision of this docstring said Article 13 was "in force August 2026". The correction was
applied everywhere else - `ShapPanel.tsx`, `README.md`, `clientCompute.ts` - and left a sentence
fragment here asserting the very date the two paragraphs above retract, in the module the README cites
as the implementation. Nothing lints prose, so it survived; that is the argument for stating the
retracted claim explicitly rather than deleting it silently.)

For **linear / logistic** models the Shapley value has an exact closed form,

    phi_i = w_i * (x_i - E[x_i]),     base = w . E[x] + b,     sum_i phi_i + base = model(x)

so attributions are *exact* and cheap — no sampling, and no need to decrypt under FHE: the model
owner explains on the plaintext weights, or the client explains on its own features. The
background expectation ``E[x]`` can be produced by **secure aggregation** rather than shared raw
data (cf. *Federated SHAP*, Machine Learning 2026; additive-HE + Shapley FL) — see
:mod:`~qsh.ai.federated`. Combined with the ML-DSA :class:`SignatureStage`, the explanation
becomes a **tamper-evident, signed** audit record (the report's XAI / audit-trail requirement).

A ``gradient_x_input`` fallback is provided for small non-linear (MLP) models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Explanation:
    """A feature-attribution explanation for one prediction."""

    prediction: int
    base_value: float          # model output on the baseline E[x]
    margin: float              # explained model output (class logit / one-vs-rest margin)
    attributions: list         # [{"feature", "value", "contribution"}] for every feature
    top: list                  # top-k features by |contribution|
    method: str

    def efficiency_gap(self) -> float:
        """|margin - base - sum(contributions)| — the SHAP local-accuracy axiom (≈0 = exact)."""
        return abs(self.margin - self.base_value
                   - sum(a["contribution"] for a in self.attributions))


def linear_shap(weight, bias, x, *, baseline=None, feature_names=None,
                top_k: int = 5, positive_class: int = 1) -> Explanation:
    """Exact Shapley attributions for a linear model ``logits = W x + b``.

    Explains the **one-vs-rest margin** of the predicted class (for a 2-class model, the
    margin of ``positive_class`` over the other), so a positive contribution pushes *toward*
    that class. ``baseline`` is the reference ``E[x]`` (defaults to zeros).
    """
    W = np.atleast_2d(np.asarray(weight, dtype=float))
    b = np.asarray(bias, dtype=float).ravel()
    n_in = W.shape[1]
    x = np.asarray(x, dtype=float).ravel()[:n_in]
    base_x = (np.zeros(n_in) if baseline is None
              else np.asarray(baseline, dtype=float).ravel()[:n_in])

    logits = W @ x + b
    pred = int(np.argmax(logits))
    if W.shape[0] == 2:
        other = 1 - positive_class
        w_eff, b_eff = W[positive_class] - W[other], float(b[positive_class] - b[other])
    else:                                            # one-vs-rest for the predicted class
        rest = [j for j in range(W.shape[0]) if j != pred]
        w_eff, b_eff = W[pred] - W[rest].mean(0), float(b[pred] - b[rest].mean())

    phi = w_eff * (x - base_x)
    # full precision kept so the local-accuracy axiom (sum(phi)+base == margin) holds exactly;
    # callers round for display (see ExplainabilityStage).
    base_value = float(w_eff @ base_x + b_eff)
    margin = float(w_eff @ x + b_eff)
    names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(n_in)]
    attrs = [{"feature": names[i] if i < len(names) else f"feature_{i}",
              "value": float(x[i]), "contribution": float(phi[i])}
             for i in range(n_in)]
    top = sorted(attrs, key=lambda a: abs(a["contribution"]), reverse=True)[:top_k]
    return Explanation(prediction=pred, base_value=base_value, margin=margin,
                       attributions=attrs, top=top, method="exact-linear-shap")


def explain_linear_model(model, x, *, baseline=None, feature_names=None,
                         top_k: int = 5) -> Explanation:
    """Convenience wrapper for :class:`qsh.fhe.ckks_inference.LinearModel` (``weight``, ``bias``)."""
    return linear_shap(model.weight, model.bias, x, baseline=baseline,
                       feature_names=feature_names, top_k=top_k)


def gradient_x_input(forward, x, *, eps: float = 1e-4, feature_names=None,
                     top_k: int = 5) -> Explanation:
    """Gradient×input attribution for a small black-box scalar model (MLP fallback).

    ``forward(x) -> float`` returns the positive-class score; gradients are finite-difference.
    Not exact (non-linear), but a standard, cheap local explanation.
    """
    x = np.asarray(x, dtype=float).ravel()
    fx = float(forward(x))
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        grad[i] = (float(forward(xp)) - fx) / eps
    phi = grad * x
    names = list(feature_names) if feature_names else [f"feature_{i}" for i in range(x.size)]
    attrs = [{"feature": names[i] if i < len(names) else f"feature_{i}",
              "value": round(float(x[i]), 6), "contribution": round(float(phi[i]), 6)}
             for i in range(x.size)]
    top = sorted(attrs, key=lambda a: abs(a["contribution"]), reverse=True)[:top_k]
    return Explanation(prediction=int(fx >= 0.5), base_value=0.0, margin=round(fx, 6),
                       attributions=attrs, top=top, method="gradient-x-input")
