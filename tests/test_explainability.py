"""Tests for the XAI layer: exact linear-SHAP attributions + signed explanations."""

import numpy as np

from qsh.ai.explainability import explain_linear_model, gradient_x_input, linear_shap
from qsh.fhe.ckks_inference import LinearModel


def test_linear_shap_satisfies_efficiency_axiom():
    # logits = W x + b  (2-class). SHAP local accuracy: sum(phi) + base == margin.
    rng = np.random.default_rng(0)
    W = rng.normal(size=(2, 8)); b = rng.normal(size=2)
    x = rng.normal(size=8)
    exp = linear_shap(W, b, x, top_k=8)
    assert exp.efficiency_gap() < 1e-9
    total = sum(a["contribution"] for a in exp.attributions)
    np.testing.assert_allclose(total, exp.margin - exp.base_value, atol=1e-9)


def test_linear_shap_ranks_the_dominant_feature_first():
    # class-1 logit driven almost entirely by feature 3
    W = np.zeros((2, 6)); W[1, 3] = 5.0
    b = np.zeros(2)
    x = np.ones(6)
    exp = linear_shap(W, b, x, feature_names=[f"x{i}" for i in range(6)], top_k=3)
    assert exp.prediction == 1
    assert exp.top[0]["feature"] == "x3"
    assert exp.top[0]["contribution"] > 0     # pushes toward the positive (fraud/positive) class


def test_explain_linear_model_wrapper_matches():
    rng = np.random.default_rng(1)
    model = LinearModel(rng.normal(size=(2, 8)), rng.normal(size=2))
    x = rng.normal(size=8)
    exp = explain_linear_model(model, x)
    assert exp.method == "exact-linear-shap"
    assert len(exp.attributions) == 8 and len(exp.top) == 5
    assert exp.efficiency_gap() < 1e-9


def test_gradient_x_input_fallback_runs():
    # simple non-linear scalar model: sigmoid(2*x0 - x2)
    def fwd(v):
        return 1.0 / (1.0 + np.exp(-(2 * v[0] - v[2])))

    exp = gradient_x_input(fwd, np.array([1.0, 0.0, -1.0, 0.5]), top_k=2)
    assert exp.method == "gradient-x-input"
    assert len(exp.top) == 2
    # feature 0 (coef +2) should be among the strongest drivers
    assert any(t["feature"] == "feature_0" for t in exp.top)
