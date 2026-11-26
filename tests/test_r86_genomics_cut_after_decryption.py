"""The genomics cut is the key holder's in the Python tier too (Round 86, batch 6; the owner's choice of 2026-11-26).

A person's high-risk cut is their own 1000 Genomes super-population's, so a bias the evaluator added would name that
group. The browser adds it after decryption (tests/test_r84_genomics_full_score_in_the_browser.py). Until this change the
full deployment added the subject's cut under encryption in both places it evaluates the score, while its comment said
the cut was subtracted after decryption. Now:

  - the CKKS stage (`qsh.fhe.ckks_inference.encrypted_linear_inference`): for a `LinearModel` whose bias is the key
    holder's (`bias_after_decryption`), nothing goes into slot 0 under encryption and the key holder adds the bias to
    the value it decrypts;
  - the MPC-FHE benchmark row's hybrid (`qsh.mpc.secret_inference`): for a `DenseSpec` with `key_holder_bias`, the
    server masks the dot product alone and the key holder adds the bias to its own share, so the opened class is the
    same and the server never holds the cut.

Each is checked by recording what the evaluator is handed, on a toy model and through the genomics pipeline, and every
decision against the plaintext score and cut. Every other model keeps its bias under encryption, as before.
"""

from __future__ import annotations

import numpy as np
import pytest


def _toy(after: bool = True):
    """Two logits [0, w.x - cut] with the subject 0.75 above its cut, a margin no CKKS error comes near."""
    from qsh.fhe.ckks_inference import LinearModel

    rng = np.random.default_rng(7)
    w = rng.normal(size=12)
    x = rng.integers(0, 3, size=12).astype(float)
    cut = float(w @ x) - 0.75
    return LinearModel(np.vstack([np.zeros_like(w), w]), np.array([0.0, -cut]), bias_after_decryption=after), x


def test_the_ckks_stage_hands_the_evaluator_no_bias(monkeypatch) -> None:
    pytest.importorskip("tenseal")
    from qsh.fhe import masking
    from qsh.fhe.ckks_inference import encrypted_linear_inference

    seen: list[float] = []
    real = masking.masked_decrypt

    def spy(backend, ct, slot0, bits, view=None):
        seen.append(float(slot0))
        return real(backend, ct, slot0, bits, view)

    monkeypatch.setattr(masking, "masked_decrypt", spy)
    model, x = _toy()
    out = encrypted_linear_inference(model, x)
    assert seen == [0.0, 0.0], f"the evaluator put {seen} into slot 0: the key holder's bias reached it"
    plain = model.weight @ x + model.bias
    assert out["bias_after_decryption"] is True and out["prediction"] == int(np.argmax(plain)) == 1
    assert out["max_abs_error"] < 1e-5 and np.allclose(out["logits"], plain, atol=1e-5)
    # A model whose bias is the server's keeps it under encryption, as every other use case does.
    seen.clear()
    server_model, _ = _toy(after=False)
    encrypted_linear_inference(server_model, x)
    assert seen == [0.0, float(server_model.bias[1])], seen


def test_the_mpc_hybrid_hands_the_server_no_bias(monkeypatch) -> None:
    pytest.importorskip("tenseal")
    from qsh.mpc import secret_inference as si

    if not si.mpyc_available():
        pytest.skip("MPyC is not installed, so the hybrid cannot run here")
    from qsh.fhe.backends import get_backend
    from qsh.mpc.layers import DenseSpec

    seen: list[list[float]] = []
    real = si._fhe_linear

    def spy(backend, x, W, b, **kw):
        seen.append(np.asarray(b, dtype=float).tolist())
        return real(backend, x, W, b, **kw)

    monkeypatch.setattr(si, "_fhe_linear", spy)
    model, x = _toy()
    spec = DenseSpec(model.weight, model.bias, activation="none", key_holder_bias=True)
    r = si.hybrid_private_inference([spec], x, get_backend("tenseal"), use_mpc=True)
    assert seen == [[0.0, 0.0]], f"the server was handed the bias {seen}"
    assert r["prediction"] == 1 and r["decision_matches_plaintext"], r


def test_a_key_holders_bias_must_be_finite() -> None:
    """Added in the clear, a non-finite bias would give NaN logits that every comparison fails silently: it is refused."""
    pytest.importorskip("tenseal")
    from qsh.fhe.ckks_inference import LinearModel, encrypted_linear_inference
    from qsh.mpc.layers import DenseSpec

    with pytest.raises(ValueError, match="not finite"):
        DenseSpec(np.zeros((2, 3)), [0.0, float("nan")], activation="none", key_holder_bias=True)
    with pytest.raises(ValueError, match="not finite"):
        encrypted_linear_inference(LinearModel(np.zeros((2, 3)), np.array([0.0, np.nan]), bias_after_decryption=True),
                                   np.zeros(3))


def test_the_genomics_model_and_its_benchmark_layer_keep_the_cut() -> None:
    from qsh.benchmark.runner import usecase_arch_layers
    from qsh.usecases import genomics

    _, ctx = genomics.build({"seed": 1})
    model = ctx.data["linear_model"]
    assert model.bias_after_decryption is True and model.bias[0] == 0.0 and model.bias[1] < 0, model.bias
    layers, _, params = usecase_arch_layers("genomics", "lightweight", base=model, n_in=model.n_in)
    assert len(layers) == 1 and layers[0].key_holder_bias and layers[0].activation == "none"
    assert np.array_equal(layers[0].bias, model.bias) and np.array_equal(layers[0].weight, model.weight)
    assert params == model.weight.size + model.bias.size


def _genomics_run(monkeypatch, config: dict) -> dict:
    """The genomics pipeline with what the evaluator is handed recorded: the CKKS stage's slot 0 and the hybrid's bias."""
    from qsh.fhe import masking
    from qsh.mpc import secret_inference as si
    from qsh.usecases import genomics

    inside_hybrid: list[int] = []
    stage_slot0: list[float] = []
    server_bias: list[list[float]] = []
    real_decrypt, real_linear = masking.masked_decrypt, si._fhe_linear

    def decrypt(backend, ct, slot0, bits, view=None):
        if not inside_hybrid:                       # the CKKS stage's decryptions, not the hybrid's masked ones
            stage_slot0.append(float(slot0))
        return real_decrypt(backend, ct, slot0, bits, view)

    def linear(backend, x, W, b, **kw):
        server_bias.append(np.asarray(b, dtype=float).tolist())
        inside_hybrid.append(1)
        try:
            return real_linear(backend, x, W, b, **kw)
        finally:
            inside_hybrid.pop()

    monkeypatch.setattr(masking, "masked_decrypt", decrypt)
    monkeypatch.setattr(si, "_fhe_linear", linear)
    pipe, ctx = genomics.build(config)
    pipe.run(ctx)
    logs = [e.message for e in pipe.bus.history if e.type.value == "log" and e.stage == "private_inference"]
    return {"ctx": ctx, "stage_slot0": stage_slot0, "server_bias": server_bias, "logs": logs,
            "mpyc": si.mpyc_available()}


def _the_subjects_decision_and_log_hold(r: dict) -> None:
    ctx = r["ctx"]
    model, x = ctx.data["linear_model"], np.asarray(ctx.data["features"], dtype=float)
    assert r["stage_slot0"] and all(v == 0.0 for v in r["stage_slot0"]), f"the CKKS stage put {r['stage_slot0']} in slot 0"
    plain = model.weight @ x[: model.n_in] + model.bias
    assert ctx.artifacts["private_prediction"] == int(np.argmax(plain)), (ctx.artifacts["private_prediction"], plain)
    assert any("its bias the key holder's own, added after decryption, so the evaluator never holds it" in m
               for m in r["logs"]), r["logs"]


def test_the_genomics_run_never_hands_the_evaluator_the_cut(monkeypatch) -> None:
    """At the lightweight size the MPC-FHE row's hybrid runs the subject's own model, so its server must get no bias."""
    pytest.importorskip("tenseal")
    r = _genomics_run(monkeypatch, {"seed": 1, "model_size": "lightweight"})
    _the_subjects_decision_and_log_hold(r)
    if r["mpyc"]:
        assert "mpc_fhe_row" in r["ctx"].artifacts, "the MPC-FHE row did not run, so its hybrid was not checked"
        assert r["server_bias"] and all(v == 0.0 for row in r["server_bias"] for v in row), (
            f"the hybrid's server was handed the bias {r['server_bias']}")


def test_at_the_default_size_the_hybrid_never_sees_the_cut(monkeypatch) -> None:
    """The default size prices the row on a random layer for cost only (`usecase_arch_layers`); the cut is not in it."""
    pytest.importorskip("tenseal")
    r = _genomics_run(monkeypatch, {"seed": 1})
    _the_subjects_decision_and_log_hold(r)
    cut = float(r["ctx"].data["linear_model"].bias[1])
    assert all(abs(v - cut) > 1e-6 for row in r["server_bias"] for v in row), (
        f"the subject's cut {cut} reached the hybrid's server: {r['server_bias']}")
