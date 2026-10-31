"""The key holder's FULL decode of a masked conversion gives away nothing about the server's weights.

WHY THIS EXISTS

`_fhe_linear` added the bias and the MPC mask as one SCALAR, which TenSEAL adds to every slot, and
`ct.dot` leaves running partial sums of the products in the other slots. The client holds the
secret key, so it can decode all 4096 slots, not just the one the harness reads: slot k minus slot
k + 1 cancels the mask and is exactly x_k * w_k, and since the client knows x it recovers the
server's weight row from ONE query (confirmed locally, scratchpad audit82/live-checks.md).
`tests/test_mpc.py` recorded slot 0 only, so it could not see this.

WHAT IS ASSERTED

- Decoding every slot with the client's own secret key (SEAL's decryptor and encoder directly,
  not the backend's decrypt), no linear combination of the non-output slots predicts any product
  x_k * w_k out of sample, over many queries, and the adjacent-slot differences that recovered
  them exactly are uncorrelated with them. The convolution path is held to the same test.
- The imaginary parts are flooded too: they carry nothing but CKKS error when unmasked, and that
  error depends on the computation; the decode shows them at the masks' width.
- The decision path still agrees with plaintext, and each run reports the bound on the full view,
  the CKKS error it measured and the flood's margin over it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tenseal")

import tenseal.sealapi as sa  # noqa: E402

from qsh.fhe.backends import get_backend  # noqa: E402
from qsh.mpc import secret_inference  # noqa: E402

#: Queries per attack. Under the null the sample correlation has a standard deviation near
#: 1/sqrt(T) = 0.06, so the 0.3 bound below sits five of those away from it; the unmasked
#: conversion this replaces gives a correlation of 1.000 and an R^2 of 1.000.
TRIALS = 240
CORR_BOUND = 0.3
R2_BOUND = 0.2


class _ClientDecode:
    """Wraps a backend and decodes, with the CLIENT's key, every slot of whatever it decrypts."""

    def __init__(self, inner):
        self.inner, self.name, self.views = inner, inner.name, []
        sc = inner.context.seal_context().data
        self._dec = sa.Decryptor(sc, inner.context.secret_key().data)
        self._enc = sa.CKKSEncoder(sc)

    def decrypt(self, ct):
        raw = ct.ciphertext()[0] if hasattr(ct, "ciphertext") else ct
        pt = sa.Plaintext()
        self._dec.decrypt(raw, pt)
        self.views.append(np.asarray(self._enc.decode_complex(pt)))
        return self.inner.decrypt(ct)

    def __getattr__(self, attr):
        return getattr(self.inner, attr)


def _non_output_features(view: np.ndarray, reach: int) -> np.ndarray:
    """Real and imaginary parts of the slots a rotation-sum over `reach` products can touch."""
    idx = np.r_[1:reach, view.size - reach + 1:view.size]
    return np.concatenate([view[idx].real, view[idx].imag])


def _held_out_r2(features: np.ndarray, target: np.ndarray) -> float:
    """R^2 on the second half of an ordinary least-squares fit on the first half."""
    half = len(target) // 2
    a = np.column_stack([features, np.ones(len(target))])
    coef, *_ = np.linalg.lstsq(a[:half], target[:half], rcond=None)
    resid = target[half:] - a[half:] @ coef
    return 1.0 - float(resid @ resid) / float(((target[half:] - target[half:].mean()) ** 2).sum())


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _assert_no_product_leaks(views, xs, w, label: str) -> None:
    views = np.asarray(views)
    n = w.size
    feats = np.stack([_non_output_features(v, n) for v in views])
    for k in range(n):
        target = xs[:, k] * w[k]
        r2 = _held_out_r2(feats, target)
        assert r2 < R2_BOUND, f"{label}: the non-output slots predict x_{k} * w_{k} (held-out R^2 {r2:.3f})"
        if k < n - 1:
            # The attack of record: slot k minus slot k + 1 is exactly the k-th product, unmasked.
            diff = (views[:, k] - views[:, k + 1]).real
            c = _corr(diff, target)
            assert abs(c) < CORR_BOUND, f"{label}: slot {k} - slot {k + 1} tracks x_{k} * w_{k} (corr {c:.3f})"


def test_no_linear_combination_of_the_other_slots_recovers_a_weight():
    rng = np.random.default_rng(84)
    spy = _ClientDecode(get_backend("tenseal"))
    w = rng.normal(size=8) * 0.4
    W, b = w[None, :], np.array([0.05])
    bits = secret_inference._mask_bits(64)
    xs = rng.normal(size=(TRIALS, 8))
    for x in xs:
        r = secret_inference._masks((1,), bits)
        y = secret_inference._fhe_linear(spy, x, W, b, mask=r)
        # The share the MPC receives still adds up to the true value with the server's -r.
        assert y[0] - r[0] == pytest.approx(float(w @ x) + 0.05, abs=1e-4)
    assert len(spy.views) == TRIALS
    _assert_no_product_leaks(spy.views, xs, w, "dense")


def test_the_imaginary_parts_are_flooded_too():
    rng = np.random.default_rng(85)
    spy = _ClientDecode(get_backend("tenseal"))
    bits = secret_inference._mask_bits(64)
    secret_inference._fhe_linear(spy, rng.normal(size=8), rng.normal(size=(1, 8)), np.zeros(1),
                                 mask=secret_inference._masks((1,), bits))
    view = spy.views[0]
    # Unmasked, the imaginary parts are the CKKS error, about 1e-6 here. Flooded, they are uniform
    # on [-2^b, 2^b), whose median magnitude is 2^(b-1); 2^(b-3) leaves room for sampling.
    for part, name in ((view.real, "real"), (view.imag, "imaginary")):
        assert np.median(np.abs(part)) > 2.0 ** (bits - 3), f"the {name} parts are not masked"
        assert np.max(np.abs(part)) <= 2.0 ** bits * (1 + 1e-9), f"a {name} part exceeds the mask width"


def test_the_convolution_path_masks_every_slot():
    """One 3x3 kernel over a 3x3 input: one patch, one output, nine products in the rotation-sum."""
    from qsh.fhe.conv import fhe_conv2d
    from qsh.mpc.layers import ConvSpec

    rng = np.random.default_rng(86)
    spy = _ClientDecode(get_backend("tenseal"))
    kernel = rng.normal(size=(1, 1, 3, 3)) * 0.4
    conv = ConvSpec(kernel, np.array([0.1]), in_shape=(1, 3, 3), stride=1, padding=0)
    bits = secret_inference._mask_bits(64)
    xs = rng.normal(size=(TRIALS, 9))
    for x in xs:
        m = secret_inference._masks((1, 1, 1), bits)
        out = fhe_conv2d(spy, x.reshape(1, 3, 3), conv, mask=m)
        assert out.ravel()[0] - m.ravel()[0] == pytest.approx(float(kernel.ravel() @ x) + 0.1, abs=1e-4)
    _assert_no_product_leaks(spy.views, xs, kernel.ravel(), "conv")


@pytest.mark.parametrize("arch", ["mlp", "cnn"])
def test_the_decision_still_agrees_and_the_run_reports_the_full_view(arch):
    pytest.importorskip("mpyc")
    from qsh.mpc import ConvSpec, DenseSpec, FlattenSpec, PoolSpec, hybrid_private_inference

    rng = np.random.default_rng(87)
    if arch == "mlp":
        layers = [(rng.normal(size=(4, 8)) * 0.3, rng.normal(size=4) * 0.1),
                  (rng.normal(size=(2, 4)) * 0.3, rng.normal(size=2) * 0.1)]
        x = rng.normal(size=8)
        decrypted, opened = 4 + 2, 4                 # linear outputs; ReLU outputs the MPC returns
    else:
        layers = [ConvSpec(rng.normal(size=(2, 1, 3, 3)) * 0.3, rng.normal(size=2) * 0.1,
                           in_shape=(1, 6, 6), stride=1, padding=1, activation="relu"),
                  PoolSpec(pool="max", size=2, stride=2), FlattenSpec(),
                  DenseSpec(rng.normal(size=(3, 18)) * 0.2, rng.normal(size=3) * 0.1, activation="none")]
        x = rng.normal(size=(1, 6, 6)) * 0.3
        decrypted, opened = 2 * 6 * 6 + 3, 2 * 6 * 6 + 2 * 3 * 3
    r = hybrid_private_inference(layers, x, get_backend("tenseal"))
    assert r["decision_matches_plaintext"] is True and r["logit_max_abs_error"] < 1e-4
    assert r["client_learns"] == "the class index"
    assert r["masked_slots"] == "every slot, real and imaginary parts"
    assert r["masked_values"] == decrypted + opened
    # The full view: both parts of all 4096 slots of every ciphertext the client decrypts, plus each
    # value the MPC opens to it. Larger than the slot-0 bound, and still far below one in a million.
    assert r["view_values"] == decrypted * 2 * 4096 + opened
    assert r["mask_statistical_distance"] < r["view_statistical_distance"] < 1e-6
    # The flood is sized to the CKKS error: at least the MPC's own 2^-40 below it.
    assert 0 < r["ckks_error_max"] < 1e-4
    assert r["flood_margin_bits"] >= 40
