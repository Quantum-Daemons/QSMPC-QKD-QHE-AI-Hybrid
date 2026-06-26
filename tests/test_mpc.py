"""Tests for the MPC (secret-sharing) non-linear layer + the hybrid FHE-linear/MPC-non-linear path."""

import numpy as np
import pytest

pytest.importorskip("mpyc")

from qsh.mpc import hybrid_private_inference, secure_argmax, secure_relu
from qsh.fhe.backends import get_backend


def test_secure_relu_matches_plaintext():
    v = [-1.5, 2.0, 0.5, -0.25, 3.0, 0.0]
    out = secure_relu(v)
    np.testing.assert_allclose(out, [max(0.0, x) for x in v], atol=1e-2)


def test_secure_argmax_matches_plaintext():
    v = [-1.0, 0.3, 2.7, 2.69, -3.0]
    assert secure_argmax(v) == int(np.argmax(v))


def test_hybrid_inference_is_faithful_and_reports_split():
    rng = np.random.default_rng(0)
    # 8 -> 4 (ReLU) -> 2 (argmax): real hidden non-linearity under MPC
    layers = [(rng.normal(size=(4, 8)) * 0.2, rng.normal(size=4) * 0.1),
              (rng.normal(size=(2, 4)) * 0.2, rng.normal(size=2) * 0.1)]
    r = hybrid_private_inference(layers, rng.normal(size=8), get_backend("tenseal"), use_mpc=True)
    assert r["decision_matches_plaintext"] is True          # FHE+MPC preserve the plaintext decision
    assert r["fhe_linear_ms"] > 0 and r["mpc_nonlinear_ms"] > 0
    assert r["mpc_comparisons"] >= 6                         # 4 ReLU + 2 argmax
    assert r["mpc_comm_kb_estimated"] > 0                    # non-linear MPC has communication
    assert r["mpc_engine"] == "mpyc-shamir"


def test_hybrid_plaintext_fallback_matches():
    # the client-demo path uses use_mpc=False (plaintext non-linear); decision must still match
    rng = np.random.default_rng(1)
    layers = [(rng.normal(size=(4, 8)) * 0.2, rng.normal(size=4) * 0.1),
              (rng.normal(size=(2, 4)) * 0.2, rng.normal(size=2) * 0.1)]
    x = rng.normal(size=8)
    a = hybrid_private_inference(layers, x, get_backend("tenseal"), use_mpc=True)
    b = hybrid_private_inference(layers, x, get_backend("tenseal"), use_mpc=False)
    assert a["prediction"] == b["prediction"]
    assert b["mpc_comm_kb_estimated"] == 0.0                 # no MPC comm in the fallback
    assert a["arch_class"] == "mlp"                          # dense stack -> mlp


# --------------------------------------------------------------------------- #
#  Genuine FHE convolution (im2col -> dot_plain) + CNN hybrid (Round 10)
# --------------------------------------------------------------------------- #
def test_fhe_conv_matches_plaintext():
    """A genuine encrypted convolution (per-patch encrypt + dot_plain per out-channel) equals the
    plaintext convolution to within CKKS noise — no rotations, no new crypto."""
    from qsh.fhe.conv import conv2d_plaintext, fhe_conv2d
    from qsh.mpc.layers import ConvSpec

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 8, 8)) * 0.3
    W = rng.normal(size=(3, 1, 3, 3)) * 0.3
    b = rng.normal(size=3) * 0.1
    conv = ConvSpec(W, b, in_shape=(1, 8, 8), stride=1, padding=1)
    enc = fhe_conv2d(get_backend("tenseal"), x, conv)
    assert enc.shape == (3, 8, 8)
    np.testing.assert_allclose(enc, conv2d_plaintext(x, conv), atol=1e-2)


def test_secure_maxpool_matches_plaintext():
    """Max-pooling under MPyC secure comparison equals plaintext max-pool + counts comparisons."""
    from qsh.fhe.conv import maxpool2d
    from qsh.mpc import secure_maxpool
    from qsh.mpc.layers import PoolSpec

    rng = np.random.default_rng(2)
    planes = rng.normal(size=(2, 4, 4))
    pool = PoolSpec(pool="max", size=2, stride=2)
    out, n_cmp = secure_maxpool(planes, pool)
    np.testing.assert_allclose(out, maxpool2d(planes, 2, 2), atol=1e-2)
    assert n_cmp == 2 * (2 * 2) * (2 * 2 - 1)                # channels * windows * (win-1)


def test_hybrid_cnn_is_faithful_and_reports_split():
    """A 1-block CNN run as genuine FHE-conv + MPC-non-linear preserves the plaintext decision and
    reports a real FHE-linear / MPC-non-linear split; arch_class is reported as 'cnn'."""
    from qsh.mpc import ConvSpec, DenseSpec, FlattenSpec, PoolSpec, plaintext_forward

    rng = np.random.default_rng(3)
    img = rng.normal(size=(1, 8, 8)) * 0.3
    specs = [
        ConvSpec(rng.normal(size=(2, 1, 3, 3)) * 0.3, rng.normal(size=2) * 0.1,
                 in_shape=(1, 8, 8), stride=1, padding=1, activation="relu"),
        PoolSpec(pool="avg", size=2, stride=2),
        FlattenSpec(),
        DenseSpec(rng.normal(size=(4, 2 * 4 * 4)) * 0.2, rng.normal(size=4) * 0.1, activation="relu"),
        DenseSpec(rng.normal(size=(2, 4)) * 0.2, rng.normal(size=2) * 0.1, activation="none"),
    ]
    r = hybrid_private_inference(specs, img, get_backend("tenseal"), use_mpc=True)
    assert r["arch_class"] == "cnn"
    assert r["decision_matches_plaintext"] is True
    assert r["fhe_linear_ms"] > 0 and r["mpc_nonlinear_ms"] > 0
    assert r["mpc_comparisons"] > 0 and r["mpc_comm_kb_estimated"] > 0
    assert r["prediction"] == int(np.argmax(plaintext_forward(specs, img)))
