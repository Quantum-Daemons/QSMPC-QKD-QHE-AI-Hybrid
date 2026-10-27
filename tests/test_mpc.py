"""Tests for the MPC (secret-sharing) non-linear layer + the hybrid FHE-linear/MPC-non-linear path."""

import numpy as np
import pytest

pytest.importorskip("mpyc")

from qsh.fhe.backends import get_backend
from qsh.mpc import hybrid_private_inference, secure_argmax, secure_relu


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
    assert r["mpc_comparisons"] == 5                         # 4 ReLU + 1 for the argmax of 2
    # Real parties since Round 75: three processes, t = 1, and bytes that crossed a socket.
    assert r["mpc_engine"] == "mpyc-shamir, 3 parties (t=1), loopback TCP"
    assert (r["mpc_parties"], r["mpc_threshold"], r["mpc_sec_param"]) == (3, 1, 40)
    assert r["mpc_sessions"] == 2 and r["mpc_comm_kb"] > 0
    # Starting the parties is the harness's cost and is reported apart from the protocol's.
    assert r["mpc_session_overhead_ms"] > 0


def test_hybrid_plaintext_fallback_matches():
    # the client-demo path uses use_mpc=False (plaintext non-linear); decision must still match
    rng = np.random.default_rng(1)
    layers = [(rng.normal(size=(4, 8)) * 0.2, rng.normal(size=4) * 0.1),
              (rng.normal(size=(2, 4)) * 0.2, rng.normal(size=2) * 0.1)]
    x = rng.normal(size=8)
    a = hybrid_private_inference(layers, x, get_backend("tenseal"), use_mpc=True)
    b = hybrid_private_inference(layers, x, get_backend("tenseal"), use_mpc=False)
    assert a["prediction"] == b["prediction"]
    assert b["mpc_comm_kb"] == 0.0 and b["mpc_sessions"] == 0   # no MPC in the fallback
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
    assert r["mpc_comparisons"] > 0 and r["mpc_comm_kb"] > 0
    assert r["prediction"] == int(np.argmax(plaintext_forward(specs, img)))



# --------------------------------------------------------------------------------------------- #
#  The MPyC tier runs as real parties (Round 75)
# --------------------------------------------------------------------------------------------- #
# It ran as ONE in-process party with MPyC's defaults (m = 1, t = 0, k = 30): a degree-0 Shamir share
# is the value itself, so the "secure" comparison computed in the clear and sent nothing.


def test_the_parties_are_processes_that_exchange_bytes():
    from qsh.mpc.parties import run_parties
    r = run_parties("relu", [1.5, -2.0, 0.25])
    assert r["result"] == [1.5, 0.0, 0.25]
    assert r["parties"] == 3 and r["threshold"] == 1 and r["sec_param"] == 40
    assert len(r["per_party_bytes"]) == 3 and all(b > 0 for b in r["per_party_bytes"])


def test_each_party_is_given_only_its_own_inputs(monkeypatch):
    """The client's file carries its values, the server's its share and mask, the helper's nothing."""
    import json
    import subprocess as sp

    from qsh.mpc import parties
    seen = {}
    real_popen = sp.Popen

    def spy(cmd, env=None, **kw):
        seen[cmd[cmd.index("-I") + 1]] = json.loads(open(env["QSH_MPC_JOB"]).read())
        return real_popen(cmd, env=env, **kw)

    monkeypatch.setattr(parties.subprocess, "Popen", spy)
    r = parties.run_parties("relu", [10.5, 3.0], server_share=[-10.0, -5.0], out_mask=[100.0, 200.0])
    assert r["result"] == [100.5, 200.0]                    # ReLU(0.5, -2.0) + the server's mask
    assert seen["0"]["share"] == [10.5, 3.0] and "out_mask" not in seen["0"]
    assert seen["1"]["share"] == [-10.0, -5.0] and seen["1"]["out_mask"] == [100.0, 200.0]
    assert "share" not in seen["2"] and "out_mask" not in seen["2"], seen["2"]


def test_a_dishonest_majority_threshold_is_refused():
    from qsh.mpc.parties import run_parties
    with pytest.raises(ValueError, match="honest majority"):
        run_parties("relu", [1.0], m=3, threshold=2)
    with pytest.raises(ValueError, match="at least two parties"):
        run_parties("relu", [1.0], m=1)


def test_no_mpc_call_falls_back_to_one_party():
    """The in-process single-party path is gone; nothing imports MPyC's runtime in the caller."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "qsh" / "mpc" / "secret_inference.py").read_text()
    assert "from mpyc.runtime import mpc" not in src and "_import_mpc" not in src


def test_the_ports_come_from_below_the_ephemeral_range():
    """A listening port inside it can be taken first by an outgoing connection's source port: 5 of
    300 sessions failed that way in Round 75, each with party 1 unable to bind base + 1."""
    from qsh.mpc import parties
    lo, _ = parties._ephemeral_range()
    for _ in range(200):
        base = parties._free_base_port(3)
        assert 10_000 <= base and base + 2 < lo, base


def test_a_party_that_cannot_bind_ends_the_run_at_once(monkeypatch):
    """A failed party is seen when it exits, not at the timeout, and a lost port is retried."""
    import socket
    import time

    from qsh.mpc import parties
    base = parties._free_base_port(3)
    with socket.socket() as squat:
        squat.bind(("0.0.0.0", base + 1))               # party 1's port, taken
        squat.listen()
        monkeypatch.setattr(parties, "_free_base_port", lambda m: base)
        t0 = time.monotonic()
        with pytest.raises(parties._PortTaken, match="address already in use"):
            parties.run_parties("relu", [1.0], timeout_s=60.0)
        assert time.monotonic() - t0 < 20.0, "the run waited for its timeout"


def test_a_lost_port_costs_one_more_attempt(monkeypatch):
    from qsh.mpc import parties
    real, calls = parties._session, []

    def flaky(*args):
        calls.append(1)
        if len(calls) == 1:
            raise parties._PortTaken("party 1 exit 1: address already in use")
        return real(*args)

    monkeypatch.setattr(parties, "_session", flaky)
    r = parties.run_parties("relu", [-1.0, 2.0])
    assert r["result"] == [0.0, 2.0] and r["attempts"] == 2


def test_the_protocol_time_excludes_starting_the_parties():
    from qsh.mpc.parties import run_parties
    r = run_parties("argmax", [0.1, 0.9, 0.5])
    assert r["result"] == 1 and 0 < r["mpc_ms"] < r["wall_ms"]


# --------------------------------------------------------------------------------------------- #
#  The conversions are masked (Round 75)
# --------------------------------------------------------------------------------------------- #
# They were decrypt-then-share and open-to-everyone: the key holder decrypted every linear output in
# the clear and every party received every ReLU result.


class _Recording:
    """A backend that records everything the key holder decrypts."""

    def __init__(self, inner):
        self.inner, self.name, self.seen = inner, inner.name, []

    def decrypt(self, ct):
        out = self.inner.decrypt(ct)
        self.seen.extend(float(v) for v in out[:1])
        return out

    def __getattr__(self, attr):
        return getattr(self.inner, attr)


def _cnn(rng):
    from qsh.mpc import ConvSpec, DenseSpec, FlattenSpec, PoolSpec
    return [
        ConvSpec(rng.normal(size=(2, 1, 3, 3)) * 0.3, rng.normal(size=2) * 0.1,
                 in_shape=(1, 6, 6), stride=1, padding=1, activation="relu"),
        PoolSpec(pool="max", size=2, stride=2), FlattenSpec(),
        DenseSpec(rng.normal(size=(4, 18)) * 0.2, rng.normal(size=4) * 0.1, activation="relu"),
        DenseSpec(rng.normal(size=(3, 4)) * 0.2, rng.normal(size=3) * 0.1, activation="none"),
    ]


def test_the_key_holder_decrypts_only_masked_values():
    """Every value decrypted is an intermediate plus a mask of up to 2^30: none is near its size."""
    rng = np.random.default_rng(7)
    be = _Recording(get_backend("tenseal"))
    r = hybrid_private_inference(_cnn(rng), rng.normal(size=(1, 6, 6)) * 0.3, be)
    assert r["decision_matches_plaintext"] is True and r["client_learns"] == "the class index"
    seen = np.abs(np.asarray(be.seen))
    # 2 * 6 * 6 conv outputs + 4 + 3 dense outputs; the intermediates themselves are below 2.
    assert seen.size == 2 * 6 * 6 + 4 + 3
    assert np.median(seen) > 2.0 ** 20, f"median |decrypted| = {np.median(seen):.3g}"


def test_the_parties_receive_shares_that_sum_to_the_intermediates(monkeypatch):
    """The client's MPC input is masked; with the server's share it adds up to the true value."""
    import json
    import subprocess as sp

    from qsh.mpc import parties
    from qsh.mpc.layers import plaintext_forward
    jobs = []
    real_popen = sp.Popen

    def spy(cmd, env=None, **kw):
        jobs.append((cmd[cmd.index("-I") + 1], json.loads(open(env["QSH_MPC_JOB"]).read())))
        return real_popen(cmd, env=env, **kw)

    monkeypatch.setattr(parties.subprocess, "Popen", spy)
    rng = np.random.default_rng(8)
    layers = [(rng.normal(size=(4, 8)) * 0.3, rng.normal(size=4) * 0.1),
              (rng.normal(size=(2, 4)) * 0.3, rng.normal(size=2) * 0.1)]
    x = rng.normal(size=8)
    hybrid_private_inference(layers, x, get_backend("tenseal"))
    by_session = [dict(jobs[i:i + 3]) for i in range(0, len(jobs), 3)]
    assert len(by_session) == 2                                  # the hidden ReLU, then the argmax
    relu = by_session[0]
    client, server = np.asarray(relu["0"]["share"]), np.asarray(relu["1"]["share"])
    hidden = layers[0][0] @ x + layers[0][1]                    # what the ReLU really runs on
    np.testing.assert_allclose(client + server, hidden, atol=1e-4)
    assert np.min(np.abs(client)) > 1e3 and "share" not in relu["2"]
    assert "out_mask" in relu["1"] and "out_mask" not in by_session[1]["1"]   # no mask on an index
    assert plaintext_forward(layers, x).size == 2


def test_the_fallback_says_what_it_gives_away():
    rng = np.random.default_rng(1)
    layers = [(rng.normal(size=(2, 8)) * 0.2, rng.normal(size=2) * 0.1)]
    r = hybrid_private_inference(layers, rng.normal(size=8), get_backend("tenseal"), use_mpc=False)
    assert r["client_learns"] == "every intermediate value" and "mask_bits" not in r


def test_each_run_reports_the_masks_bound():
    rng = np.random.default_rng(9)
    layers = [(rng.normal(size=(4, 8)) * 0.2, rng.normal(size=4) * 0.1),
              (rng.normal(size=(2, 4)) * 0.2, rng.normal(size=2) * 0.1)]
    r = hybrid_private_inference(layers, rng.normal(size=8), get_backend("tenseal"))
    assert r["mask_bits"] == 30 and r["masked_values"] == 4 + 4 + 2   # z, ReLU(z), the logits
    assert r["mask_statistical_distance"] == pytest.approx(
        r["masked_values"] * r["max_abs_masked_value"] / 2.0 ** 30, rel=1e-3)
    assert r["mask_statistical_distance"] < 1e-7
    assert r["logit_max_abs_error"] < 1e-4


def test_a_mask_too_wide_for_the_fixed_point_is_refused(monkeypatch):
    from qsh.config import get_config
    from qsh.mpc import secret_inference
    monkeypatch.setattr(get_config().benchmark, "mpc_mask_bits", 31)
    with pytest.raises(ValueError, match="does not fit 64-bit fixed point"):
        secret_inference._mask_bits(64)


def test_an_index_is_never_masked():
    from qsh.mpc.parties import run_parties
    with pytest.raises(ValueError, match="class index"):
        run_parties("argmax", [1.0, 2.0], out_mask=[0.5])
