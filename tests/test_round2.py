"""Tests for Round-2 features: config, crypto-agility (sig), pause/resume/step,
MCP bridge, Pauli concealment, OpenFHE bootstrap."""

import threading
import time

import pytest

from qsh.config import get_config, load_config
from qsh.orchestration.crypto_agility import default_registry
from qsh.orchestration.events import EventType
from qsh.orchestration.pipeline import PipelineState, build_pipeline
from qsh.pqc.mcp_bridge import mcp_demo
from qsh.qhe.qotp import pauli_concealment_demo


# --- config ---------------------------------------------------------------
def test_config_loads_with_defaults():
    c = load_config()
    assert c.qkd.pulses == 200_000
    assert c.qhe.sk_eps == pytest.approx(1e-3)
    assert c.orchestration.step_delay > 0
    assert "kem" in c.crypto.to_dict() if hasattr(c.crypto, "to_dict") else True


def test_config_singleton():
    assert get_config() is get_config()


# --- crypto-agility: signature stage --------------------------------------
def test_signature_stage_signs_and_verifies():
    pipe, ctx = build_pipeline("demo", {"sig": "ml-dsa-87"})
    pipe.run(ctx)
    sig = ctx.artifacts.get("result_signature")
    assert sig is not None
    assert sig["algorithm"] == "ML-DSA-87"
    assert sig["verified"] is True
    assert pipe.bus.metrics().get("signature_verified") == 1


def test_crypto_registry_exposes_all_families():
    reg = default_registry()
    for fam in ("kem", "qkd", "fhe", "sig"):
        assert reg.options(fam)
        assert reg.default(fam) in reg.options(fam)


# --- MCP quantum-safe agent channel ---------------------------------------
def test_mcp_demo_authenticated_and_tamper_evident():
    r = mcp_demo()
    assert r["authenticated"] is True
    assert r["tamper_rejected"] is True
    assert r["signature_alg"].startswith("ML-DSA")


def test_mcp_demo_with_qkd_mix():
    r = mcp_demo(qkd_key=b"q" * 32)
    assert r["authenticated"] is True
    assert r["qkd_mixed"] is True


# --- Pauli-gate concealment (circuit privacy) -----------------------------
def test_pauli_concealment_hides_result():
    rep = pauli_concealment_demo(n=3, seed=1)
    assert rep["fidelity_correct_key"] == pytest.approx(1.0, abs=1e-6)
    assert rep["fidelity_wrong_key"] < 0.9
    assert rep["circuit_private"]


# --- pause / resume / step ------------------------------------------------
def _wait(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_pause_step_resume_flow():
    pipe, ctx = build_pipeline("demo")
    pipe.request_pause()                     # begin paused (step budget 0)
    t = threading.Thread(target=lambda: pipe.run(ctx, step_delay=0.0), daemon=True)
    t.start()
    # pauses before any stage completes
    assert _wait(lambda: pipe.state is PipelineState.PAUSED)
    assert not any(e.type is EventType.STAGE_END for e in pipe.bus.history)
    # one step -> exactly one stage completes, then re-pauses
    pipe.request_step()
    assert _wait(lambda: sum(e.type is EventType.STAGE_END for e in pipe.bus.history) == 1)
    assert _wait(lambda: pipe.state is PipelineState.PAUSED)
    # resume -> runs to completion
    pipe.request_resume()
    assert _wait(lambda: pipe.state is PipelineState.DONE, timeout=10.0)
    t.join(timeout=5.0)


def test_stop_from_pause():
    pipe, ctx = build_pipeline("demo")
    pipe.request_pause()
    t = threading.Thread(target=lambda: pipe.run(ctx, step_delay=0.0), daemon=True)
    t.start()
    assert _wait(lambda: pipe.state is PipelineState.PAUSED)
    pipe.request_stop()
    assert _wait(lambda: pipe.state is PipelineState.STOPPED, timeout=5.0)
    t.join(timeout=5.0)


# --- OpenFHE CKKS bootstrap (commercial-OK; needs the openfhe wheel + recent libstdc++) ---
@pytest.mark.slow
def test_openfhe_bootstrap():
    try:
        from qsh.fhe.backends import OpenFHEBackend

        b = OpenFHEBackend(use_bootstrap=True)
    except ImportError:
        pytest.skip("openfhe not importable on this host (needs Ubuntu-24.04-class libstdc++)")
    if not b.bootstrap_enabled:
        pytest.skip("bootstrap not available")
    ct = b.encrypt([0.5, 0.25, 0.1])
    out = b.decrypt(b.bootstrap(ct))[:3]
    assert out[0] == pytest.approx(0.5, abs=1e-2)
