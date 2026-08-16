"""Tests for Round-2 features: config, crypto-agility (sig), pause/resume/step,
MCP bridge, Pauli concealment, OpenFHE bootstrap."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from qsh.config import get_config, load_config
from qsh.orchestration.crypto_agility import default_registry
from qsh.orchestration.events import EventType
from qsh.orchestration.pipeline import PipelineState, build_pipeline
from qsh.pqc.mcp_bridge import mcp_demo
from qsh.qhe.qotp import pauli_output_concealment_demo


# --- config ---------------------------------------------------------------
def test_config_loads_with_defaults():
    c = load_config()
    # This assertion pins the value the YAML and the dataclass must agree on, and the value moved
    # twice for reasons worth recording here rather than only in the config.
    #
    # Round 46 raised it 200_000 -> 2_000_000 and justified the raise with a "~5e3-1.1e4 finite-key
    # positive-key band" attributed to Rusca et al. (2018). Round 47 refuted that band on both
    # counts: the paper states no positivity threshold at all, and `src/qsh/qkd/finite_key.py`, which
    # implements that paper's own equations, puts the threshold at ~1e5 sifted bits - about 4e7
    # pulses at the intensities shipped until this round - on the 50 km channel this project ships.
    # At 2_000_000 pulses the single-photon lower bound is NEGATIVE, so Round 46's block carried no
    # composable key at any parameter setting; the raise was made against a number that was not in
    # the paper it cited. `src/qsh/config/defaults.py` carries the full record, the corrected
    # citations, and the margin the current 20_000_000 is sized for.
    assert c.qkd.pulses == 20_000_000
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


# --- Pauli OUTPUT concealment (NOT circuit privacy) -----------------------
# This test used to assert `fidelity_correct_key == 1.0` and then `circuit_private`, which read as
# evidence for a claim it never tested: the demo applies no circuit, and the correct-key fidelity is
# 1.0 by construction because decrypt(encrypt(x)) == x. A test whose assertions cannot fail is worse
# than a missing test, because it is counted as coverage. Assert the falsifiable property, and pin the
# tautological one as the self-check it is.
def test_pauli_mask_conceals_the_state_from_a_wrong_key():
    rep = pauli_output_concealment_demo(n=3, seed=1)
    assert rep["fidelity_wrong_key"] < 0.9
    assert rep["output_concealed"]
    # Stated in the payload so no caller can mistake this for circuit privacy.
    assert rep["applies_a_circuit"] is False


def test_roundtrip_selfcheck_is_a_tautology_and_labelled_as_one():
    """encrypt/decrypt are inverses, so this is 1.0 for every input and every key.

    Kept as a regression guard on `encrypt`/`decrypt`, and asserted separately from the concealment
    claim so that the two are never conflated again.
    """
    for seed in (1, 2, 3):
        rep = pauli_output_concealment_demo(n=3, seed=seed)
        assert rep["roundtrip_fidelity_selfcheck"] == pytest.approx(1.0, abs=1e-6)
        # The verdict must not depend on the tautology: a wrong key is the only thing tested.
        assert "circuit_private" not in rep


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
#
# This test used to have two ways to report success without checking anything: it skipped on
# ImportError, and it skipped when `bootstrap_enabled` was False -- which the backend set by
# swallowing every exception from its bootstrap setup. On this EL9 host neither fired, because
# the real failure was a SIGSEGV inside EvalBootstrapSetup that no except clause can catch; it
# killed the whole run instead. Now: the work happens in a subprocess (so a crash is a failure,
# not a dead suite), `bootstrap_enabled` is never silently False (the backend raises), and the
# only skip left is "openfhe genuinely will not import" -- which conftest.py cross-checks, so
# the skip cannot cover for a host where openfhe imports fine.
_BOOTSTRAP_PROBE = Path(__file__).with_name("_openfhe_bootstrap_probe.py")

#: Set to "1" on any host where bootstrapping is supposed to work (CI, the Docker image, the
#: demo VPS). Turns even the legitimate "openfhe missing" skip into a failure there.
REQUIRE_ENV = "QSH_REQUIRE_OPENFHE_BOOTSTRAP"


def _probe_payload(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"probe printed no JSON payload; stdout was:\n{stdout}")


@pytest.mark.slow
def test_openfhe_bootstrap():
    proc = subprocess.run(
        [sys.executable, str(_BOOTSTRAP_PROBE)],
        capture_output=True, text=True, timeout=900,
    )
    required = os.environ.get(REQUIRE_ENV) == "1"

    if proc.returncode == 42:                      # openfhe not importable -- the honest skip
        reason = _probe_payload(proc.stdout).get("import_error", "<no detail>")
        msg = (f"openfhe does not import on this host, so CKKS bootstrapping cannot be "
               f"exercised. Underlying error: {reason}. On RHEL/AlmaLinux 9 the published "
               f"wheel needs a newer libstdc++ than the system provides -- fix it with "
               f"`bash env/build_openfhe_el9.sh` (see requirements-fhe.txt).")
        if required:
            pytest.fail(f"{REQUIRE_ENV}=1 but {msg}")
        pytest.skip(msg)

    if proc.returncode < 0:                        # died on a signal; POSIX reports -N
        sig = signal.Signals(-proc.returncode)
        pytest.fail(
            f"OpenFHE bootstrapping crashed the interpreter: {sig.name} "
            f"({proc.returncode}). This is the defect this test exists to catch -- a native "
            f"crash, not an exception, so nothing upstream could catch or skip it. Check the "
            f"crypto-context parameters in qsh/fhe/backends.py (OpenFHEBackend.__init__ "
            f"documents which ones are load-bearing).\nstderr:\n{proc.stderr[-2000:]}"
        )

    assert proc.returncode == 0, (
        f"bootstrap probe exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )

    d = _probe_payload(proc.stdout)
    expected = d["expected"]

    # The ciphertext really was depleted, and bootstrapping really did buy levels back.
    assert d["level_before"] == d["total_depth"] - 2, d
    assert d["levels_regained"] > 0, (
        f"EvalBootstrap returned a ciphertext at the same level it was given "
        f"({d['level_before']}), so nothing was refreshed: {d}")

    # Values survive the round trip. Bootstrapping is approximate, hence the tolerance: the
    # depleted ciphertext decrypts near-exactly, the bootstrapped one to about two decimals.
    assert d["depleted"] == pytest.approx(expected, abs=1e-6), d
    assert d["refreshed"] == pytest.approx(expected, abs=1e-2), d

    # And the refreshed ciphertext is usable -- one more multiply, which the depleted one
    # could not have afforded.
    assert d["reused"] == pytest.approx([2 * v for v in expected], abs=2e-2), d
