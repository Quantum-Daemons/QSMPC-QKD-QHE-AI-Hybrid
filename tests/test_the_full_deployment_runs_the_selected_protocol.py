"""The full deployment runs the protocol and the profile the page selected, as the browser does.

WHY THIS EXISTS (Round 80, the user's choice to fix it in the same round). A survey of both
pipelines for the README's use-case flowcharts found two settings the page offers that the Python
pipeline ignored, while the browser honoured both:

- `qkd_protocol`: choosing E91 reached only the OptiQKD stage, so the fibre and satellite QKD stages
  ran BB84 whatever the page said. One control, two protocols, depending on the transport.
- the PQC-only profile on satellite: the fibre stage skipped QKD under it, but the free-space QKD and
  QSDC stages ran the downlink anyway, reporting a key over a channel the profile says is not there.
  OptiQKD likewise tuned a channel the run had just disabled.

WHAT IS ASSERTED

- E91 runs on both QKD stages when selected: an entangled source, a CHSH S, the asymptotic key of
  `run_e91` (no decoy bound, so no block), mixed into the session key when it reaches 256 bits.
- The browser's E91 gives the same S and the same expected QBER on the same channel; its key count
  takes the expected pair count where Python samples it, so the two agree within a few percent.
- BB84 stays the default and unchanged.
- Under PQC-only no quantum stage runs on satellite, and OptiQKD does not run on any use case.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from qsh.config import get_config
from qsh.qkd.channel import ChannelParams
from qsh.qkd.e91 import run_e91

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _telecom(cfg: dict) -> dict:
    from qsh.usecases import telecom_ids

    return telecom_ids.run_demo({"seed": 1, **cfg})["metrics"]


def _satellite(cfg: dict) -> dict:
    from qsh.usecases import satellite_comms

    return satellite_comms.run_demo({"seed": 1, **cfg})


def test_e91_runs_on_the_fibre_stage_when_selected() -> None:
    m = _telecom({"qkd_protocol": "e91"})
    q = get_config().qkd
    expected = run_e91(q.pulses, ChannelParams(), visibility=q.e91_visibility, seed=1)
    assert m["qkd_decoy_states"] == "none (entangled source)"
    assert m["chsh_s"] == pytest.approx(round(expected.chsh_s, 2)) and m["chsh_s"] > 2
    assert m["secure_key_bits"] == expected.secure_key_bits > 256
    assert "qkd_block" not in m and "qkd_finite_key_bits" not in m      # no decoy bound, no block
    assert m["qkd_mixed"] == 1


def test_bb84_stays_the_default_and_unchanged() -> None:
    m = _telecom({})
    assert m["qkd_decoy_states"] == "one decoy" and "chsh_s" not in m
    # 5,184,652 until Round 82: the default statistics became the tighter of Chernoff and Hoeffding
    # per count (5,214,327), and (A16) took tau0 off delta(n_Z), which alone gives 5,174,318.
    assert m["secure_key_bits"] == m["qkd_finite_key_bits"] == 5_214_327
    assert m["qkd_fk_deviation"] == "tighter"
    assert _telecom({"fk_deviation": "hoeffding"})["secure_key_bits"] == 5_174_318
    assert m["qkd_block"] == get_config().qkd.finite_key_block


def test_e91_runs_on_the_satellite_stage_when_selected() -> None:
    m = _satellite({"qkd_protocol": "e91"})["metrics"]
    assert m["qkd_decoy_states"] == "none (entangled source)" and m["chsh_s"] > 2
    assert "qkd_block" not in m and "qkd_finite_key_bits" not in m
    assert m["secure_key_bits"] > 256 and m["qkd_mixed"] == 1


def test_pqc_only_runs_no_quantum_stage_on_satellite() -> None:
    r = _satellite({"qkd": "pqc-only"})
    m = r["metrics"]
    for name in ("qber", "secure_key_bits", "skr_per_pulse", "qkd_mixed", "qsdc_secrecy_capacity",
                 "qsdc_link_budget_limited", "qsdc_eavesdrop_detected", "telemetry_anomaly_detected"):
        assert name not in m, f"{name} was reported under the PQC-only profile"
    logs = " ".join(e.get("message", "") for e in r.get("events", []) if e.get("type") == "log")
    if logs:   # the demo returns its events; when it does, the skip says why
        assert "PQC-only" in logs


#: What the OptiQKD stage reports when it tunes a channel. Its `qkd_optimize_ms` is the pipeline's
#: timing of every stage, emitted even for one that returns at once, so it is not among them.
OPTIQKD_OUTPUTS = ("qkd_baseline_skr", "qkd_optimized_skr", "qkd_optimized_qber",
                   "qkd_skr_improvement_pct", "qkd_convergence_evals")


def test_optiqkd_does_not_tune_a_disabled_channel() -> None:
    tuned = _telecom({"qkd_optimize": True})
    assert all(k in tuned for k in OPTIQKD_OUTPUTS), "the stage did not run with a channel: the check below proves nothing"
    m = _telecom({"qkd": "pqc-only", "qkd_optimize": True})
    assert not any(k in m for k in OPTIQKD_OUTPUTS), sorted(k for k in OPTIQKD_OUTPUTS if k in m)
    assert "qber" not in m


@pytest.fixture(scope="module")
def browser_e91() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(f"export {{ e91, FIBER }} from {json.dumps(str(WEB / 'src/lib/quantumSim'))};\n"
                         f"export {{ qkdCfg }} from {json.dumps(str(WEB / 'src/lib/config'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + """
const q = m.qkdCfg();
const r = m.e91(q.pulses, { ...m.FIBER }, 1, q.e91_visibility);
console.log(JSON.stringify({ s: r.chshS, e0: r.e0, bits: r.secureKeyBits }));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True,
                                         text=True).stdout)


def test_python_and_the_browser_run_the_same_e91(browser_e91) -> None:
    q = get_config().qkd
    py = run_e91(q.pulses, ChannelParams(), visibility=q.e91_visibility, seed=1)
    assert browser_e91["s"] == pytest.approx(py.chsh_s, rel=1e-12)
    assert browser_e91["e0"] == pytest.approx(py.qber, rel=1e-12)       # Python's QBER is the expected one
    assert browser_e91["bits"] == pytest.approx(py.secure_key_bits, rel=0.03)
