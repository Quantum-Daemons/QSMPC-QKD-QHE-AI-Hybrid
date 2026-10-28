"""The fibre detector set is Rusca et al. 2018's, value by value, and its dead time is modelled.

WHY THIS EXISTS

Until Round 76 the fibre set was four "modelling choices" - detector efficiency 0.25, 1e-5 dark
counts per gate, a 1.5 percent error floor, and a 100 MHz clock borrowed from a 2 km free-space
paper - matching no published source. Against the measured 50 km link the config already cited
(Lucamarini et al., Opt. Express 21, 24550 (2013)), the model detected 2.2x that paper's counts and
its per-pulse rate was 3.1x the paper's BB84 rate. By the user's decision the set is now the one
Rusca, Boaron, Grunenfelder, Martin & Zbinden state (Appl. Phys. Lett. 112, 171104 (2018),
arXiv:1801.03443): the paper whose 1-decoy finite-key analysis `qsh/qkd/finite_key.py` implements.
It is a simulation set (SNSPDs), not a measured link.

That set comes with a 1 GHz clock and a 100 ns dead time, and at 1 GHz the dead time is not a
detail: on the 50 km default it removes 37 percent of the detections. Python's `ChannelParams` also
carried its own literal copies of the old values, which `orchestration/stages.py` relied on for two
of the five, so a config change would have moved the browser and left the Python stage behind.

WHAT IS ASSERTED

- Each fibre value equals the figure Rusca et al. state, quoted beside it.
- `ChannelParams` takes its defaults from the config, and a downlink takes the downlink's clock and
  dead time rather than the fibre's.
- The dead-time factor is the non-paralysable form of their Eq. (B3), and both languages' BB84 runs
  charge it: the per-pulse rate is the one-decoy bound times the factor, the same in Python and in
  the browser.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import fields
from pathlib import Path

import pytest

from qsh.config import get_config
from qsh.qkd.bb84 import QKDParams, _decoy_secure_fraction, e_det_for_bound, run_bb84
from qsh.qkd.channel import ChannelParams, FreeSpaceParams, dead_time_factor

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

#: (config key, value, Rusca et al. 2018's words for it)
RUSCA_2018 = [
    ("detector_efficiency", 0.5, "an efficiency (eta_det around 50%)"),
    ("dark_count_prob", 1e-8, "dark-count rate (DCR) of 10 Hz which correspond to pDC = 10^-8"),
    ("misalignment", 0.01, "We choose a typical value pErr of 1%"),
    ("detector_dead_time_s", 100e-9, "a dead-time tau_DT = 100 ns"),
    ("source_rate_hz", 1e9, "a system working at a repetition rate of 1 GHz"),
    ("fiber_loss_db_per_km", 0.2, "Table I: 26 dB at 100 km, 46 dB at 200 km - 0.2 dB per km"),
]


@pytest.mark.parametrize("key,value,quote", RUSCA_2018, ids=[k for k, _, _ in RUSCA_2018])
def test_each_value_is_ruscas(key: str, value: float, quote: str) -> None:
    assert getattr(get_config().qkd, key) == pytest.approx(value), f"{key}: Rusca 2018 says {quote!r}"


def test_the_channel_defaults_are_the_config() -> None:
    q, c = get_config().qkd, ChannelParams()
    for name in ("distance_km", "fiber_loss_db_per_km", "detector_efficiency", "dark_count_prob",
                 "misalignment", "source_rate_hz"):
        assert getattr(c, name) == pytest.approx(getattr(q, name)), name
    assert c.dead_time_s == pytest.approx(q.detector_dead_time_s)
    fs = FreeSpaceParams()
    for name in (f.name for f in fields(FreeSpaceParams)):
        assert getattr(fs, name) == pytest.approx(getattr(q, f"free_space_{name}")), name
    src = (REPO / "src" / "qsh" / "qkd" / "channel.py").read_text()
    assert "detector_efficiency: float = 0.25" not in src and "= 1e-5" not in src


def test_a_downlink_keeps_its_own_clock_and_dead_time() -> None:
    q = get_config().qkd
    d = ChannelParams(distance_km=800.0, free_space=FreeSpaceParams())
    assert d.source_rate_hz == pytest.approx(q.free_space_source_rate_hz)
    assert d.dead_time_s == pytest.approx(q.free_space_detector_dead_time_s)
    # And its own detectors (Round 77): they were the fibre set's on any channel built like this one.
    for name in ("detector_efficiency", "dark_count_prob", "misalignment"):
        assert getattr(d, name) == pytest.approx(getattr(q, f"free_space_{name}")), name
    assert ChannelParams(free_space=FreeSpaceParams()).distance_km == pytest.approx(q.free_space_distance_km)
    assert dead_time_factor(d, 0.5) == 1.0          # Liao et al. state none, so none is modelled


def _gains(c: ChannelParams) -> tuple[float, float, float, float]:
    q = get_config().qkd
    mu = q.mu_signal
    nu = min(max(1e-4, q.mu_decoy), mu * 0.9)
    eta, y0 = c.transmittance(), 2 * c.dark_count_prob
    return mu, nu, y0 + 1 - math.exp(-eta * mu), y0 + 1 - math.exp(-eta * nu)


def test_the_factor_is_the_non_paralysable_form() -> None:
    """c = 1 / (1 + R t D) with D one of two detectors' share of the clicks: 0.625 at 50 km."""
    c = ChannelParams()
    _, _, q_mu, q_nu = _gains(c)
    click = 0.5 * (q_mu + q_nu)
    expected = 1.0 / (1.0 + c.source_rate_hz * c.dead_time_s * click / 2.0)
    assert dead_time_factor(c, click) == pytest.approx(expected, rel=1e-15)
    assert dead_time_factor(c, click) == pytest.approx(0.625, abs=0.001)
    # And the clock decides it: 1/c - 1 is R t D, so a tenth of the clock is a tenth of that term
    # (the same link at 100 MHz loses 5.7% where 1 GHz loses 37.5%).
    slow = ChannelParams(source_rate_hz=1e8)
    fast_term = 1 / dead_time_factor(c, click) - 1
    assert 1 / dead_time_factor(slow, click) - 1 == pytest.approx(fast_term / 10, rel=1e-12)


def test_the_bb84_rate_is_the_bound_times_the_factor() -> None:
    c = ChannelParams()
    mu, nu, q_mu, q_nu = _gains(c)
    bound = _decoy_secure_fraction(c.transmittance(), mu, nu, e_det_for_bound(c), 2 * c.dark_count_prob,
                                   get_config().qkd.error_correction_eff, 0.5)[0]
    run = run_bb84(200_000, c, seed=0, qkd_params=QKDParams(mu_signal=mu, mu_decoy=nu))
    assert run.secret_key_rate == pytest.approx(bound * dead_time_factor(c, 0.5 * (q_mu + q_nu)), rel=1e-12)
    assert run.secret_key_rate == pytest.approx(2.853e-3, rel=1e-3)


def test_the_browser_charges_the_same_dead_time() -> None:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(f"export {{ bb84, deadTimeFactor, FIBER }} from {json.dumps(str(WEB / 'src/lib/quantumSim'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(
            f"import * as m from {json.dumps(bundle.as_uri())};\n"
            "const out = [10, 50, 100].map((km) => {\n"
            "  const p = { ...m.FIBER, distanceKm: km };\n"
            "  return { km, rate: m.bb84(200000, p, 0).secretKeyRate,\n"
            "           slow: m.bb84(200000, { ...p, sourceRateHz: 1e8 }, 0).secretKeyRate };\n"
            "});\n"
            "console.log(JSON.stringify(out));\n")
        rows = json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True,
                                         text=True).stdout)
    q = get_config().qkd
    for row in rows:
        c = ChannelParams(distance_km=float(row["km"]))
        py = run_bb84(200_000, c, seed=0, qkd_params=QKDParams(mu_signal=q.mu_signal, mu_decoy=q.mu_decoy))
        assert row["rate"] == pytest.approx(py.secret_key_rate, rel=1e-9), row
        assert row["slow"] > row["rate"], f"a slower clock should lose fewer detections: {row}"
