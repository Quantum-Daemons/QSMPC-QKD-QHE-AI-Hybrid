"""The fibre key rate is the bound for the protocol it runs, and the QBER shown is the one the bound uses.

WHY THIS EXISTS

The protocol sends a signal and ONE weak decoy, so a receiver never measures the vacuum yield Y0.
Until Round 75 both implementations used Ma, Qi, Zhao & Lo's (2005) section 3.4 estimator, which
needs Y0, and fed it the channel model's `2 * dark_count_prob` - a value the protocol cannot know.
The docstring disclosed that and called the result "tighter than section 3.5" without the size:
measured at the shipped intensities, 1.3 percent high at 50 km, 12.6 percent at 100 km, 40 percent at
120 km, with a maximum distance of 140.9 km against the paper's own one-decoy 134.7 km. Section 3.5's
"tighter bound", Eq. (41), sets Y0 = 0 in the estimate, because Eve chooses it.

Separately, the BB84 run showed `intrinsic_qber`, whose background term `0.5 p_dark / eta` counts one
detector and a whole photon per pulse, beside a bound that used Ma's Eq. (11) E_mu. The QBER shown
and the QBER the rate was computed from differed by 6 percent at 50 km and 2.8x at 140 km, and the
abort test read the smaller one.

WHAT IS ASSERTED

- Python's `_decoy_secure_fraction` equals Eq. (41) written out here independently from the paper.
- It is strictly below the section 3.4 known-Y0 estimate wherever the background matters, so a
  revert to the old estimator fails.
- The observed QBER the run reports is the bound's E_mu, in Python and in the browser.
- The decoy algebra exists once per language: no other file carries its own transcription. The
  browser optimiser's private copy, the one a past fix missed, now calls the shared function.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from qsh.config import get_config
from qsh.qkd.bb84 import _decoy_secure_fraction, e_det_for_bound, signal_qber
from qsh.qkd.channel import ChannelParams

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _h2(x: float) -> float:
    return 0.0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def _eq41(eta: float, mu: float, nu: float, e_det: float, y0: float, f: float, q: float) -> float:
    """Ma, Qi, Zhao & Lo, PRA 72, 012326 (2005), section 3.5, Eq. (41) into Eq. (1)."""
    q_mu = y0 + 1 - math.exp(-eta * mu)
    q_nu = y0 + 1 - math.exp(-eta * nu)
    e_mu = (0.5 * y0 + e_det * (1 - math.exp(-eta * mu))) / q_mu
    e_nu = (0.5 * y0 + e_det * (1 - math.exp(-eta * nu))) / q_nu
    y1 = mu / (mu * nu - nu * nu) * (q_nu * math.exp(nu) - q_mu * math.exp(mu) * nu * nu / (mu * mu))
    if y1 <= 0:
        return 0.0
    e1 = min(0.5, e_nu * q_nu * math.exp(nu) / (y1 * nu))
    return max(0.0, q * (-q_mu * f * _h2(e_mu) + y1 * mu * math.exp(-mu) * (1 - _h2(e1))))


def _known_y0(eta: float, mu: float, nu: float, e_det: float, y0: float, f: float, q: float) -> float:
    """Section 3.4's estimator with the model's Y0 - what shipped until Round 75."""
    q_mu = y0 + 1 - math.exp(-eta * mu)
    q_nu = y0 + 1 - math.exp(-eta * nu)
    e_mu = (0.5 * y0 + e_det * (1 - math.exp(-eta * mu))) / q_mu
    e_nu = (0.5 * y0 + e_det * (1 - math.exp(-eta * nu))) / q_nu
    y1 = mu / (mu * nu - nu * nu) * (q_nu * math.exp(nu) - q_mu * math.exp(mu) * nu * nu / (mu * mu)
                                     - (mu * mu - nu * nu) / (mu * mu) * y0)
    e1 = min(0.5, (e_nu * q_nu * math.exp(nu) - 0.5 * y0) / (y1 * nu))
    return max(0.0, q * (-q_mu * f * _h2(e_mu) + y1 * mu * math.exp(-mu) * (1 - _h2(e1))))


def _shipped():
    c = get_config().qkd
    mu = c.mu_signal
    return mu, min(max(1e-4, c.mu_decoy), mu * 0.9), c.error_correction_eff


@pytest.mark.parametrize("distance_km", [1.0, 25.0, 50.0, 100.0, 120.0, 130.0])
@pytest.mark.parametrize("eve", [0.0, 0.1])
def test_the_bound_is_eq_41(distance_km: float, eve: float) -> None:
    mu, nu, f = _shipped()
    p = ChannelParams(distance_km=distance_km, eve_intercept_fraction=eve)
    args = (p.transmittance(), mu, nu, e_det_for_bound(p), 2 * p.dark_count_prob, f, 0.5)
    assert _decoy_secure_fraction(*args)[0] == pytest.approx(_eq41(*args), rel=1e-12, abs=1e-18)


def test_the_bound_is_below_the_known_vacuum_yield_estimate() -> None:
    mu, nu, f = _shipped()
    for distance_km, at_least in ((50.0, 1.005), (100.0, 1.10), (120.0, 1.30)):
        p = ChannelParams(distance_km=distance_km)
        args = (p.transmittance(), mu, nu, e_det_for_bound(p), 2 * p.dark_count_prob, f, 0.5)
        ours, old = _decoy_secure_fraction(*args)[0], _known_y0(*args)
        assert old / ours >= at_least, f"{distance_km} km: {old:.4e} / {ours:.4e} = {old / ours:.4f}"


@pytest.mark.parametrize("distance_km", [10.0, 50.0, 100.0, 140.0])
def test_the_qber_shown_is_the_qber_the_bound_uses(distance_km: float) -> None:
    mu, nu, f = _shipped()
    p = ChannelParams(distance_km=distance_km, eve_intercept_fraction=0.05)
    e_mu = _decoy_secure_fraction(p.transmittance(), mu, nu, e_det_for_bound(p),
                                  2 * p.dark_count_prob, f, 0.5)[1]
    assert signal_qber(p, mu) == pytest.approx(e_mu, rel=1e-15)


def test_the_browser_shows_the_same_qber_its_bound_uses() -> None:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            "export { signalQber, decoySecureFraction, eDetForBound, transmittance, FIBER } from "
            f"{json.dumps(str(WEB / 'src/lib/quantumSim'))};\n"
            f"export {{ qkdCfg }} from {json.dumps(str(WEB / 'src/lib/config'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True,
                       capture_output=True)
        runner.write_text(
            f"import * as m from {json.dumps(bundle.as_uri())};\n"
            "const c = m.qkdCfg(); const mu = c.mu_signal, nu = Math.min(Math.max(1e-4, c.mu_decoy), mu * 0.9);\n"
            "const out = [10, 50, 100, 140].map((km) => {\n"
            "  const p = { ...m.FIBER, distanceKm: km, eveFraction: 0.05 };\n"
            "  const b = m.decoySecureFraction(m.transmittance(p), mu, nu, m.eDetForBound(p), 2 * p.darkCount, c.error_correction_eff, 0.5);\n"
            "  return { shown: m.signalQber(p, mu), used: b.eMu, rate: b.perPulse };\n"
            "});\n"
            "console.log(JSON.stringify(out));\n")
        rows = json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True,
                                         text=True).stdout)
    for km, row in zip([10.0, 50.0, 100.0, 140.0], rows, strict=True):
        assert row["shown"] == pytest.approx(row["used"], rel=1e-15), (km, row)
        mu, nu, f = _shipped()
        p = ChannelParams(distance_km=km, eve_intercept_fraction=0.05)
        py = _decoy_secure_fraction(p.transmittance(), mu, nu, e_det_for_bound(p),
                                    2 * p.dark_count_prob, f, 0.5)[0]
        assert row["rate"] == pytest.approx(py, rel=1e-12, abs=1e-18), (km, row["rate"], py)


def test_the_decoy_algebra_exists_once_per_language() -> None:
    """`mu * nu - nu * nu` is the Y1 bound's denominator: one Python and one TypeScript home."""
    homes = []
    for path in list((REPO / "src").rglob("*.py")) + list((WEB / "src").rglob("*.ts")):
        if re.search(r"mu\s*\*\s*nu\s*-\s*nu\s*\*\s*nu", path.read_text()):
            homes.append(str(path.relative_to(REPO)))
    assert sorted(homes) == ["src/qsh/qkd/bb84.py", "web/src/lib/quantumSim.ts"], homes


def test_both_runs_show_the_signal_qber() -> None:
    """The run's observed QBER is drawn around `signal_qber` / `signalQber`, not `intrinsic_qber`."""
    py = (REPO / "src" / "qsh" / "qkd" / "bb84.py").read_text()
    ts = (WEB / "src" / "lib" / "quantumSim.ts").read_text()
    assert "qber_phys = signal_qber(params, mu)" in py, "the Python decoy path no longer shows E_mu"
    assert "const eDet = signalQber(p, mu);" in ts, "the browser bb84() no longer shows E_mu"
