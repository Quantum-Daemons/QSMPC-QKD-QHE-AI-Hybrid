"""The eavesdropper's disturbance is COMPOSED with the channel error, in both languages, at all four sites.

WHY THIS EXISTS

Until 2026-10-24 both languages ADDED intercept-resend's 0.25 f to the channel error. Two flips on one
bit cancel, so independent error sources compose as a + b - 2ab; the sum overcounted by 2ab. At 50 km
with a full attack the E91 model QBER read 0.2722 where composition gives 0.2649, and S 1.29 for 1.33.
The change had to reach four sites at once - `intrinsic_qber` and `e_det_for_bound` in Python, their
mirrors in `quantumSim.ts` - because the abort reads one and the rate reads the other, and the
cross-language rate test compares the two languages at every Eve fraction.

The same round found four tooltip figures for QSDC secrecy capacity that had been literals since
before two formula corrections, and a log line still quoting the superseded pooled criterion. Those
are now computed by the function the satellite stage runs, which this file also holds in place.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _strip(src: str, py: bool = False) -> str:
    if py:
        src = re.sub(r'"""[\s\S]*?"""', " ", src)
        return re.sub(r"#[^\n]*", " ", src)
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(?<![:\"'`])//[^\n]*", " ", src)


def test_all_four_sites_call_the_one_helper_per_language() -> None:
    py_channel = _strip((REPO / "src/qsh/qkd/channel.py").read_text(), py=True)
    py_bb84 = _strip((REPO / "src/qsh/qkd/bb84.py").read_text(), py=True)
    ts = _strip((WEB / "src/lib/quantumSim.ts").read_text())
    assert re.search(r"def compose_independent_errors\(a: float, b: float\)", py_channel)
    assert re.search(r"return a \+ b - 2\.0 \* a \* b", py_channel)
    assert "compose_independent_errors(base, eve)" in py_channel, "intrinsic_qber does not compose"
    assert "compose_independent_errors(params.misalignment" in py_bb84, "e_det_for_bound does not compose"
    assert re.search(r"export function composeIndependentErrors\(a: number, b: number\)", ts)
    assert "composeIndependentErrors(base, 0.25 * p.eveFraction)" in ts, "intrinsicQber does not compose"
    assert "composeIndependentErrors(p.misalignment, 0.25 * p.eveFraction)" in ts, "eDetForBound does not compose"
    # The linear forms are gone from both languages.
    assert "base + eve)" not in py_channel and "misalignment + 0.25 * params.eve_intercept_fraction" not in py_bb84
    assert "base + 0.25 * p.eveFraction" not in ts and "p.misalignment + 0.25 * p.eveFraction)" not in ts


def test_python_values_compose() -> None:
    from qsh.qkd.bb84 import e_det_for_bound
    from qsh.qkd.channel import ChannelParams, compose_independent_errors, intrinsic_qber

    assert compose_independent_errors(0.1, 0.25) == pytest.approx(0.3, abs=1e-15)
    p0 = ChannelParams()
    for f in (0.0, 0.2, 0.5, 1.0):
        p = ChannelParams(eve_intercept_fraction=f)
        base = intrinsic_qber(p0)
        assert intrinsic_qber(p) == pytest.approx(base + f / 4 - 2 * base * f / 4, abs=1e-15)
        assert e_det_for_bound(p) == pytest.approx(
            p.misalignment + f / 4 - 2 * p.misalignment * f / 4, abs=1e-15)


def _browser(expr: str) -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        entry = Path(tmp) / "probe.ts"
        entry.write_text(
            f'import * as q from {json.dumps(str(WEB / "src/lib/quantumSim"))};\n'
            f"console.log(JSON.stringify({expr}));\n")
        js = subprocess.run([str(ESBUILD), str(entry), "--bundle", "--platform=node", "--log-level=error"],
                            cwd=WEB, check=True, capture_output=True, text=True).stdout
        out = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True).stdout
        return json.loads(out.strip().splitlines()[-1])


def test_the_browser_matches_python_and_keeps_the_chsh_relation() -> None:
    from qsh.qkd.channel import ChannelParams, intrinsic_qber

    got = _browser(
        "{ eve1: q.intrinsicQber({ ...q.FIBER, distanceKm: 50, misalignment: 0.015, eveFraction: 1 }),"
        "  e91: (() => { const r = q.e91(20000000, { ...q.FIBER, distanceKm: 50, misalignment: 0.015,"
        "        eveFraction: 1 }, 1); return { s: r.chshS, e0: r.e0 }; })() }")
    py = intrinsic_qber(ChannelParams(distance_km=50, misalignment=0.015, eve_intercept_fraction=1.0))
    assert got["eve1"] == pytest.approx(py, abs=1e-12), (got, py)
    # S = 2 sqrt(2) (1 - 2 e0) holds with an eavesdropper too, at the composed error rate.
    assert got["e91"]["s"] == pytest.approx(2 * math.sqrt(2) * (1 - 2 * got["e91"]["e0"]), abs=1e-12)
    # 0.264775 on Round 76's Rusca detector set; 0.264872 on the set before it, whose dark counts
    # added 2e-4 to the base error at this misalignment.
    assert got["e91"]["e0"] == pytest.approx(0.264775, abs=5e-6), "the composed Eve=1 E91 error rate moved"


def test_no_tooltip_quotes_a_secrecy_capacity_literal() -> None:
    panel = _strip((WEB / "src/components/ParameterPanel.tsx").read_text())
    literals = re.findall(r"C_s\s*(?:=|at)\s*[+-]?\d+\.\d+", panel)
    assert not literals, f"a QSDC figure is quoted as a literal again: {literals}"
    assert panel.count("csAt(") >= 3, "the tooltips no longer compute their figures"
    stage = _strip((WEB / "src/lib/clientCompute.ts").read_text())
    assert "requires h(QBER) < 1/(1+g)" not in stage, "the superseded pooled criterion is back in the log"
    assert "qsdcLink(satChannel(config))" in stage, "the satellite stage no longer uses the shared link function"
