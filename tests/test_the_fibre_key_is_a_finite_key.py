"""Every fibre key is the composable finite key of a block, and every site reports the same one.

WHY THIS EXISTS (Round 80, the user's choice). After Round 79 the satellite page reported a
composable finite key while the fibre pages reported the asymptotic bound times the pulses - 28,534
bits for the 2e7-pulse block at 50 km, where the same block's composable key is 9,811. Two pages, two
meanings of "secure bits". By the user's choice the fibre key is now a block's finite key too, at
every fibre site: the use-case pipelines, the two-party demo and its Validations page, and the Python
API, in both languages. The model computes it (`run_bb84` / `bb84()`), so no caller can report the
other quantity; E91, which has no decoy bound here, keeps its asymptotic count.

WHAT IS ASSERTED

- `run_bb84`'s and `bb84()`'s key is Rusca et al.'s 1-decoy bound on the block's expected counts,
  for every block kind ("pa_bits", "seconds", "pulses"), and the two languages agree to the bit and on
  the block's size, at several distances, with and without an eavesdropper.
- The configured block is what every fibre caller takes when it names none: the API's demo, the
  two-party benchmark and the pipeline stage report one key for one channel.
- The key is below the asymptotic product, grows with the block, and is zero where the bound has no
  key; the stage then says how many blocks, pooled, would carry one.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import pytest

from qsh.config import get_config
from qsh.qkd.bb84 import QKDParams, run_bb84
from qsh.qkd.channel import ChannelParams
from qsh.qkd.finite_key import (
    FiniteKeyParams,
    fibre_block_pulses,
    finite_key_length,
    finite_key_over,
    params_for_source,
)

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

#: (km, eve, kind) cases both languages run.
CASES = [(km, eve, kind) for km in (10.0, 50.0, 100.0) for eve in (0.0, 0.05) for kind in ("pa_bits", "seconds", "pulses")]


def _block(ch: ChannelParams, kind: str, run_pulses: float = 2e7) -> float:
    q = get_config().qkd
    return fibre_block_pulses(ch, params_for_source(QKDParams()), kind=kind, run_pulses=run_pulses,
                              seconds=q.finite_key_block_seconds, pa_bits=q.finite_key_block_pa_bits)


def _python(km: float, eve: float, kind: str) -> dict:
    ch = ChannelParams(distance_km=km, eve_intercept_fraction=eve)
    block = _block(ch, kind)
    r = run_bb84(200_000, ch, seed=1, qkd_params=QKDParams(), block_pulses=block)
    return dict(bits=r.secure_key_bits, block=r.block_pulses, skr=r.secret_key_rate)


@pytest.fixture(scope="module")
def browser() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            f"export {{ bb84, FIBER }} from {json.dumps(str(WEB / 'src/lib/quantumSim'))};\n"
            f"export {{ fibreBlockPulses, finiteKeySourceOf }} from {json.dumps(str(WEB / 'src/lib/finiteKey'))};\n"
            f"export {{ qkdCfg, webCfg }} from {json.dumps(str(WEB / 'src/lib/config'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + f"""
const q = m.qkdCfg();
const cases = {json.dumps(CASES)};
const out = cases.map(([km, eve, kind]) => {{
  const ch = {{ ...m.FIBER, distanceKm: km, eveFraction: eve }};
  const block = m.fibreBlockPulses(ch, m.finiteKeySourceOf(ch, q.finite_key_eps_sec), {{ kind, runPulses: 2e7,
    seconds: q.finite_key_block_seconds, paBits: q.finite_key_block_pa_bits }});
  const r = m.bb84(200000, ch, 1, undefined, {{ blockPulses: block }});
  return {{ bits: r.secureKeyBits, block: r.blockPulses, skr: r.secretKeyRate }};
}});
// The configured default, as every fibre caller that names no block takes it (the two-party demo).
const demo = [50, 100].map((km) => {{
  const r = m.bb84(m.webCfg().demo_alice_bob_pulses, {{ ...m.FIBER, distanceKm: km }}, 1);
  return {{ bits: r.secureKeyBits, block: r.blockPulses }};
}});
console.log(JSON.stringify({{ cases: out, demo }}));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True,
                                         text=True).stdout)


@pytest.mark.parametrize("i", range(len(CASES)), ids=[f"{km:g}km-eve{eve}-{k}" for km, eve, k in CASES])
def test_both_languages_compute_the_same_fibre_key(browser, i) -> None:
    km, eve, kind = CASES[i]
    py, js = _python(km, eve, kind), browser["cases"][i]
    assert js["block"] == pytest.approx(py["block"], rel=1e-12)
    assert js["bits"] == py["bits"], (km, eve, kind, js["bits"], py["bits"])


def test_the_key_is_rusca_s_bound_on_the_block() -> None:
    ch = ChannelParams()
    for kind in ("pa_bits", "seconds", "pulses"):
        block = _block(ch, kind)
        r = run_bb84(200_000, ch, seed=1, qkd_params=QKDParams(), block_pulses=block)
        bound = finite_key_length(block, ch, FiniteKeyParams())
        assert r.secure_key_bits == int(bound.key_bits), kind
        assert 0 < r.secure_key_bits < r.secret_key_rate * block, kind     # below the asymptotic product


def test_every_fibre_caller_takes_the_configured_block(browser) -> None:
    """The API's demo, the two-party benchmark and the browser's demo: one channel, one block, one key."""
    import sys

    sys.path.insert(0, str(REPO))
    from api.inference_service import qkd_demo

    q = get_config()
    for km, js in zip((50.0, 100.0), browser["demo"], strict=True):
        ch = ChannelParams(distance_km=km)
        block = _block(ch, q.qkd.finite_key_block, run_pulses=q.web.demo_alice_bob_pulses)
        expected = finite_key_over([(block, ch)], params_for_source(QKDParams()))
        bits = int(expected.key_bits) if math.isfinite(expected.key_bits) and expected.key_bits > 0 else 0
        assert qkd_demo(km, 0.0)["secure_key_bits"] == bits == js["bits"], km
        assert js["block"] == pytest.approx(block, rel=1e-12)
        if km == 50.0:                                  # the benchmark's own channel
            from qsh.benchmark.alice_bob_e2e import run_alice_bob_e2e

            assert run_alice_bob_e2e(seed=0)["qkd_secure_key_bits"] == bits


def test_the_stage_reports_its_block_and_what_a_key_needs() -> None:
    from qsh.usecases import telecom_ids

    m = telecom_ids.run_demo({"seed": 1})["metrics"]
    q = get_config().qkd
    assert m["qkd_block"] == q.finite_key_block
    assert m["secure_key_bits"] == m["qkd_finite_key_bits"] > 256 and m["qkd_mixed"] == 1
    assert m["qkd_finite_key_bound"] == "Rusca et al. 2018 (one decoy)" and m["qkd_eps_sec"] == 1e-9
    assert m["qkd_block_units_for_key"] == 1
    short = telecom_ids.run_demo({"seed": 1, "qkd_block": "pulses", "qkd_distance_km": 100.0})["metrics"]
    assert short["secure_key_bits"] == 0 and short["qkd_mixed"] == 0
    assert short["qkd_block_units_for_key"] == 2           # two 2e7-pulse blocks at 100 km carry one


def test_the_key_grows_with_the_block() -> None:
    ch = ChannelParams()
    keys = [run_bb84(200_000, ch, seed=1, qkd_params=QKDParams(), block_pulses=n).secure_key_bits
            for n in (2e7, 1e8, 1e9, 1e10)]
    assert keys == sorted(keys) and keys[0] > 0
