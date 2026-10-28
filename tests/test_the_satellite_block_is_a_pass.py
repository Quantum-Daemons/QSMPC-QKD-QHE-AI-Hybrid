"""The satellite's key is the composable finite key of a pass, and both languages compute the same one.

WHY THIS EXISTS (Round 79, the user's choice). The satellite stage reported the asymptotic bound
times a fixed 2e7 pulses - a fifth of a second of Micius's 100 MHz clock - as its key. The user asked
for the block to be sized by link time, with a finite-key length, and for the published practice to
choose between seconds at the page's range and a whole pass. Every flown experiment found computes
its finite key over one pass (Liao et al. 2017 and 2018; Tiangong-2; Khmelev et al. 2023) or over
fixed blocks inside one orbit (Li et al. 2025, Jinan-1); none computes one on seconds of a low-orbit
pass. So the default block is one pass peaking at the page's range, with passes pooled and seconds at
the range offered beside it (`QKDConfig.free_space_block`).

WHAT IS ASSERTED

- The Python pass (`qkd/satellite_pass.py`) is the browser's (`satNetwork.ts::simulatePass`): the
  same duration and the same finite key over it, for every downlink profile, both decoy settings and
  every block kind; and the browser's bound (`finiteKey.ts`) is Python's to 1e-9.
- The stage's key IS the block's finite key: one pass by default; zero under the sampled run's abort;
  pooling passes carries more than the same passes one at a time; ten seconds at 800 km carry none,
  and the run says how many seconds would.
- Against flown passes, with nothing fitted. Liao's 19 December 2016 pass at 1e-9: 0.55 of the
  published 300,939 bits. Khmelev et al.'s Micius-Zvenigorod session: the same fraction, 0.56, of its
  629,000 bits (computed with Chernoff bounds), so the gap is the analysis, consistently. Jinan-1's
  four published passes (Li et al. 2025, Table 1): within a factor of 2.5, and within 25% on the pass
  whose count rate fixed the profile's one calibrated constant.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import pytest

from qsh.config.profiles import load_profiles
from qsh.orchestration.stages import Context
from qsh.qkd.channel import link_elevation_deg
from qsh.qkd.finite_key import block_counts, blocks_for_positive_key, finite_key_over
from qsh.qkd.satellite_pass import simulate_pass
from qsh.usecases import satellite_comms
from qsh.usecases.satellite_comms import (
    _sat_block,
    _sat_channel,
    _sat_finite_key_params,
    _sat_source,
)

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

PROFILES = list(load_profiles("downlink"))
#: (config, label) cases both languages run: every profile with and without the vacuum at 800 km, and
#: every block kind on the default link.
CASES = ([({"sat_profile": pid, "sat_vacuum_decoy": vac}, f"{pid}/vac{vac}") for pid in PROFILES for vac in (0, 1)]
         + [({"sat_block": "passes", "sat_block_passes": 3}, "3 passes"),
            ({"sat_block": "seconds", "sat_block_seconds": 10}, "10 s"),
            ({"sat_block": "seconds", "sat_block_seconds": 100}, "100 s"),
            ({"sat_eps_sec": 1e-10}, "eps 1e-10"),
            ({"sat_distance_km": 1500}, "1500 km"),
            ({"sat_distance_km": 600, "sat_min_elevation_deg": 15}, "600 km, 15 deg")])


def _python(cfg: dict) -> dict:
    ctx = Context(usecase="satellite_comms", config=dict(cfg))
    ch, src = _sat_channel(ctx), _sat_source(ctx)
    block = _sat_block(ctx, ch)
    p = _sat_finite_key_params(ctx, src)
    z, x = block_counts(block.unit, p)
    r = finite_key_over(((n * block.copies, c) for n, c in block.unit), p)
    return dict(key=r.key_bits, units=blocks_for_positive_key(z, x, p), seconds=block.seconds,
                pulses=block.pulses, top=block.max_elevation_deg)


@pytest.fixture(scope="module")
def browser() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            f"export * from {json.dumps(str(WEB / 'src/lib/finiteKey'))};\n"
            f"export {{ satChannel, satBlock }} from {json.dumps(str(WEB / 'src/lib/satLink'))};\n"
            f"export {{ liaoPass }} from {json.dumps(str(WEB / 'src/lib/satNetwork'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + f"""
const cases = {json.dumps([c for c, _ in CASES])};
const out = cases.map((cfg) => {{
  const ch = m.satChannel(cfg);
  const block = m.satBlock(cfg, ch);
  const src = m.finiteKeySourceOf(ch, Number(cfg.sat_eps_sec ?? 1e-9));
  const [z, x] = m.blockCounts(block.unit, src);
  const r = m.keyLength(m.scaleCounts(z, block.copies), m.scaleCounts(x, block.copies), src);
  return {{ key: Number.isFinite(r.keyBits) ? r.keyBits : null, units: m.blocksForPositiveKey(z, x, src),
           seconds: block.copies * block.unitSeconds,
           pulses: block.copies * block.unit.reduce((a, [n]) => a + n, 0), top: block.maxElevationDeg }};
}});
const link = m.satChannel({{}});
const liao = m.liaoPass(link);
const src = m.finiteKeySourceOf(link, 1e-9);
const lk = m.keyLength(...m.blockCounts(liao.segments, src), src);
console.log(JSON.stringify({{ cases: out, liao: {{ key: lk.keyBits, s: liao.durationS }} }}));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True,
                                         text=True).stdout)


@pytest.mark.parametrize("i", range(len(CASES)), ids=[label for _, label in CASES])
def test_both_languages_compute_the_same_block_key(browser, i) -> None:
    cfg, label = CASES[i]
    py, js = _python(cfg), browser["cases"][i]
    assert js["seconds"] == pytest.approx(py["seconds"], rel=1e-12), label
    assert js["pulses"] == pytest.approx(py["pulses"], rel=1e-12), label
    assert js["units"] == py["units"], label
    if math.isfinite(py["key"]):
        assert js["key"] == pytest.approx(py["key"], rel=1e-9, abs=1e-6), label
    else:
        assert js["key"] is None, label
    if py["top"] is None:
        assert js["top"] is None
    else:
        assert js["top"] == pytest.approx(py["top"], rel=1e-12), label


def test_the_python_pass_is_liaos_and_the_test_oracles(browser) -> None:
    """Liao's window through `simulate_pass` at the browser's step agrees with the browser, and with
    the independent 4,000-step integration of the bound's own test file to a tenth of a per cent."""
    import sys

    sys.path.insert(0, str(REPO / "tests"))
    from test_the_vacuum_decoy_bound_is_lim_2014 import _liao_pass

    ch = _sat_channel(Context(usecase="satellite_comms", config={}))
    fs = ch.free_space
    liao = simulate_pass(ch, link_elevation_deg(fs, 645.0), window_elevation_deg=link_elevation_deg(fs, 1200.0),
                         step_s=0.25)
    key = finite_key_over(liao.segments).key_bits
    assert liao.duration_s == pytest.approx(browser["liao"]["s"], rel=1e-12)
    assert key == pytest.approx(browser["liao"]["key"], rel=1e-9)
    assert key == pytest.approx(finite_key_over(_liao_pass()).key_bits, rel=1e-3)
    assert 0.5 < key / 300_939 < 0.6


def _run(cfg: dict) -> dict:
    return satellite_comms.run_demo({"seed": 1, **cfg})["metrics"]


def test_the_stage_reports_the_pass_as_its_key() -> None:
    m = _run({})
    assert m["qkd_block"] == "pass" and m["qkd_block_copies"] == 1
    assert m["secure_key_bits"] == m["qkd_finite_key_bits"] > 256 and m["qkd_mixed"] == 1
    assert m["qkd_finite_key_bound"].startswith("Lim et al. 2014") and m["qkd_eps_sec"] == 1e-9
    # A pass peaking at the default 800 km: 35.8 degrees, about 408 s above the 10 degree mask.
    assert m["sat_pass_max_elevation_deg"] == pytest.approx(35.84, abs=0.01)
    assert m["qkd_block_s"] == pytest.approx(408.5, abs=0.5)
    assert m["qkd_block_units_for_key"] == 1


def test_pooling_beats_the_same_passes_one_at_a_time() -> None:
    """Sidhu et al. 2022: the key of M pooled passes is at least M times one pass's."""
    one, three = _run({})["secure_key_bits"], _run({"sat_block": "passes", "sat_block_passes": 3})
    assert three["secure_key_bits"] > 3 * one
    assert three["qkd_block_s"] == pytest.approx(3 * _run({})["qkd_block_s"], abs=0.2)


def test_seconds_at_a_range_carry_no_key_where_a_pass_does() -> None:
    m = _run({"sat_block": "seconds"})
    assert m["qkd_block_s"] == 10.0 and m["secure_key_bits"] == 0 and m["qkd_mixed"] == 0
    assert m["qkd_block_units_for_key"] == 33          # seconds at 800 km before the bound turns
    assert _run({"sat_block": "seconds", "sat_block_seconds": 33})["secure_key_bits"] > 0


def test_the_sampled_abort_still_decides() -> None:
    m = _run({"sat_eve": 0.5})
    assert m["qber"] > 0.11 and m["secure_key_bits"] == 0 and m["qkd_mixed"] == 0


def _pass_key(profile: str, top: float, window: float | None = None) -> float:
    ctx = Context(usecase="satellite_comms", config={"sat_profile": profile})
    ch, src = _sat_channel(ctx), _sat_source(ctx)
    p = simulate_pass(ch, top, window_elevation_deg=window)
    return finite_key_over(p.segments, _sat_finite_key_params(ctx, src)).key_bits


def test_khmelevs_session_shows_the_same_analysis_gap_as_liaos() -> None:
    """Khmelev et al. (arXiv:2310.17476): 629,000 bits at 1e-9 with Chernoff bounds, over the session
    whose sifted rate fell from 20.7 kbit/s at 600 km to 2.9 at 1,100 km."""
    ch = _sat_channel(Context(usecase="satellite_comms", config={"sat_profile": "micius_2022_zvenigorod"}))
    fs = ch.free_space
    ratio = _pass_key("micius_2022_zvenigorod", link_elevation_deg(fs, 600.0), link_elevation_deg(fs, 1100.0)) / 629_000
    assert ratio == pytest.approx(0.564, abs=0.02)
    ch0 = _sat_channel(Context(usecase="satellite_comms", config={}))
    liao_ratio = finite_key_over(simulate_pass(ch0, link_elevation_deg(ch0.free_space, 645.0),
                                               window_elevation_deg=link_elevation_deg(ch0.free_space, 1200.0),
                                               step_s=0.25).segments).key_bits / 300_939
    assert abs(ratio - liao_ratio) < 0.05, (ratio, liao_ratio)


#: Li et al., Nature 2025 (Jinan-1), Table 1: highest elevation and final key per pass.
JINAN_PASSES = {"20 Sep": (41.8, 330_240), "25 Sep": (34.1, 406_784), "28 Sep": (77.7, 571_392),
                "29 Sep": (52.7, 436_096)}


def test_jinan1_passes_are_within_a_factor_of_two_and_a_half() -> None:
    ratios = {d: _pass_key("jinan1_2022_jinan", top) / fin for d, (top, fin) in JINAN_PASSES.items()}
    assert all(0.8 < r < 2.5 for r in ratios.values()), ratios
    # The pass whose peak count rate fixed the profile's one calibrated constant (Round 78).
    assert ratios["25 Sep"] == pytest.approx(1.0, abs=0.25), ratios
