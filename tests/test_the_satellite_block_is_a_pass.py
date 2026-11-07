"""The satellite's key is the composable finite key of a pass, and both languages compute the same one.

WHY THIS EXISTS (Round 79, the user's choice). The satellite stage reported the asymptotic bound
times a fixed 2e7 pulses - a fifth of a second of Micius's 100 MHz clock - as its key. The user asked
for the block to be sized by link time, with a finite-key length, and for the published practice to
choose between seconds at the page's range and a whole pass. Every flown experiment found computes
its finite key over one pass (Liao et al. 2017 and 2018; Tiangong-2; Khmelev et al. 2024) or over
fixed blocks inside one orbit (Li et al. 2025, Jinan-1); none computes one on seconds of a low-orbit
pass. So the default block is one pass peaking at the page's range, with passes pooled and seconds at
the range offered beside it (`QKDConfig.free_space_block`).

WHAT IS ASSERTED

- The Python pass (`qkd/satellite_pass.py`) is the browser's (`satNetwork.ts::simulatePass`): the
  same duration and the same finite key over it, for every downlink profile, both decoy settings and
  every block kind, on one key basis and on both (Round 84); and the browser's bound (`finiteKey.ts`)
  is Python's to 1e-9.
- The stage's key IS the block's finite key: one pass by default; zero under the sampled run's abort;
  pooling passes carries more than the same passes one at a time; ten seconds at 800 km carry a
  hundredth of a pass's key at most - none under the published Hoeffding statistics - and the run
  says how many seconds a key needs.
- Against flown passes, with nothing fitted, under each statistics and each key basis (Round 84).
  Lim et al. 2014's protocol keys the Z basis only; the published keys came from both bases (Liao et
  al. 2017's from every sifted bit; Khmelev et al. 2024: "both bases are used for key generation").
  Under Chernoff-type statistics, the satellite's default, Liao's 19 December 2016 pass at 1e-9 reads
  0.75 of the published 300,939 bits on one basis and 1.50 on both, and the both-bases key is within 3
  percent of the asymptotic figure: the drop from the asymptotic row to the one-basis row is 95 percent
  the basis convention, not the statistics (54 percent under Hoeffding, which reads 0.55 and 1.08).
  Khmelev et al.'s Micius-Zvenigorod session reads 1.34 and 2.67 of its final 310,400 bits (1.40 and 2.80
  until Round 86 counted it from the 28 deg where their tracking started), which
  subtract a detector-efficiency mismatch the model does not have (629,000 bits before it). Jinan-1's
  four passes (arXiv:2408.10994v1, Table 1): within a factor of 2.5 under Hoeffding, and within 25% on
  the pass whose count rate fixed the profile's one calibrated constant; within a factor of 3 under
  the default, whose whole-pass key the preprint's 500-kbit blocks cannot reach.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from qsh.config.profiles import load_profiles
from qsh.orchestration.stages import Context
from qsh.qkd.channel import link_elevation_deg
from qsh.qkd.finite_key import (
    VacuumDecoyParams,
    block_counts,
    blocks_for_positive_key,
    finite_key_over,
    key_length,
)
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
    # Both key bases (Round 84), on the same block.
    both = key_length(z.scaled(block.copies), x.scaled(block.copies), replace(p, key_basis="both"))
    return dict(key=r.key_bits, both=both.key_bits, both_bases=both.basis_keys,
                units=blocks_for_positive_key(z, x, p), seconds=block.seconds,
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
            f"export {{ liaoPass, simulatePass }} from {json.dumps(str(WEB / 'src/lib/satNetwork'))};\n"
            f"export {{ elevationDeg }} from {json.dumps(str(WEB / 'src/lib/quantumSim'))};\n"
            f"export {{ qkdCfg }} from {json.dumps(str(WEB / 'src/lib/config'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + f"""
const cases = {json.dumps([c for c, _ in CASES])};
const out = cases.map((cfg) => {{
  const ch = m.satChannel(cfg);
  const block = m.satBlock(cfg, ch);
  // The satellite's own statistics, as the page's stage passes them (the #78 fix): without them the
  // bound falls back to the fibre's default and the two languages part.
  const src = m.finiteKeySourceOf(ch, Number(cfg.sat_eps_sec ?? 1e-9), undefined,
                                  cfg.sat_fk_deviation ?? m.qkdCfg().free_space_finite_key_deviation);
  const [z, x] = m.blockCounts(block.unit, src);
  const r = m.keyLength(m.scaleCounts(z, block.copies), m.scaleCounts(x, block.copies), src);
  // Both key bases (Round 84): the same block, the source's own option.
  const b = m.keyLength(m.scaleCounts(z, block.copies), m.scaleCounts(x, block.copies), {{ ...src, keyBasis: "both" }});
  const fin = (v) => (Number.isFinite(v) ? v : null);
  return {{ key: fin(r.keyBits), both: fin(b.keyBits), bothBases: (b.basisKeys ?? []).map(fin),
           units: m.blocksForPositiveKey(z, x, src),
           seconds: block.copies * block.unitSeconds,
           pulses: block.copies * block.unit.reduce((a, [n]) => a + n, 0), top: block.maxElevationDeg }};
}});
const link = m.satChannel({{}});
const liao = m.liaoPass(link);
const keys = (ch, segs, dev) => {{
  const one = m.finiteKeySourceOf(ch, 1e-9, undefined, dev);
  const both = m.finiteKeySourceOf(ch, 1e-9, undefined, dev, "both");
  const [z, x] = m.blockCounts(segs, one);
  return {{ one: m.keyLength(z, x, one).keyBits, both: m.keyLength(z, x, both).keyBits }};
}};
const src = m.finiteKeySourceOf(link, 1e-9, undefined, m.qkdCfg().free_space_finite_key_deviation);
const lk = m.keyLength(...m.blockCounts(liao.segments, src), src);
// Khmelev et al.'s session on its own profile: 600 km at the top, the window at 1,100 km.
const zv = m.satChannel({{ sat_profile: "micius_2022_zvenigorod", sat_eve: 0 }});
const zfs = zv.freeSpace;
const kh = m.simulatePass(zv, m.elevationDeg(600, zfs.orbitAltitudeKm, zfs.stationAltitudeKm),
  {{ windowElevationDeg: m.elevationDeg(1100, zfs.orbitAltitudeKm, zfs.stationAltitudeKm) }});
const devs = ["chernoff", "hoeffding", "tighter"];
console.log(JSON.stringify({{ cases: out, liao: {{ key: lk.keyBits, s: liao.durationS, asym: liao.keyBits,
  keys: Object.fromEntries(devs.map((d) => [d, keys(link, liao.segments, d)])) }},
  khmelev: {{ s: kh.durationS, keys: Object.fromEntries(devs.map((d) => [d, keys(zv, kh.segments, d)])) }} }}));
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
    # Both key bases (Round 84): the same sum, and the same two keys under it.
    for got, want in [(js["both"], py["both"])] + list(zip(js["bothBases"], py["both_bases"], strict=True)):
        if math.isfinite(want):
            assert got == pytest.approx(want, rel=1e-9, abs=1e-6), label
        else:
            assert got is None, label
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
    # The satellite's own default statistics (the #78 fix), which the browser's stage passes too; the bare
    # class default is the fibre's.
    from qsh.config import get_config

    sat = VacuumDecoyParams(deviation=get_config().qkd.free_space_finite_key_deviation)
    key = finite_key_over(liao.segments, sat).key_bits
    assert liao.duration_s == pytest.approx(browser["liao"]["s"], rel=1e-12)
    assert key == pytest.approx(browser["liao"]["key"], rel=1e-9)
    assert key == pytest.approx(finite_key_over(_liao_pass(), sat).key_bits, rel=1e-3)
    assert 0.7 < key / 300_939 < 0.8                     # the satellite's default statistics
    published = finite_key_over(liao.segments, VacuumDecoyParams(deviation="hoeffding")).key_bits
    assert 0.5 < published / 300_939 < 0.6               # Hoeffding, as Lim et al. published it


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
    # Four significant figures each (Round 80), so 1225 against 3 x 408.5.
    assert three["qkd_block_s"] == pytest.approx(3 * _run({})["qkd_block_s"], rel=1e-3)


def test_seconds_at_a_range_carry_far_less_than_a_pass() -> None:
    """Ten seconds held at 800 km against the pass peaking there. Under the published Hoeffding
    statistics they carry no key and a key needs 33; under the default, which bounds the small counts
    by their own Chernoff deviation (Round 82), they carry about a hundredth of the pass's and a key
    needs 7."""
    m = _run({"sat_block": "seconds", "sat_fk_deviation": "hoeffding"})
    assert m["qkd_block_s"] == 10.0 and m["secure_key_bits"] == 0 and m["qkd_mixed"] == 0
    assert m["qkd_block_units_for_key"] == 33          # seconds at 800 km before the bound turns
    assert _run({"sat_block": "seconds", "sat_block_seconds": 33,
                 "sat_fk_deviation": "hoeffding"})["secure_key_bits"] > 0
    d, whole = _run({"sat_block": "seconds"}), _run({})["secure_key_bits"]
    assert d["qkd_fk_deviation"] == "chernoff" and d["qkd_block_units_for_key"] == 7
    assert 0 < d["secure_key_bits"] < 0.02 * whole, (d["secure_key_bits"], whole)


def test_the_sampled_abort_still_decides() -> None:
    m = _run({"sat_eve": 0.5})
    assert m["qber"] > 0.11 and m["secure_key_bits"] == 0 and m["qkd_mixed"] == 0


def _pass_key(profile: str, top: float, window: float | None = None, deviation: str | None = None) -> float:
    ctx = Context(usecase="satellite_comms", config={"sat_profile": profile, "sat_fk_deviation": deviation})
    ch, src = _sat_channel(ctx), _sat_source(ctx)
    p = simulate_pass(ch, top, window_elevation_deg=window)
    return finite_key_over(p.segments, _sat_finite_key_params(ctx, src)).key_bits


#: The published keys the Network tab sets the model beside. Liao et al., Nature 549, 43 (2017): the
#: pass of 19 December 2016. Khmelev et al., Opt. Express 32(7), 11964 (2024), the Micius-Zvenigorod
#: session of 1 March 2022, quoted from arXiv:2310.17476v1: the final key, after the detector-
#: efficiency-mismatch analysis (the abstract's "310 kbits"), and the key before that analysis, both
#: with Chernoff bounds at 1e-9 and error correction at f_EC 1.44.
LIAO_FINAL, KHMELEV_FINAL, KHMELEV_BEFORE_MISMATCH = 300_939, 310_400, 629_000

#: (one basis, both bases) as fractions of Liao's key and of Khmelev et al.'s final key, per statistics.
PUBLISHED_FRACTIONS = {"chernoff": ((0.751, 1.498), (1.402, 2.798)),
                       "hoeffding": ((0.546, 1.082), (1.143, 2.273))}


def _two_keys(cfg: dict, segments, deviation: str) -> tuple[float, float]:
    """The pass's key on one basis and on both, under the profile's own source and the run's bound."""
    ctx = Context(usecase="satellite_comms", config={**cfg, "sat_fk_deviation": deviation})
    p = _sat_finite_key_params(ctx, _sat_source(ctx))
    z, x = block_counts(segments, p)
    return key_length(z, x, p).key_bits, key_length(z, x, replace(p, key_basis="both")).key_bits


def test_the_published_keys_were_counted_on_both_bases(browser) -> None:
    """Liao et al. 2017 and Khmelev et al. 2024 keyed both bases; Lim et al.'s protocol keys one.

    Round 82 read the one-basis key's fractions of the published keys - 0.75 of Liao's and 0.69 of
    Khmelev's 629,000 under Chernoff, 0.55 and 0.56 under Hoeffding - as gaps the statistics partly
    closed. The fractions stand; the reading did not. The asymptotic row keys every sifted signal
    detection in both bases, so the step from it to the one-basis finite key is mostly the basis
    convention: 95 percent under Chernoff, 54 under Hoeffding. Counted as the papers counted, the
    model's key exceeds both published keys, and what remains between them is the analysis: Liao et
    al. name none, allow for intensity fluctuations and correct errors with a Hamming code; Khmelev et
    al. correct at f_EC 1.44 against 1.16 here and subtract a detector-efficiency mismatch this model
    does not have. Both languages give the same keys."""
    ch0 = _sat_channel(Context(usecase="satellite_comms", config={}))
    liao = simulate_pass(ch0, link_elevation_deg(ch0.free_space, 645.0),
                         window_elevation_deg=link_elevation_deg(ch0.free_space, 1200.0), step_s=0.25)
    zv = _sat_channel(Context(usecase="satellite_comms", config={"sat_profile": "micius_2022_zvenigorod"}))
    kh = simulate_pass(zv, link_elevation_deg(zv.free_space, 600.0),
                       window_elevation_deg=link_elevation_deg(zv.free_space, 1100.0))
    assert kh.duration_s == pytest.approx(browser["khmelev"]["s"], rel=1e-12)
    asym = browser["liao"]["asym"]                       # the Network tab's asymptotic row
    keys = {}
    for dev in ("chernoff", "hoeffding", "tighter"):
        keys[dev] = (_two_keys({}, liao.segments, dev),
                     _two_keys({"sat_profile": "micius_2022_zvenigorod"}, kh.segments, dev))
        (l1, l2), (k1, k2) = keys[dev]
        for got, want in ((browser["liao"]["keys"][dev], (l1, l2)), (browser["khmelev"]["keys"][dev], (k1, k2))):
            assert (got["one"], got["both"]) == pytest.approx(want, rel=1e-9), dev
        # Each basis at half the budget: twice the one-basis key, less the price of halving eps.
        assert 1.98 < l2 / l1 < 2.0 and 1.98 < k2 / k1 < 2.0, (dev, l2 / l1, k2 / k1)
    for dev, ((lo, lb), (ko, kb)) in PUBLISHED_FRACTIONS.items():
        (l1, l2), (k1, k2) = keys[dev]
        assert (l1 / LIAO_FINAL, l2 / LIAO_FINAL) == pytest.approx((lo, lb), abs=0.01), dev
        assert (k1 / KHMELEV_FINAL, k2 / KHMELEV_FINAL) == pytest.approx((ko, kb), abs=0.01), dev
        # The published keys sit between the two conventions: above one basis, below both.
        assert l1 < LIAO_FINAL < l2 and k1 < KHMELEV_BEFORE_MISMATCH < k2, dev
    (l1, l2), (k1, _) = keys["chernoff"]
    assert k1 / KHMELEV_BEFORE_MISMATCH == pytest.approx(0.692, abs=0.01)
    # The drop from the asymptotic row to the one-basis key is mostly the basis convention.
    assert (l2 - l1) / (asym - l1) == pytest.approx(0.952, abs=0.01)
    assert 0.97 < l2 / asym < 1.0
    (h1, h2), _ = keys["hoeffding"]
    assert (h2 - h1) / (asym - h1) == pytest.approx(0.541, abs=0.01)
    # The union of two statistics reads as Chernoff on a pass, as it does on one basis.
    assert keys["tighter"][0][1] == pytest.approx(l2, rel=1e-3)
    assert keys["tighter"][1][1] == pytest.approx(keys["chernoff"][1][1], rel=1e-3)


#: Li et al. (Jinan-1), arXiv:2408.10994v1, Table 1: highest elevation and final key per pass. The
#: version of record (Nature 640, 47 (2025)) reports up to 1.07 Mbit in a single pass and moves the
#: per-pass results to its Extended Data; these are the preprint's.
JINAN_PASSES = {"20 Sep": (41.8, 330_240), "25 Sep": (34.1, 406_784), "28 Sep": (77.7, 571_392),
                "29 Sep": (52.7, 436_096)}


def test_jinan1_passes_are_within_a_factor_of_two_and_a_half() -> None:
    ratios = {d: _pass_key("jinan1_2022_jinan", top, deviation="hoeffding") / fin
              for d, (top, fin) in JINAN_PASSES.items()}
    assert all(0.8 < r < 2.5 for r in ratios.values()), ratios
    # The pass whose peak count rate fixed the profile's one calibrated constant (Round 78).
    assert ratios["25 Sep"] == pytest.approx(1.0, abs=0.25), ratios
    # The default statistics (Round 82) only add key here, and the whole pass's key stays within a
    # factor of 3 of the published one, computed on 500-kbit blocks inside the orbit.
    default = {d: _pass_key("jinan1_2022_jinan", top) / fin for d, (top, fin) in JINAN_PASSES.items()}
    assert all(ratios[d] <= default[d] < 3.0 for d in ratios), (ratios, default)
