"""The satellite Network tab's model (`web/src/lib/satNetwork.ts`) against what was measured.

WHY THIS EXISTS (Round 77). A downlink has no link past the horizon - 2,571 km of slant range at a
500 km orbit - and none below the 10 degree minimum elevation. By the user's request the page now
models what carries a key further: the trusted satellite relay Micius flew (Liao et al., Phys. Rev.
Lett. 120, 030501 (2018)) and a single-plane constellation with inter-satellite links (Vergoossen et
al., arXiv:1903.07845 (2019)), both on top of a pass model. That is the browser's code, the code a
visitor runs, so it is executed here and held to published figures and to closed-form geometry.

WHAT IS ASSERTED

- The pass model, run on the Downlink tab's own bound, reproduces this suite's independent Python
  integration of Liao et al. 2017's 19 December 2016 pass, and so Liao's 3,551,136 detections and
  1,671,072 sifted bits within 1 percent; its duration is within 2 percent of the 273 s.
- On Khmelev et al.'s Zvenigorod pass (arXiv:2310.17476), counted the same way, the sifted rate at
  600 km is within 0.5 dB of the measured 20.7 kbit/s, and at 1,100 km it reads 1.75 dB above the
  measured 2.9 kbit/s - the same shortfall in slope Liao's pass shows, stated rather than tuned away.
- Liao 2018's geometry: a 94-minute orbit, Xinglong to Graz 7,600 km and to Nanshan 2,500 km from the
  paper's coordinates, no single satellite seeing Xinglong and Graz at once, the relayed key the
  shorter of the two, and the XOR round trip exact.
- The constellation's geometry in closed form: neighbour distance, line-of-sight clearance, the
  fewest satellites that see each other over the Earth, the shadow fraction; and its link runs the
  same BB84 bound over a channel with no atmosphere.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
R = 6371.0


@pytest.fixture(scope="module")
def net() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            f"export * from {json.dumps(str(WEB / 'src/lib/satNetwork'))};\n"
            f"export {{ satChannel }} from {json.dumps(str(WEB / 'src/lib/satLink'))};\n"
            f"export {{ bb84Rate }} from {json.dumps(str(WEB / 'src/lib/quantumSim'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + r"""
const link = m.satChannel({});
const liao = m.liaoPass(link);
const zv = m.satChannel({ sat_profile: "micius_2022_zvenigorod" });
const el = (r, g) => { const a = 6371 + g, b = 6871; return Math.asin((b*b - a*a - r*r) / (2*a*r)) * 180 / Math.PI; };
const zvPass = m.simulatePass(zv, el(600, 0.198), { windowElevationDeg: el(1100, 0.198), stepS: 0.25 });
const [X, N, G] = m.LIAO_2018_STATIONS;
const relay = m.trustedRelay(link, X, G, 60, 40);
const kA = new Uint8Array([1, 2, 3, 250]), kB = new Uint8Array([9, 8, 7, 6]);
const xor = m.xorRelay(kA, kB);
const isl = m.islFromProfile("micius_type_2019");
const ring = [9, 10, 16].map((n) => m.stringOfPearls(n, 500, isl));
const ch10 = m.islChannel(isl, ring[1].neighbourKm);
const cube = m.stringOfPearls(10, 500, m.islFromProfile("cubesat_naughton_2019"));
const out = {
  periodS: m.orbitalPeriodS(500),
  liao: { s: liao.durationS, det: liao.detections, sift: liao.siftedBits, key: liao.keyBits,
          top: liao.maxElevationDeg, edge: liao.windowElevationDeg },
  zv: { first: zvPass.samples[0].siftedRate, top: zvPass.samples[Math.floor(zvPass.samples.length / 2)].siftedRate },
  range10: m.rangeAtElevationKm(10, 500, 0.89), range0: m.rangeAtElevationKm(0, 500, 0.89),
  xg: m.groundDistanceKm(X, G), xn: m.groundDistanceKm(X, N), shared: m.sharedViewMaxKm(link),
  relay: { key: relay.keyBits, a: relay.a.keyBits, b: relay.b.keyBits, wait: relay.waitH, sharedView: relay.sharedView },
  xor: { broadcast: Array.from(xor.broadcast), recovered: Array.from(xor.recovered) },
  ring: ring.map((c) => ({ n: c.n, d: c.neighbourKm, low: c.chordMinAltitudeKm, los: c.losOk, min: c.minSatellites,
                           loss: c.islLossDb, perPulse: c.islPerPulse, bps: c.islBitPerS, ecl: c.eclipseFraction,
                           pair: c.pairEclipseFraction, orbitS: c.islSecondsPerOrbit, period: c.periodS })),
  rate10: m.bb84Rate(ch10).perPulse,
  cube: { bps: cube.islBitPerS, loss: cube.islLossDb },
  noEclipse: m.eclipseHalfAngleRad(500, 80),
};
console.log(JSON.stringify(out));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True,
                                         text=True).stdout)


def test_the_orbit_is_micius_s(net) -> None:
    # Liao et al. 2018: "The satellite flies along a sun-synchronized orbit which circles Earth every
    # 94 minutes".
    assert net["periodS"] / 60 == pytest.approx(94.0, abs=1.0)
    assert net["range10"] == pytest.approx(1692.4, abs=0.5)    # the 10 degree edge
    assert net["range0"] == pytest.approx(2571.0, abs=1.0)     # the horizon


def test_the_pass_is_liaos_counted_like_for_like(net) -> None:
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    from test_the_satellite_channel_is_sourced import DETECTIONS, PASS_S, SIFTED, _pass

    oracle = _pass()
    liao = net["liao"]
    assert liao["s"] == pytest.approx(oracle["seconds"], rel=1e-9)
    assert liao["s"] == pytest.approx(PASS_S, rel=0.02)
    # The browser's integration and this suite's own Python one, to the step size.
    assert liao["det"] == pytest.approx(oracle["detections"], rel=1e-5)
    assert liao["sift"] == pytest.approx(oracle["sifted"], rel=1e-5)
    assert liao["det"] == pytest.approx(DETECTIONS, rel=0.01)
    assert liao["sift"] == pytest.approx(SIFTED, rel=0.01)
    # The window Liao's figure shows: from 1200 km (19.8 degrees) to the 645 km top (48.9).
    assert (liao["top"], liao["edge"]) == pytest.approx((48.90, 19.76), abs=0.01)
    # The secret key is asymptotic and the measured one finite-key, so the model must exceed it -
    # per pulse sent, by about half (462 kbit against 300,939 bits).
    assert 1.0 < liao["key"] / 300_939 < 2.0


def test_the_zvenigorod_pass_is_checked_the_same_way(net) -> None:
    """Khmelev et al.: "the resulting sifted key rate changes from around 20.7 kbit/s at 600 km to
    2.9 kbit/s at 1,100 km", sifted on signal states (their Eq. 3, p_alpha = p_s)."""
    at_600 = 10 * math.log10(net["zv"]["top"] / 20_700)
    at_1100 = 10 * math.log10(net["zv"]["first"] / 2_900)
    assert abs(at_600) < 0.5, f"{at_600:+.2f} dB at 600 km"
    assert at_1100 == pytest.approx(1.75, abs=0.05), "the model's slope is about 2 dB shallow, as on Liao's pass"


def test_liao_2018_geometry_and_the_relay(net) -> None:
    # "The distances from Xinglong to Nanshan and Graz are 2500 km and 7600 km"
    assert net["xg"] == pytest.approx(7600, rel=0.01)
    assert net["xn"] == pytest.approx(2500, rel=0.02)
    # One satellite's footprint at 10 degrees: 2 R psi, psi = acos(a/b cos 10) - 10 degrees.
    a, b = R + 0.89, R + 500.0
    psi = math.acos(a / b * math.cos(math.radians(10))) - math.radians(10)
    assert net["shared"] == pytest.approx(2 * R * psi, rel=1e-9)
    assert net["relay"]["sharedView"] is False                 # 7,621 km against 3,122 km
    assert net["relay"]["key"] == pytest.approx(min(net["relay"]["a"], net["relay"]["b"]))
    assert net["relay"]["a"] > net["relay"]["b"]               # a 60 degree pass outdoes a 40 degree one
    # Same local time at each station: 117.58 E to 15.49 E is 102.08 degrees, 6.8 hours.
    assert net["relay"]["wait"] == pytest.approx(102.084 / 15, abs=0.01)
    k_a, k_b = [1, 2, 3, 250], [9, 8, 7, 6]
    assert net["xor"]["broadcast"] == [x ^ y for x, y in zip(k_a, k_b, strict=True)]
    assert net["xor"]["recovered"] == k_b


def test_the_constellation_geometry_is_closed_form(net) -> None:
    b = R + 500.0
    for row in net["ring"]:
        n = row["n"]
        assert row["d"] == pytest.approx(2 * b * math.sin(math.pi / n), rel=1e-12)
        assert row["low"] == pytest.approx(b * math.cos(math.pi / n) - R, rel=1e-12)
        assert row["min"] == math.ceil(math.pi / math.acos((R + 100.0) / b))
        assert row["los"] == (row["low"] >= 100.0)
    assert [r["los"] for r in net["ring"]] == [False, True, True]   # 9 dip to 86 km, 10 clear 164 km
    assert net["ring"][0]["min"] == 10
    # The shadow: asin(R / (R + h)) / pi at beta = 0, 35.7 minutes of 94.5; a link needs both ends.
    ecl = math.asin(R / b) / math.pi
    for row in net["ring"]:
        assert row["ecl"] == pytest.approx(ecl, rel=1e-9)
        assert row["pair"] == pytest.approx(ecl - 1 / row["n"], rel=1e-9)
    assert net["noEclipse"] == 0.0                                    # at beta 80 the orbit never darkens


def test_the_link_is_the_downlink_bound_without_an_atmosphere(net) -> None:
    ten = net["ring"][1]
    # Diffraction for 10 urad over the neighbour distance into 0.3 m, 3 dB pointing, 16% optics.
    expected = 20 * math.log10(10e-6 * ten["d"] * 1e3 / 0.3) + 3.0 - 10 * math.log10(0.16)
    assert ten["loss"] == pytest.approx(expected, rel=1e-9)
    assert ten["perPulse"] == pytest.approx(net["rate10"], rel=1e-12)
    assert ten["bps"] == pytest.approx(net["rate10"] * 1e8, rel=1e-12)   # per pulse sent (Round 78)
    assert ten["orbitS"] == pytest.approx(ten["pair"] * ten["period"], rel=1e-12)
    # CubeSat-class terminals (Naughton 2019) make no key at that spacing.
    assert net["cube"]["bps"] == 0.0 and net["cube"]["loss"] > ten["loss"]
