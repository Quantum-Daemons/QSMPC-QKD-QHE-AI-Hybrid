"""The downlink model against every measured night, and the two Round 78 options checked as stated.

WHY THIS EXISTS (Round 78, the user's choices). Round 77 found the model about 2 dB shallow at the
far end of the two Micius passes it was checked on, and the user asked for a sourced loss term that
grows at low elevation, and for Jinan-1 as a template, each to be verified first. The verification
found:

- No sourced term closes the gap on both passes. On Khmelev et al.'s Zvenigorod pass the model fits
  the whole rising half and misses only the falling half, so that residual is not elevation at all;
  and across the 23 passes in Liao et al.'s Extended Data Table 2 the model's peak is centred, the
  nights scattering around it by 2.4 dB. By the user's choice the turbulence coupling the research
  ranked best-sourced is offered as an OPTION (a field stop the station does not publish), not as a
  default.
- Jinan-1 needs one constant calibrated: built only from stated values it over-predicts the peak of
  its 25 September 2022 pass by 3.67 dB. What that pass then checks independently is the fall with
  elevation, which the calibration does not touch.

WHAT IS ASSERTED

- Liao's 23 nights: the default link's peak is centred on them (median within 0.5 dB), they scatter by
  under 3 dB, none is more than 5 dB away; the 8 dB profile sits on the conservative side. The browser
  computes the same per-night rates.
- The coupling term: its closed form, zero when off and at the zenith, growing as the satellite sinks,
  the same in both languages; with the 10 urad template Liao's pass falls 9.6 dB from 645 to 1200 km,
  against 8.4 without and 10.8 measured.
- Jinan-1: the calibrated constant reproduces the pass's peak count rate; the fall with elevation over
  that pass matches within 0.5 dB; on four Jinan passes the model's asymptotic key exceeds every
  published finite key, by ratios that spread less than a factor of two.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path

import pytest

from qsh.config import frontend_config
from qsh.orchestration.stages import Context
from qsh.qkd.channel import ChannelParams, FreeSpaceParams, turbulence_coupling_db
from qsh.usecases.satellite_comms import _sat_channel, _sat_source

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"


def _nights() -> list[dict]:
    s = frontend_config()["validation"]["liao_2017_xinglong_passes"]
    return [dict(zip(s["columns"], r, strict=True)) for r in s["rows"]]


def _link(profile: str | None = None, **cfg) -> tuple[ChannelParams, object]:
    ctx = Context(usecase="satellite_comms", config={**({"sat_profile": profile} if profile else {}), **cfg})
    return _sat_channel(ctx), _sat_source(ctx)


def _sifted_at(profile: str | None, km: float) -> float:
    """Signal-state sifted bit/s at one range, as Liao counted (`satNetwork.ts::siftedRateAt`)."""
    ch, src = _link(profile, sat_distance_km=km, sat_min_elevation_deg=0.0)
    eta, y0 = ch.transmittance(), 2 * ch.dark_count_prob
    return 0.5 * src.p_signal * (y0 + 1 - math.exp(-eta * src.mu_signal)) * ch.source_rate_hz


def _ratios_db(profile: str | None) -> list[float]:
    return [10 * math.log10(n["peak_sifted_khz"] * 1e3 / _sifted_at(profile, n["shortest_distance_km"]))
            for n in _nights()]


def test_the_23_nights_are_the_published_table() -> None:
    nights = _nights()
    assert len(nights) == 23
    dec19 = next(n for n in nights if n["date"] == "19/12/2016")
    assert (dec19["max_elevation_deg"], dec19["shortest_distance_km"], dec19["peak_sifted_khz"]) == (
        47.79, 645.08, 14.1)


def test_the_default_link_is_centred_on_the_nights() -> None:
    r = _ratios_db(None)
    mean = statistics.mean(r)
    assert abs(statistics.median(r)) < 0.5, f"median {statistics.median(r):+.2f} dB"
    assert statistics.pstdev(r) < 3.0, f"nights scatter by {statistics.pstdev(r):.2f} dB"
    assert max(abs(x) for x in r) < 5.0
    assert 8 <= sum(x > 0 for x in r) <= 15
    assert abs(mean) < 1.0
    # The 8 dB upper value is the conservative reading: most measured nights beat it.
    up = _ratios_db("micius_2017_upper_atmosphere")
    assert 0.5 < statistics.median(up) < 1.5 and sum(x > 0 for x in up) >= 13


def _fraction(fs: FreeSpaceParams, elevation: float) -> float:
    r0_500 = 0.98 * 500e-9 / (fs.seeing_arcsec * math.pi / 648000)
    r0 = r0_500 * (fs.wavelength_nm / 500) ** 1.2 * math.sin(math.radians(elevation)) ** 0.6
    fwhm = 0.98 * fs.wavelength_nm * 1e-9 / r0
    return 1 - math.exp(-4 * math.log(2) * (fs.field_stop_urad * 1e-6 / fwhm) ** 2)


def test_the_coupling_term_is_its_closed_form() -> None:
    off = FreeSpaceParams()
    assert off.field_stop_urad == 0.0 and turbulence_coupling_db(off, 1200.0) == 0.0
    fs = FreeSpaceParams(field_stop_urad=10.0)
    assert turbulence_coupling_db(fs, 498.0) == pytest.approx(0.0, abs=1e-12)    # inside the zenith range
    from qsh.qkd.channel import link_elevation_deg
    for km in (645.0, 1200.0, 1600.0):
        e = link_elevation_deg(fs, km)
        assert turbulence_coupling_db(fs, km) == pytest.approx(
            10 * math.log10(_fraction(fs, 90.0) / _fraction(fs, e)), rel=1e-12)
    vals = [turbulence_coupling_db(fs, km) for km in (600.0, 900.0, 1200.0, 1500.0)]
    assert vals == sorted(vals) and vals[0] < 0.1 < 1.0 < vals[2] < 1.5
    # Zvenigorod's receiver accepts 25 urad, where the term is a tenth of a dB.
    assert turbulence_coupling_db(FreeSpaceParams(field_stop_urad=25.0), 1200.0) < 0.15


def test_the_field_stop_template_steepens_liaos_pass() -> None:
    drop = lambda p: 10 * math.log10(_sifted_at(p, 645.0) / _sifted_at(p, 1200.0))  # noqa: E731
    assert drop(None) == pytest.approx(8.41, abs=0.05)
    assert drop("micius_2017_field_stop") == pytest.approx(9.56, abs=0.05)   # Liao: ~10.8
    # It is station optics: QSDC's Eve gap leaves it out, as it leaves out the 16% optics.
    ch, _ = _link("micius_2017_field_stop", sat_distance_km=1200.0)
    assert ch.receiver_optics_db() == pytest.approx(
        -10 * math.log10(0.16) + turbulence_coupling_db(ch.free_space, 1200.0), rel=1e-12)


# --- Jinan-1 (Li et al., Nature 2025): the 25 September 2022 pass, read from Fig. 4b -------------
# Counts per second at the top (about 823 km, where Table 1's 34.1 deg puts it), where QKD ends on
# the falling branch (1462 km) and where it starts (1630 km), over a background of about 650/s.
JINAN_PEAK, JINAN_BACKGROUND = (823.3, 40_700.0), 650.0
JINAN_EDGES = {1462.0: 9_380.0, 1630.0: 6_294.0}


def _detections(km: float, **override) -> float:
    ch, src = _link("jinan1_2022_jinan", sat_distance_km=km, sat_min_elevation_deg=0.0)
    if override:
        ch.free_space.receiver_optics_efficiency = override["eff"]
    eta, y0 = ch.transmittance(), 2 * ch.dark_count_prob
    q = lambda mu: y0 + 1 - math.exp(-eta * mu)  # noqa: E731
    return (src.p_signal * q(src.mu_signal) + src.p_decoy * q(src.mu_decoy)
            + (1 - src.p_signal - src.p_decoy) * y0) * ch.source_rate_hz


def test_jinan1_has_one_calibrated_constant() -> None:
    peak = JINAN_PEAK[1] - JINAN_BACKGROUND
    assert _detections(JINAN_PEAK[0]) == pytest.approx(peak, rel=0.02)
    # Only the stated efficiencies (49% indoors x 80% gating) would read 3.67 dB high.
    stated = _detections(JINAN_PEAK[0], eff=0.49 * 0.8)
    assert 10 * math.log10(stated / peak) == pytest.approx(3.67, abs=0.05)


def test_jinan1_falls_with_elevation_as_measured() -> None:
    """Independent of the calibration, which scales every range alike."""
    model_peak = _detections(JINAN_PEAK[0])
    for km, counts in JINAN_EDGES.items():
        measured = 10 * math.log10((JINAN_PEAK[1] - JINAN_BACKGROUND) / (counts - JINAN_BACKGROUND))
        model = 10 * math.log10(model_peak / _detections(km))
        assert model == pytest.approx(measured, abs=0.5), f"{km} km: {model:.2f} against {measured:.2f} dB"


@pytest.fixture(scope="module")
def browser() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    nights = [n["shortest_distance_km"] for n in _nights()]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            f"export * from {json.dumps(str(WEB / 'src/lib/satNetwork'))};\n"
            f"export {{ satChannel }} from {json.dumps(str(WEB / 'src/lib/satLink'))};\n"
            f"export {{ turbulenceCouplingDb }} from {json.dumps(str(WEB / 'src/lib/quantumSim'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + f"""
const link = m.satChannel({{}});
const stop = m.satChannel({{ sat_profile: "micius_2017_field_stop" }});
const jinan = m.satChannel({{ sat_profile: "jinan1_2022_jinan" }});
const passes = [["20 Sep", 41.8, 330240], ["25 Sep", 34.1, 406784], ["28 Sep", 77.7, 571392], ["29 Sep", 52.7, 436096]];
console.log(JSON.stringify({{
  nights: {json.dumps(nights)}.map((km) => m.siftedRateAt(link, km)),
  coupling: [645, 1200, 1600].map((km) => m.turbulenceCouplingDb({{ ...stop, distanceKm: km }})),
  jinan: passes.map(([d, top, fin]) => ({{ d, fin, key: m.simulatePass(jinan, top).keyBits }})),
}}));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True,
                                         text=True).stdout)


def test_the_browser_draws_the_same_nights_and_coupling(browser) -> None:
    for km, rate in zip((n["shortest_distance_km"] for n in _nights()), browser["nights"], strict=True):
        assert rate == pytest.approx(_sifted_at(None, km), rel=1e-9), km
    ch, _ = _link("micius_2017_field_stop")
    for km, db in zip((645.0, 1200.0, 1600.0), browser["coupling"], strict=True):
        assert db == pytest.approx(turbulence_coupling_db(ch.free_space, km), rel=1e-9), km


def test_jinan1_keys_exceed_every_published_finite_key(browser) -> None:
    """The model's key is the asymptotic bound per pulse sent and the paper's is finite-key, so the
    model must exceed it on every pass; how far varies with the night (2.9x to 5.1x here), as
    Micius's own nights do. The paper's sifted packets are a quarter of the detections and it does
    not say which states they count, so part of the factor may be protocol, not physics."""
    ratios = [p["key"] / p["fin"] for p in browser["jinan"]]
    assert all(r > 1.0 for r in ratios), ratios
    assert max(ratios) / min(ratios) < 2.0, ratios
