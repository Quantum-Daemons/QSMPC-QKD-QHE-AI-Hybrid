"""One-way quasi-QSDC (STIKE) on the satellite page: the formula, the book-keeping, both languages and the page (Round 86).

Two-way QSDC does not close over a LEO downlink (Qi et al.'s Eq. 8; `tests/test_usecases.py`). Pan et al., Sci. Adv. 11, eadt4627
(2025), send the message one way under a one-time pad from a pre-shared pool, and loss then costs rate rather than secrecy. The
figures reproduced here are Sun et al., arXiv:2602.06624 (2026), Table S1.
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
from qsh.orchestration.events import EventType
from qsh.qkd.bb84 import _decoy_secure_fraction, _h2, e_det_for_bound
from qsh.qkd.channel import ChannelParams, FreeSpaceParams
from qsh.qkd.stike import pulses_per_bit, stike_rates, stike_send
from qsh.usecases import satellite_comms

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
LINK = ChannelParams(distance_km=800, free_space=FreeSpaceParams())
MICIUS = {"mu": 0.8, "nu": 0.1, "vacuum": True}
LINK_OPTICS_DB = LINK.receiver_optics_db()


def _rates(params: ChannelParams = LINK):
    q = get_config().qkd
    return stike_rates(params, MICIUS["mu"], MICIUS["nu"], q.error_correction_eff, vacuum=MICIUS["vacuum"])


def _send(rates, *, pool=128_000_000, votes=20, seed=1, message=b"S" * 74):
    return stike_send(message, rates, p_signal=0.5, source_rate_hz=1e8, pool_bits=pool, votes_per_bit=votes, seed=seed)


def test_the_gains_of_sun_et_al_are_reproduced() -> None:
    # Table S1: 17.83 dB of channel, 6.5 dB inside Bob, a 20% detector, 1e-6 dark counts, mu 0.71, nu 0.28.
    eta = 10 ** (-(17.83 + 6.5) / 10) * 0.2
    y0 = 2e-6
    assert y0 + 1 - math.exp(-eta * 0.71) == pytest.approx(5.31e-4, rel=0.02)
    assert y0 + 1 - math.exp(-eta * 0.28) == pytest.approx(2.09e-4, rel=0.01)


def test_their_measured_rates_put_eq3_inside_their_key_rates() -> None:
    # Measured gains and error rates of Table S1, one signal per pulse at 1.25 GHz; the two days generated key at 13.51 and 30.42 kbps.
    mu, nu, f = 0.71, 0.28, 1.22
    q_mu, q_nu, e_mu, e_nu = 5.31e-4, 2.09e-4, 0.0287, 0.0401
    y0, e0 = 2e-6, 0.5
    q1 = mu * mu * math.exp(-mu) / (mu * nu - nu * nu) * (q_nu * math.exp(nu) - q_mu * math.exp(mu) * nu * nu / mu / mu - (mu * mu - nu * nu) / (mu * mu) * y0)
    e1 = mu * math.exp(-mu) * (e_nu * q_nu * math.exp(nu) - e0 * y0) / (nu * q1)
    cs = 0.5 * q_mu * (-f * _h2(e_mu) + q1 / q_mu * (1 - _h2(e1)))
    assert 13_510 < cs * 1.25e9 < 30_420


def test_eq_3_is_the_key_rate_the_qkd_stage_uses() -> None:
    r = _rates()
    q = get_config().qkd
    y0 = 2.0 * LINK.dark_count_prob
    from qsh.qkd.bb84 import e_det_for_bound
    direct, e_mu, _ = _decoy_secure_fraction(LINK.transmittance(), 0.8, 0.1, e_det_for_bound(LINK), y0, q.error_correction_eff, 0.5, vacuum=True)
    assert r.secrecy == direct and r.qber == e_mu


def test_the_default_downlink_closes_one_way_where_it_does_not_two_way() -> None:
    r = _rates()
    assert r.secrecy > 0 and 3e-5 < r.secrecy < 4e-5
    res = _send(r)
    assert res.status == "delivered" and res.delivered == b"S" * 74


def test_the_pool_is_charged_as_the_protocol_charges_it() -> None:
    r = _rates()
    res = _send(r)
    per_bit = pulses_per_bit(r, 20)
    assert res.key_in_flight_bits == res.signal_pulses == 592 * per_bit
    assert res.key_consumed_bits + res.key_returned_bits == res.key_in_flight_bits
    assert res.pool_after_bits == res.pool_before_bits - res.key_consumed_bits + res.key_regenerated_bits
    # Their P_rec = 1 - Q_mu / 2, the share of the reservation that comes back.
    assert res.key_returned_bits / res.key_in_flight_bits == pytest.approx(1 - r.gain / 2, abs=1e-5)
    assert res.key_consumed_bits == pytest.approx(592 * 20, rel=0.05)


def test_a_pool_smaller_than_the_reservation_sends_nothing() -> None:
    r = _rates()
    res = _send(r, pool=10_000_000)
    assert res.status == "pool_too_small" and res.delivered is None and res.key_consumed_bits == 0 and res.pool_after_bits == 10_000_000


def test_an_eavesdropper_takes_the_key_rate_and_nothing_is_spent() -> None:
    attacked = ChannelParams(distance_km=800, free_space=FreeSpaceParams(), eve_intercept_fraction=0.5)
    r = _rates(attacked)
    res = _send(r)
    assert r.secrecy == 0 and res.status == "no_secure_rate" and res.key_consumed_bits == 0


def test_too_little_spreading_cannot_decode() -> None:
    res = _send(_rates(), votes=1, seed=2)
    assert res.status == "undecodable" and res.delivered is None and res.erased_bits > 0 and res.key_consumed_bits > 0


@pytest.fixture(scope="module")
def browser() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bundle, runner = d / "bundle.mjs", d / "run.mjs"
        entry = (f"export * from {json.dumps(str(WEB / 'src/lib/stike'))};\n"
                 f"export {{ FREE_SPACE }} from {json.dumps(str(WEB / 'src/lib/quantumSim'))};\n"
                 f"export {{ USECASE_SPECS }} from {json.dumps(str(WEB / 'src/lib/useCaseFlows'))};\n")
        subprocess.run([str(ESBUILD), "--bundle", "--loader=ts", "--format=esm", "--platform=node", '--define:import.meta.env={"BASE_URL":"/"}',
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True, input=entry.encode())
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + """
const rates = m.stikeRates(m.FREE_SPACE);
const opts = { pSignal: 0.5, sourceRateHz: 1e8, votesPerBit: 20, seed: 1 };
const step = m.USECASE_SPECS.satellite_comms.steps.find((s) => s.stageId === "qsdc_telemetry");
const link = { qsdc_link_budget_limited: 1, qsdc_eq8_bound: -80.82, channel_loss_db: 32.2, qsdc_eve_gain_g: 532.66 };
const tag = (extra) => ({ label: step.label({ ...link, ...extra }, {}), ok: step.ok({ ...link, ...extra }, {}) });
console.log(JSON.stringify({ rates,
  curves: m.qsdcCurves(m.FREE_SPACE),
  atAPoint: m.qsdcCurves(m.FREE_SPACE, undefined, [23.55]),
  delivered: m.stikeSend(74, rates, { ...opts, poolBits: 128000000 }),
  small: m.stikeSend(74, rates, { ...opts, poolBits: 1000 }),
  canvas: { sent: tag({ stike_status: "delivered", stike_delivered: 1, stike_secrecy_per_signal: 3.356e-5, stike_key_consumed_bits: 11803, stike_key_regenerated_bits: 3306 }),
            small: tag({ stike_status: "pool_too_small", stike_delivered: 0 }), none: tag({}) },
}));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True).stdout)


def test_the_rates_agree_across_languages(browser) -> None:
    py, ts = _rates(), browser["rates"]
    for a, b in ((py.secrecy, ts["secrecy"]), (py.reliable, ts["reliable"]), (py.retained, ts["retained"]), (py.gain, ts["gain"]), (py.qber, ts["qber"])):
        assert a == pytest.approx(b, rel=1e-9)
    assert pulses_per_bit(py, 20) * 592 == browser["delivered"]["signalPulses"]


def test_the_plotted_curves_are_the_three_formulas(browser) -> None:
    # Recomputed here from the Python estimator, so the plot cannot drift from the bounds the stage uses.
    q = get_config().qkd
    c = browser["curves"]
    optics = LINK_OPTICS_DB
    detector = -10 * math.log10(LINK.detector_efficiency)
    assert c["rows"][0]["lossDb"] == pytest.approx(optics + detector, abs=1e-9)
    assert c["linkLossDb"] == pytest.approx(LINK.loss_db() + detector, rel=1e-9)
    seen = 0
    for row in c["rows"]:
        loss = row["lossDb"]
        eta = 10 ** (-loss / 10)
        y0 = 2 * LINK.dark_count_prob
        secrecy, e_mu, _ = _decoy_secure_fraction(eta, MICIUS["mu"], MICIUS["nu"], e_det_for_bound(LINK), y0, q.error_correction_eff, 0.5, vacuum=MICIUS["vacuum"])
        gain = y0 + 1 - math.exp(-eta * MICIUS["mu"])
        reliable = 0.5 * gain * (1 - q.error_correction_eff * _h2(e_mu))
        g = 10 ** ((loss - optics) / 10)
        two_way = gain * (1 - _h2(e_mu) - g * _h2(min(0.5, 2 * e_mu)))
        for value, key in ((secrecy, "oneWaySecrecy"), (reliable, "oneWayReliable"), (two_way, "twoWay")):
            if value > 0:
                assert row[key] == pytest.approx(value, rel=1e-9), (loss, key)
                seen += 1
            else:
                assert row[key] is None, (loss, key)
    assert seen > 100


def test_two_way_closes_where_the_formula_says_and_short_of_this_link(browser) -> None:
    c = browser["curves"]
    # At a floor of e, 1 - h(e) - g h(2e) = 0 gives the g it closes at, and g is the loss past the station's optics.
    e = _rates().qber
    closes_at = LINK_OPTICS_DB + 10 * math.log10((1 - _h2(e)) / _h2(2 * e))
    assert c["twoWayClosesToDb"] == pytest.approx(closes_at, abs=0.6)
    assert c["twoWayClosesToDb"] < c["linkLossDb"] - 10
    assert any(r["oneWaySecrecy"] is not None and r["lossDb"] > c["linkLossDb"] for r in c["rows"])


# The source's own terms: distance or channel in dB, 6.5 dB in Bob, a 20% detector, and the rate over a 1.25 GHz clock.
PUBLISHED = {
    "Pan et al. 2025, 50.3 km fibre": (50.3 * 0.2, 34.08),
    "Pan et al. 2025, 81.1 km fibre": (81.1 * 0.2, 11.36),
    "Pan et al. 2025, 104.8 km fibre": (104.8 * 0.2, 2.38),
    "Sun et al. 2026, 1.4 km air and 10 km fibre": (17.83, 4.22),
}


def test_a_published_point_replaces_the_nearest_sample(browser) -> None:
    # So hovering the point reads it, and not a grid row a fraction of a pixel away.
    near = [r for r in browser["atAPoint"]["rows"] if abs(r["lossDb"] - 23.55) < 0.5]
    assert sum(r["lossDb"] == 23.55 for r in near) == 1 and all(r["oneWaySecrecy"] is not None for r in near)
    assert min(abs(r["lossDb"] - 23.55) for r in near if r["lossDb"] != 23.55) >= 0.25


def test_the_published_points_are_the_papers() -> None:
    points = {p["label"]: p for p in get_config().qkd.qsdc_published_points}
    assert set(points) == set(PUBLISHED)
    for label, (channel_db, kbps) in PUBLISHED.items():
        assert points[label]["loss_db"] == pytest.approx(channel_db + 6.5 + 10 * math.log10(5), abs=0.01), label
        assert points[label]["bit_per_pulse"] == pytest.approx(kbps * 1e3 / 1.25e9, rel=1e-3), label


def test_the_panel_sits_with_the_range_chart_on_the_satellite_page() -> None:
    page = (WEB / "src/pages/UseCasePage.tsx").read_text(encoding="utf-8")
    assert "<QsdcLinkPanel channel={sat} metrics={metrics} />" in page


def test_the_browser_charges_the_pool_the_same_way(browser) -> None:
    d = browser["delivered"]
    assert d["status"] == "delivered" and d["keyConsumedBits"] + d["keyReturnedBits"] == d["keyInFlightBits"]
    # Only what Bob retained is spent: a binomial draw, so five standard deviations is a fixed band and no seed is chosen.
    expected = d["rates"]["retained"] * d["signalPulses"]
    assert abs(d["keyConsumedBits"] - expected) < 5 * math.sqrt(expected)
    assert browser["small"]["status"] == "pool_too_small" and browser["small"]["keyConsumedBits"] == 0


def test_the_canvas_says_what_became_of_the_telemetry(browser) -> None:
    c = browser["canvas"]
    assert c["sent"]["ok"] is True and "STIKE" in c["sent"]["label"] and "pre-shared" in c["sent"]["label"] and "QSDC NOT AVAILABLE" in c["sent"]["label"]
    assert c["small"]["ok"] is False and "too small" in c["small"]["label"] and "PQC session key" in c["small"]["label"]
    assert c["none"]["ok"] is False and "falls back to the PQC session key" in c["none"]["label"]


def test_the_stage_sends_the_telemetry_and_says_how() -> None:
    r = satellite_comms.run_demo({"seed": 1})
    m = r["metrics"]
    assert r["telemetry_delivered"] is False and r["telemetry_quasi_qsdc"] is True
    assert m["stike_status"] == "delivered" and m["stike_delivered"] == 1 and m["stike_signal_pulses"] > 0
    assert m["stike_pool_after_bits"] < 128_000_000


def _stage_logs(config: dict, out: Path) -> list[str]:
    pipe, ctx = satellite_comms.build({"seed": 1, **config})
    pipe.run(ctx, out_dir=str(out))
    return [e.message for e in pipe.bus.history if e.type == EventType.LOG and e.stage == "qsdc_telemetry"]


def test_a_run_that_sent_the_telemetry_one_way_does_not_say_it_fell_back(tmp_path) -> None:
    # The browser's twin of this is `test_the_browser_log_says_what_became_of_the_telemetry`, on the pipeline run in node.
    sent = _stage_logs({}, tmp_path)
    assert not any("falls back" in line for line in sent), sent
    assert "carries the telemetry instead" in sent[-2] and sent[-1].startswith("Quasi-QSDC (one-way STIKE) delivered"), sent
    small = _stage_logs({"sat_stike_pool_mbit": 1}, tmp_path)
    assert "falls back" in small[-2] and "carries the telemetry instead" not in small[-2], small
    assert small[-1].startswith("Quasi-QSDC (one-way STIKE) pool_too_small") and small[-1].endswith("falls back to the PQC session key."), small
    none = _stage_logs({"sat_stike_pool_mbit": 0}, tmp_path)
    assert "falls back" in none[-1] and not any("Quasi-QSDC" in line for line in none), none


def test_both_stages_report_the_same_quasi_qsdc_metrics() -> None:
    names = re.compile(r'"(stike_[a-z0-9_]+)"')
    client = set(names.findall((WEB / "src/lib/clientCompute.ts").read_text(encoding="utf-8")))
    stage = set(names.findall((REPO / "src/qsh/usecases/satellite_comms/__init__.py").read_text(encoding="utf-8")))
    assert client and client == stage


def test_the_controls_reach_the_stage() -> None:
    none = satellite_comms.run_demo({"seed": 1, "sat_stike_pool_mbit": 0})
    assert none["telemetry_quasi_qsdc"] is None and "stike_status" not in none["metrics"]
    small = satellite_comms.run_demo({"seed": 1, "sat_stike_pool_mbit": 1})
    assert small["metrics"]["stike_status"] == "pool_too_small" and small["telemetry_quasi_qsdc"] is False
    fewer = satellite_comms.run_demo({"seed": 1, "sat_stike_votes": 10})
    assert fewer["metrics"]["stike_signal_pulses"] < satellite_comms.run_demo({"seed": 1})["metrics"]["stike_signal_pulses"]


def test_the_telemetry_length_is_the_message_length() -> None:
    assert get_config().web.demo_sat_telemetry_bytes == len(satellite_comms._DEFAULT_TELEMETRY)
