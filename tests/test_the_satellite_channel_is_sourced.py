"""The satellite downlink's parameters are Micius's, each traceable to Liao et al. 2017, and the model
reproduces Liao's own pass when it is counted the way Liao counted it.

WHY THIS EXISTS

Until Round 75 the config called its free-space defaults "Micius's published configuration" when
they were not:
- the 1.2 m aperture belongs to Yin et al.'s 2017 stations, not Liao's Xinglong (1 m);
- the loss budget had no receiver-optics term, Liao's "~16%", about 8 dB;
- the dark counts were 200x Micius's and the error floor more than 2x, justified by a 2 km
  horizontal ground link.

Too little loss and too much noise offset each other, and the page's ~5.5 kbit/s at 800 km was the
product of that cancellation. The satellite stage also kept its own literal copies of three of the
values, so retuning the config would have moved the browser and left the Python stage behind.

ROUND 77 found the check that replaced it was not like-for-like either. It compared Liao's "~12
kbit/s at 645 km" with half of every detection at mu = 0.8, as if every pulse were a signal. Liao
sent "a high mu_s, a moderate mu_1 and a zero mu_2 (vacuum), sent with probabilities of 50%, 25%, and
25%", and counted SIGNAL states only: 1,671,072 sifted of 3,551,136 detections is 0.47057, where
signal-only sifting at those probabilities gives 0.47059 and all-state sifting 0.5. So the old check
was a factor of two generous, and its "within 10 percent" was an accident of that factor. Counted
Liao's way, the fixed 5 dB atmosphere it ran was 2.9 dB low at 645 km and 2.5 dB high at 1200 km.

WHAT IS ASSERTED

- Each free-space value equals the figure Liao et al. state, which is quoted beside it here.
- The loss budget carries the station optics, the atmosphere scales by air mass from its anchor at
  1200 km, and there is no link below the 10 degree minimum or past the horizon.
- QSDC's Eve-to-Bob gap leaves the station optics out, as Qi et al.'s back channel did.
- The satellite stage reads the channel and the source from the selected profile, not from literals,
  and runs Micius's 0.8 / 0.1 with the vacuum state.
- Integrated over Liao's 19 December 2016 pass and counted like-for-like, the model gives Liao's
  detections and sifted bits within 1 percent: 3,519,419 against 3,551,136. Not fitted: 5.5 dB is
  the midpoint of Liao's stated 3-8 dB. The pass's shape is stated rather than hidden: the model
  falls 8.4 dB from 645 to 1200 km where Liao's rate fell 10.8, so it reads 2.0 dB high at 1200 km.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qsh.config import get_config
from qsh.qkd.channel import (
    EARTH_RADIUS_KM,
    ChannelParams,
    FreeSpaceParams,
    atmospheric_loss_db,
    elevation_deg,
    in_view,
)

REPO = Path(__file__).resolve().parents[1]

#: (config key, value, Liao et al. 2017's words for it)
LIAO_2017 = [
    ("free_space_rx_aperture_m", 1.0, "a Ritchey-Chretien telescope with an aperture of 1 m"),
    ("free_space_beam_divergence_urad", 10.0, "near-diffraction-limited far-field divergence of ~10 urad"),
    ("free_space_detector_efficiency", 0.5, "four single-photon detectors (efficiency 50%"),
    ("free_space_receiver_optics_efficiency", 0.16,
     "overall optical efficiency including the receiving telescope and the fiber coupling ... ~16%"),
    ("free_space_dark_count_prob", 5e-8, "dark counts <25 Hz ... within a 2-ns time window"),
    ("free_space_misalignment", 0.011, "QBER with an average of 1.1%"),
    ("free_space_source_rate_hz", 1e8, "laser pulses (848.6 nm, 100 MHz, 0.2 ns)"),
    ("free_space_orbit_altitude_km", 500.0, "orbiting at an altitude of ~500 km"),
    ("free_space_station_altitude_km", 0.89, "Xinglong ground observatory station ... altitude 890m"),
    ("free_space_atmospheric_ref_km", 1200.0, "at 1200 km the loss due to atmospheric absorption ..."),
    ("free_space_min_elevation_deg", 10.0,
     "A single-orbit experiment ends when the satellite reaches 10 deg elevation angle"),
    ("free_space_mu_signal", 0.8, "mu_s=0.8, mu_1=0.1, mu_2=0"),
    ("free_space_mu_decoy", 0.1, "mu_s=0.8, mu_1=0.1, mu_2=0"),
    ("free_space_p_signal", 0.5, "sent with probabilities of 50%, 25%, and 25%, respectively"),
    ("free_space_p_decoy", 0.25, "sent with probabilities of 50%, 25%, and 25%, respectively"),
    ("free_space_vacuum_decoy", True, "a high mu_s, a moderate mu_1 and a zero mu_2 (vacuum)"),
]

#: Liao et al. 2017, the orbit of 19 December 2016: "the minimal (maximal) separation of 645 km
#: (1200 km). Within a duration of 273 s for the QKD data collection, the ground station collected
#: 3,551,136 detection events, and thereof 1,671,072 bits of sifted keys".
PASS_MIN_KM, PASS_MAX_KM, PASS_S = 645.0, 1200.0, 273.0
DETECTIONS, SIFTED = 3_551_136, 1_671_072


@pytest.mark.parametrize("key,value,quote", LIAO_2017, ids=[k for k, _, _ in LIAO_2017])
def test_each_value_is_liaos(key: str, value: float, quote: str) -> None:
    assert getattr(get_config().qkd, key) == pytest.approx(value), f"{key}: Liao 2017 says {quote!r}"


def test_atmosphere_and_pointing_sit_inside_liaos_estimates() -> None:
    q = get_config().qkd
    assert 3.0 <= q.free_space_atmospheric_db <= 8.0, "Liao: 3 dB to 8 dB at 1200 km"
    assert q.free_space_atmospheric_db == pytest.approx(5.5), "the midpoint, by the user's choice"
    assert q.free_space_pointing_db <= 3.0, "Liao: less than 3 dB"


def _downlink(distance_km: float, **free_space) -> ChannelParams:
    q = get_config().qkd
    return ChannelParams(
        distance_km=distance_km, detector_efficiency=q.free_space_detector_efficiency,
        dark_count_prob=q.free_space_dark_count_prob, misalignment=q.free_space_misalignment,
        free_space=FreeSpaceParams(**free_space))


def test_the_budget_carries_the_station_optics_and_an_air_mass_atmosphere() -> None:
    c = _downlink(800.0)
    fs = c.free_space
    geometric = -20 * math.log10(1.0 / (10e-6 * 800e3))
    assert c.receiver_optics_db() == pytest.approx(-10 * math.log10(0.16))
    assert c.loss_db() == pytest.approx(geometric + atmospheric_loss_db(fs, 800.0) + 3.0
                                        + c.receiver_optics_db())
    assert c.loss_db() == pytest.approx(32.213, abs=1e-3)
    # The anchor, and Kasten & Young's air mass at the elevation each range implies.
    assert atmospheric_loss_db(fs, 1200.0) == pytest.approx(5.5, rel=1e-12)
    assert elevation_deg(800.0, 500.0, 0.89) == pytest.approx(35.84, abs=0.01)
    assert atmospheric_loss_db(fs, 800.0) == pytest.approx(3.192, abs=1e-3)
    # Liao's own diffraction estimate at 1200 km is 22 dB; this law gives 21.6.
    assert -20 * math.log10(1.0 / (10e-6 * 1200e3)) == pytest.approx(22.0, abs=0.5)


def test_no_link_below_the_minimum_elevation_or_past_the_horizon() -> None:
    fs = FreeSpaceParams()
    # 10 degrees at a 500 km orbit seen from 890 m is a slant range of 1,692.4 km; the horizon 2,571.
    assert elevation_deg(1692.4, 500.0, 0.89) == pytest.approx(10.0, abs=0.01)
    assert in_view(fs, 1690.0) and not in_view(fs, 1700.0)
    assert math.isfinite(_downlink(1690.0).loss_db())
    assert _downlink(1700.0).loss_db() == float("inf") and _downlink(1700.0).transmittance() == 0.0
    assert elevation_deg(2600.0, 500.0, 0.89) is None
    # The cut is a setting, not a constant: at 5 degrees 1,700 km is a link again.
    assert math.isfinite(_downlink(1700.0, min_elevation_deg=5.0).loss_db())


def test_the_satellite_stage_reads_the_channel_from_the_profile() -> None:
    from qsh.orchestration.stages import Context
    from qsh.usecases.satellite_comms import _SAT_DEFAULTS, _sat_channel, _sat_source

    for key in ("sat_distance_km", "sat_detector_efficiency", "sat_dark_count_prob",
                "sat_misalignment", "sat_atmospheric_db", "sat_pointing_db", "sat_profile",
                "sat_min_elevation_deg", "sat_vacuum_decoy"):
        assert _SAT_DEFAULTS[key] is None, f"{key} is a literal again: {_SAT_DEFAULTS[key]}"
    ctx = Context(usecase="satellite_comms", config={})
    src = _sat_source(ctx)
    assert (src.mu_signal, src.mu_decoy, src.vacuum_decoy, src.p_signal, src.p_decoy) == (
        0.8, 0.1, True, 0.5, 0.25), "the satellite does not run Micius's own source"
    # Another profile is a different link: the 8 dB upper value moves the budget by its air mass share.
    up = _sat_channel(Context(usecase="satellite_comms",
                              config={"sat_profile": "micius_2017_upper_atmosphere"}))
    base = _sat_channel(ctx)
    assert up.loss_db() - base.loss_db() == pytest.approx(
        2.5 * atmospheric_loss_db(base.free_space, 800.0) / 5.5, rel=1e-9)
    # Switching the vacuum off hands its share to the decoy.
    off = _sat_source(Context(usecase="satellite_comms", config={"sat_vacuum_decoy": False}))
    assert (off.vacuum_decoy, off.p_decoy) == (False, 0.5)


def test_the_satellite_rate_is_the_vacuum_plus_weak_bound() -> None:
    from qsh.qkd.bb84 import QKDParams, _decoy_secure_fraction, e_det_for_bound, run_bb84

    c = ChannelParams(distance_km=800.0, free_space=FreeSpaceParams())
    eta, y0 = c.transmittance(), 2 * c.dark_count_prob
    f = get_config().qkd.error_correction_eff
    vac = QKDParams(mu_signal=0.8, mu_decoy=0.1, vacuum_decoy=True, p_signal=0.5, p_decoy=0.25)
    run = run_bb84(200_000, c, seed=0, qkd_params=vac)
    bound = _decoy_secure_fraction(eta, 0.8, 0.1, e_det_for_bound(c), y0, f, 0.5, vacuum=True)[0]
    one = _decoy_secure_fraction(eta, 0.8, 0.1, e_det_for_bound(c), y0, f, 0.5)[0]
    assert run.secret_key_rate == pytest.approx(bound, rel=1e-12)
    assert run.secret_key_rate == pytest.approx(3.3559e-5, rel=1e-4)
    # The measured vacuum yield can only tighten the estimate, and costs nothing with no background.
    assert bound > one
    assert (_decoy_secure_fraction(eta, 0.8, 0.1, 0.011, 0.0, f, 0.5, vacuum=True)[0]
            == pytest.approx(_decoy_secure_fraction(eta, 0.8, 0.1, 0.011, 0.0, f, 0.5)[0], rel=1e-15))


def _pass(**free_space) -> dict:
    """Liao's 19 December 2016 pass, integrated by this test's own geometry.

    A circular orbit at the profile's altitude; the station's closest approach is 645 km; the data
    window runs while the range is under 1200 km. Earth's rotation is left out (about 0.35 km/s at
    Xinglong against the satellite's 7.6), so the pass is symmetric. Counted as Liao counted: states
    at 50 / 25 / 25, sifted = matching bases on SIGNAL states.
    """
    q = get_config().qkd
    a, b = EARTH_RADIUS_KM + q.free_space_station_altitude_km, EARTH_RADIUS_KM + q.free_space_orbit_altitude_km
    w = math.sqrt(398600.4418 / b ** 3)                       # rad/s, circular orbit
    psi = lambda r: math.acos((a * a + b * b - r * r) / (2 * a * b))  # noqa: E731  (central angle)
    psi0 = psi(PASS_MIN_KM)
    half = math.acos(math.cos(psi(PASS_MAX_KM)) / math.cos(psi0)) / w
    rate, mu, nu = q.free_space_source_rate_hz, q.free_space_mu_signal, q.free_space_mu_decoy
    p_s, p_d = q.free_space_p_signal, q.free_space_p_decoy
    det = sift = 0.0
    n = 4000                                                  # trapezoid rule, converged to < 1 bit
    h = 2 * half / n
    for i in range(n + 1):
        t = -half + i * h
        wt = h / 2 if i in (0, n) else h
        r = math.sqrt(a * a + b * b - 2 * a * b * math.cos(psi0) * math.cos(w * t))
        c = _downlink(r, **free_space)
        eta, y0 = c.transmittance(), 2 * c.dark_count_prob
        q_mu, q_nu = y0 + 1 - math.exp(-eta * mu), y0 + 1 - math.exp(-eta * nu)
        det += (p_s * q_mu + p_d * q_nu + (1 - p_s - p_d) * y0) * rate * wt
        sift += 0.5 * p_s * q_mu * rate * wt
    end = {}
    for r in (PASS_MIN_KM, PASS_MAX_KM):
        c = _downlink(r, **free_space)
        end[r] = 0.5 * p_s * (2 * c.dark_count_prob + 1 - math.exp(-c.transmittance() * mu)) * rate
    return {"seconds": 2 * half, "detections": det, "sifted": sift, "at_645": end[PASS_MIN_KM],
            "at_1200": end[PASS_MAX_KM]}


def test_liao_counted_signal_states_only() -> None:
    q = get_config().qkd
    mu, nu, p_s, p_d = q.free_space_mu_signal, q.free_space_mu_decoy, q.free_space_p_signal, q.free_space_p_decoy
    signal_only = 0.5 * p_s * mu / (p_s * mu + p_d * nu)      # the low-transmittance limit
    assert SIFTED / DETECTIONS == pytest.approx(signal_only, abs=1e-4)
    assert abs(SIFTED / DETECTIONS - 0.5) > 0.02, "all-state sifting would give 0.5"


def test_the_model_reproduces_liaos_pass_counted_like_for_like() -> None:
    """Not fitted: 5.5 dB is the midpoint of the 3-8 dB Liao states for 1200 km."""
    p = _pass()
    assert p["seconds"] == pytest.approx(PASS_S, rel=0.02)            # 276.5 s
    assert p["detections"] == pytest.approx(DETECTIONS, rel=0.01)     # 3,519,419, -0.9%
    assert p["sifted"] == pytest.approx(SIFTED, rel=0.01)             # 1,655,574, -0.9%
    # "~12 kbit/s at 645 km": within 0.5 dB (10,881 bit/s)
    assert abs(10 * math.log10(p["at_645"] / 12_000)) < 0.5
    # "~1 kbit/s at 1200 km": the model reads high there, by 2.0 dB (1,572 bit/s) - the cost of a
    # midpoint anchor that matches the pass total, stated rather than tolerated silently.
    assert 10 * math.log10(p["at_1200"] / 1_000) == pytest.approx(1.96, abs=0.05)
    drop = 10 * math.log10(p["at_645"] / p["at_1200"])
    assert drop == pytest.approx(8.41, abs=0.05), "Liao's rate fell 10.8 dB over the same ranges"


def test_the_upper_atmosphere_profile_is_the_conservative_reading() -> None:
    """Liao's 8 dB upper value: closest to the pass's slope, 28 percent fewer detections."""
    p = _pass(atmospheric_db=8.0)
    assert p["detections"] / DETECTIONS - 1 == pytest.approx(-0.280, abs=0.005)
    assert 10 * math.log10(p["at_645"] / p["at_1200"]) == pytest.approx(9.78, abs=0.05)
    assert abs(10 * math.log10(p["at_1200"] / 1_000)) < 0.6


def test_the_qsdc_gap_leaves_out_the_station_optics() -> None:
    src = (REPO / "src" / "qsh" / "qkd" / "qsdc.py").read_text()
    assert "params.loss_db() - params.receiver_optics_db()" in src
