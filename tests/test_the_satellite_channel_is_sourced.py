"""The satellite downlink's parameters are Micius's, each traceable to Liao et al. 2017.

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

WHAT IS ASSERTED

- Each free-space value equals the figure Liao et al. state, which is quoted beside it here.
- The loss budget carries the station optics, and QSDC's Eve-to-Bob gap leaves them out, as Qi et
  al.'s back channel did.
- The satellite stage reads the channel from the config, not from literals of its own.
- The model reproduces a Micius measurement it was not fitted to: at Micius's own signal intensity
  (0.8), the sifted rate at 645 km is within 10 percent of Liao's "~12 kbit/s".
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qsh.config import get_config
from qsh.qkd.channel import ChannelParams, FreeSpaceParams

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
]


@pytest.mark.parametrize("key,value,quote", LIAO_2017, ids=[k for k, _, _ in LIAO_2017])
def test_each_value_is_liaos(key: str, value: float, quote: str) -> None:
    assert getattr(get_config().qkd, key) == pytest.approx(value), f"{key}: Liao 2017 says {quote!r}"


def test_atmosphere_and_pointing_sit_inside_liaos_estimates() -> None:
    q = get_config().qkd
    assert 3.0 <= q.free_space_atmospheric_db <= 8.0, "Liao: 3 dB to 8 dB at 1200 km"
    assert q.free_space_pointing_db <= 3.0, "Liao: less than 3 dB"


def _downlink(distance_km: float) -> ChannelParams:
    q = get_config().qkd
    return ChannelParams(
        distance_km=distance_km, detector_efficiency=q.free_space_detector_efficiency,
        dark_count_prob=q.free_space_dark_count_prob, misalignment=q.free_space_misalignment,
        free_space=FreeSpaceParams(
            beam_divergence_urad=q.free_space_beam_divergence_urad,
            rx_aperture_m=q.free_space_rx_aperture_m, atmospheric_db=q.free_space_atmospheric_db,
            pointing_db=q.free_space_pointing_db,
            receiver_optics_efficiency=q.free_space_receiver_optics_efficiency))


def test_the_budget_carries_the_station_optics() -> None:
    c = _downlink(800.0)
    geometric = -20 * math.log10(1.0 / (10e-6 * 800e3))
    assert c.receiver_optics_db() == pytest.approx(-10 * math.log10(0.16))
    assert c.loss_db() == pytest.approx(geometric + 5.0 + 3.0 + c.receiver_optics_db())
    # Liao's own diffraction estimate at 1200 km is 22 dB; this law gives 21.6.
    assert -20 * math.log10(1.0 / (10e-6 * 1200e3)) == pytest.approx(22.0, abs=0.5)


def test_the_satellite_stage_reads_the_channel_from_config() -> None:
    from qsh.usecases.satellite_comms import _SAT_DEFAULTS
    for key in ("sat_distance_km", "sat_detector_efficiency", "sat_dark_count_prob",
                "sat_misalignment", "sat_atmospheric_db", "sat_pointing_db"):
        assert _SAT_DEFAULTS[key] is None, f"{key} is a literal again: {_SAT_DEFAULTS[key]}"


def test_the_model_reproduces_micius_sifted_rate_at_645_km() -> None:
    """Not fitted to it: 12.2 kbit/s from the model at Micius's mu = 0.8 against Liao's ~12."""
    c = _downlink(645.0)
    mu, rate_hz = 0.8, get_config().qkd.free_space_source_rate_hz
    sifted = 0.5 * (2 * c.dark_count_prob + 1 - math.exp(-c.transmittance() * mu)) * rate_hz
    assert sifted == pytest.approx(12_000, rel=0.10), f"{sifted:.0f} bit/s"


def test_the_qsdc_gap_leaves_out_the_station_optics() -> None:
    src = (REPO / "src" / "qsh" / "qkd" / "qsdc.py").read_text()
    assert "params.loss_db() - params.receiver_optics_db()" in src
