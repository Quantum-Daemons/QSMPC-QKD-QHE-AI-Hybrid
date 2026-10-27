"""Quantum channel model for QKD simulation.

Captures the real-world impairments the report's WS1 (AI-QKD parameter
optimization) must track: fibre loss, polarization/phase drift, detector dark
counts, and an optional intercept-resend eavesdropper. Outputs feed the QBER /
SKR telemetry consumed by the AI optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _cfg(name: str):
    """A default read from `QKDConfig` when the dataclass is built, not copied into this file.

    These defaults were LITERALS - 0.25, 1e-5, 0.015 - equal to the config only because nobody had
    changed it. `orchestration/stages.py` builds its fibre channel from three config keys and lets the
    rest default, so a config change reached the browser (which reads config) and left the Python
    stage on the literals. Round 76's change of detector set is what would have split them.
    """
    def read() -> float:
        from ..config import get_config
        return float(getattr(get_config().qkd, name))
    return field(default_factory=read)


@dataclass
class FreeSpaceParams:
    """Diffraction terms of a satellite downlink, for links that are not fibre.

    Mirrors ``FreeSpaceSpec`` in ``web/src/lib/quantumSim.ts``; both sets of defaults come from
    ``QKDConfig.free_space_*`` (Liao et al. 2017, value by value).
    """

    beam_divergence_urad: float = _cfg("free_space_beam_divergence_urad")
    rx_aperture_m: float = _cfg("free_space_rx_aperture_m")
    atmospheric_db: float = _cfg("free_space_atmospheric_db")     # crossed once - NOT per km
    pointing_db: float = _cfg("free_space_pointing_db")
    #: Receiving telescope plus fibre coupling, "~16%" in Liao 2017 - its own budget line since
    #: Round 75, which the model did not have.
    receiver_optics_efficiency: float = _cfg("free_space_receiver_optics_efficiency")


@dataclass
class ChannelParams:
    """Physical channel + detector parameters. Defaults are the fibre set in `QKDConfig`."""

    distance_km: float = _cfg("distance_km")
    fiber_loss_db_per_km: float = _cfg("fiber_loss_db_per_km")
    detector_efficiency: float = _cfg("detector_efficiency")
    dark_count_prob: float = _cfg("dark_count_prob")              # per gate, per detector
    misalignment: float = _cfg("misalignment")                    # baseline optical error
    eve_intercept_fraction: float = 0.0       # 0..1 fraction of pulses Eve intercept-resends
    free_space: FreeSpaceParams | None = None  # set for a downlink; None keeps the fibre law
    #: The source clock and the detectors' dead time, which together decide how many detections
    #: arrive while a detector is still recovering (`dead_time_factor`). Left unset they take the
    #: link's own: the fibre set's, or the downlink's when `free_space` is given - so a downlink built
    #: without them does not inherit the fibre's 1 GHz and 100 ns.
    source_rate_hz: float | None = None
    dead_time_s: float | None = None

    def __post_init__(self) -> None:
        from ..config import get_config
        q = get_config().qkd
        downlink = self.free_space is not None
        if self.source_rate_hz is None:
            self.source_rate_hz = float(q.free_space_source_rate_hz if downlink else q.source_rate_hz)
        if self.dead_time_s is None:
            self.dead_time_s = float(q.free_space_detector_dead_time_s if downlink
                                     else q.detector_dead_time_s)

    def loss_db(self) -> float:
        """One-way channel loss. Two physical regimes, and one law for both was a real error.

        FIBRE is attenuated per kilometre, so loss is linear in distance. Unchanged, and what
        ``free_space=None`` still gives.

        A SATELLITE DOWNLINK is dominated by diffraction: the beam leaves with far-field divergence
        ``theta``, has spread to roughly ``theta * r`` across on arrival, and the ground telescope
        captures ``(D_rx / (theta * r))**2`` - an inverse SQUARE law, not a linear one. The
        atmosphere adds a few fixed dB because a downlink crosses it once, near the ground, rather
        than along the whole path.

        WHY THIS EXISTS. The browser replaced the linear model in an earlier round and this file did
        not, so the same nominal 800 km downlink measured 24.48 dB there and 28.0 dB here
        (``sat_loss_db_per_km = 0.035``, the very rate ``quantumSim.channelLossDb``'s comment records
        as rejected: it reads plausibly at 800 km (beside a "~27 dB" this comment once credited to Micius; Liao et al.
        state no total) and then diverges
        badly - 42 dB at 1200 km where physics gives about 28, 175 against 40 at 5000). Nothing
        surfaced the disagreement because BOTH sides clamped secrecy capacity to 0 on the abort path,
        so each reported "0" for what were really -51.58 and -138.93 bit/carrier. Reporting the sign
        is what exposed it.

        Liao et al., Nature 549, 43 (2017) / arXiv:1707.00542.
        """
        if self.free_space is None:
            return self.distance_km * self.fiber_loss_db_per_km
        fs = self.free_space
        beam_diameter_m = (fs.beam_divergence_urad * 1e-6) * (self.distance_km * 1e3)
        # Capped at 1: inside the near field the receiver cannot capture more than the whole beam.
        captured = min(1.0, fs.rx_aperture_m / max(beam_diameter_m, 1e-9))
        geometric_db = -20.0 * float(np.log10(captured))
        optics_db = -10.0 * float(np.log10(max(fs.receiver_optics_efficiency, 1e-12)))
        return geometric_db + fs.atmospheric_db + fs.pointing_db + optics_db

    def receiver_optics_db(self) -> float:
        """The ground station's own optics (telescope + fibre coupling) in dB; 0 on fibre.

        Part of `loss_db`, because it removes signal photons, and NOT part of QSDC's Eve-to-Bob gap
        `g` (see `qsdc.py`): Qi et al.'s system lost about 13 dB in its optical elements, of a 25.1 dB
        total, while their g counted 4.1 dB of back channel and detector.
        """
        if self.free_space is None:
            return 0.0
        return -10.0 * float(np.log10(max(self.free_space.receiver_optics_efficiency, 1e-12)))

    def transmittance(self) -> float:
        """End-to-end channel + detector transmittance (probability a pulse is detected)."""
        return self.detector_efficiency * 10 ** (-self.loss_db() / 10)


def dead_time_factor(params: ChannelParams, click_prob: float) -> float:
    """The fraction of detections a dead detector still registers, after Rusca et al. 2018 Eq. (B3).

    ``c = 1 / (1 + R * D * t_DT)``, with ``R`` the source clock, ``t_DT`` the dead time and ``D`` the
    probability per pulse that a given detector clicks, before the correction: the non-paralysable
    dead-time model.

    Where this departs from (B3) as printed, and why. Rusca et al. write ``D`` as the sum of their
    Eq. (B2), which counts only detections where Alice and Bob both chose Z and which already
    carries ``c``. Taken literally that makes (B3) implicit, and taking ``D`` before the correction
    is the reading under which it is the standard formula. A detector is dead after ANY click,
    though, matched basis or not, so ``D`` here counts every click: Bob has two detectors - the
    model's background yield is ``2 * p_dark`` - each receiving half of them, so
    ``D = click_prob / 2`` with ``click_prob`` the total detection probability per pulse at Bob,
    averaged over what Alice sends. At this project's symmetric bases that charges twice the dead
    time the Z-only sum would, the conservative side.

    Signal and noise clicks are lost alike, so the QBER is unchanged and every gain, and with it the
    key rate, is multiplied by ``c``.
    """
    rt = float(params.source_rate_hz) * float(params.dead_time_s)
    if rt <= 0.0:
        return 1.0
    return 1.0 / (1.0 + rt * 0.5 * float(click_prob))


def compose_independent_errors(a: float, b: float) -> float:
    """Error rate after two INDEPENDENT bit-flip sources: a(1 - b) + b(1 - a) = a + b - 2ab.

    Two flips on the same bit cancel, so independent error sources do not simply add. Until
    2026-10-24 the eavesdropper's disturbance was ADDED to the channel error, which overcounts by 2ab:
    at 50 km with a full intercept-resend attack the model QBER read 0.2722 where this gives 0.2649.
    `quantumSim.ts::composeIndependentErrors` is the browser twin; both languages' two call sites
    (`intrinsic_qber` / `e_det_for_bound` here and in `bb84.py`) use it, which is what the
    cross-language rate test holds them to.
    """
    return a + b - 2.0 * a * b


def intrinsic_qber(params: ChannelParams) -> float:
    """Analytic baseline QBER from misalignment + dark counts, composed with any eavesdropping.

    Intercept-resend on a fraction f of pulses flips a bit with probability 0.25 * f (the textbook
    BB84 disturbance), independently of the channel's own errors, so the two are composed with
    `compose_independent_errors` rather than added. The channel's base term (misalignment plus the
    dark-count share) stays a sum: it is a small-error approximation this model has always used,
    and composing it too would move every Eve-free figure for no physical gain at these rates.
    """
    base = params.misalignment + 0.5 * params.dark_count_prob / max(params.transmittance(), 1e-9)
    eve = 0.25 * params.eve_intercept_fraction
    return float(min(0.5, compose_independent_errors(base, eve)))
