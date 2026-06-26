"""Quantum channel model for QKD simulation.

Captures the real-world impairments the report's WS1 (AI-QKD parameter
optimization) must track: fibre loss, polarization/phase drift, detector dark
counts, and an optional intercept-resend eavesdropper. Outputs feed the QBER /
SKR telemetry consumed by the AI optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelParams:
    """Physical channel + detector parameters."""

    distance_km: float = 50.0
    fiber_loss_db_per_km: float = 0.2          # standard SMF-28 @1550nm
    detector_efficiency: float = 0.25
    dark_count_prob: float = 1e-5              # per gate
    misalignment: float = 0.015               # baseline optical error (polarization drift)
    eve_intercept_fraction: float = 0.0       # 0..1 fraction of pulses Eve intercept-resends

    def transmittance(self) -> float:
        """End-to-end channel + detector transmittance (probability a pulse is detected)."""
        loss_db = self.distance_km * self.fiber_loss_db_per_km
        return self.detector_efficiency * 10 ** (-loss_db / 10)


def intrinsic_qber(params: ChannelParams) -> float:
    """Analytic baseline QBER from misalignment + dark counts + any eavesdropping.

    Intercept-resend on a fraction f of pulses adds ~f * 0.25 to the error rate
    (the textbook BB84 disturbance of an intercept-resend attack).
    """
    base = params.misalignment + 0.5 * params.dark_count_prob / max(params.transmittance(), 1e-9)
    eve = 0.25 * params.eve_intercept_fraction
    return float(min(0.5, base + eve))
