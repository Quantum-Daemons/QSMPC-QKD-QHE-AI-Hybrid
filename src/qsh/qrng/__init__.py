"""Quantum random-number generation (entropy source).

Default = a local state-vector quantum-measurement simulation (honest: simulated, not hardware),
with SP 800-22-style health checks + von Neumann / Toeplitz post-processing. Optional real-hardware
source = the ANU QRNG API (opt-in, online). Feeds the BB84 entropy via :func:`quantum_seed`.
"""

from .entropy_source import quantum_seed
from .health import HealthReport, health_suite
from .simulator import (
    QRNGResult,
    generate_quantum_bits,
    measure_plus_qubits,
    toeplitz_extract,
    von_neumann,
)

__all__ = [
    "generate_quantum_bits",
    "measure_plus_qubits",
    "von_neumann",
    "toeplitz_extract",
    "QRNGResult",
    "health_suite",
    "HealthReport",
    "quantum_seed",
]
