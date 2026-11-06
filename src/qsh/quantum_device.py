"""Which state-vector simulator the Python tiers' PennyLane circuits run on, from one setting: ``quantum_sim.device``.

Four sites wrote ``qml.device("default.qubit", ...)`` (Round 86 found them), so the GPU simulators the full deployment can
install had no way in. ``default.qubit`` stays the default: it runs everywhere, CI included, and its backpropagation is
what trains the hybrid network end to end. On this machine a GPU simulator overtakes it before any register size these
circuits use matters (docs/models-and-benchmarks.md, "State-vector simulators, measured"), so the choice is for larger
experiments, not for the shipped models.

A device PennyLane cannot load is an error naming the setting, never a quiet return to ``default.qubit``.
"""

from __future__ import annotations

from .config import get_config

#: Gradients each device supports for the hybrid network. `default.qubit` differentiates by backpropagation through its
#: own arithmetic; the Lightning simulators by the adjoint method, which they implement natively.
_GRADIENT = {"default.qubit": "backprop", "lightning.qubit": "adjoint", "lightning.gpu": "adjoint",
             "lightning.kokkos": "adjoint"}


def device_name() -> str:
    """The configured simulator."""
    return get_config().quantum_sim.device


def make_device(n_qubits: int):
    """A PennyLane device on ``n_qubits`` wires, from the configured simulator."""
    import pennylane as qml

    name = device_name()
    try:
        return qml.device(name, wires=n_qubits)
    except Exception as e:  # PennyLane raises DeviceError, ImportError or a plugin's own error
        raise RuntimeError(
            f"quantum_sim.device is {name!r}, and PennyLane cannot load it here ({e}). "
            "`lightning.gpu` needs the packages in requirements-gpu.txt and an NVIDIA GPU.") from e


def gradient_method() -> str:
    """The ``diff_method`` a trainable QNode uses on the configured simulator."""
    name = device_name()
    if name not in _GRADIENT:
        raise RuntimeError(f"quantum_sim.device is {name!r}; no gradient method is recorded for it in qsh.quantum_device")
    return _GRADIENT[name]
