"""The Python tiers' circuits run on one configured simulator, and every simulator gives them the same numbers.

WHY THIS EXISTS (Round 86, the owner's choice of the whole GPU path). Four sites wrote `qml.device("default.qubit", ...)`, so
`lightning.gpu`, which runs on this machine's Blackwell GPU, had no way in. Each site now asks `qsh.quantum_device`. A different
simulator must not change a result beyond rounding, so each circuit is evaluated on two and compared.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from qsh.config import get_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def device(monkeypatch):
    def use(name: str) -> None:
        monkeypatch.setattr(get_config().quantum_sim, "device", name)
    return use


def test_the_default_stays_the_portable_simulator() -> None:
    from qsh.config.defaults import QSHConfig

    assert QSHConfig().quantum_sim.device == "default.qubit"


def test_no_module_names_a_simulator_itself() -> None:
    offenders = [str(p.relative_to(REPO)) for p in (REPO / "src/qsh").rglob("*.py")
                 if p.name != "quantum_device.py" and re.search(r"qml\.device\(\s*[\"']", p.read_text(encoding="utf-8"))]
    assert not offenders, f"these choose their own simulator: {offenders}"


def test_a_simulator_pennylane_cannot_load_is_an_error_not_the_default(device) -> None:
    from qsh.quantum_device import make_device

    device("no.such.device")
    with pytest.raises(RuntimeError, match="quantum_sim.device is 'no.such.device'"):
        make_device(2)


def test_the_qcnn_gives_the_same_output_on_lightning(device) -> None:
    from qsh.qhe.qcnn import QCNN

    x = np.linspace(-1.0, 1.0, 8)
    reference = QCNN(n_qubits=8, seed=3)
    out_default = reference.forward(x) if hasattr(reference, "forward") else reference(x)
    device("lightning.qubit")
    other = QCNN(n_qubits=8, seed=3)
    out_lightning = other.forward(x) if hasattr(other, "forward") else other(x)
    assert np.allclose(np.asarray(out_default, dtype=float), np.asarray(out_lightning, dtype=float), atol=1e-9)


def test_the_hybrid_network_trains_on_lightning_by_the_adjoint_method(device) -> None:
    import torch

    from qsh.ai.models import HybridQNN

    torch.manual_seed(0)
    x = torch.randn(5, 6)
    net_default = HybridQNN(n_features=6, n_classes=2, n_qubits=4, n_layers=2)
    state = net_default.state_dict()
    device("lightning.qubit")
    net_lightning = HybridQNN(n_features=6, n_classes=2, n_qubits=4, n_layers=2)
    net_lightning.load_state_dict(state)
    assert torch.allclose(net_default(x), net_lightning(x), atol=1e-6)
    loss = net_lightning(x).sum()
    loss.backward()
    grads = [p.grad for p in net_lightning.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "the adjoint gradient did not reach the weights"


def test_a_malformed_config_file_is_an_error(tmp_path) -> None:
    from qsh.config import load_config

    bad = tmp_path / "config.yaml"
    bad.write_text("quantum_sim: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_config(bad)


def test_lightning_gpu_matches_where_it_is_installed(device) -> None:
    pytest.importorskip("pennylane_lightning.lightning_gpu", reason="requirements-gpu.txt is not installed here")
    from qsh.qhe.qcnn import QCNN

    x = np.linspace(-1.0, 1.0, 8)
    ref = QCNN(n_qubits=8, seed=3)
    device("lightning.gpu")
    try:
        gpu = QCNN(n_qubits=8, seed=3)
    except RuntimeError as e:
        pytest.skip(f"no usable GPU: {e}")
    f = (lambda m: m.forward(x)) if hasattr(ref, "forward") else (lambda m: m(x))
    assert np.allclose(np.asarray(f(ref), dtype=float), np.asarray(f(gpu), dtype=float), atol=1e-9)
