"""Tests for the AI layer: models, training, secure aggregation, federated learning."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qsh.ai.federated import SecureAggregator, federated_train, partition
from qsh.ai.models import build_model
from qsh.ai.train import train_model
from qsh.data.synthetic import synthetic_transactions


def test_models_build_and_forward():
    import torch

    cnn = build_model("cnn", n_classes=2)
    assert cnn(torch.randn(2, 1, 28, 28)).shape == (2, 2)
    mlp = build_model("mlp", n_features=8, n_classes=2)
    assert mlp(torch.randn(3, 8)).shape == (3, 2)
    tr = build_model("transformer", n_classes=2, seq_len=16)
    assert tr(torch.randn(2, 16)).shape == (2, 2)


def test_secure_aggregation_masks_cancel():
    agg = SecureAggregator(5)
    updates = [np.random.default_rng(i).normal(size=30) for i in range(5)]
    masked = [agg.mask_update(c, u) for c, u in enumerate(updates)]
    recovered = agg.aggregate(masked)
    assert np.allclose(recovered, np.sum(updates, axis=0), atol=1e-9)
    # an individual masked update must differ from the raw update (server can't see it)
    assert not np.allclose(masked[0], updates[0])


def test_centralized_training_learns():
    X, y = synthetic_transactions(n=1500, seed=0)
    res = train_model(X, y, kind="mlp", epochs=8, seed=0)
    assert res.test_acc > 0.8


def test_federated_matches_centralized_ballpark():
    X, y = synthetic_transactions(n=2000, seed=0)
    parts = partition(X[:1600], y[:1600], n_clients=4, iid=True, seed=0)
    fr = federated_train(parts, kind="mlp", rounds=6, eval_data=(X[1600:], y[1600:]),
                         secure_aggregation=True, seed=0)
    assert fr.secure_aggregation
    assert fr.global_acc > 0.8


def test_quantum_federated_learning_trains():
    """QFL: a hybrid quantum-classical (PennyLane VQC + classical) model federated via FedAvg with
    quantum-safe secure aggregation learns a separable task (the report's QSMPC-QKD-QHE-QFL path)."""
    pytest.importorskip("pennylane")
    rng = np.random.default_rng(0)
    n = 240
    X = np.vstack([rng.normal(-1.0, 0.7, (n // 2, 6)),
                   rng.normal(1.0, 0.7, (n // 2, 6))]).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    parts = partition(X[:180], y[:180], n_clients=3, iid=True, seed=0)
    fr = federated_train(parts, kind="qfl", rounds=5, local_epochs=1, lr=0.1,
                         eval_data=(X[180:], y[180:]), secure_aggregation=True,
                         model_kwargs={"n_qubits": 4, "n_layers": 2}, seed=0)
    assert fr.secure_aggregation
    assert fr.n_clients == 3
    assert len(fr.round_acc) == 5
    assert fr.global_acc > 0.75              # genuinely learns (smoke saw ~0.98)
    assert fr.round_acc[-1] >= fr.round_acc[0]  # improves over rounds


def test_onnx_export_parity(tmp_path):
    pytest.importorskip("onnxscript")
    from qsh.ai.inference import export_onnx, verify_onnx

    X, y = synthetic_transactions(n=800, seed=0)
    res = train_model(X, y, kind="mlp", epochs=4, seed=0)
    p = export_onnx(res.model, X[:1], tmp_path / "m.onnx")
    assert verify_onnx(p, res.model, X[:4])
