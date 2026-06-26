"""Small, FHE/QHE-friendly neural networks (PyTorch).

Models are deliberately small so they (a) train in seconds on the Blackwell GPU,
(b) export cleanly to ONNX for in-browser inference (onnxruntime-web), and (c) keep
homomorphic-evaluation depth shallow for the MPC-FHE benchmark. A square activation
option is provided because x^2 is the canonical FHE-friendly non-linearity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SquareActivation(nn.Module):
    """x -> x^2  (degree-2 polynomial; cheap under CKKS)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class SmallCNN(nn.Module):
    """Compact, configurable CNN for small grayscale images (MedMNIST-style).

    Defaults reproduce the original 28x28 two-block net (channels ``(8, 16)``, avg-pool, 64-wide
    head) so existing training/ONNX export is unchanged. The kwargs let the genuine-FHE path build
    the tiny **FHE-feasible** config — ``SmallCNN(one_block=True, channels=(4,), img_size=14,
    hidden=16)`` = one conv block (1->4 ch, 3x3, pad 1), avg-pool/2 -> 7x7, flatten(196),
    Linear(196->16), ReLU, Linear(16->2) — which runs in seconds as a real encrypted convolution.
    """

    def __init__(self, n_classes: int = 2, in_ch: int = 1, fhe_friendly: bool = False,
                 channels: tuple[int, ...] = (8, 16), pool: str = "avg", img_size: int = 28,
                 hidden: int = 64, one_block: bool = False) -> None:
        super().__init__()
        act = SquareActivation if fhe_friendly else nn.ReLU
        Pool = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        if one_block:
            channels = channels[:1]
        feats: list[nn.Module] = []
        c, s = in_ch, img_size
        for oc in channels:
            feats += [nn.Conv2d(c, oc, 3, padding=1), act(), Pool(2)]
            c, s = oc, s // 2
        self.features = nn.Sequential(*feats)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(c * s * s, hidden), act(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TinyTransformer(nn.Module):
    """Tiny Transformer encoder classifier for short token / feature sequences."""

    def __init__(
        self,
        n_classes: int = 2,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        seq_len: int = 16,
        vocab: int | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model) if vocab else nn.Linear(1, d_model)
        self.is_token = vocab is not None
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_token:
            h = self.embed(x.long())
        else:
            h = self.embed(x.unsqueeze(-1).float())
        h = h + self.pos[:, : h.size(1)]
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class TabularMLP(nn.Module):
    """MLP for tabular features (e.g. finance fraud / genomics SNP vectors)."""

    def __init__(self, n_features: int, n_classes: int = 2, hidden: int = 64,
                 fhe_friendly: bool = False) -> None:
        super().__init__()
        act = SquareActivation if fhe_friendly else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), act(),
            nn.Linear(hidden, hidden // 2), act(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridQNN(nn.Module):
    """Hybrid quantum-classical classifier — the per-client model for **Quantum Federated Learning**
    (the report's QSMPC-QKD-QHE-QFL). A classical encoder feeds a PennyLane **variational quantum
    circuit** (VQC) head, then a classical read-out:

        Linear -> tanh -> Linear -> tanh  ->  VQC(AngleEmbedding + StronglyEntanglingLayers, <Z> per wire)  ->  Linear

    The VQC reuses the qml_ids ansatz; PennyLane's **torch** interface (``diff_method="backprop"`` on
    ``default.qubit``) makes the whole network — including the quantum weights — backprop-trainable, so
    the existing FedAvg loop (torch SGD) federates the quantum parameters too, averaged across sites
    under quantum-safe secure aggregation. Honest scope: a small simulated VQC (not fault-tolerant QFL).

    ``pennylane`` is imported lazily so the base model factory has no hard quantum dependency.
    """

    def __init__(self, n_features: int, n_classes: int = 2, n_qubits: int = 4,
                 n_layers: int = 2, hidden: int = 16) -> None:
        super().__init__()
        import pennylane as qml

        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)}
        self.pre = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, n_qubits), nn.Tanh()
        )
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.head = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import math

        h = self.pre(x) * math.pi          # map tanh output into the angle range [-pi, pi]
        q = self.qlayer(h)
        return self.head(q)


def build_model(kind: str, **kwargs) -> nn.Module:
    """Factory: kind in {cnn, transformer, mlp, qfl}."""
    kind = kind.lower()
    if kind == "cnn":
        return SmallCNN(**kwargs)
    if kind == "transformer":
        return TinyTransformer(**kwargs)
    if kind == "mlp":
        return TabularMLP(**kwargs)
    if kind == "qfl":
        return HybridQNN(**kwargs)
    raise ValueError(f"unknown model kind: {kind!r}")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
