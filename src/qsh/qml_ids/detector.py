"""Intrusion detectors for QKD attack telemetry (WS2: QML-IDS).

Two detectors with the same interface:
  * :class:`ClassicalIDS` -- RandomForest baseline (the report's Phase-1 classical SOTA).
  * :class:`QuantumIDS`   -- a PennyLane variational quantum classifier (angle-encoded,
    the WS2 research target; smaller parameter count).

Both report accuracy, macro-F1, and the binary (attack vs normal) F1.

WHICH NUMBER IS THE HEADLINE, because the choice flatters one detector. This docstring used to name
the report's binary F1 >= 0.93 as the criterion these detectors are tracked against, and call the
quantum one "the report's selling point". Measured on the shipped corpus, ``QuantumIDS`` scores
binary F1 0.9333 - clearing that bar by 0.003 - while its MACRO F1 is 0.4667 against a RandomForest
at 0.9866. An eight-class detector that collapses attack types into "something happened" can pass a
binary threshold while being close to useless per class, and this project treats an aggregate that
hides a broken class as a defect rather than a result. ``models/metrics.json`` records
``quantum_beats_classical: false``; read macro-F1 as the headline and the binary figure as the
report's historical target, not as evidence of a working multi-class detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .attacks import binary_labels, generate


@dataclass
class IDSReport:
    detector: str
    accuracy: float
    macro_f1: float
    binary_f1: float
    n_params: int


class ClassicalIDS:
    """RandomForest multi-class IDS baseline."""

    name = "random-forest"

    def __init__(self, n_estimators: int = 200, seed: int = 0) -> None:
        from sklearn.ensemble import RandomForestClassifier

        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    @property
    def n_params(self) -> int:
        return sum(t.tree_.node_count for t in self.clf.estimators_)


class QuantumIDS:
    """Variational quantum classifier (PennyLane) for attack detection.

    Angle-encodes the (PCA-reduced) features on ``n_qubits`` and trains a small
    StronglyEntanglingLayers ansatz. Few parameters by design -- the report's
    "low-resource implementation" claim for Hybrid QLSTM / QSVM.
    """

    name = "variational-qnn"

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, seed: int = 0) -> None:
        import pennylane as qml
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler

        self.qml = qml
        self.n_qubits = n_qubits
        self.pca = PCA(n_components=n_qubits, random_state=seed)
        self.scaler = MinMaxScaler((0, np.pi))
        self.dev = qml.device("default.qubit", wires=n_qubits)
        shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.2, shape)
        self.bias = 0.0

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, w):
            qml.AngleEmbedding(x, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def _encode(self, X, fit=False):
        Z = self.pca.fit_transform(X) if fit else self.pca.transform(X)
        return self.scaler.fit_transform(Z) if fit else self.scaler.transform(Z)

    def fit(self, X, y, *, epochs: int = 25, lr: float = 0.1):
        from pennylane import numpy as pnp

        Xe = self._encode(X, fit=True)
        yb = binary_labels(y) * 2 - 1                      # {-1, +1}
        w = pnp.array(self.weights, requires_grad=True)
        b = pnp.array(self.bias, requires_grad=True)
        opt = self.qml.AdamOptimizer(lr)

        def loss(w, b):
            preds = pnp.array([self._circuit(x, w) + b for x in Xe])
            return pnp.mean((preds - yb) ** 2)

        for _ in range(epochs):
            w, b = opt.step(loss, w, b)
        self.weights, self.bias = np.array(w), float(b)
        return self

    def predict(self, X):
        Xe = self._encode(X)
        preds = np.array([self._circuit(x, self.weights) + self.bias for x in Xe])
        return (preds > 0).astype(int)                     # binary attack/normal

    @property
    def n_params(self) -> int:
        return int(self.weights.size) + 1


class QuantumKernelIDS:
    """Quantum-kernel SVM (QSVC) for attack detection — the 2026 QML-IDS standard (QSVC/QRF survey;
    "QSVM best" for QKD-attack telemetry). A PennyLane fidelity feature map defines a quantum kernel
    ``k(x1,x2) = |<phi(x1)|phi(x2)>|^2``; an sklearn SVC trains on the precomputed kernel.

    CLASSICAL IN CLOSED FORM. The feature map is AngleEmbedding alone (one rotation per wire, no
    entangling gate), so ``phi(x)`` is a product state and the kernel equals the classical product
    kernel ``prod_i cos^2((x1_i - x2_i) / 2)`` exactly (checked to 2.2e-16 on 200 random pairs at
    4 wires). It costs O(n) per pair classically; no quantum resource is involved.
    """

    name = "quantum-svc"

    def __init__(self, n_qubits: int = 4, seed: int = 0) -> None:
        import pennylane as qml
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.svm import SVC

        self.qml = qml
        self.n_qubits = n_qubits
        self.pca = PCA(n_components=n_qubits, random_state=seed)
        self.scaler = MinMaxScaler((0, np.pi))
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):                        # fidelity kernel: embed x1, un-embed x2
            qml.AngleEmbedding(x1, wires=range(n_qubits))
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))

        self._kc = kernel_circuit
        self.svc = SVC(kernel="precomputed")

    def _kernel(self, A, B):
        return np.array([[float(self._kc(a, b)[0]) for b in B] for a in A])   # probs[0] = |<phi(a)|phi(b)>|^2

    def _encode(self, X, fit=False):
        Z = self.pca.fit_transform(X) if fit else self.pca.transform(X)
        return self.scaler.fit_transform(Z) if fit else self.scaler.transform(Z)

    def fit(self, X, y):
        self._Xtr = self._encode(X, fit=True)
        self.svc.fit(self._kernel(self._Xtr, self._Xtr), binary_labels(y))
        return self

    def predict(self, X):
        Xe = self._encode(X)
        return self.svc.predict(self._kernel(Xe, self._Xtr))

    @property
    def n_params(self) -> int:
        return int(self.svc.n_support_.sum())              # support vectors define the quantum decision boundary


def evaluate_quantum_svc(n_per_class: int = 60, seed: int = 0,
                         test_fraction: float | None = None) -> IDSReport:
    """Train + evaluate the quantum-kernel SVM IDS (binary attack/normal). Kept small (the kernel is
    O(n^2) circuit evaluations) — the 2026 QSVC standard, demonstrated on the QKD-attack telemetry."""
    if test_fraction is None:
        from ..config import get_config

        test_fraction = get_config().ids.test_fraction
    X, y, _ = generate(n_per_class, seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_fraction, random_state=seed, stratify=y)
    ids = QuantumKernelIDS(seed=seed).fit(Xtr, ytr)
    pred = ids.predict(Xte)
    yb = binary_labels(yte)
    return IDSReport(ids.name, float((pred == yb).mean()),
                     float(f1_score(yb, pred, average="macro")), float(f1_score(yb, pred)), ids.n_params)


def evaluate_classical(n_per_class: int = 300, seed: int = 0,
                       test_fraction: float | None = None) -> IDSReport:
    """Train + evaluate the classical multi-class IDS."""
    if test_fraction is None:
        from ..config import get_config

        test_fraction = get_config().ids.test_fraction
    X, y, _ = generate(n_per_class, seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_fraction, random_state=seed, stratify=y)
    ids = ClassicalIDS(seed=seed).fit(Xtr, ytr)
    pred = ids.predict(Xte)
    return IDSReport(
        ids.name,
        float((pred == yte).mean()),
        float(f1_score(yte, pred, average="macro")),
        float(f1_score(binary_labels(yte), binary_labels(pred))),
        ids.n_params,
    )


def evaluate_quantum(n_per_class: int = 120, seed: int = 0,
                     test_fraction: float | None = None, **fit_kw) -> IDSReport:
    """Train + evaluate the variational quantum IDS (binary attack/normal)."""
    if test_fraction is None:
        from ..config import get_config

        test_fraction = get_config().ids.test_fraction
    X, y, _ = generate(n_per_class, seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_fraction, random_state=seed, stratify=y)
    ids = QuantumIDS(seed=seed).fit(Xtr, ytr, **fit_kw)
    pred = ids.predict(Xte)
    yb = binary_labels(yte)
    return IDSReport(
        ids.name,
        float((pred == yb).mean()),
        float(f1_score(yb, pred, average="macro")),
        float(f1_score(yb, pred)),
        ids.n_params,
    )
