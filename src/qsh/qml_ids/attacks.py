"""QKD attack-scenario telemetry generator (WS2: QML-IDS).

Synthesizes labelled feature vectors for the eight attack scenarios studied in
Al-Kuwari et al., *Resisting QKD Attacks Using QML*, arXiv:2509.14282 (IET Quantum
Comm. 2026): each attack leaves a characteristic signature in the QKD telemetry
(QBER level/variance, sift ratio, detector-click statistics, timing jitter,
multi-photon rate). These features train the intrusion detector.

The report carefully separates two noise sources: (a) QKD *channel* environment
noise and (b) the NISQ hardware noise of the *QML inference engine*. This module
models (a); the detector exposes a hook for (b).
"""

from __future__ import annotations

import numpy as np

ATTACKS = [
    "normal",
    "intercept_resend",
    "photon_number_splitting",
    "trojan_horse",
    "detector_blinding",
    "wavelength_trojan",
    "rng_manipulation",
    "combined",
]

# per-attack signature: (qber_mean, qber_std, sift_ratio, multiphoton_rate, timing_jitter)
_SIGNATURES = {
    "normal":                 (0.015, 0.004, 0.50, 0.02, 0.05),
    "intercept_resend":       (0.135, 0.020, 0.50, 0.02, 0.06),
    "photon_number_splitting":(0.030, 0.008, 0.42, 0.18, 0.05),
    "trojan_horse":           (0.022, 0.006, 0.49, 0.05, 0.20),
    "detector_blinding":      (0.005, 0.003, 0.62, 0.03, 0.08),
    "wavelength_trojan":      (0.028, 0.010, 0.48, 0.06, 0.15),
    "rng_manipulation":       (0.018, 0.030, 0.50, 0.02, 0.05),
    "combined":               (0.090, 0.035, 0.45, 0.12, 0.18),
}

FEATURES = ["qber_mean", "qber_std", "sift_ratio", "multiphoton_rate", "timing_jitter",
            "dark_count_rate", "click_asymmetry"]


def generate(n_per_class: int = 400, seed: int = 0
             ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, label_names) of labelled QKD-session features for 8 classes."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls, name in enumerate(ATTACKS):
        qm, qs, sr, mp, tj = _SIGNATURES[name]
        qber = rng.normal(qm, qs, n_per_class).clip(0, 0.5)
        qstd = rng.normal(qs, qs * 0.3, n_per_class).clip(1e-4, 0.1)
        sift = rng.normal(sr, 0.02, n_per_class).clip(0.2, 0.8)
        multi = rng.normal(mp, mp * 0.3 + 1e-3, n_per_class).clip(0, 1)
        jit = rng.normal(tj, 0.02, n_per_class).clip(0, 1)
        dark = rng.normal(1e-5 * (3 if name == "detector_blinding" else 1), 5e-6, n_per_class).clip(0, None)
        asym = rng.normal(0.5 if name == "detector_blinding" else 0.02, 0.03, n_per_class).clip(0, 1)
        X.append(np.stack([qber, qstd, sift, multi, jit, dark, asym], axis=1))
        y.append(np.full(n_per_class, cls))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], ATTACKS


def binary_labels(y: np.ndarray) -> np.ndarray:
    """Collapse to attack(1) vs normal(0) for the binary IDS metric."""
    return (y != ATTACKS.index("normal")).astype(np.int64)
