"""Synthetic data generators (privacy-safe, dependency-free).

Used when real datasets are unavailable or carry usage/re-identification
restrictions. The transaction generator follows the AMLSim/PaySim spirit (a small
fraction of structured fraud); the genotype generator emits plausible SNP matrices
with a phenotype signal. Higher-fidelity generators exist - Synthea (EHR), IBM AMLSim
(AML graphs), MOSTLY-AI (tabular) - and none is used here: Round 17 removed the AMLSim and
Synthea submodules, which nothing imported (docs/development.md).
"""

from __future__ import annotations

import numpy as np


def synthetic_transactions(n: int = 4000, fraud_rate: float = 0.05, seed: int = 0
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Transaction feature matrix with a learnable fraud signal.

    Features: [log_amount, hour, n_prior_24h, dest_risk, velocity, country_mismatch,
               new_device, amount_zscore]. Fraud shifts amount/velocity/risk.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def block(n_, fraud):
        amt = rng.lognormal(3.5 + (1.2 if fraud else 0), 1.0, n_)
        hour = rng.integers(0, 24, n_).astype(float)
        prior = rng.poisson(2 + (6 if fraud else 0), n_).astype(float)
        risk = rng.beta(2, 5, n_) + (0.4 if fraud else 0)
        vel = rng.gamma(2, 1.0 + (3 if fraud else 0), n_)
        mismatch = (rng.random(n_) < (0.5 if fraud else 0.05)).astype(float)
        new_dev = (rng.random(n_) < (0.4 if fraud else 0.08)).astype(float)
        z = (np.log1p(amt) - 3.5) / 1.0
        return np.stack([np.log1p(amt), hour, prior, np.clip(risk, 0, 1), vel,
                         mismatch, new_dev, z], axis=1)

    X = np.vstack([block(n_legit, False), block(n_fraud, True)]).astype(np.float32)
    y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(np.int64)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def synthetic_genotypes(n: int = 1000, n_snps: int = 64, seed: int = 0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Genotype matrix in {0,1,2} (minor-allele count) with a phenotype signal."""
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.5, n_snps)               # minor allele frequency per SNP
    G = rng.binomial(2, maf, size=(n, n_snps)).astype(np.float32)
    causal = rng.choice(n_snps, size=max(1, n_snps // 8), replace=False)
    weights = rng.normal(0, 1, causal.size)
    logits = (G[:, causal] - 2 * maf[causal]) @ weights
    p = 1 / (1 + np.exp(-logits / np.std(logits)))
    y = (rng.random(n) < p).astype(np.int64)
    return G, y


def synthetic_images(n: int = 512, size: int = 28, seed: int = 0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Two-class 28x28 grayscale images (blob vs stripes) for offline CNN demos."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, size, size), np.float32)
    y = rng.integers(0, 2, n)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        if y[i] == 0:  # gaussian blob
            cx, cy = rng.integers(8, 20, 2)
            X[i, 0] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 20.0)
        else:          # stripes
            X[i, 0] = (np.sin((xx + rng.integers(0, 6)) * 0.9) > 0).astype(np.float32)
        X[i, 0] += rng.normal(0, 0.05, (size, size))
    return np.clip(X, 0, 1).astype(np.float32), y.astype(np.int64)
