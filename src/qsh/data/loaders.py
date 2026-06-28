"""Dataset loaders for the finance / medical / genomics use cases.

Prefers free public data (MedMNIST CC-BY, scikit-learn bundled sets); falls back to
synthetic generators (:mod:`~qsh.data.synthetic`) when a download is unavailable or a
dataset has usage restrictions. All loaders return NumPy arrays ``(X, y)``.
"""

from __future__ import annotations

import numpy as np


def load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    """Wisconsin breast-cancer (medical tabular, bundled with scikit-learn)."""
    from sklearn.datasets import load_breast_cancer as _lbc
    from sklearn.preprocessing import StandardScaler

    ds = _lbc()
    X = StandardScaler().fit_transform(ds.data).astype(np.float32)
    return X, ds.target.astype(np.int64)


def load_medmnist(name: str = "pneumoniamnist", split: str = "train"
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Load a MedMNIST 2D dataset (real data; CC-BY 4.0). Returns X ``(N, 1, 28, 28)`` float32 in
    ``[0, 1]`` and integer labels.

    The full Docker image **bakes the dataset at build time** so this needs no network at runtime.
    If MedMNIST is genuinely unavailable (e.g. an offline CI box with no cached data), it falls back
    to a clearly-logged **synthetic** image set — that path is for offline tests only and is NEVER a
    silent substitute for real data in a real run (per the project's no-easy-fallback policy).
    """
    try:
        import medmnist
        from medmnist import INFO

        info = INFO[name]
        cls = getattr(medmnist, info["python_class"])
        ds = cls(split=split, download=True)
        X = ds.imgs.astype(np.float32) / 255.0
        if X.ndim == 3:
            X = X[:, None, :, :]                       # (N,1,28,28)
        else:
            X = np.transpose(X, (0, 3, 1, 2))
        y = ds.labels.reshape(-1).astype(np.int64)
        return X, y
    except Exception as exc:  # pragma: no cover - offline-CI fallback only (logged, not silent)
        import logging

        logging.getLogger("qsh").warning(
            "MedMNIST '%s' unavailable (%s) — falling back to SYNTHETIC images. This is an "
            "offline/CI path ONLY and is NOT real medical data; the full Docker image bakes the "
            "real dataset so production runs use it.", name, exc)
        from .synthetic import synthetic_images

        return synthetic_images(n=512)


def load_finance_fraud(n: int = 4000, fraud_rate: float = 0.05, seed: int = 0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic transaction features + fraud labels (PaySim/AMLSim-style).

    See :mod:`~qsh.data.synthetic`. Use real PaySim/credit-card-fraud CSVs in
    production; the synthetic generator keeps the demo dependency-free + privacy-safe.
    """
    from .synthetic import synthetic_transactions

    return synthetic_transactions(n=n, fraud_rate=fraud_rate, seed=seed)


def load_genomic(n: int = 1000, n_snps: int = 64, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic SNP genotype matrix + phenotype labels.

    Real demos can substitute 1000 Genomes (CC-BY) / OpenSNP (CC0); we generate
    synthetic genotypes to avoid re-identification risk in the public demo.
    """
    from .synthetic import synthetic_genotypes

    return synthetic_genotypes(n=n, n_snps=n_snps, seed=seed)


# Bundled real Polygenic Risk Scores (PGS Catalog, CC-BY 4.0). Each is a real published per-SNP
# effect-weight table → the model is a genuine linear PRS = Σ weight_i · dosage_i, which runs as an
# exact CKKS dot-product (the published "HE-applied-to-PRS" pattern; FHE-native, no surrogate).
_PRS_FILES = {"breast_cancer": "pgs000001_breast_cancer.tsv"}


def load_prs(name: str = "breast_cancer") -> tuple[list[str], list[str], np.ndarray]:
    """Load a real PGS-Catalog Polygenic Risk Score → ``(rsids, effect_alleles, effect_weights)``.

    The weights are the real published effect sizes (PGS000001 = a 77-SNP breast-cancer PRS, Mavaddat
    et al.). Bundled under ``data/genomics/`` so it runs offline (CC-BY needs attribution → DATA_LICENSES).
    """
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "data" / "genomics" / _PRS_FILES[name]
    rsids: list[str] = []
    eff: list[str] = []
    weights: list[float] = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.startswith("rsID") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rsids.append(parts[0]); eff.append(parts[1]); weights.append(float(parts[3]))
    return rsids, eff, np.asarray(weights, dtype=float)


def load_satellite_telemetry(n_windows: int = 600, n_channels: int = 6, window: int = 8,
                             anomaly_rate: float = 0.15, seed: int = 0):
    """ESA-ADB-style multi-channel satellite telemetry windows + anomaly labels.

    Channels (periodic signal + sensor noise): battery_v, temp_c, payload_w, attitude_err, solar_a,
    downlink_snr. Anomalies are injected level-shifts/spikes (the dominant real spacecraft-telemetry
    anomaly modes). Real **ESA-ADB** telemetry (Zenodo, CC BY 3.0 IGO, commercial-OK) is substitutable —
    the anomaly-detection model + the encrypted-inference path are identical. Returns ``(X (N, C·W) float32,
    y (N,) ∈ {0,1}, channel_names)``.
    """
    rng = np.random.default_rng(seed)
    names = ["battery_v", "temp_c", "payload_w", "attitude_err", "solar_a", "downlink_snr"][:n_channels]
    t = np.arange(window)
    X = np.zeros((n_windows, n_channels * window), dtype=np.float32)
    y = np.zeros(n_windows, dtype=np.int64)
    for i in range(n_windows):
        anom = rng.random() < anomaly_rate
        chans = []
        for c in range(n_channels):
            base = np.sin(2 * np.pi * (t + rng.random() * window) / window) * (0.5 + 0.4 * c)
            base = base + rng.normal(0, 0.05, window)
            if anom and rng.random() < 0.6:                # inject a spike / level-shift
                base[rng.integers(0, window)] += rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 3.0)
            chans.append(base)
        X[i] = np.concatenate(chans)
        y[i] = int(anom)
    return X, y, names


def load_genomic_prs(n_individuals: int = 400, seed: int = 0, name: str = "breast_cancer"):
    """Real PRS model + a cohort of **real 1000 Genomes** genotype dosages.

    Returns ``(dosages (N, M) ∈ {0,1,2}, weights (M,), rsids, effect_alleles, freq (M,))``. The PRS
    **weights are real** (PGS Catalog, CC-BY) AND the genotype dosages are **real 1000 Genomes phase-3**
    effect-allele counts for exactly these SNPs (fetched once from the Ensembl REST API over IGSR / 1000
    Genomes — open, no-auth, commercial-OK — and bundled as ``data/genomics/g1k_dosages.tsv``). The PRS is
    therefore a genuine score over real human genotypes. Offline-CI fallback (only if the bundled real file
    is absent): Hardy-Weinberg sampling, **logged as NOT real** — never a silent substitute.
    """
    from pathlib import Path

    rsids, eff, weights = load_prs(name)
    m = len(weights)
    g1k = Path(__file__).resolve().parents[3] / "data" / "genomics" / "g1k_dosages.tsv"
    try:
        rows = [ln for ln in g1k.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        header = rows[0].split("\t")[1:]                          # rsIDs present in the real file
        col = {rs: j for j, rs in enumerate(header)}
        idx = [col[rs] for rs in rsids]                           # align real columns to the PGS SNP order
        data = [[int(p) for p in ln.split("\t")[1:]] for ln in rows[1:]]
        dosages = np.asarray([[r[j] for j in idx] for r in data], dtype=float)
        if n_individuals and len(dosages) > n_individuals:
            dosages = dosages[:n_individuals]
        freq = dosages.mean(axis=0) / 2.0                        # empirical real effect-allele frequency
        return dosages, weights, rsids, eff, freq
    except Exception as exc:                                      # pragma: no cover - offline-CI fallback only
        import logging
        logging.getLogger("qsh").warning(
            "Real 1000G dosages unavailable (%s) -> Hardy-Weinberg sampling (offline/CI fallback, NOT real "
            "genotypes; the bundled data/genomics/g1k_dosages.tsv supplies real data in production).", exc)
        rng = np.random.default_rng(seed)
        freq = rng.uniform(0.1, 0.5, m)
        dosages = rng.binomial(2, freq, size=(n_individuals, m)).astype(float)
        return dosages, weights, rsids, eff, freq
