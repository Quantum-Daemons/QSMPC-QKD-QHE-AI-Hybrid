"""Dataset loaders for the finance / medical / genomics use cases.

Prefers free public data (MedMNIST CC-BY, scikit-learn bundled sets); falls back to
synthetic generators (:mod:`~qsh.data.synthetic`) when a download is unavailable or a
dataset has usage restrictions. All loaders return NumPy arrays ``(X, y)``.
"""

from __future__ import annotations

import numpy as np


def load_breast_cancer(scaler_fit_fraction: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Wisconsin breast-cancer (medical tabular, bundled with scikit-learn).

    The scaler is fitted on the first ``scaler_fit_fraction`` of rows and applied to all of them.

    This was ``StandardScaler().fit_transform(ds.data)`` over all 569 rows until Round 34 - the
    textbook leakage, with every consumer's test rows contributing to the constants applied to
    themselves. Two facts bound how much it mattered and neither excuses it. The published medical
    model does not come through here: ``scripts/train/medical_breastmnist.py`` trains on MedMNIST via
    ``datasets.breastmnist``, so nothing on the Hub or in ``models/metrics.json`` was contaminated.
    But the demo and benchmark paths do (``usecases/medical_fl``, ``benchmark/runner``), and the
    browser printed those numbers with nothing saying they had been scaled against their own test rows.

    A fraction rather than a caller-supplied split, because the two consumers split differently and a
    loader that silently assumed one of them would be a worse lie than the one being fixed. 0.7 is
    what both actually use, so the constants match what a correct pipeline would produce; a caller
    that splits elsewhere should pass its own. Fitting on a prefix is only sound when the rows are
    exchangeable - WDBC ships in no meaningful order, so it is sound here and would NOT be on a
    time series, where the prefix must be the chronological training window instead.
    """
    from sklearn.datasets import load_breast_cancer as _lbc
    from sklearn.preprocessing import StandardScaler

    ds = _lbc()
    cut = max(1, int(len(ds.data) * scaler_fit_fraction))
    scaler = StandardScaler().fit(ds.data[:cut])
    X = scaler.transform(ds.data).astype(np.float32)
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
    """**PaySim simulator** output -> the 8 AML features + fraud label.

    PaySim is a SIMULATOR: the rows are synthetic transactions generated from aggregated real
    mobile-money logs, not real transactions. The Kaggle dataset is titled "Synthetic Financial
    Datasets For Fraud Detection". Earlier revisions of this docstring called them real.

    Reads the bundled real sample ``data/finance/paysim_sample.tsv`` (Kaggle ``ealaxi/paysim1``,
    **CC-BY-SA-4.0**; Lopez-Rojas et al. 2016). The 8 columns are derived from PaySim fields (PaySim is a simulator: synthetic transactions)
    (amount / type / step / balances) in the order :mod:`finance_aml` expects; fraud is oversampled to
    ``fraud_rate`` for training balance (the real base rate is 0.13 %). Returns ``(X (n, 8) float32,
    y (n,) ∈ {0,1})``. Offline-CI fallback (only if the bundled file is absent): the synthetic generator,
    **logged as NOT real** — never a silent substitute (no-easy-fallback policy).
    """
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "data" / "finance" / "paysim_sample.tsv"
    try:
        rows = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        data = np.asarray([[float(v) for v in ln.split("\t")] for ln in rows[1:]], dtype=np.float32)
        X, y = data[:, :8], data[:, 8].astype(np.int64)
        if n and len(X) != n:                                    # stratified subsample to honour n + fraud_rate
            rng = np.random.default_rng(seed)
            fr, lg = np.where(y == 1)[0], np.where(y == 0)[0]
            nf = min(len(fr), int(round(n * fraud_rate)))
            pick = np.concatenate([rng.choice(fr, nf, replace=False),
                                   rng.choice(lg, min(len(lg), n - nf), replace=False)])
            rng.shuffle(pick)
            X, y = X[pick], y[pick]
        return X, y
    except Exception as exc:                                     # pragma: no cover - offline-CI fallback only
        import logging

        logging.getLogger("qsh").warning(
            "Bundled PaySim sample unavailable (%s) -> locally generated transactions (offline/CI fallback, "
            "NOT the PaySim simulator output, "
            "data; the bundled data/finance/paysim_sample.tsv supplies the PaySim simulator sample in "
            "production).", exc)
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


# Bundled real Polygenic Risk Scores (PGS Catalog; licence is PER SCORE, not blanket CC-BY - see
# DATA_LICENSES). Each is a real published per-SNP effect-weight table → the model is a genuine linear
# PRS = Σ weight_i · dosage_i, which runs as an exact CKKS dot-product (the published
# "HE-applied-to-PRS" pattern; FHE-native, no surrogate).
#: name -> (scoring file, dosage file). Two scores are bundled and they are keyed DIFFERENTLY, which is
#: the whole reason this is a mapping rather than a filename.
#:
#: `breast_cancer_prs313` is PGS000004 / PRS313_BC (Mavaddat 2018, 313 variants) and is **ACTIVE** - what
#: the pipeline, the browser payload, the trainer and the registry all score. The score in force is
#: `DataConfig.genomics_prs_score`, not a default written into any call site.
#:
#: `breast_cancer` is PGS000001 / PRS77_BC, keyed by rsID, and is **RETIRED** - kept because its dosage
#: file is the one small enough to commit in full, so it is what CI exercises for the rsID-keyed code
#: path, and because retiring the data along with the score would delete the evidence for the switch.
#:
#: WHAT THIS COMMENT USED TO SAY, AND WHY IT IS WORTH RECORDING. It said PRS77 "is what both the Python
#: pipeline and the browser payload currently score" and that PRS313 was "STAGED, NOT ACTIVE ... a use
#: case scoring PRS313 in Python while showing PRS77 in the browser would be worse than either alone."
#: Round 31 switched four surfaces and missed the pipeline, so for one round the shipped state was the
#: exact inverse of that sentence - PRS313 in the browser, PRS77 in Python - and the sentence was still
#: here arguing against it. A comment that describes an intention outlives the intention; this one now
#: names the config field instead, so the answer is looked up rather than remembered.
#:
#: PGS000004 publishes NO rsIDs - all 313 harmonised rows have an empty `hm_rsID` with
#: `hm_source=Author-reported` - so it is keyed by `chr:pos:effect_allele` and joined to genotypes by
#: allele match against the phase-3 VCF. Reverse-mapping to rsIDs was measured and rejected: 70% of its
#: positions carry one rsID, 28% carry several, allele matching resolves about half of those, and the
#: remaining ~15% would have had to be guessed - which would score a DIFFERENT variant's genotypes
#: while every number on the page still looked reasonable.
#: The score in force. Every current caller passes ``name`` explicitly, so this default is not
#: exercised today - which is exactly why it was still pointing at the RETIRED score after Round 31.
#: A dead default is a trap for the next caller, not a harmless leftover.
#:
#: Kept as a literal rather than read from ``DataConfig`` so these loaders stay pure functions with no
#: config dependency; ``tests/test_prs_pipeline_parity.py`` asserts the two agree, so there is one
#: authority and a check rather than two authorities and a hope.
ACTIVE_PRS = "breast_cancer_prs313"

_PRS_FILES = {
    "breast_cancer": ("pgs000001_breast_cancer.tsv", "g1k_dosages.tsv"),
    "breast_cancer_prs313": ("pgs000004_breast_cancer.tsv", "g1k_dosages_prs313.tsv"),
}


def load_prs(name: str = ACTIVE_PRS) -> tuple[list[str], list[str], np.ndarray]:
    """Load a real PGS-Catalog Polygenic Risk Score → ``(keys, effect_alleles, effect_weights)``.

    The weights are the real published effect sizes (PGS000004 = a 313-variant breast-cancer PRS,
    Mavaddat et al. 2018). Bundled under ``data/genomics/`` so it runs offline; the licence is per score,
    not blanket CC-BY, and attribution requirements are in DATA_LICENSES.

    ``keys`` are rsIDs for a score that publishes them and ``chr:pos:effect_allele`` for one that does
    not. The first return value was called ``rsids`` until Round 32, which was true of PGS000001 and
    false of the score now in force.
    """
    from pathlib import Path

    score_file, _ = _PRS_FILES[name]
    path = Path(__file__).resolve().parents[3] / "data" / "genomics" / score_file
    keys: list[str] = []
    eff: list[str] = []
    weights: list[float] = []
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        row = dict(zip(header, parts, strict=False))
        # Two layouts, one join. Whatever key is built here must match the dosage file's column
        # headers, which is what makes the alignment in `load_real_genotypes` a dict lookup rather
        # than a positional assumption.
        if "rsID" in row:
            keys.append(row["rsID"])
            eff.append(row["effect_allele"])
            weights.append(float(row["effect_weight"]))
        elif "chr_name" in row and "chr_position" in row:
            ea = row["effect_allele"]
            keys.append(f"{row['chr_name']}:{row['chr_position']}:{ea}")
            eff.append(ea)
            weights.append(float(row["effect_weight"]))
        else:
            raise ValueError(
                f"{path.name}: row has neither an rsID nor chr_name/chr_position, so it cannot be "
                f"joined to a dosage column: {parts[:4]}")
    assert keys, f"no variant rows parsed from {path.name}"
    return keys, eff, np.asarray(weights, dtype=float)


def load_satellite_telemetry(n_windows: int = 600, n_channels: int = 6, window: int = 8,
                             anomaly_rate: float = 0.15, seed: int = 0):
    """**Real NASA SMAP/MSL** spacecraft telemetry -> multi-channel windows + real anomaly labels.

    Reads the bundled real sample ``data/satellite/smap_msl_telemetry.tsv`` — ``n_channels`` real NASA
    telemetry channels (col 0 of each Telemanom test stream) + the real labeled-anomaly flag per timestep
    (Telemanom; Hundman et al. 2018, NASA JPL). **No provider states a licence** for this telemetry --
    17 U.S.C. 105 removes US copyright only and is not a grant; see DATA_LICENSES.md. Sliding length-``window``
    windows are stacked **channel-major** to match the synthetic layout + the use case's ``feature_names``.
    A window is anomalous if any of its real timesteps is. Returns ``(X (N, C·W) float32, y (N,) ∈ {0,1},
    channel_names)``. Offline-CI fallback (only if the bundled file is absent): an ESA-ADB-style synthetic
    generator, **logged as NOT real** — never a silent substitute (no-easy-fallback policy).
    """
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "data" / "satellite" / "smap_msl_telemetry.tsv"
    try:
        rows = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        header = rows[0].split("\t")                              # C channel names + "anomaly"
        names = header[:-1][:n_channels]
        c = len(names)
        arr = np.asarray([[float(v) for v in ln.split("\t")] for ln in rows[1:]], dtype=np.float32)
        telem, anom = arr[:, :c], arr[:, -1].astype(int)         # (L, C) real telemetry, (L,) real labels
        n = max(0, len(telem) - window + 1)
        if n == 0:
            raise ValueError("bundled telemetry shorter than the window")
        X = np.zeros((n, c * window), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            X[i] = np.concatenate([telem[i:i + window, j] for j in range(c)])   # channel-major
            y[i] = int(anom[i:i + window].max())
        if n_windows and n > n_windows:
            X, y = X[:n_windows], y[:n_windows]
        return X, y, names
    except Exception as exc:                                     # pragma: no cover - offline-CI fallback only
        import logging

        logging.getLogger("qsh").warning(
            "Real NASA SMAP/MSL telemetry unavailable (%s) -> SYNTHETIC ESA-ADB-style generator (offline/CI "
            "fallback, NOT real data; the bundled data/satellite/smap_msl_telemetry.tsv supplies real NASA "
            "telemetry in production).", exc)
        rng = np.random.default_rng(seed)
        names = ["battery_v", "temp_c", "payload_w", "attitude_err", "solar_a", "downlink_snr"][:n_channels]
        t = np.arange(window)
        X = np.zeros((n_windows, n_channels * window), dtype=np.float32)
        y = np.zeros(n_windows, dtype=np.int64)
        for i in range(n_windows):
            is_anom = rng.random() < anomaly_rate
            chans = []
            for ci in range(n_channels):
                base = np.sin(2 * np.pi * (t + rng.random() * window) / window) * (0.5 + 0.4 * ci)
                base = base + rng.normal(0, 0.05, window)
                if is_anom and rng.random() < 0.6:               # inject a spike / level-shift
                    base[rng.integers(0, window)] += rng.choice([-1.0, 1.0]) * rng.uniform(1.5, 3.0)
                chans.append(base)
            X[i] = np.concatenate(chans)
            y[i] = int(is_anom)
        return X, y, names


def load_genomic_prs(n_individuals: int = 400, seed: int = 0, name: str = ACTIVE_PRS):
    """Real PRS model + a cohort of **real 1000 Genomes** genotype dosages.

    Returns ``(dosages (N, M) ∈ {0,1,2}, weights (M,), rsids, effect_alleles, freq (M,))``. The PRS
    **weights are real** (PGS Catalog, licence per score) AND the genotype dosages are **real 1000 Genomes phase-3**
    effect-allele counts for exactly these SNPs (fetched once from the Ensembl REST API over IGSR / 1000
    Genomes — open, no-auth, commercial-OK — and bundled as ``data/genomics/g1k_dosages.tsv``). The PRS is
    therefore a genuine score over real human genotypes. Offline-CI fallback (only if the bundled real file
    is absent): Hardy-Weinberg sampling, **logged as NOT real** — never a silent substitute.
    """
    from pathlib import Path

    rsids, eff, weights = load_prs(name)
    m = len(weights)
    _, dosage_file = _PRS_FILES[name]
    root = Path(__file__).resolve().parents[3] / "data" / "genomics"
    g1k = root / dosage_file
    if not g1k.exists():
        # A COMMITTED SLICE OF REAL GENOTYPES, not a synthetic substitute. PRS313's full 2,504-row
        # matrix is gitignored because DATA_LICENSES section 4 justifies about 30,800 genotype cells
        # and the full matrix is 24x that. So on a fresh clone and in CI the file above is simply
        # absent, and the `except Exception` below would return Hardy-Weinberg samples - the documented
        # fallback, correctly loud, but it meant the PRS313 path (now the default for both the browser
        # and the trainer) had ZERO CI coverage while a maintainer's run reported green.
        #
        # The slice holds the same cell budget in a different shape - 97 individuals x 313 variants
        # against 400 x 77 for PRS77 - so CI reads real genotypes and the redistribution argument is
        # unchanged. Tried only when the full file is missing, so a maintainer still gets all 2,504.
        slice_file = root / f"{Path(dosage_file).stem}_slice.tsv"
        if slice_file.exists():
            g1k = slice_file
    try:
        rows = [ln for ln in g1k.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        header = rows[0].split("\t")
        col = {name: j for j, name in enumerate(header)}
        missing = [k for k in rsids if k not in col]
        if missing:
            raise KeyError(f"{g1k.name} has no column for {len(missing)} of {len(rsids)} score "
                           f"variants, e.g. {missing[:3]}")
        idx = [col[k] for k in rsids]        # align real columns to the score's variant order, by NAME
        dosages = np.asarray(
            [[int(ln.split("\t")[j]) for j in idx] for ln in rows[1:]], dtype=float)
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
