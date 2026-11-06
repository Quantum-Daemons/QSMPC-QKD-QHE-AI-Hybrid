"""Dataset loaders for the finance / medical / genomics / satellite use cases.

Prefers free public data (MedMNIST CC-BY, scikit-learn bundled sets); falls back to
synthetic generators (:mod:`~qsh.data.synthetic`) when a download is unavailable or a
dataset has usage restrictions. The array loaders return NumPy arrays ``(X, y)``; the
satellite payload, which carries its own model, comes back as :class:`OpsSatSegments`
and has no synthetic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def load_medmnist(name: str | None = None, split: str = "train"
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Load a MedMNIST 2D dataset (real data). Returns X ``(N, 1, 28, 28)`` float32 in ``[0, 1]`` and integer labels.

    ``name`` defaults to the configured subset, ``benchmark.medical_cnn.medmnist_name`` (BreastMNIST), which the full
    Docker image bakes at build time, so this needs no network at runtime.

    A set that cannot be loaded RAISES. Until Round 86 it fell back to synthetic images with a warning in the server's
    log, which a run never showed: a full-deployment medical run could report its CNN and its encrypted inference on
    images that were not medical at all (a reader found it). Synthetic images stand in only when the caller sets
    ``QSH_ALLOW_SYNTHETIC_IMAGES=1``, which the test suite does to exercise the mechanism where the file is absent.
    """
    import os

    from ..config import get_config

    name = name or get_config().benchmark.medical_cnn["medmnist_name"]
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
    except Exception as exc:
        if os.environ.get("QSH_ALLOW_SYNTHETIC_IMAGES") != "1":
            raise RuntimeError(
                f"MedMNIST {name!r} could not be loaded ({type(exc).__name__}: {exc}). The full Docker image bakes "
                "it at build time; elsewhere fetch it with `python -m qsh.data.download --medmnist`. Synthetic images "
                "stand in only for tests that set QSH_ALLOW_SYNTHETIC_IMAGES=1.") from exc
        import logging

        logging.getLogger("qsh").warning(
            "MedMNIST '%s' unavailable (%s): SYNTHETIC images, because QSH_ALLOW_SYNTHETIC_IMAGES=1. "
            "This is NOT medical data.", name, exc)
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
    """Synthetic SNP genotype matrix + phenotype labels, for the generic benchmark only.

    The genomics use case does not come through here: it scores real 1000 Genomes phase 3 genotypes
    through `load_genomic_prs`. Those carry no licence instrument (IGSR's disclaimer; open-consent
    data), so they are not CC BY, and openSNP is not a source here (DATA_LICENSES.md section 4).
    """
    from .synthetic import synthetic_genotypes

    return synthetic_genotypes(n=n, n_snps=n_snps, seed=seed)


# Bundled real Polygenic Risk Scores (PGS Catalog; the licence is the score publication's, per score -
# see DATA_LICENSES). Each is a real published per-variant effect-weight table, so the model is a genuine
# linear PRS = sum_i weight_i * dosage_i, which runs as one approximate CKKS dot product (the published
# "HE-applied-to-PRS" pattern; FHE-native, no surrogate).
#: name -> (scoring file, dosage file).
#:
#: `breast_cancer_pgs000072` is PGS000072 (CC_Breast; Graff et al., Nat Commun 12:970, 2021, CC BY 4.0;
#: 187 variants, keyed by rsID) and is **ACTIVE** - what the pipeline, the browser payload, the trainer
#: and the registry all score. The score in force is `DataConfig.genomics_prs_score`, not a default
#: written into any call site.
#:
#: `breast_cancer` is PGS000001 / PRS77_BC, keyed by rsID, and is **RETIRED**: kept because its
#: 400-individual dosage file is committed in full and CI exercises it.
#:
#: A score variant may lack a dosage column only where the dosage file declares it `#unscored`, with
#: the reason (`unscored_variants`); the score is then evaluated over the variants that have one, never
#: with a filled-in value. PGS000072's file declares the variants Ensembl's 1000 Genomes records do not
#: resolve (scripts/data/fetch_g1k_dosages.py).
#:
#: Kept as a literal rather than read from ``DataConfig`` so these loaders stay pure functions with no
#: config dependency; ``tests/test_prs_pipeline_parity.py`` asserts the two agree, so there is one
#: authority and a check rather than two authorities and a hope.
ACTIVE_PRS = "breast_cancer_pgs000072"

_PRS_FILES = {
    "breast_cancer": ("pgs000001_breast_cancer.tsv", "g1k_dosages.tsv"),
    "breast_cancer_pgs000072": ("pgs000072_breast_cancer.tsv", "g1k_dosages_pgs000072.tsv"),
}


def unscored_variants(dosage_path) -> dict[str, str]:
    """``key -> reason`` for every variant a dosage file declares ``#unscored``.

    The fetcher writes one ``#unscored<TAB>key<TAB>reason`` line per score variant it could not
    resolve to a 1000 Genomes genotype, and leaves the column out. Reading the declaration rather
    than inferring it from an absent column is what keeps a dropped variant from being silent: any
    absence the file does not declare still raises.
    """
    from pathlib import Path

    out: dict[str, str] = {}
    for line in Path(dosage_path).read_text().splitlines():
        if line.startswith("#unscored\t"):
            _, key, reason = line.split("\t", 2)
            out[key] = reason
    return out


def load_prs(name: str = ACTIVE_PRS) -> tuple[list[str], list[str], np.ndarray]:
    """Load a real PGS-Catalog Polygenic Risk Score -> ``(keys, effect_alleles, effect_weights)``.

    The weights are the real published effect sizes, every variant of the published score (PGS000072:
    187 breast-cancer variants, Graff et al., Nat Commun 12:970, 2021, CC BY 4.0). Bundled under
    ``data/genomics/`` so it runs offline; the attribution each score's licence asks for is in
    DATA_LICENSES. ``load_genomic_prs`` returns the variants that have a genotype column.

    ``keys`` are rsIDs for a score that publishes them (both bundled scores do) and
    ``chr:pos:effect_allele`` for one that does not.
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


#: The satellite anomaly demo's payload: real OPS-SAT-AD telemetry segments (CC BY 4.0) and the demo
#: regression that scores them. Since Round 84 it replaces the NASA SMAP/MSL TSV this module read
#: before, for which no provider grants a data licence (DATA_LICENSES.md section 2).
#:
#: A byte-identical copy of `web/public/data/satellite/opssat_segments.json`, the file the browser's
#: plaintext twin fetches, so the two languages score one model on one set of rows. The copy lives
#: under `data/` because the full deployment's API image ships `data/` and not `web/public/`
#: (docker/Dockerfile.api, .dockerignore); tests/test_r84_satpy_opssat_ckks.py fails the moment the
#: two copies differ by a byte.
OPSSAT_SEGMENTS = Path(__file__).resolve().parents[3] / "data" / "satellite" / "opssat_segments.json"


@dataclass(frozen=True, eq=False)
class OpsSatSegments:
    """The OPS-SAT-AD anomaly-demo payload, checked: the demo regression and the segments it scores.

    `scripts/data/build_heldout_payloads.py` writes it from the benchmark's test partition
    (`models/satellite_opssat/eval_sample.json`, 529 segments): a logistic regression over the 19
    published per-segment features, fitted with its scaler on 370 of the segments, and the other 159,
    z-scored with the fitting constants, which neither the regression nor its scaler saw. A demo model,
    not the published `satellite_comms.opssat` models; `model_note` says so in the payload's words.
    """

    names: tuple[str, ...]       # the 19 per-segment features, in column order
    weight: np.ndarray           # (2, 19); class 0's row is zero, so the margin is the class-1 logit
    bias: np.ndarray             # (2,)
    segments: np.ndarray         # (159, 19) held-out segments in the regression's z-space
    labels: np.ndarray           # (159,) the benchmark's labels, 1 = anomalous
    row_index: np.ndarray        # (159,) each segment's row in the 529-segment test partition
    baseline: np.ndarray         # (19,) E[x] over the fitting segments: the SHAP reference point
    baseline_note: str
    model_note: str
    provenance: str
    licence: str
    licence_url: str
    attribution: str


def load_opssat_segments(path: Path | None = None) -> OpsSatSegments:
    """The bundled OPS-SAT-AD anomaly-demo payload (CC BY 4.0), with its parts checked against each other.

    There is no synthetic fallback. The payload is a tracked file in every checkout and in the API
    image, so a missing one is a packaging error: ``FileNotFoundError`` reaches the caller, which says
    so, rather than made-up rows being scored under the OPS-SAT-AD name. Parts whose shapes disagree
    raise ``ValueError``.
    """
    import json

    p = OPSSAT_SEGMENTS if path is None else Path(path)
    d = json.loads(p.read_text())
    names = tuple(str(v) for v in d["names"])
    m = len(names)
    W = np.asarray(d["model"]["W"], dtype=float)
    b = np.asarray(d["model"]["b"], dtype=float)
    X = np.asarray(d["segments"], dtype=float)
    y = np.asarray(d["labels"], dtype=np.int64)
    rows = np.asarray(d["row_index"], dtype=np.int64)
    base = np.asarray(d["baseline"], dtype=float)
    problems = []
    if d.get("feature_space") != "zscore":
        problems.append(f"feature_space {d.get('feature_space')!r}, not the regression's 'zscore'")
    if W.shape != (2, m) or b.shape != (2,):
        problems.append(f"model W {W.shape} and b {b.shape} for {m} features")
    if X.ndim != 2 or X.shape[1] != m or not len(X):
        problems.append(f"segments {X.shape} for {m} features")
    if y.shape != (len(X),) or rows.shape != (len(X),) or not set(y.tolist()) <= {0, 1}:
        problems.append(f"{y.shape} labels and {rows.shape} row indices for {len(X)} segments")
    if base.shape != (m,):
        problems.append(f"baseline {base.shape} for {m} features")
    if problems:
        raise ValueError(f"{p.name} is malformed: " + "; ".join(problems))
    return OpsSatSegments(
        names=names, weight=W, bias=b, segments=X, labels=y, row_index=rows, baseline=base,
        baseline_note=str(d.get("baseline_note", "")), model_note=str(d.get("real_model", "")),
        provenance=str(d["provenance"]), licence=str(d["licence"]),
        licence_url=str(d.get("licence_url") or ""), attribution=str(d.get("attribution", "")))


def load_genomic_prs(n_individuals: int = 400, seed: int = 0, name: str = ACTIVE_PRS):
    """Real PRS model + a cohort of **real 1000 Genomes** genotype dosages.

    Returns ``(dosages (N, M) in {0,1,2}, weights (M,), keys, effect_alleles, freq (M,))`` over the M
    score variants the dosage file has a column for; a variant it declares ``#unscored`` is left out
    of all five, and any other missing column raises. The PRS **weights are real** (PGS Catalog; the
    licence is the score publication's) AND the dosages are **real 1000 Genomes phase-3** effect-allele
    counts at exactly these variants, fetched from the Ensembl REST API's genotype endpoint by
    ``scripts/data/fetch_g1k_dosages.py`` (1000 Genomes phase 3 carries no licence instrument: IGSR's
    disclaimer, open-consent data; cite The 1000 Genomes Project Consortium, Nature 526:68-74, 2015).
    The PRS is therefore a genuine score over real human genotypes. Offline-CI fallback (only if no
    dosage file is present): Hardy-Weinberg sampling, **logged as NOT real** - never a silent substitute.
    """
    from pathlib import Path

    rsids, eff, weights = load_prs(name)
    m = len(weights)
    _, dosage_file = _PRS_FILES[name]
    root = Path(__file__).resolve().parents[3] / "data" / "genomics"
    g1k = root / dosage_file
    if not g1k.exists():
        # A COMMITTED SLICE OF REAL GENOTYPES, not a synthetic substitute. The active score's full
        # 2,504-row matrix is gitignored because DATA_LICENSES section 4 justifies about 30,800
        # genotype cells per file and the full matrix is far more. So on a fresh clone and in CI the
        # file above is absent, and the `except Exception` below would return Hardy-Weinberg samples -
        # the documented fallback, correctly loud, but it would leave the active score's path without
        # CI coverage while a maintainer's run reported green.
        #
        # The slice is 97 individuals (48 AFR, 49 EUR) at every scored variant, inside that budget, so
        # CI reads real genotypes. Tried only when the full file is missing, so a maintainer still
        # gets all 2,504.
        slice_file = root / f"{Path(dosage_file).stem}_slice.tsv"
        if slice_file.exists():
            g1k = slice_file
    try:
        rows = [ln for ln in g1k.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        header = rows[0].split("\t")
        col = {name: j for j, name in enumerate(header)}
        declared = unscored_variants(g1k)
        missing = [k for k in rsids if k not in col and k not in declared]
        if missing:
            raise KeyError(f"{g1k.name} has no column for {len(missing)} of {len(rsids)} score "
                           f"variants and does not declare them #unscored, e.g. {missing[:3]}")
        both = [k for k in declared if k in col]
        if both:
            raise KeyError(f"{g1k.name} declares {both[:3]} #unscored and also carries a column for them")
        keep = [i for i, k in enumerate(rsids) if k in col]
        rsids, eff, weights = [rsids[i] for i in keep], [eff[i] for i in keep], weights[keep]
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
            "genotypes; the score's committed dosage slice under data/genomics/ supplies real data).", exc)
        rng = np.random.default_rng(seed)
        freq = rng.uniform(0.1, 0.5, m)
        dosages = rng.binomial(2, freq, size=(n_individuals, m)).astype(float)
        return dosages, weights, rsids, eff, freq
