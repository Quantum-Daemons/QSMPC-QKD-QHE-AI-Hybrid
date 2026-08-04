"""The per-use-case model registry — the single source of truth for what runs where.

Every use case has up to three model tiers, and the *relationship between them* is the point:

  T1 ``crypto``    the FHE/MPC-feasible student. Small because homomorphic evaluation is, and
                   will remain, cost-bounded: the largest model ever demonstrated end-to-end
                   under CKKS is 8B parameters at a ~10^4x runtime gap (IACR ePrint 2026/935).
  T2 ``industry``  the real literature model for that industry, trained locally and published to
                   the Hugging Face Hub as safetensors + ONNX (fp32 server / int8 browser).
  T3 ``large``     models too big to run under encryption at all — served remotely or in-browser
                   via WebGPU, and labelled as plaintext.

The measured quantity that ties T1 to T2 is ``crypto_fidelity``: how often the encrypted student
agrees with the plaintext teacher on a held-out split, and how much task metric that costs. That
number is the honest answer to "what does the privacy actually cost", and it is measured per run
rather than asserted.

Two kinds of number live here and they must not be confused:

  * **Literature baselines** are constants with a citation. They are what a re-implementation is
    trying to beat or match, and they are checked into this file with their source.
  * **Achieved metrics** are produced by a training run and written to ``models/metrics.json``
    by ``qsh.models.train``. They are NEVER written here by hand. If a model has not been trained,
    :attr:`ModelSpec.achieved` is ``None`` and the UI must say so rather than print a placeholder.

This module feeds ``qsh.config.frontend_config()`` so the SPA reads the same repo ids, revisions,
baselines and licences the backend uses. No model id, URL or metric may be hard-coded in the web app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

#: Hugging Face namespace the T2 artefacts are published under.
HF_NAMESPACE = "Daemons-Q"

#: Where ``qsh.models.train`` writes achieved metrics. Small, git-tracked, machine-written.
METRICS_PATH = Path(__file__).resolve().parents[3] / "models" / "metrics.json"


# --------------------------------------------------------------------------- #
#  Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    """Provenance and licence for the data a model was trained on.

    ``publishable`` gates whether trained weights may be pushed to a PUBLIC Hub repo. The Hub grants
    every user a "perpetual, irrevocable, worldwide, royalty-free, non-exclusive" licence over public
    repo content, so a licence mistake cannot be walked back. Anything not clearly commercial-OK stays
    private/local, and that decision is recorded here rather than in a person's memory.
    """

    name: str
    licence: str
    url: str
    licence_checked: str          # ISO date the licence text was last read at ``url``
    publishable: bool             # may derived weights go to a PUBLIC HF repo?
    notes: str = ""


@dataclass(frozen=True)
class MetricSpec:
    """The metric a model is judged on, plus the published numbers it is measured against.

    ``name`` is deliberately explicit (``minority_class_f1``, not ``f1``; ``event_wise_f0_5``, not
    ``f1``). Four of the five industries here have a well-documented failure mode where the
    aggregate metric looks excellent while the class that matters is broken, so the UI prints the
    specific metric and, where one exists, the misleading aggregate right next to it.
    """

    name: str
    display: str
    baseline: float | None            # the published number a re-implementation should beat
    baseline_source: str
    ceiling: float | None = None      # best credible published number (not necessarily reachable)
    ceiling_source: str = ""
    companion: str = ""               # the aggregate metric to show alongside, to expose the gap
    higher_is_better: bool = True


@dataclass(frozen=True)
class ArtifactSpec:
    """One file inside a Hub repo."""

    path: str                     # path within the repo, e.g. "onnx/model_int8.onnx"
    fmt: str                      # "onnx" | "safetensors" | "json"
    precision: str                # "fp32" | "int8" | "none"
    serves: str                   # "browser" | "server" | "both"


@dataclass(frozen=True)
class ModelSpec:
    usecase: str
    tier: str                     # "t1_crypto" | "t2_industry" | "t3_large"
    key: str                      # stable id, unique across the registry
    label: str
    architecture: str
    params: int | None            # None when not a parametric model (e.g. a PRS weight vector)
    dataset: DatasetSpec
    metric: MetricSpec
    artifacts: tuple[ArtifactSpec, ...] = ()
    hf_repo: str | None = None    # None => not published (T1 students are computed, not downloaded)
    hf_revision: str | None = None  # pin a 40-char commit sha before any browser fetch
    runs_encrypted: bool = False
    notes: str = ""

    @property
    def hf_repo_id(self) -> str | None:
        return f"{HF_NAMESPACE}/{self.hf_repo}" if self.hf_repo else None

    def artifact(self, serves: str, precision: str) -> ArtifactSpec | None:
        for a in self.artifacts:
            if a.precision == precision and a.serves in (serves, "both"):
                return a
        return None


# --------------------------------------------------------------------------- #
#  Datasets
# --------------------------------------------------------------------------- #
# Licences below were read at the URL given on the date given. Re-read before any public push:
# PGS Catalog in particular is per-score, and IGSR explicitly disclaims blanket openness.
DS_AMLWORLD = DatasetSpec(
    name="IBM AMLworld HI-Small",
    licence="CDLA-Sharing-1.0",
    url="https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
    licence_checked="2026-08-03",
    publishable=True,
    notes=(
        "5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. "
        "SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is "
        "synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. "
        "Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not "
        "because it is real data - no commercial-OK corpus of real laundering transactions exists."
    ),
)
DS_PAYSIM = DatasetSpec(
    name="PaySim (Kaggle ealaxi/paysim1)",
    licence="CC-BY-SA-4.0",
    url="https://www.kaggle.com/datasets/ealaxi/paysim1",
    licence_checked="2026-08-03",
    publishable=False,
    notes=(
        "Retained for the bundled browser demo sample only. Whether ShareAlike propagates from a "
        "training set to model weights is legally unsettled, so PaySim-derived weights are NOT "
        "published; AMLworld (CDLA) is the publishable finance model."
    ),
)
DS_CAMELYON17 = DatasetSpec(
    name="Camelyon17-WILDS",
    licence="CC0-1.0",
    url="https://wilds.stanford.edu/datasets/",
    licence_checked="2026-08-03",
    publishable=True,
    notes=(
        "5 hospitals. The official split trains on 3 and holds 2 out as OOD val/test — federating "
        "across all 5 would destroy the OOD test set, so the federation spans the 3 training centres."
    ),
)
DS_BREASTMNIST = DatasetSpec(
    name="MedMNIST v2 BreastMNIST",
    licence="CC-BY-4.0",
    url="https://medmnist.com/",
    licence_checked="2026-08-03",
    publishable=True,
    notes="780 images at 28x28. The tiny tier; also the source of the FHE-feasible T1 student.",
)
DS_PGS_CATALOG = DatasetSpec(
    name="PGS Catalog scoring files",
    licence="per-score; EMBL-EBI Terms of Use by default",
    url="https://www.pgscatalog.org/downloads/",
    licence_checked="2026-08-03",
    publishable=False,
    notes=(
        "NOT blanket CC-BY-4.0. Every scoring file carries a '#license' header and individual scores "
        "can be CC BY-NC-ND. PGS000001 itself carries the Catalog default terms, not CC-BY. An earlier "
        "revision of this note claimed the header was re-read by qsh.models.licences.audit_pgs_score() "
        "before publishing; no such module exists and no such check runs. publishable=False is what "
        "actually prevents publication, and it is enforced by scripts/publish_hf.py."
    ),
)
DS_1000G = DatasetSpec(
    name="1000 Genomes phase 3 (IGSR)",
    licence="open access, but not public domain",
    url="https://www.internationalgenome.org/IGSR_disclaimer",
    licence_checked="2026-08-03",
    publishable=False,
    notes=(
        "IGSR: 'restrictions on different pieces of data within IGSR and rights claimed on pieces of "
        "data vary'. Allele dosages are used locally; no per-sample genotype is ever published."
    ),
)
DS_ESA_ADB = DatasetSpec(
    name="ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46",
    licence="CC-BY-3.0-IGO",
    url="https://zenodo.org/records/12528696",
    licence_checked="2026-08-03",
    publishable=True,
    notes=(
        "Missions are anonymised Mission1/2/3 — they are not named Sentinel anything. 17.5 years, "
        "1.19% annotated. Mission3 is excluded upstream for triviality."
    ),
)
DS_SMAP_MSL = DatasetSpec(
    name="NASA SMAP/MSL spacecraft telemetry (Telemanom)",
    licence="US Government work (17 U.S.C. 105)",
    url="https://github.com/khundman/telemanom",
    licence_checked="2026-08-03",
    publishable=True,
    notes=(
        "Retained as the bundled browser sample. Kept for continuity, but ESA-ADB is the model "
        "dataset: SMAP/MSL is evaluated almost universally with point-adjusted F1, which TSB-AD "
        "(NeurIPS 2024) shows a uniform random score can beat."
    ),
)
DS_CICIDS2017_FIXED = DatasetSpec(
    name="CIC-IDS2017, Engelen-corrected redistribution (WTMC 2021)",
    licence="no licence stated by either provider",
    url="https://intrusion-detection.distrinet-research.be/WTMC2021/",
    licence_checked="2026-08-03",
    publishable=False,
    notes=(
        "CIC-IDS2017 states no licence at all — only that it is 'publicly available for researchers' "
        "plus a citation request — and the Engelen redistribution states no terms either. Absent an "
        "affirmative grant, neither is safe to bundle or redistribute, so this is local training and "
        "benchmarking only. Engelen et al. found >25% of flows are artefacts and several attack "
        "classes were mislabelled."
    ),
)
DS_WDBC = DatasetSpec(
    name="Wisconsin Diagnostic Breast Cancer (UCI / sklearn)",
    licence="CC-BY-4.0",
    url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    licence_checked="2026-08-03",
    publishable=True,
    notes=(
        "UCI relicensed its repository to CC BY 4.0; the dataset page states it explicitly. Ships in "
        "scikit-learn, which is why it is the bundled medical sample. Attribution is still required — "
        "'public domain' would be wrong."
    ),
)

#: Provenance for the four samples bundled into the public demo and printed in its run log.
#:
#: These are NOT the training corpora — finance trains on AMLworld but the browser ships PaySim, and
#: conflating the two is exactly the mistake the "(bundled demo sample)" prefix exists to prevent.
#:
#: This is the single source of truth. It reaches the browser through ``frontend_config()`` ->
#: ``config.defaults.json`` (CI-parity-checked), the shipped JSON files through
#: ``scripts/prep_real_kaggle_data.py``, and is asserted against those JSON files by
#: ``tests/test_demo_provenance.py``. Three separate copies of these strings previously drifted apart
#: and shipped a licence claim the project had already published a correction for; one authority plus
#: a test is the fix.
BUNDLED_DEMO_SAMPLES: dict[str, tuple[str, DatasetSpec]] = {
    "finance_aml": (
        "PaySim SIMULATOR output — synthetic transactions from aggregated real mobile-money logs "
        "(Kaggle ealaxi/paysim1, CC-BY-SA-4.0). ShareAlike applies to these bundled rows.",
        DS_PAYSIM,
    ),
    "medical_fl": (
        "real Wisconsin breast-cancer diagnostic measurements (UCI / scikit-learn, CC-BY-4.0)",
        DS_WDBC,
    ),
    "genomics": (
        "real 1000 Genomes phase-3 genotypes + published PGS000001 weights (PGS Catalog: per-score "
        "licence, EMBL-EBI terms by default; IGSR grants no blanket licence)",
        DS_PGS_CATALOG,
    ),
    "satellite_comms": (
        "real NASA SMAP/MSL spacecraft telemetry (Telemanom / NASA JPL). No provider states a "
        "licence; 17 U.S.C. 105 removes US copyright domestically but is not a grant — see "
        "DATA_LICENSES.md.",
        DS_SMAP_MSL,
    ),
}

#: Where each bundled sample is served from, so the parity test can find the file it must match.
DEMO_SAMPLE_FILES: dict[str, str] = {
    "finance_aml": "web/public/data/finance/paysim_sample.json",
    "medical_fl": "web/public/data/medical/breast_cancer_sample.json",
    "genomics": "web/public/data/genomics/pgs000001_prs.json",
    "satellite_comms": "web/public/data/satellite/telemetry.json",
}

DS_QKD_ATTACK_SYNTH = DatasetSpec(
    name="QKD attack telemetry (synthetic, methodology-grounded)",
    licence="generated by this repository",
    url="https://arxiv.org/abs/2509.14282",
    licence_checked="2026-08-03",
    publishable=True,
    notes=(
        "8 classes = 7 attack types + normal, per Al-Kuwari et al. (IET Quantum Communication 7(1), "
        "e70028, 2026). No commercial-OK real QKD-attack capture exists; this is labelled synthetic "
        "everywhere it is reported."
    ),
)


# --------------------------------------------------------------------------- #
#  Metrics — the published numbers each model is measured against
# --------------------------------------------------------------------------- #
M_AML = MetricSpec(
    name="minority_class_f1",
    display="minority-class F1 (laundering)",
    baseline=0.6323,
    baseline_source="GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17)",
    ceiling=0.6816,
    ceiling_source="Multi-PNA+EU, AAAI-24 vol.38 no.10 pp.11838-11846 (68.16 +/- 2.65)",
    companion="auprc",
)
M_CAMELYON = MetricSpec(
    name="ood_accuracy",
    display="OOD test accuracy (held-out hospitals)",
    baseline=0.703,
    baseline_source="ERM baseline, official WILDS leaderboard (70.3 +/- 6.4)",
    ceiling=0.946,
    ceiling_source="ContriMix DenseNet-121, official WILDS leaderboard (94.6 +/- 1.2)",
    companion="id_accuracy",
)
M_BREASTMNIST = MetricSpec(
    name="auc",
    display="AUC",
    baseline=0.901,
    baseline_source="ResNet-18 @28px, MedMNIST v2, Yang et al., Scientific Data 10:41 (2023) (AUC 0.901 / ACC 0.863)",
    ceiling=0.919,
    ceiling_source="Google AutoML Vision, MedMNIST v2 (AUC 0.919)",
    companion="accuracy",
)
M_PRS = MetricSpec(
    name="c_index_eur",
    display="C-index (European ancestry)",
    baseline=0.628,
    baseline_source=(
        "PGS Catalog PPM017270 for PGS000001: Harrell's C-index 0.628 (95% CI 0.618-0.638) from a Cox "
        "model over censored UK Biobank follow-up, n=200,195. An earlier revision called it an AUROC; "
        "the Catalog record names it 'Concordance Statistic / C-index'. A C-index is computed over "
        "time-to-event pairs and an AUROC over a fixed case/control split, so this is not directly "
        "comparable to an AUROC figure."
    ),
    ceiling=0.69,
    ceiling_source="Upper end of the published range for germline-SNP breast-cancer PRS discrimination",
    companion="",
    # Any germline-SNP PRS claiming much above ~0.70 discrimination is measuring something else.
)
M_ESA = MetricSpec(
    name="event_wise_f0_5",
    display="event-wise F0.5",
    baseline=0.786,
    baseline_source=(
        "ESA-ADB Table 2, Mission1 lightweight (ch. 41-46), best event-wise F0.5. NOTE the caption: "
        "Table 2 scores detection of ALL EVENTS excluding communication gaps - anomalies PLUS annotated "
        "rare nominal events - whereas this project scores anomalies only, and ESA-ADB splits each "
        "mission into first-half train / second-half test. So 0.786 is an indicative target, not a "
        "like-for-like number."
    ),
    ceiling=None,
    ceiling_source="",
    companion="event_wise_f0_5_full_channels",
    # The companion is the whole point: the same method scores 0.061 on the full Mission1 channel set.
)
M_IDS = MetricSpec(
    name="macro_f1",
    display="macro F1 across attack classes",
    baseline=None,
    baseline_source=(
        "No trustworthy single number: Engelen et al. (WTMC 2021 @ IEEE S&P Workshops) show aggregate "
        "accuracy sits at 0.99 both before and after correcting CIC-IDS2017 while Bot F1 moves "
        "0.60 -> 0.99 and Web Attack XSS F1 moves 0.18 -> 0.79 (Table II; the 0.11 quoted by an "
        "earlier revision of this note was XSS's ORIGINAL RECALL, not an F1). Per-class F1 is "
        "reported instead."
    ),
    companion="aggregate_accuracy",
)
M_QML_IDS = MetricSpec(
    name="macro_f1",
    display="macro F1 (synthetic QKD-attack telemetry)",
    baseline=None,
    baseline_source=(
        "Architecture showcase, not a performance claim. On THIS project's synthetic telemetry neither "
        "quantum model beats the classical random forest (RF macro F1 0.987 vs variational QNN 0.467). "
        "Published results are MIXED, not uniformly below a random forest -- an earlier revision of this "
        "note claimed they were, and the very paper cited here contradicts it: Al-Kuwari et al. "
        "(arXiv:2509.14282) Table VIII reports a hybrid QLSTM at 94.7% accuracy / 94.7% F1 against their "
        "own random-forest baseline on the same generated data. What remains well-founded is caution "
        "about evaluation quality rather than a universal ranking: arXiv:2512.05069 was withdrawn by "
        "its own authors for evaluation that was 'insufficient to fully support the conclusions'."
    ),
    companion="classical_rf_macro_f1",
)


# --------------------------------------------------------------------------- #
#  The registry
# --------------------------------------------------------------------------- #
_ONNX_PAIR = (
    ArtifactSpec("onnx/model.onnx", "onnx", "fp32", "server"),
    ArtifactSpec("onnx/model_int8.onnx", "onnx", "int8", "browser"),
    ArtifactSpec("model.safetensors", "safetensors", "fp32", "server"),
    ArtifactSpec("feature_spec.json", "json", "none", "both"),
)

#: For models where dynamic int8 quantisation has nothing to quantise. A tree ensemble keeps its
#: parameters in node attributes rather than weight tensors, so no int8 graph is produced and none is
#: uploaded -- declaring one anyway pointed the browser tier at a file that does not exist at the
#: pinned revision, which would 404 rather than degrade.
_ONNX_FP32_ONLY = (
    ArtifactSpec("onnx/model.onnx", "onnx", "fp32", "both"),
    ArtifactSpec("model.safetensors", "safetensors", "fp32", "server"),
    ArtifactSpec("feature_spec.json", "json", "none", "both"),
)

_SPECS: tuple[ModelSpec, ...] = (
    # ---------------- finance / AML ----------------
    ModelSpec(
        usecase="finance_aml",
        tier="t2_industry",
        key="finance_aml.gnn",
        label="Multi-PNA+EU transaction GNN",
        architecture="PNA with reverse message passing, multigraph port numbering, ego IDs, edge updates",
        params=None,          # filled from the trained checkpoint; AAAI-24 reports no count
        dataset=DS_AMLWORLD,
        metric=M_AML,
        artifacts=_ONNX_PAIR,
        hf_repo="qsh-finance-aml-gnn",
        notes=(
            "Message-passing needs the graph neighbourhood at inference, so the browser tier ships the "
            "stateless gradient-boosting variant instead of this one."
        ),
    ),
    ModelSpec(
        usecase="finance_aml",
        tier="t2_industry",
        key="finance_aml.gbm",
        label="Graph-feature-preprocessed gradient boosting",
        architecture="GFP (graph features) + XGBoost",
        params=None,
        dataset=DS_AMLWORLD,
        metric=M_AML,
        artifacts=_ONNX_FP32_ONLY,   # tree ensemble: no int8 graph exists to serve
        hf_repo="qsh-finance-aml-gbm",
        notes="Stateless at inference, so this is the browser-tier finance model.",
    ),
    ModelSpec(
        usecase="finance_aml",
        tier="t1_crypto",
        key="finance_aml.student",
        label="CKKS-linear fraud student",
        architecture="logistic regression distilled from the gradient-boosting teacher",
        params=None,
        dataset=DS_AMLWORLD,
        metric=M_AML,
        runs_encrypted=True,
        notes="Runs under real CKKS. crypto_fidelity is measured against the teacher every run.",
    ),
    # ---------------- medical / federated ----------------
    ModelSpec(
        usecase="medical_fl",
        tier="t2_industry",
        key="medical_fl.densenet",
        label="DenseNet-121, federated across 3 hospitals",
        architecture="DenseNet-121, FedAvg with pairwise-mask secure aggregation (qsh.ai.federated)",
        params=None,
        dataset=DS_CAMELYON17,
        metric=M_CAMELYON,
        artifacts=_ONNX_PAIR,
        hf_repo="qsh-medical-fl-densenet121",
        notes=(
            "FedAvg alone is not a privacy guarantee — gradient inversion is possible. The secure "
            "aggregation in qsh.ai.federated hides individual updates from the aggregator; it does "
            "not bound what the trained model itself leaks."
        ),
    ),
    ModelSpec(
        usecase="medical_fl",
        tier="t2_industry",
        key="medical_fl.resnet18",
        label="ResNet-18 on BreastMNIST",
        architecture="ResNet-18 @28px",
        params=None,
        dataset=DS_BREASTMNIST,
        metric=M_BREASTMNIST,
        artifacts=_ONNX_PAIR,
        hf_repo="qsh-medical-breastmnist-resnet18",
        notes="The small tier: fits the browser comfortably and has a clean published baseline.",
    ),
    ModelSpec(
        usecase="medical_fl",
        tier="t1_crypto",
        key="medical_fl.student",
        label="FHE-feasible diagnostic CNN",
        architecture="1 conv block (1->4ch, 3x3), avg-pool, dense head, 14x14 input",
        params=None,
        dataset=DS_BREASTMNIST,
        metric=M_BREASTMNIST,
        runs_encrypted=True,
        notes="Genuine encrypted convolution: im2col + CKKS dot_plain, ReLU/max-pool under MPC.",
    ),
    # ---------------- genomics ----------------
    ModelSpec(
        usecase="genomics",
        tier="t2_industry",
        key="genomics.prs",
        label="Polygenic risk score (published effect weights)",
        architecture="per-SNP effect-weight vector; Bayesian shrinkage (PRS-CS / LDpred2 family)",
        params=None,
        dataset=DS_PGS_CATALOG,
        metric=M_PRS,
        artifacts=(ArtifactSpec("weights.json", "json", "none", "both"),),
        hf_repo=None,          # publication blocked pending the per-score licence audit
        runs_encrypted=True,   # a dot product: the rare case that is both clinical standard and FHE-native
        notes=(
            "Not a neural network, and presenting a genomic foundation model as a PRS method would be "
            "the most obvious credibility hole available. Evaluation must be stratified by genetic "
            "ancestry: the European-to-African AUROC drop is the finding, not a footnote."
        ),
    ),
    # ---------------- satellite ----------------
    ModelSpec(
        usecase="satellite_comms",
        tier="t2_industry",
        key="satellite_comms.telemanom",
        label="Telemanom LSTM forecaster (per channel)",
        architecture="2 x LSTM(80), sequence length 250, dropout 0.3, non-parametric dynamic threshold with pruning",
        params=None,
        dataset=DS_ESA_ADB,
        metric=M_ESA,
        artifacts=_ONNX_PAIR,
        hf_repo="qsh-satellite-telemanom-esa",
        notes=(
            "Report event-wise F0.5, never point-adjusted F1. Always run a trivial baseline alongside: "
            "windowed isolation forest wins the ESA-ADB Mission2 lightweight subset outright."
        ),
    ),
    ModelSpec(
        usecase="satellite_comms",
        tier="t1_crypto",
        key="satellite_comms.student",
        label="CKKS-linear telemetry anomaly student",
        architecture="logistic regression over channel x timestep windows",
        params=None,
        dataset=DS_ESA_ADB,
        metric=M_ESA,
        runs_encrypted=True,
    ),
    # ---------------- telecom / defence ----------------
    ModelSpec(
        usecase="telecom_ids",
        tier="t2_industry",
        key="telecom_ids.transformer",
        label="Encoder-only Transformer IDS",
        architecture="4 layers, 8 heads, d_model 128, FFN 512, GELU, learnable [CLS]",
        params=None,
        dataset=DS_CICIDS2017_FIXED,
        metric=M_IDS,
        artifacts=_ONNX_PAIR,
        hf_repo=None,          # trained on a research-only corpus; weights stay local
        notes=(
            "Trained on the Engelen-corrected corpus locally. Weights are not published because the "
            "redistribution terms of the corrected dataset are unverified."
        ),
    ),
    ModelSpec(
        usecase="telecom_ids",
        tier="t2_industry",
        key="telecom_ids.qml",
        label="Quantum-kernel SVC + variational QNN",
        architecture="PCA -> angle embedding -> StronglyEntanglingLayers; fidelity quantum kernel",
        params=25,
        dataset=DS_QKD_ATTACK_SYNTH,
        metric=M_QML_IDS,
        artifacts=(ArtifactSpec("weights.json", "json", "none", "both"),),
        hf_repo="qsh-telecom-qml-ids",
        notes="Shown as an architecture comparison against a random forest on identical data.",
    ),
)


# --------------------------------------------------------------------------- #
#  Achieved metrics (machine-written; never hand-edited)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _achieved() -> dict:
    """Load ``models/metrics.json``. Absent file => nothing has been trained yet, which is a valid state."""
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:  # a corrupt file must be loud, not silently empty
        raise RuntimeError(f"{METRICS_PATH} is not valid JSON: {exc}") from exc


def specs() -> tuple[ModelSpec, ...]:
    """Every registered model spec, with any recorded HF revision applied."""
    recorded = _achieved()
    out = []
    for s in _SPECS:
        entry = recorded.get(s.key, {})
        rev = entry.get("hf_revision")
        out.append(replace(s, hf_revision=rev) if rev else s)
    return tuple(out)


def get(key: str) -> ModelSpec:
    for s in specs():
        if s.key == key:
            return s
    raise KeyError(f"unknown model key: {key!r}")


def for_usecase(usecase: str, tier: str | None = None) -> tuple[ModelSpec, ...]:
    return tuple(s for s in specs() if s.usecase == usecase and (tier is None or s.tier == tier))


def achieved(key: str) -> dict | None:
    """The measured metrics for ``key`` from the last training run, or ``None`` if never trained."""
    return _achieved().get(key)


def frontend_models() -> dict:
    """The registry subset the SPA needs, so no model id, URL, baseline or metric is hard-coded in TS.

    ``achieved`` is ``None`` for anything not yet trained; the UI must render that as "not trained"
    rather than substituting a baseline or a placeholder number.
    """
    out: dict[str, list] = {}
    for s in specs():
        out.setdefault(s.usecase, []).append({
            "key": s.key,
            "tier": s.tier,
            "label": s.label,
            "architecture": s.architecture,
            "params": s.params,
            "runs_encrypted": s.runs_encrypted,
            "hf_repo_id": s.hf_repo_id,
            "hf_revision": s.hf_revision,
            "artifacts": [
                {"path": a.path, "fmt": a.fmt, "precision": a.precision, "serves": a.serves}
                for a in s.artifacts
            ],
            "dataset": {
                "name": s.dataset.name,
                "licence": s.dataset.licence,
                "url": s.dataset.url,
                "publishable": s.dataset.publishable,
            },
            "metric": {
                "name": s.metric.name,
                "display": s.metric.display,
                "baseline": s.metric.baseline,
                "baseline_source": s.metric.baseline_source,
                "ceiling": s.metric.ceiling,
                "ceiling_source": s.metric.ceiling_source,
                "companion": s.metric.companion,
            },
            "achieved": achieved(s.key),
            "notes": s.notes,
        })
    return out
