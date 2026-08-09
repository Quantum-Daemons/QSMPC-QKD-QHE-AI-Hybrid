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
    # Set when a retrain superseded a published artefact: the sha that WAS pinned, kept so the UI can
    # say "retrained since publication, needs republishing" instead of the identical-looking "never
    # pinned". See train.record_metrics for why a retrain must not carry its old pin forward.
    hf_revision_stale: str | None = None
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
#: Referenced by no ModelSpec and that is correct rather than dead: the genomics bundled sample
#: carries BOTH published PGS weights and 1000 Genomes genotypes, and a sample maps to one
#: DatasetSpec, so BUNDLED_DEMO_SAMPLES names the PGS Catalog entry. This record exists so the
#: genotype half's licence caveat stays checked and dated rather than living only in prose.
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
DS_OPSSAT_AD = DatasetSpec(
    name="OPS-SAT-AD",
    licence="CC-BY-4.0",
    url="https://zenodo.org/records/12588359",
    licence_checked="2026-08-05",
    publishable=True,
    notes=(
        "2,123 annotated telemetry segments across 9 channels from ESA's OPS-SAT CubeSat, 20.4% "
        "anomalous, with 19 precomputed per-segment features and the benchmark's own train/test "
        "split. Kotowski et al., Scientific Data 12 (2025). CC BY 4.0 and 507 KB, so unlike every "
        "other corpus here the DATA can be redistributed as well as the weights. A different task "
        "from ESA-ADB: segment classification, not forecasting, so its score is not comparable "
        "with the Telemanom model's and neither is the other's baseline."
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
    licence="provider's own pages state conflicting terms",
    url="https://www.unb.ca/cic/datasets/ids-2017.html",
    licence_checked="2026-08-05",
    publishable=False,
    notes=(
        "Re-checked 2026-08-05, and the earlier wording here was too strong. Terms ARE stated, they "
        "just disagree: the Canadian Institute for Cybersecurity's institute-level FAQ says 'You may "
        "redistribute, republish, and mirror our datasets in any form; however, any use or "
        "redistribution of the data must include a citation', and places that under commercial use, "
        "while the CICIDS2017 dataset page itself says only 'publicly available for researchers' and "
        "names no licence. That is an unresolved ambiguity rather than a permission - the same class "
        "of question Round 19 settled by dropping the ImageNet dependency instead of guessing. The "
        "Engelen redistribution is separately hosted and states nothing. 5G-NIDD was evaluated as an "
        "alternative and rejected: IEEE DataPort marks it subscription-required and neither that "
        "record nor the authors' NetsLab page carries a licence string. So the shipped telecom model "
        "trains on synthetic telemetry, and no code path loads this corpus."
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
M_BREASTMNIST_R50 = MetricSpec(
    name="auc",
    display="AUC",
    baseline=0.857,
    baseline_source=("ResNet-50 @28px, MedMNIST v2, Yang et al., Scientific Data 10:41 (2023) "
                     "(AUC 0.857 / ACC 0.812). Note this is BELOW the ResNet-18 baseline of 0.901 "
                     "on the same corpus and the same 100-epoch Adam regime: on 780 images the "
                     "deeper network loses. That is the published finding, and it is why this model "
                     "exists here - to reproduce it rather than to win."),
    ceiling=0.919,
    ceiling_source="Google AutoML Vision, MedMNIST v2 (AUC 0.919)",
    companion="accuracy",
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
M_OPSSAT = MetricSpec(
    name="auc_pr",
    display="AUC PR",
    baseline=0.979,
    baseline_source=("FCNN, best of the 30 algorithms benchmarked in Kotowski et al., Scientific "
                     "Data 12 (2025), on this dataset's own test split (AUC PR 0.979, F1 0.946, "
                     "MCC 0.932). XGBOD reaches 0.975 and RF+ICCS 0.963."),
    ceiling=0.979,
    ceiling_source="FCNN, the best of the 30 published baselines - there is no higher number to cite.",
    # Area under the precision-recall curve is the headline because the positive class is 20.4% of
    # the data: accuracy would read ~0.80 for a detector that never fires at all.
    companion="f1",
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
    # fp32 serves BOTH: it is the server tier's graph, and it is also the browser's fallback when
    # the int8 build uses operators ORT Web cannot execute (dynamic quantisation alongside LSTM).
    # Marking it server-only left the browser with no runnable artefact and no way to say why.
    ArtifactSpec("onnx/model.onnx", "onnx", "fp32", "both"),
    ArtifactSpec("onnx/model_int8.onnx", "onnx", "int8", "browser"),
    ArtifactSpec("model.safetensors", "safetensors", "fp32", "server"),
    ArtifactSpec("feature_spec.json", "json", "none", "both"),
)

#: The distilled students. A linear model of tens of parameters gains nothing from int8 and the
#: plain weights are worth shipping: at this size they are readable by eye, which is part of the
#: point of an encrypted tier you are asked to trust.
_STUDENT_ARTIFACTS = (
    ArtifactSpec("onnx/model.onnx", "onnx", "fp32", "both"),
    ArtifactSpec("weights.json", "json", "fp32", "both"),
)

#: medical_fl.student is a small convolutional net rather than a linear model. It carries a
#: safetensors checkpoint AND a structural weights.json: the browser evaluates this student under
#: CKKS, and it can extract neither conv kernels from safetensors nor from ONNX without a parser
#: this project does not ship. Without the JSON the encrypted convolution ran on the student's
#: shape with a synthetic kernel pattern -- a real homomorphic convolution, but not this model's
#: inference, so crypto_fidelity stayed a number measured offline and unverifiable in a browser.
_STUDENT_ARTIFACTS_TORCH = (
    ArtifactSpec("onnx/model.onnx", "onnx", "fp32", "both"),
    ArtifactSpec("weights.json", "json", "none", "browser"),
    ArtifactSpec("model.safetensors", "safetensors", "fp32", "server"),
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
        artifacts=(),
        # NOT published, and this field says so - `hf_repo=None` is this file's own documented
        # convention for that (see the field comment on ModelSpec). It previously read
        # "qsh-finance-aml-gnn", which asserted a published repo: Daemons-Q/qsh-finance-aml-gnn
        # returns HTTP 401 while its siblings return 200, there is no models/finance_aml_gnn
        # directory, and models/metrics.json has no entry for it. So the spec described a model that
        # was never trained, and `hf_repo_id` is surfaced by the CLI and by frontend_config.
        hf_repo=None,
        notes=(
            "Reference architecture only - never trained or published here. Message-passing needs the "
            "graph neighbourhood at inference, so every tier ships the stateless gradient-boosting "
            "variant instead. Kept as a spec because it is the published state of the art this "
            "project's GBM is measured against, not because an artifact exists."
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
        artifacts=_STUDENT_ARTIFACTS,
        hf_repo="qsh-finance-aml-ckks-student",
        notes="Runs under real CKKS. crypto_fidelity is measured against the teacher every run.",
    ),
    # ---------------- medical / federated ----------------
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
        tier="t2_industry",
        key="medical_fl.resnet50",
        label="ResNet-50 on BreastMNIST (published comparison)",
        architecture="ResNet-50 @28px, 1-channel stem, trained from random initialisation",
        params=None,
        dataset=DS_BREASTMNIST,
        metric=M_BREASTMNIST_R50,
        artifacts=_ONNX_PAIR,
        hf_repo="qsh-medical-breastmnist-resnet50",
        notes=(
            "Trained to test a published claim rather than to win one. MedMNIST v2 reports ResNet-50 "
            "at AUC 0.857 against ResNet-18's 0.901 on this corpus, so the expected outcome is that "
            "the larger model scores LOWER, and it is reported either way. It replaces a DenseNet-121 "
            "entry that named a dataset with no loader, no fetch command and no artefact directory, "
            "and linked a Hugging Face repo that did not exist. Architectures benchmarked only "
            "ImageNet-pretrained (AlexNet, VGG16, DenseNet-121, EfficientNet-B4 in MedMNIST+) are "
            "deliberately absent: this project dropped ImageNet weights over redistribution terms, "
            "so their published numbers are not comparable to a from-scratch run."
        ),
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
        artifacts=_STUDENT_ARTIFACTS_TORCH,
        hf_repo="qsh-medical-breastmnist-ckks-student",
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
            "ancestry, and what is measured here is a distribution shift, NOT an AUROC drop: an "
            "earlier revision of this note called it an AUROC drop, but 1000 Genomes is a reference "
            "panel carrying no breast-cancer case/control status, so no AUROC is computable from it "
            "and genomics_prs.py records auroc=null rather than inventing one. The finding is the "
            "by_super_population.*.frac_above_global_cut rows listed above: one globally-set "
            "high-risk cut flags a far larger share of AFR than of EUR, a gap driven by "
            "allele-frequency differences rather than by risk. Read those rows, not this sentence - "
            "they are recomputed on every training run, so no figure quoted in prose here can drift "
            "away from them. The published European AUROC baseline comes from a phenotyped cohort "
            "that is access-controlled, which is why it cannot be reproduced on this page."
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
            "windowed isolation forest wins the ESA-ADB Mission2 lightweight subset outright. "
            "RETRACTION, Round 19: a Round 18 note here claimed this forecaster was about 22x WORSE "
            "than persistence. That was a measurement error of ours, not a property of the model -- "
            "the probe resampled at 60s while the model is trained at 300s (see `resample` in the "
            "achieved record). Re-measured at the trained cadence, next-step MAE is 0.0314 on the "
            "evaluation split against 0.0897 for persistence, i.e. the model is about 2.9x BETTER; "
            "0.40x / 0.35x / 0.38x on train / test / whole series. Preprocessing for any evaluation "
            "must be read from this record rather than hardcoded, which is now what "
            "scripts/prep_model_inputs.py does. "
            "SEPARATE AND STILL OPEN: event-wise F0.5 does not reproduce across identical-config "
            "retrains -- 0.4545 here against the 0.714286 recorded -- while point-adjusted F1 does "
            "(0.9937 vs 0.9945) and the isolation-forest baseline reproduces exactly (0.000248). "
            "Treat the event-wise figure as a single draw from a wide distribution, which is the "
            "instability arXiv:2607.11969 (2026-07) describes for overlap-based metrics."
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
        artifacts=_STUDENT_ARTIFACTS,
        hf_repo="qsh-satellite-telemanom-ckks-student",
        runs_encrypted=True,
        notes=(
            "The one published model with no browser evaluation sample, and the reason is the "
            "metric rather than the weights. Round 21 gave the other three students samples; this "
            "one is scored on event-wise F0.5, which is defined over contiguous runs of a timeline "
            "and cannot be computed from a shuffled set of independent rows -- which is exactly "
            "what the browser evaluator scores. Shipping a per-row metric instead would verify a "
            "number nobody published for it. "
            "It needs no scaler, unlike the other three: scripts/train/satellite_telemanom.py fits "
            "it on raw `flat_te` (the last 8 timesteps x 6 channels), so the absent `input` block "
            "in its weights.json is correct rather than the omission it looks like. "
            "Worth reading alongside its numbers: the teacher scores 0.0 event-wise F0.5 on this "
            "split and the student 0.0032, so agreement of 0.9987 is agreement about finding almost "
            "nothing."),
    ),
    ModelSpec(
        usecase="satellite_comms",
        tier="t2_industry",
        key="satellite_comms.opssat",
        label="OPS-SAT segment anomaly detector",
        architecture="gradient-boosted trees over 19 per-segment telemetry features",
        params=None,
        dataset=DS_OPSSAT_AD,
        metric=M_OPSSAT,
        artifacts=_ONNX_PAIR,
        hf_repo="qsh-satellite-opssat-ad",
        notes=(
            "Segment CLASSIFICATION on real OPS-SAT telemetry, which is a different task from the "
            "Telemanom forecaster above: this one is scored against the 30 algorithms published "
            "with the dataset, not against persistence, and the two numbers must not be read as a "
            "comparison. Chosen because its corpus is CC BY 4.0 and 507 KB - the one case in this "
            "project where the data itself can be redistributed alongside the weights."
        ),
    ),
    ModelSpec(
        usecase="satellite_comms",
        tier="t1_crypto",
        key="satellite_comms.opssat_student",
        label="CKKS-linear OPS-SAT segment student",
        architecture="logistic regression over the 19 published per-segment features",
        params=None,
        dataset=DS_OPSSAT_AD,
        metric=M_OPSSAT,
        artifacts=_STUDENT_ARTIFACTS,
        hf_repo="qsh-satellite-opssat-ckks-student",
        runs_encrypted=True,
        notes=("19 features is small enough to evaluate under CKKS as a single packed dot product, "
               "so the encrypted tier here runs on the published feature set rather than on a "
               "reduction of it."),
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
        # The published repo carries onnx/model.onnx (an ai.onnx.ml graph, so int8 is skipped:
        # a tree/kernel model keeps its parameters in node attributes). Declaring only
        # weights.json meant browserModels() found nothing runnable and the page reported
        # "nothing to verify" for a model that is published, pinned and browser-runnable.
        artifacts=(ArtifactSpec("onnx/model.onnx", "onnx", "fp32", "both"),
                   ArtifactSpec("feature_spec.json", "json", "none", "both")),
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
        stale = entry.get("hf_revision_invalidated_by_retrain")
        if rev:
            out.append(replace(s, hf_revision=rev))
        elif stale:
            out.append(replace(s, hf_revision_stale=stale))
        else:
            out.append(s)
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


#: Where each model's committed artefacts live, relative to the repo root. Defined here rather than
#: in scripts/publish_hf.py because the registry needs it too -- it reads each export report to learn
#: whether the graph is browser-runnable -- and two copies would drift on the first new model.
ARTIFACT_DIRS: dict[str, str] = {
    "finance_aml.gbm": "models/finance_aml_gbm",
    "medical_fl.resnet18": "models/medical_breastmnist_resnet18",
    "medical_fl.resnet50": "models/medical_breastmnist_resnet50",
    "genomics.prs": "models/genomics_prs",
    "satellite_comms.telemanom": "models/satellite_telemanom",
    "satellite_comms.opssat": "models/satellite_opssat",
    "satellite_comms.opssat_student": "models/satellite_opssat_student",
    "telecom_ids.qml": "models/telecom_qml_ids",
    "finance_aml.student": "models/finance_aml_student",
    "medical_fl.student": "models/medical_breastmnist_student",   # already on disk from training
    "satellite_comms.student": "models/satellite_telemanom_student",
}

#: Repo root, so ARTIFACT_DIRS resolves regardless of the caller's working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _export_report(key: str) -> dict:
    """The committed export report for a model key, or {} when it was never exported."""
    import json

    directory = ARTIFACT_DIRS.get(key)
    if not directory:
        return {}
    path = _REPO_ROOT / directory / "export_report.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:                                  # noqa: BLE001 - a malformed report is not fatal
        return {}


def _onnx_domains(key: str) -> list[str]:
    return list(_export_report(key).get("onnx_domains") or [])


def _artifact_exists(key: str, path: str) -> bool:
    """Whether the export actually produced this artefact.

    ``_ONNX_PAIR`` declares an int8 graph for every model, but int8 is genuinely absent for two
    classes: tree ensembles and ai.onnx.ml linear models keep their parameters in node attributes,
    so there is nothing to quantise, and a graph can be built, measured and then withheld for
    missing the numeric gate. Advertising a file that was never written sends the browser to a
    404 -- or worse, past the graph assertion with the wrong name.
    """
    report = _export_report(key)
    if not report:
        return True          # unexported: nothing measured either way, so do not over-claim absence
    if path.endswith("model_int8.onnx"):
        return bool(report.get("int8_path"))
    return True


def _browser_runnable(key: str, precision: str = "fp32") -> bool:
    """Whether ORT Web can execute this model's graph at the given precision.

    RECORDED AS PROVENANCE, NOT AS A GATE. `browserModels()` in the frontend deliberately does not
    consult this: the flag was built on a diagnosis that was retracted, in which unsupported
    operators were blamed for what turned out to be the JSEP build being imported unconditionally on
    a machine with no GPU adapter.

    This docstring used to claim that "the int8 Telemanom graph loads in 0.03 s under Python ORT and
    never finishes session creation in a browser, because MatMulInteger + DynamicQuantizeLinear
    alongside LSTM is not viable in ORT Web". That is false, and it is the same retracted diagnosis
    wearing a specific number. Round 21 measured it: `satellite_comms.telemanom` serves exactly that
    int8 graph to the browser, and it creates a session and runs a 1x250x6 forecast on the WebGPU
    execution provider in 153 ms. Deleting the claim rather than softening it, because a confident
    measurement that was never re-measured is how it survived three rounds.

    `int8_browser_runnable` is written by no exporter, so this returns False for every int8 artefact.
    That is honest for an unmeasured property and harmless while nothing gates on it.
    """
    report = _export_report(key)
    field = "browser_runnable" if precision != "int8" else "int8_browser_runnable"
    return bool(report.get(field, False))


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
            "hf_revision_stale": s.hf_revision_stale,
            # Whether ONNX Runtime Web can execute this graph at all. A tree ensemble is
            # ai.onnx.ml (TreeEnsembleClassifier), a domain ORT Web does not implement, so it is
            # not browser-runnable at any size. Exported so the browser can decline BEFORE
            # downloading megabytes and then hanging in session creation, which is how this was
            # discovered.
            "browser_runnable": _browser_runnable(s.key),
            "onnx_domains": _onnx_domains(s.key),
            # Only artefacts the export actually wrote. A declared-but-absent int8 graph used to
            # reach the browser, which prefers int8, and the fetch then named a file that does not
            # exist in the repo.
            "artifacts": [
                {"path": a.path, "fmt": a.fmt, "precision": a.precision, "serves": a.serves,
                 # Per-artifact, because a model can be browser-runnable at fp32 and not at int8.
                 "browser_runnable": (_browser_runnable(s.key, a.precision)
                                      if a.fmt == "onnx" else False)}
                for a in s.artifacts if _artifact_exists(s.key, a.path)
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
