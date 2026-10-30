"""The per-use-case model registry — the single source of truth for what runs where.

Every use case has up to three model tiers, and the *relationship between them* is the point:

  T1 ``crypto``    the FHE/MPC-feasible student. Small because homomorphic evaluation is
                   cost-bounded **at the operating point this demo targets**: a single device, in a
                   browser tab, evaluating CKKS through node-seal - SEAL compiled to WASM,
                   single-threaded, under the 2 GB WASM memory ceiling.

                   The scoping matters and an earlier revision of this line lacked it, claiming
                   homomorphic evaluation "is, and will remain, cost-bounded". That does not survive
                   2026: Park et al. (arXiv:2601.18511v2, 2026-06-17) run Llama-3-8B end to end under
                   CKKS, 128 encrypted tokens in 20 s for summarisation and 18 s/token generation - on
                   **eight** NVIDIA RTX PRO 6000 GPUs. This project owns one of that GPU model and its
                   FHE default is TenSEAL on CPU, while the artifact that has to run is one
                   single-threaded WASM tab, so that result sits far outside this operating point;
                   no measurement here puts a number on the gap. A 33-parameter logistic regression
                   is not a compromise here; it is the class of model that closes at this operating
                   point. State the operating point, not a prediction about the field - the scoped
                   claim is falsifiable and the unscoped one was already false.

                   AND IT DOES NOT HOLD FOR EVERY USE CASE. Round 32 found the citation that falsifies
                   it for genomics specifically: Knight et al., Cell Reports Methods 6(1):101271
                   (2026-01-26), run polygenic risk scoring under HE over **110,000 SNPs** and describe
                   it as "practical to use on a single CPU" - roughly 350x this project's 313 variants.
                   So `genomics.prs` is not small because homomorphic evaluation is expensive. It is
                   small because PGS000004 HAS 313 variants, which is a property of Mavaddat et al. 2018
                   and not of CKKS.

                   The cost argument survives where the model is not a bare dot product - finance,
                   medical and satellite all evaluate something with depth. Applying it to a linear
                   score was the kind of generalisation that reads as rigour and understates what the
                   architecture can do.

                   The primary source for depth being the binding cost is ATLAS (arXiv:2607.23478,
                   2026-07-26), which measures it directly for homomorphic transformer inference.
                   Cited HERE rather than only in references/REFERENCES.md because that file claimed
                   this docstring rested on a secondary systematisation (IACR ePrint 2026/935) when
                   Round 31 had already removed that citation from this file - a reference added for a
                   code site that was never updated to use it.

                   And it is contested. Trimming (arXiv:2608.00375, 2026-08-01) argues depth is
                   constrained by the modulus-chain structure and can be decoupled from it via an
                   auxiliary chain and rational levels. No number from it is quoted anywhere in this
                   project and none should be: it presents no experiments, closing with the statement
                   that the framework "will be further validated through concrete implementation and
                   experimental evaluation". It is recorded as a named direction that would qualify
                   this paragraph if it were demonstrated, not as a result that already does.
  T2 ``industry``  the real literature model for that industry, trained locally and published to
                   the Hugging Face Hub as safetensors + ONNX (fp32 server / int8 browser).
  T3 ``large``     models too big to run under encryption at all — served remotely or in-browser
                   via WebGPU, and labelled as plaintext.

Two measured numbers tie T1 to T2, and they come from different places.

  * OFFLINE, at training time: ``crypto_fidelity`` (``agreement`` and the metric delta recorded in
    ``models/metrics.json``) is how often the student reproduces the plaintext teacher's decision
    on a held-out split, and how much task metric that costs. Both predictions are computed in
    plaintext, so this measures the distillation, not the encryption.
  * PER RUN, in the browser: ``encrypted_agreement`` is how often the CKKS evaluation of the model
    the page encrypts reproduces that same model's plaintext top-1 decision over the rows it scores.
    That is what CKKS noise costs. It is not student against teacher.

Neither number bounds what private inference costs in general. A metric lost to the student is the
cost of the student this project chose (exact encrypted tree-ensemble inference exists), not of
encryption itself.

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
    #: The credit line CC BY requires wherever the data, or a model trained on it, is published.
    #:
    #: SEPARATE FROM ``notes`` because the two have different audiences and different force. ``notes``
    #: is this project's own adjudication and does not leave Python -- ``frontend_models()`` exports
    #: name/licence/url/publishable and nothing else, so a visitor to a use-case page saw
    #: "OPS-SAT-AD -- CC BY 4.0 [source]" and no author at all. That is a licence condition going
    #: unmet on a public site, not an untidy caption. This field is threaded to the browser, to the
    #: Hugging Face model card and to the evaluation sample's provenance, which are the three places
    #: a reader meets the data.
    #:
    #: Defaulted to "" so no call site breaks, but every CC BY corpus here carries one and adding a
    #: fifth without one should look wrong: this project's single most-repeated defect is a fix that
    #: reached one of two twins, and a per-dataset obligation is exactly that shape.
    attribution: str = ""


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
    #: Set when the measured figure and ``baseline`` share a metric NAME but are not the same
    #: measurement, so a bar chart placing them side by side is comparing protocols rather than
    #: methods. ``MeasuredAgainstPublished`` renders this under the plot and marks the published bar;
    #: empty means the caption's like-for-like claim stands.
    #:
    #: Explicit per model, NOT inferred from the presence of a note field: ``sample_size_note`` reads
    #: as a warning on ``satellite_comms.telemanom`` and as a REASSURANCE on ``telecom_ids.nidd``, so
    #: keying off it would suppress a valid comparison and is the two-fields-one-word trap again.
    comparison_caveat: str = ""


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
    # Non-empty when the model publishes NO browser evaluation sample on purpose, and this is why.
    # The browser reads it and does not request `eval_sample.json`, which until Round 71 it fetched
    # from the Hub on every satellite run and logged as a 404 - an absence decided in Round 21 that
    # read in the console like a missing file.
    eval_sample_reason: str = ""
    # Which split the published evaluation sample was drawn from, and how, in the words the run log
    # prints. Declared per model since Round 71, because the log said "held-out" for every model
    # while two samples were drawn from the whole corpus, training rows included: the finance
    # teacher's 8,192 rows (and so the student's, which reuses them) and Telemanom's 128 windows.
    # scripts/prep_model_inputs.py writes this into the sample it builds, so the file and the page
    # say the same thing.
    eval_split: str = ""
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
    # CDLA-Sharing-1.0 section 3.1 requires the attribution notices to travel with the data and with
    # anything published from it. Empty until Round 51, alongside `DS_ESA_ADB` - the two corpora whose
    # licences are NOT spelled `CC-BY-4.0`, which is exactly what Round 49's backfill matched on.
    attribution=(
        "Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering "
        "Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money "
        "Laundering (AML), CDLA-Sharing-1.0."
    ),
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
    licence="CC-BY-4.0 asserted by MedMNIST; the BUSI authors publish no licence",
    url="https://medmnist.com/",
    licence_checked="2026-10-26",
    publishable=True,
    notes=(
        "780 images at 28x28. The tiny tier; also the source of the FHE-feasible T1 student. "
        "TWO problems, both upstream of MedMNIST, both checked 2026-08-10. "
        "LICENCE: MedMNIST licenses ITSELF CC BY 4.0, but BreastMNIST is a resize of BUSI "
        "(Al-Dhabyani et al., Data in Brief 28:104863, 2020) and the CC BY there attaches to the "
        "ARTICLE - PMC carries 'This is an open access article under the CC BY license' and states "
        "no separate licence for the data, while the dataset host (scholar.cu.edu.eg) asks only "
        "'If you use this dataset, please cite'. A citation request is not a grant. CHECKED AGAIN "
        "2026-10-26 at every source the authors publish: the host page asks the same in every "
        "capture from 2021 to 2026; the article's supplements hold the Baheya Hospital ethics "
        "committee's approval to publish the dataset publicly and the waiver of consent, and no "
        "licence; the authors made no repository deposit; and every Kaggle mirror carries a licence "
        "its uploader chose, none of them an author. So the BUSI authors publish no licence: the "
        "data was released publicly with no terms, and the only licence anyone states is "
        "MedMNIST's CC BY 4.0 for its derived dataset. The three HF repos stay public on that "
        "basis, by decision (2026-10-26). That differs from CIC-IDS2017, declined, and ImageNet, "
        "dropped: their providers state conflicting or restrictive terms, and BUSI's state none. "
        "CONTAMINATION: Pawlowska, Karwat & Zolek (Letter to the Editor, Data in Brief 48:109247, "
        "2023-05-19) found 235 duplicates (~19% of 780), >=8 images in BOTH benign and malignant, "
        "70 axilla images that are not breast tissue, and warn of train/test leakage. MEASURED HERE "
        "on the actual npz rather than assumed: at 28x28 exactly ONE pair survives as a pixel-exact "
        "duplicate - train[115] label=0 (MALIGNANT) and test[76] label=1 (BENIGN). The same image, "
        "trained as malignant and scored as benign. That is 1 of 156 test images, 0.64%, which the "
        "model cannot get right and which no amount of training fixes. Loosening to RMSE<=0.05 "
        "finds 216 near-duplicate pairs, 120 crossing a split boundary and 6 both crossing AND "
        "contradicting - but those are threshold-dependent and the exact pair is the only hard "
        "number. So the letter's 19% does NOT wholesale survive downsampling; a real floor on this "
        "benchmark does."
    ),
    # Both layers, because the corpus is a derivative and CC BY credit runs through to the source.
    # The second sentence is not decoration: the licence note above records that the upstream grant
    # is what is actually unverifiable, so naming BUSI is what lets a reader check that for
    # themselves rather than take this project's word for it.
    # MedMNIST asks for BOTH papers to be cited (its README's citation section): v2 and the v1
    # conference paper.
    attribution=(
        "Yang, Shi, Wei, Liu, Zhao, Ke, Pfister & Ni, 'MedMNIST v2 - A large-scale lightweight "
        "benchmark for 2D and 3D biomedical image classification', Scientific Data 10:41 (2023), "
        "and Yang, Shi & Ni, 'MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark "
        "for Medical Image Analysis', IEEE ISBI 2021; CC BY 4.0. BreastMNIST is derived from "
        "Al-Dhabyani, Gomaa, Khaled & Fahmy, 'Dataset of breast ultrasound images', Data in Brief "
        "28:104863 (2020)."
    ),
)
DS_PGS_CATALOG = DatasetSpec(
    name="PGS Catalog scoring files",
    licence="per-score; EMBL-EBI Terms of Use by default",
    url="https://www.pgscatalog.org/downloads/",
    licence_checked="2026-08-03",
    publishable=False,
    notes=(
        "NOT blanket CC-BY-4.0. Round 46 corrected this note: scoring files carry NO '#license' "
        "header at all - a raw download of PGS000004 today has 14 '#' lines and none of them is a "
        "licence. The licence lives only in the REST record, as an EMBL-EBI Terms-of-Use "
        "pass-through, and individual scores can be CC BY-NC-ND. `scripts/train/genomics_prs.py` "
        "has said this correctly since it was written; this note and DATA_LICENSES.md contradicted "
        "the working code. PGS000001 itself carries the Catalog default terms, not CC-BY. An earlier "
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
    url="https://doi.org/10.5281/zenodo.12588358",
    licence_checked="2026-08-05",
    publishable=True,
    notes=(
        "2,123 annotated telemetry segments across 9 channels from ESA's OPS-SAT CubeSat, 20.4% "
        "anomalous, with 18 published per-segment features plus a `sampling` metadata column, and the benchmark's own train/test "
        "split. That split is stratified random, not temporal, so scores on it do not measure "
        "generalisation over time. Ruszczak et al., Scientific Data 12 (2025). CC BY 4.0, which "
        "permits redistributing the data with credit; the per-segment feature table is about "
        "507 KB, the raw segment file much larger. A different task "
        "from ESA-ADB: segment classification, not forecasting, so its score is not comparable "
        "with the Telemanom model's and neither is the other's baseline."
    ),
    # Its DATA is redistributed here -- the whole published test partition ships inside two HF repos
    # as eval_sample.json -- so the CC BY credit is a condition on this project's own distribution,
    # not only a courtesy to a reader. It is not the only such corpus: WDBC, PaySim, AMLworld,
    # ESA-ADB, 5G-NIDD, BreastMNIST and gnomAD rows ship too.
    attribution=(
        "Ruszczak, Kotowski, Evans & Nalepa, 'The OPS-SAT benchmark for detecting anomalies in "
        "satellite telemetry', Scientific Data 12, 710 (2025), doi:10.1038/s41597-025-05035-3, "
        "CC BY 4.0."
    ),
)
DS_ESA_ADB = DatasetSpec(
    name="ESA Anomaly Dataset (ESA-ADB), Mission1 lightweight channels 41-46",
    licence="CC-BY-3.0-IGO",
    url="https://zenodo.org/records/12528696",
    licence_checked="2026-08-03",
    publishable=True,
    # CC BY 3.0 IGO REQUIRES ATTRIBUTION, and this field was empty until Round 51 while two models
    # trained against this corpus were published. Round 49 added `attribution` and backfilled "all
    # four CC BY corpora" - it matched on the string `CC-BY-4.0` and this licence is `CC-BY-3.0-IGO`,
    # so the one corpus whose licence is spelled differently is the one the backfill missed. Eighth
    # instance of a fix reaching every site but one.
    #
    # Kotowski first for the PAPER, De Canio first on the Zenodo RECORD - both correct, and
    # `references/REFERENCES.md` records why citing the paper for the benchmark and the Zenodo DOI for
    # the data is not an inconsistency. Thirteen places in this repository once credited OPS-SAT-AD to
    # "Kotowski et al."; the DOI and the Zenodo id are the reliable discriminators, not the surname.
    # The benchmark README asks for the DMLR journal version to be cited; the Zenodo record names
    # its own three creators, in this order.
    attribution=(
        "Kotowski et al., 'European Space Agency Dataset and Benchmark for Real-World Anomaly "
        "Detection in Spacecraft Time Series', Journal of Data-centric Machine Learning Research "
        "(DMLR), 2026 (earlier version: arXiv:2406.17826). Data: De Canio, Kotowski & Haskamp, "
        "'ESA Anomaly Dataset', Zenodo record 12528696 (2024), CC BY 3.0 IGO."
    ),
    notes=(
        # THE SPAN IS SOURCED. Table 1 of arXiv:2406.17826v2 gives the anonymised durations:
        # Mission1 14 years, Mission2 3.5, both 17.5 - so the 17.5 once carried here was the two
        # missions' total, not Mission1's. `scripts/publish_hf.py` copies `DatasetSpec.notes`
        # verbatim into the model card, so this sentence reaches the Hub.
        "Missions are anonymised Mission1/2/3 - they are not named Sentinel anything. "
        "Mission1 holds 14 years of telemetry and Mission2 3.5 (anonymised durations, "
        "arXiv:2406.17826v2 Table 1; the paper sizes the corpus by data points instead, because "
        "sampling rates vary and the timeline is anonymised). "
        "1.19% of points annotated across Mission1 and Mission2 (1.80% in Mission1). "
        "Mission3 is excluded upstream for triviality. The CC BY 3.0 IGO term "
        "appears on the Zenodo landing page but NOT in its REST metadata, which returns no licence "
        "at all - an API-only check reads this corpus as unlicensed."
    ),
)
DS_SMAP_MSL = DatasetSpec(
    name="NASA SMAP/MSL spacecraft telemetry (Telemanom)",
    licence="no licence granted for the data",
    url="https://github.com/khundman/telemanom",
    licence_checked="2026-10-30",
    # `publishable` gates derived weights, and no published model trains on this corpus. Whether a
    # sample whose data carries no licence should stay bundled at all is an open decision.
    publishable=True,
    notes=(
        "NO SOURCE GRANTS A LICENCE FOR THIS DATA. The Telemanom repository's LICENSE.txt reads "
        "'Copyright (c) 2018, California Institute of Technology' and grants BSD-style terms for "
        "its software; it says nothing about the data. Nor is the data shown to be a US "
        "Government work: 17 U.S.C. 105 covers works prepared by federal officers or employees, "
        "and JPL is a NASA centre operated by Caltech. "
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
        "Engelen redistribution is separately hosted and states nothing. 5G-NIDD was evaluated as "
        "an alternative and rejected here on the same grounds, and THAT REJECTION IS WITHDRAWN: it "
        "is CC BY 4.0 with access_type Open at the authors' repository of record (Fairdata/Metax, "
        "DOI 10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361), and the subscription-required IEEE "
        "DataPort listing the rejection rested on is a mirror, not the source. CC BY 4.0 requires "
        "attribution wherever the data or a model trained on it is published; see DATA_LICENSES.md "
        "section 1, which is the owner of that adjudication - this note is a copy of it and copies "
        "are what went stale. 5G-NIDD does not change this spec either way: it is network-layer "
        "flow statistics, a different problem from QKD-attack classification, so the shipped "
        "telecom model still trains on synthetic telemetry and no code path loads this corpus."
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
    # The note directly above has said "attribution is still required" since Round 21 while there
    # was no field to put it in, so the requirement was recorded and not discharged. This is it.
    attribution=(
        "Wolberg, Mangasarian, Street & Street, 'Breast Cancer Wisconsin (Diagnostic)', UCI "
        "Machine Learning Repository (1993), doi:10.24432/C5DW2B, CC BY 4.0."
    ),
)

# The corpus this project rejected on licence grounds and then un-rejected: see DS_CICIDS2017_FIXED,
# which owns that retraction. Adopted here as its own model rather than as a replacement for the
# QKD-attack classifier, because it is a different problem.
DS_5G_NIDD = DatasetSpec(
    name="5G-NIDD (BTS1_BTS2_fields_preserved.csv)",
    licence="CC-BY-4.0",
    url="https://doi.org/10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361",
    licence_checked="2026-08-16",
    publishable=True,
    notes=(
        "1,215,890 real 5G network flows x 113 columns, counted in this file rather than quoted, "
        "9 classes: Benign 477,737 | UDPFlood 457,340 | HTTPFlood 140,812 | SlowrateDoS 73,124 | "
        "TCPConnectScan 20,052 | SYNScan 20,043 | UDPScan 15,906 | SYNFlood 9,721 | ICMPFlood 1,155. "
        "LICENCE VERIFIED AT THE SOURCE, not at a mirror: the Metax record for dataset "
        "9d13ef28-2ca7-44b0-9950-225359afac65 reads access_type Open, license CC-BY-4.0 and "
        "restriction_grounds []. The IEEE DataPort listing whose subscription wall this project "
        "originally read as a licence restriction is a MIRROR; the Fairdata record is the "
        "authors' repository of record. "
        "THE EXPORT CHANGED IN ROUND 50 AND THAT IS THE MOST IMPORTANT LINE HERE. This is "
        "`BTS1_BTS2_fields_preserved.csv`, 113 columns. The release also ships `Combined.csv` at 52 "
        "columns, which is what this project trained on until Round 50. Same rows, same classes, "
        "same distribution - and all 52 of Combined's columns are informationally identical in the "
        "wide file, equal distinct-value counts on every one, so Combined is a strict column "
        "SUBSET. It omits `IdleTime`, and that single column is worth 22 accuracy points: 0.9829 "
        "with it, 0.7616 without. Five other columns Combined lacks (`Dir`, `Flgs`, `TcpOpt`, "
        "`sVlan`, `dVlan`) are worth 0.0006 accuracy between them - which is redundancy rather than "
        "emptiness, since `Dir` alone separates ICMPFlood at one-vs-rest AUC 0.9811 and `TcpOpt` "
        "separates SlowrateDoS at 0.9433. The other 41 columns already carry what they carry. "
        "SIXTY-SIX OF THE 113 COLUMNS ARE UNUSABLE and the loader drops them with reasons attached "
        "(qsh.models.datasets.NIDD_DROPPED, republished in the model's feature_spec.json): 3 labels "
        "(`Attack Type`, its binary collapse `Label`, and `Attack Tool` which scores accuracy 0.9615 "
        "on its own), 9 row identifiers, 10 host fingerprints (addresses, MACs, ports, IP ids), 5 "
        "columns bit-identical to `Dur` on every row (`RunTime`/`Mean`/`Sum`/`Min`/`Max`), and 39 "
        "that hold exactly one state. 47 features remain. "
        "THE 39 SINGLE-STATE COLUMNS ARE WORTH A SEPARATE SENTENCE because their names promise "
        "signal: 30 of them are 100% NULL in this release, and that includes every SIntPkt/DIntPkt "
        "inter-packet-time column and every jitter column. They are empty, not weak. "
        "THE ROW IDENTIFIERS REMAIN A FINDING, with a narrower claim than Round 49 made. "
        "`Unnamed: 0` is the CSV's own index and restoring it alone takes accuracy from 0.9829 to "
        "0.9997. That is what a row counter is worth on this corpus. Whether published figures "
        "lean on such columns is an OPEN question: the dataset paper's 9-class baselines "
        "(Samarakoon et al., arXiv:2212.01298) rank the Argus sequence number `Seq` and `Offset` "
        "among their top-10 inputs, and Wang, Fok & Thing's 28 inputs include `Offset`, both "
        "columns this project drops as row identifiers; no published ablation settles it. See "
        "M_NIDD's baseline_source below."
    ),
    # Taken VERBATIM from the Metax record's own `bibliographic_citation`, not assembled here. An
    # assembled credit line is a place to guess an author list; the depositors published the form
    # they want to be cited in, and using it is both safer and more correct.
    attribution=(
        "Y. Siriwardhana et al., 'Descriptor: 5G Wireless Network Intrusion Detection Dataset "
        "(5G-NIDD)', in IEEE Data Descriptions, doi:10.1109/IEEEDATA.2025.3592888. Dataset: "
        "doi:10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361, CC BY 4.0."
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
        "real 1000 Genomes phase-3 genotypes + published PGS000004 (PRS313) weights (PGS Catalog: "
        "per-score licence, EMBL-EBI terms by default; IGSR grants no blanket licence). 30 of 2,504 "
        "individuals ship in the browser payload - DATA_LICENSES section 4 justifies a slice of that "
        "size, not the full matrix. Variants keyed by chr:pos:effect_allele because PGS000004 "
        "publishes no rsIDs.",
        DS_PGS_CATALOG,
    ),
    "satellite_comms": (
        "real NASA SMAP/MSL spacecraft telemetry (Telemanom / NASA JPL). No source grants a "
        "licence for the data: the Telemanom LICENSE is a Caltech copyright covering its "
        "software, and JPL is operated by Caltech, so the data is not shown to be a US Government "
        "work under 17 U.S.C. 105 either - see DATA_LICENSES.md.",
        DS_SMAP_MSL,
    ),
}

#: Where each bundled sample is served from, so the parity test can find the file it must match.
DEMO_SAMPLE_FILES: dict[str, str] = {
    "finance_aml": "web/public/data/finance/paysim_sample.json",
    "medical_fl": "web/public/data/medical/breast_cancer_sample.json",
    "genomics": "web/public/data/genomics/pgs000004_prs313.json",
    "satellite_comms": "web/public/data/satellite/telemetry.json",
}

#: `licence` says what is true today: NO licence is granted. "Generated by this repository" says
#: where the rows come from, not what anyone may do with them, and that phrase used to stand in
#: this field - and so in the public Hub card's licence name - as if it were a grant.
DS_QKD_ATTACK_SYNTH = DatasetSpec(
    name="QKD attack telemetry (synthetic, methodology-grounded)",
    licence="no licence granted; synthetic rows generated by this repository's own code",
    url="https://arxiv.org/abs/2509.14282",
    licence_checked="2026-10-30",
    publishable=True,
    notes=(
        "8 classes = 7 attack types + normal, per Al-Kuwari et al. (IET Quantum Communication 7(1), "
        "e70028, 2026). The class list follows the paper; the 7 features and their signature "
        "values are this project's own design. No commercial-OK real QKD-attack capture exists; "
        "this is labelled synthetic everywhere it is reported. No licence is granted for these "
        "rows or for the weights trained on them."
    ),
)


# --------------------------------------------------------------------------- #
#  Metrics — the published numbers each model is measured against
# --------------------------------------------------------------------------- #
M_AML = MetricSpec(
    name="minority_class_f1",
    display="minority-class F1 (laundering)",
    baseline=0.6323,
    # The protocol differs from this project's, and the sentence says so because the comparison
    # caveat field is not set for this metric (setting it changes how the chart renders).
    baseline_source=(
        "GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17). Same metric and "
        "corpus, different protocol: that figure uses a 60/20/20 split by transaction order and "
        "IBM's Graph Feature Preprocessor updated over the test window; this project tests on the "
        "last 30 percent of transactions (fit 56, validation 14) with its own graph features, "
        "whose structural block is fitted on the training window."),
    # 0.7401 is MEGA-PNA's row and NOT the table's best: the same Table 2 prints MEGA-GenAgg at
    # 74.88 and MEGA(PNA)-GIN at 74.69. The text below says so; moving the number is a data change.
    ceiling=0.7401,
    ceiling_source=(
        "MEGA-PNA, arXiv:2412.00241v2 Table 2 'Minority-class F1 scores (%) on AML edge "
        "classification task', AML Small HI (74.01 +/- 1.55); the paper's v3 is accepted for "
        "publication in TMLR (2026). It is not the table's highest row: the same table prints "
        "MEGA-GenAgg at 74.88 +/- 0.38 and MEGA(PNA)-GIN at 74.69 +/- 0.50, and Multi-PNA (68.16 "
        "+/- 2.65) below all three."),
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
                     "exists here - to test it rather than to win, under a training regime that "
                     "differs from the paper's (see the model's notes)."),
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
    name="auroc_eur",
    display="AUROC (European ancestry)",
    baseline=0.633,
    baseline_source=(
        "PGS Catalog PPM001647 for PGS000004 (PRS313): AUROC 0.633 (95% CI 0.624-0.641), n=400,812 "
        "Europeans - 4,340 cases and 396,472 controls - from Jia et al., JNCI Cancer Spectrum 2020 "
        "(doi:10.1093/jncics/pkaa021). Verified at the Catalog REST record 2026-08-11. "
        "TWO CAVEATS THAT MAKE THIS A CITATION AND NOT A TARGET, both from the record itself: the "
        "model is fitted WITH COVARIATES - 'family history of cancer (in first-degree relatives), "
        "genotyping array' - which a bare weighted dot product does not have; and the cohort was "
        "'missing 25 variants (8%) based on availability of variants in the cohort', so the published "
        "figure is for 288 of the 313 variants, not all of them. "
        "This replaced PGS000001's C-index 0.628 (PPM017270) when the score changed. That figure was a "
        "Harrell's C-index over censored follow-up, not an AUROC, and the two are not directly "
        "comparable - a C-index is computed over time-to-event pairs and an AUROC over a fixed "
        "case/control split. PGS000004 publishes a genuine AUROC, so the switch also removes that "
        "comparability problem rather than only changing the number."
    ),
    ceiling=0.668,
    ceiling_source=(
        "Highest AUROC among PGS000004's OWN Catalog performance records: 0.668 (PPM012913, n=302,425, "
        "European). NOT THE BARE SCORE: that record evaluates PGS000004 combined with the iCARE-Lit "
        "classical risk factors (age at menarche, parity, BMI, family history and others) for "
        "5-year absolute risk (Hurson et al., Int J Epidemiol 2022), so a weighted dot product "
        "alone is not expected to reach it. "
        "Measured across all 28 AUROC records for this score on 2026-08-11, not taken from a "
        "range quoted elsewhere - a first attempt at this line read the first page of results only and "
        "put the ceiling at 0.6573, which is a real record but not the maximum. European records span "
        "0.5877-0.668 over 22 entries; the largest cohort is PPM012918 at n=5,328,392 with 0.665. "
        "The previous ceiling of 0.69 was chosen against a C-index scale, so it was never a bound on an "
        "AUROC for this score even though it happens to sit close to one. "
        "WORTH READING BESIDE THIS USE CASE'S FINDING: the lowest of the 28 is 0.55 (PPM005171, "
        "n=3,801, 'African unspecified'), against 0.5877-0.668 for European samples. The published "
        "record set shows the same ancestry gap this page demonstrates by a different route."
    ),
    companion="",
    # Any germline-SNP PRS claiming much above ~0.70 discrimination is measuring something else.
)
M_OPSSAT = MetricSpec(
    name="auc_pr",
    display="AUC PR",
    baseline=0.979,
    baseline_source=("FCNN, best of the 30 algorithms benchmarked in Ruszczak et al., Scientific "
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
    baseline=0.968,
    baseline_source=(
        "ESA-ADB Supplementary Table 9, Mission1 lightweight (ch. 41-46), Telemanom-ESA-Pruned: "
        "corrected event-wise F0.5 0.968 (precision 0.999, recall 0.862) for detection of ANOMALIES "
        "ALONE - which is what this project scores, so this is the like-for-like row. Earlier rounds "
        "cited Table 2's 0.786 instead. That number is real and correctly attributed, but Table 2 "
        "scores ALL EVENTS excluding communication gaps, i.e. anomalies PLUS annotated rare nominal "
        "events, so it was the wrong comparator and the lenient one. Verified against the paper "
        "(arXiv:2406.17826 v2) in Round 24. FOUR protocol differences remain and none of them are "
        "closed by changing the number: (1) ESA-ADB's metric multiplies event precision by the "
        "TIME-LEVEL true-negative rate, which is why most of its baselines read <0.001 - this "
        "project records that corrected value as `corrected_event_wise_f0_5` beside the uncorrected "
        "one; (2) ESA-ADB splits each mission in half, this project tests on the last 30 percent; "
        "(3) ESA-ADB's Mission1 lightweight test partition holds 65 annotated events of which 29 are "
        "anomalies, against 3 in the window scored here; (4) ESA-ADB resamples Mission1 at "
        "0.033 Hz, about one sample per 30 s, chosen from the dominant rate of channels 41-46, "
        "while this model resamples to 300 s, so its 250-step window spans about 20.8 hours "
        "where the benchmark's would span about 2.1, and short events can vanish. So this remains "
        "an indicative target, and the sample-size gap is roughly tenfold."
    ),
    ceiling=None,
    ceiling_source="",
    companion="event_wise_f0_5_full_channels",
    # The companion is the whole point: the same method scores 0.061 on the full Mission1 channel set -
    # but that is a TABLE 2 figure, whose matched lightweight partner is Table 2's 0.786, not the 0.968
    # three lines above (Supp. Table 9, anomalies alone, whose own full-channel number is 0.039). The
    # channel-scope collapse only means something within one table and one event set.
    comparison_caveat=(
        "The two bars are event-wise F0.5 under the same overlap rule, and that is where the "
        "agreement stops. This project's scored window contains THREE labelled anomaly events; "
        "ESA-ADB's Mission1 lightweight test partition contains 29, roughly a tenfold difference in "
        "the denominator. None of the three is caught here (tp 0, fp 2, fn 3), so the score is 0 "
        "and another false alarm cannot lower it; catching one event with the same two false alarms "
        "would lift it to 0.33. The splits differ too (last 30 percent here, half the mission "
        "there), so does the resampling (300 s here, about 30 s there), and ESA-ADB multiplies "
        "event precision by the time-level true-negative rate, which this figure does not. Treat "
        "0.968 as an indicative target, not as a score this bar was competing against."
    ),
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
        "its own authors for evaluation that was 'insufficient to fully support the conclusions'. That "
        "single withdrawal is thin support for a field-wide claim, so the anchor is now a survey: "
        "Kaissar, Bou Nassif & Bouridane, Future Internet 18(5):234 (2026-04-27, CC BY 4.0, DOI "
        "10.3390/fi18050234), systematic to January 2026, finds that 'most current approaches rely on "
        "simulated quantum environments and legacy datasets, while evaluation practices remain "
        "inconsistent across studies' - active but methodologically early, which is a different and "
        "better-founded statement than 'worse than a random forest'. The Al-Kuwari paper is now "
        "peer-reviewed: IET Quantum Communication 7(1), DOI 10.1049/qtc2.70028 (2026-02-23). And the "
        "nearest strong 2026 result is CLASSICAL, which corroborates the framing here - but it is "
        "a DIFFERENT problem and Round 46 stopped calling it the same one. IACR ePrint 2026/1282 "
        "reaches 96.9% recall / 97.6% precision with physics-aware temporal features plus XGBoost "
        "against 17.3% for static QBER thresholding - but its abstract, read at source, is "
        "entanglement-based BBM92, a 24-dimensional feature space over 30-second sliding windows, "
        "binary detection of blended sub-threshold attacks on simulated FSO telemetry. This model "
        "classifies 8 BB84 attack TYPES from per-session physical observables. Neither the "
        "protocol, the label space nor the feature space matches, so it is an anchor, not a "
        "baseline."
    ),
    companion="classical_rf_macro_f1",
)
M_NIDD = MetricSpec(
    name="macro_f1",
    display="macro F1 across 9 network-attack classes",
    # `baseline=None` IS THE DECISION, and it follows M_IDS above rather than being an omission.
    # `ModelPanel` renders `metric.baseline` beside whichever recorded key matches `metric.name`, so
    # `baseline=0.9961` here would print "macro_f1 0.9266, baseline 0.9961" - a macro F1 against an
    # ACCURACY, which is the exact failure tests/test_metric_baseline_consistency.py exists for. The
    # published figures for this corpus are accuracies; they belong in prose, below.
    baseline=None,
    baseline_source=(
        "Two published figures on 5G-NIDD are usually quoted - Wang, Fok & Thing report 99.61% "
        "accuracy for a classical CNN and 99.71% for a hybrid quantum-classical Quan-ConvCNN - and "
        "BOTH are accuracies on the BINARY benign-versus-malicious task, while this model is scored "
        "on 9 classes, which is why no number appears in `baseline`. The 9-class anchors are the "
        "dataset paper's random forest at 99.48% accuracy on its top-10 features (Samarakoon et "
        "al., arXiv:2212.01298) and BiTAD (Future Internet 17(11):482) at 90.47%. Caveats. (1) The Wang, Fok & Thing authors' own claim is about "
        "generalisation to UNSEEN attacks rather than about the headline; "
        "`references/REFERENCES.md` WS2 records this. (2) Accuracy is the wrong comparator on a "
        "corpus whose two majority classes are 39.3% and 37.6% of the rows. (3) MEASURED HERE: an "
        "identical gradient-boosted model reaches accuracy 0.9997 with the corpus's integer row "
        "counters restored, and 0.9829 with every identifier and every host fingerprint dropped; "
        "the column that lifts it from the 0.7608 of the 52-column export is one flow statistic, "
        "`IdleTime`, which that export does not carry. (4) Whether published figures lean on row "
        "identifiers is OPEN: those top-10 inputs include the Argus sequence number `Seq` and "
        "`Offset`, Wang, Fok & Thing's inputs include `Offset`, both columns are dropped here as "
        "identifiers, and no published ablation settles it. (5) This model's "
        "figures come from a stratified random split, which flatters: when Wang, Fok & Thing train "
        "on one base station and test on the other, their accuracies fall from above 98% to about "
        "78-82%. These figures are an anchor and not a target that has been beaten or missed."
    ),
    companion="aggregate_accuracy",
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
            "Reference architecture only - never trained or published here. Message passing needs "
            "each account's graph neighbourhood at scoring time, which this demo's browser tier does "
            "not hold, so every tier ships the gradient-boosting model, which scores one precomputed "
            "feature row per transaction. That is a choice for this operating point, not an "
            "impossibility: Bilgi et al. (arXiv:2610.18054, preprint) run a 2-layer GNN under CKKS "
            "on AMLworld HI-Small at 5.31 s per inference on one GPU, with the graph structure "
            "visible to the server. Kept as a spec for the architecture it names. It is no longer "
            "the best published figure on this split: the MEGA rows above it are (see the ceiling)."
        ),
    ),
    ModelSpec(
        usecase="finance_aml",
        tier="t2_industry",
        key="finance_aml.gbm",
        label="Graph-feature gradient boosting",
        # NOT IBM's Graph Feature Preprocessor (GFP): that ships in `snapml` under a proprietary
        # licence and is excluded here. scripts/train/finance_aml.py reimplements graph features.
        architecture=("per-account and structural graph features computed by this project "
                      "(not IBM's Graph Feature Preprocessor) + XGBoost"),
        params=None,
        dataset=DS_AMLWORLD,
        metric=M_AML,
        artifacts=_ONNX_FP32_ONLY,   # tree ensemble: no int8 graph exists to serve
        hf_repo="qsh-finance-aml-gbm",
        eval_split=(
            "a draw from the held-out temporal test window, the last 30 percent of the "
            "transactions in time order, enriched to 2 percent laundering"),
        notes=("The browser-tier finance model: it scores one feature row per transaction with no "
               "message passing at scoring time. That row is not stateless - its per-account and "
               "structural features are computed from the account's earlier transactions - so a "
               "deployment needs that history to score a new transaction."),
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
        eval_split=(
            "the rows of finance_aml.gbm's sample, a draw from the held-out temporal test window, "
            "the last 30 percent of the transactions in time order, enriched to 2 percent "
            "laundering"),
        # `runs_encrypted=True` describes what this student was built for, not what any code path
        # runs today; see the notes.
        notes=("Built to run under CKKS, but no code path here encrypts THIS model. The public "
               "page's CKKS step evaluates the 8-feature linear model bundled with the PaySim "
               "sample and re-scores this 33-parameter AMLworld student in plaintext ONNX; the "
               "full deployment's finance pipeline encrypts its own PaySim-trained surrogate. Its "
               "agreement with the teacher (`agreement`) is measured offline, at training time, in "
               "plaintext; each browser run measures `encrypted_agreement`, CKKS against plaintext "
               "for the model it encrypts."),
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
        eval_split="the official BreastMNIST test split, whole",
        notes=("The small tier: fits the browser comfortably and has a clean published baseline. "
               "Its training regime is not the baseline's: torchvision's 7x7 stride-2 stem with "
               "max-pool, early-stopped at 39 of at most 60 epochs at lr 3e-4 (manifest.json), "
               "where MedMNIST v2's 28-px ResNets use a 3x3 stride-1 stem without max-pool and 100 "
               "epochs at lr 1e-3. Close, not like for like."),
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
        eval_split="the official BreastMNIST test split, whole, the same rows as medical_fl.resnet18",
        notes=(
            "Trained to test a published claim rather than to win one. MedMNIST v2 reports ResNet-50 "
            "at AUC 0.857 against ResNet-18's 0.901 on this corpus, so the expected outcome is that "
            "the larger model scores LOWER, and it is reported either way. The regime here is not "
            "the published one: torchvision's 7x7 stride-2 stem with max-pool, early-stopped at 27 "
            "of at most 60 epochs at lr 3e-4, against MedMNIST v2's 3x3 stride-1 stem without "
            "max-pool and 100 epochs at lr 1e-3. It replaces a DenseNet-121 "
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
        eval_split="the official BreastMNIST test split, whole, at 14x14",
        notes=("Genuine encrypted convolution (im2col + CKKS dot_plain), but not end to end under "
               "CKKS. In the browser each layer's output is decrypted, ReLU and average pooling run "
               "in plaintext in the client, and the class is the argmax of decrypted logits; the "
               "page's MPC argmax runs on the WDBC logistic regression, not on this CNN. The full "
               "deployment's hybrid path, which runs a federated CNN of this shape rather than these "
               "weights, keeps ReLU on secret shares (MPyC) and pools each share locally; its masked "
               "conversion protects only CKKS slot 0 (see qsh.mpc.secret_inference)."),
    ),
    # ---------------- genomics ----------------
    ModelSpec(
        usecase="genomics",
        tier="t2_industry",
        key="genomics.prs",
        label="Polygenic risk score (published effect weights)",
        # PGS000004's development method per the PGS Catalog REST record (method_name and
        # method_params), read 2026-10-30. Genome-wide significance (P<5e-8) is PGS000001's.
        architecture=("per-variant effect-weight vector over 313 variants (PGS000004: "
                      "hard-thresholding stepwise forward regression, p < 1e-5)"),
        params=None,
        dataset=DS_PGS_CATALOG,
        metric=M_PRS,
        artifacts=(ArtifactSpec("weights.json", "json", "none", "both"),),
        hf_repo=None,          # publication blocked pending the per-score licence audit
        runs_encrypted=True,   # a dot product: the rare case that is both clinical standard and FHE-native
        notes=(
            "WHAT RUNS UNDER CKKS DIFFERS BY TIER. The browser's CKKS step is a sign test on the 8 "
            "largest-effect of the 313 variants (W = [-w/2, +w/2], b = 0), not the 313-variant "
            "score; the full deployment evaluates all 313 against a 75th-percentile cut. "
            "Not a neural network, and presenting a genomic foundation model as a PRS method would be "
            "the most obvious credibility hole available. Evaluation must be stratified by genetic "
            "ancestry, and what is measured here is a distribution shift, NOT an AUROC drop: an "
            "earlier revision of this note called it an AUROC drop, but 1000 Genomes is a reference "
            "panel carrying no breast-cancer case/control status, so no AUROC is computable from it "
            "and genomics_prs.py records auroc=null rather than inventing one. PGS000004 (PRS313) is Mavaddat "
            "et al. (AJHG 2018), 313 variants - not PRS-CS or LDpred2, which are Bayesian-shrinkage "
            "methods this score does not use. An earlier architecture string named that family and "
            "contradicted this project's own training script. Until Round 31 this page ran PGS000001, the "
            "77-variant Mavaddat 2015 score selected by genome-wide significance (P<5e-8); the switch is "
            "recorded because the variant count and the baseline both changed with it. The finding is the "
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
        eval_split=(
            "windows drawn at random from the held-out temporal test window, the last 30 percent "
            "of the resampled timeline"),
        notes=(
            "Report event-wise F0.5, never point-adjusted F1. Always run a trivial baseline alongside: "
            "a windowed isolation forest scores highest on the ESA-ADB Mission2 lightweight subset "
            "in the paper's Table 2 (all events, 0.949), though not in Supplementary Table 9 "
            "(anomalies only, 0.000). "
            "RETRACTION, Round 19: a Round 18 note here claimed this forecaster was about 22x WORSE "
            "than persistence. That was a measurement error of ours, not a property of the model -- "
            "the probe resampled at 60s while the model is trained at 300s (see `resample` in the "
            "achieved record). Preprocessing for any evaluation must be read from this record "
            "rather than hardcoded, which is now what scripts/prep_model_inputs.py does. "
            "RETRAINED IN ROUND 71. The weights published until then were trained with a z-score "
            "fitted on the whole series, test window included: the trainer was fixed in Round 34 "
            "and the model was never retrained, so its feature_spec.json still carried whole-series "
            "constants. With the constants fitted on the training window, next-step MAE over all "
            "441,600 test windows is 0.0250 against 0.0764 for persistence, about 3.1x better. The "
            "event-wise figure fell to 0 (tp=0 fp=2 fn=3): the detector still flags 2 points of "
            "441,600 and neither lands in one of the 3 scored events, where the two leaky runs had "
            "caught one and recorded 0.714286 and 0.454545. Point-adjusted F1 fell with it, from "
            "0.9937 to 0, which is how much one flagged point inside a long event carries under "
            "point adjustment; the isolation-forest baseline reproduces exactly (0.000248). Treat "
            "the event-wise figure as a single draw from a wide distribution: three events cannot "
            "rank detectors. With 0 of 3 events detected, the exact (Clopper-Pearson) 95% interval "
            "for recall runs from 0 to 0.71."
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
        eval_sample_reason=(
            "its metric, event-wise F0.5, is defined over contiguous runs of a timeline and cannot "
            "be re-scored from independent rows, which is what the browser evaluator scores"),
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
            "Worth reading alongside its numbers: since the Round 71 retrain the teacher and the "
            "student both score 0.0 event-wise F0.5 on this split, so their agreement is agreement "
            "about finding nothing, and there is no fidelity claim to make. No code path here "
            "evaluates this student under CKKS today: the satellite page has no private-inference "
            "stage, and the full deployment's telemetry stage encrypts its own SMAP/MSL model."),
    ),
    ModelSpec(
        usecase="satellite_comms",
        tier="t2_industry",
        key="satellite_comms.opssat",
        label="OPS-SAT segment anomaly detector",
        architecture="gradient-boosted trees over 18 published per-segment features + `sampling`",
        params=None,
        dataset=DS_OPSSAT_AD,
        metric=M_OPSSAT,
        artifacts=_ONNX_PAIR,
        hf_repo="qsh-satellite-opssat-ad",
        eval_split="the benchmark's own test partition, whole",
        notes=(
            "Segment CLASSIFICATION on real OPS-SAT telemetry, which is a different task from the "
            "Telemanom forecaster above: this one is scored against the 30 algorithms published "
            "with the dataset, not against persistence, and the two numbers must not be read as a "
            "comparison. Chosen because its corpus is CC BY 4.0 and small, so the whole test "
            "partition can be redistributed alongside the weights. The benchmark's split is "
            "stratified random, so these scores do not measure generalisation over time. The "
            "card's metrics are the scikit-learn model's before export; the published ONNX graph "
            "scores accuracy 0.9509 on the same 529 test segments against 0.9660 recorded here, "
            "so the browser's re-score reproduces the graph, not the card."
        ),
    ),
    ModelSpec(
        usecase="satellite_comms",
        tier="t1_crypto",
        key="satellite_comms.opssat_student",
        label="CKKS-linear OPS-SAT segment student",
        architecture=("logistic regression over the quadratic expansion (209 terms) of the 18 "
                      "published per-segment features + `sampling`"),
        params=None,
        dataset=DS_OPSSAT_AD,
        metric=M_OPSSAT,
        artifacts=_STUDENT_ARTIFACTS,
        hf_repo="qsh-satellite-opssat-ckks-student",
        runs_encrypted=True,
        eval_split="the benchmark's own test partition, whole",
        notes=("19 columns is small enough to evaluate under CKKS as a single packed dot product, "
               "so the encrypted tier here runs on the published feature set rather than on a "
               "reduction of it - and it had room to spare. The student now fits over the QUADRATIC "
               "expansion of those features, 209 terms, still one packed dot product at unchanged "
               "multiplicative depth, because the expansion is a function of the plaintext input "
               "and is computed before encryption. The ring carries orders of magnitude more slots "
               "than this feature map needs, so the cipher was never the binding constraint "
               "here - the feature map was. Measured: "
               "minority-class F1 0.8649 -> 0.9043, AUC-PR 0.9275 -> 0.9681 (0.968055 against the "
               "teacher's 0.968107, equal at four decimals). No code path here evaluates this "
               "student under CKKS today: the satellite page has no private-inference stage, and "
               "the browser re-scores the published ONNX graph in plaintext."),
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
            "NOT TRAINED IN THIS REPOSITORY, and an earlier revision of this note said otherwise. It "
            "read 'Trained on the Engelen-corrected corpus locally', which nothing here corroborates: "
            "there is no `telecom_ids.transformer` key in `models/metrics.json`, no entry in "
            "`ARTIFACT_DIRS`, no directory under `models/`, and no test asserting any of it. The "
            "`ModelPanel` printed that sentence directly beside its own 'Not trained in this build - no "
            "measured metrics', so the same panel asserted both. A local run nobody can point at is not "
            "a training claim. Weights would in any case not be published: the redistribution terms of "
            "the corrected dataset are unverified, which is why `hf_repo` is None."
        ),
    ),
    ModelSpec(
        usecase="telecom_ids",
        tier="t2_industry",
        key="telecom_ids.qml",
        # The key keeps its name; the label says what the published artefact IS. The ONNX graph on
        # the Hub, and the one the browser runs and verifies, is the 200-tree random forest
        # (scripts/train/telecom_ids.py exports `rf`). The two quantum models are trained offline
        # and recorded in models/metrics.json; no page runs them.
        label="Random-forest QKD-attack classifier, with quantum-kernel SVC and QNN comparisons",
        architecture=("published ONNX: a 200-tree scikit-learn random forest over 7 telemetry "
                      "features. Offline comparisons, not in the ONNX: a quantum-kernel SVC (PCA "
                      "-> angle embedding; its fidelity kernel is exactly the classical product "
                      "kernel prod cos^2((x_i - x'_i)/2)) and a variational QNN "
                      "(StronglyEntanglingLayers, 25 parameters)"),
        # 25 is the QNN's parameter count, not the forest's.
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
        eval_split="the held-out quarter of the seed-42 stratified split, whole",
        notes=("The published model and its headline accuracy are the random forest's. The "
               "quantum-kernel SVC (macro F1 0.832) and the variational QNN (0.467) are an "
               "architecture comparison recorded at training time on identical data; neither runs "
               "on the page. The SVC's kernel uses no entangling gate, so it is computable "
               "classically in closed form."),
    ),
    ModelSpec(
        usecase="telecom_ids",
        tier="t2_industry",
        key="telecom_ids.nidd",
        label="5G-NIDD network-attack classifier (9 classes)",
        architecture="gradient-boosted trees over 47 flow-statistic features",
        params=None,
        dataset=DS_5G_NIDD,
        metric=M_NIDD,
        # `_ONNX_FP32_ONLY`, like finance_aml.gbm. A tree ensemble exports as a single
        # ai.onnx.ml TreeEnsembleClassifier whose thresholds live in node attributes, so
        # `is_quantizable()` returns False and NO int8 graph is ever written. Declaring one would
        # point the browser -- which prefers int8 -- at a file that 404s at the pinned revision.
        artifacts=_ONNX_FP32_ONLY,
        hf_repo="qsh-telecom-5g-nidd-gbm",
        eval_split="a per-class-floored draw from the held-out quarter of the seed-42 stratified split",
        notes=(
            "NETWORK-LAYER flow classification on real 5G traffic. This is a DIFFERENT TASK from "
            "the QKD-layer attack classifier on this same page, with zero overlap in label space "
            "or feature space, so neither score is the other's baseline. "
            "REFITTED IN ROUND 50 ON A WIDER EXPORT OF THE SAME CORPUS, which is the fact that "
            "explains every number here. It trained on `Combined.csv` (52 columns) and scored "
            "accuracy 0.7608; it now trains on `BTS1_BTS2_fields_preserved.csv` (113 columns) and "
            "scores about 0.9829. The rows are identical and the method is unchanged. One column "
            "does all of it: `IdleTime`, a flow statistic the narrow export omits. Removing it "
            "alone from the wide fit returns the model to 0.7616, and that ablation is re-measured "
            "on every run as `idle_time_ablation`. It is not an identifier - 17 distinct integer "
            "values over 1.2 million rows - and it separates SlowrateDoS, an attack defined by "
            "idling, at one-vs-rest AUC 0.9574. "
            "READ THE TWO NUMBERS THE OTHER WAY ROUND FROM EVERYWHERE ELSE IN THIS PROJECT - STILL. "
            "Every other model here reports an aggregate that flatters and a per-class figure that "
            "does not. On this corpus macro F1 is the OPTIMISTIC number and aggregate accuracy the "
            "pessimistic one, and the refit shrank that inversion without removing it: the gap was "
            "0.1655 (0.9263 macro F1 against 0.7608 accuracy) and is about 0.0121 (0.9950 against "
            "0.9829). The mechanism is the same one as before, not a new one - Benign and UDPFlood "
            "are 39.3% and 37.6% of the rows and remain the only pair that confuses, so accuracy "
            "is dominated by them while the macro average is carried by seven classes at 0.9999 or "
            "better. The residual error is 1.71% of the test split against 23.92% before, and it is "
            "one direction: UDPFlood predicted as Benign. "
            "THE LEAK COUNTERFACTUAL IS STILL RECORDED EVERY RUN and its claim is now narrower. "
            "Restoring the corpus's integer row identifiers takes accuracy to about 0.9997, the "
            "same shape of result Engelen et al. (WTMC 2021) produced for CIC-IDS2017. That shows "
            "what a row counter is worth on this corpus. It does not show what the published "
            "figures used, and 0.9829 is reachable here with no identifiers at all; whether those "
            "figures lean on identifiers stays open -- see M_NIDD's baseline_source."
        ),
    ),
    #: Telecom's T1 tier, added in Round 33. Until then `runs_encrypted=True` held on five specs and
    #: none of them was telecom, so this was the only use case with no encrypted tier - which meant
    #: the standing per-industry question about the private-inference artifact class had no artifact
    #: to judge here, and `references/model-choice-vs-2026-practice.md` answered a model-quality
    #: question in its place.
    #:
    #: `hf_repo=None` and the weights are BUNDLED, following `genomics.prs` - the other CKKS tier
    #: with no Hub repo. 64 parameters do not need a pinned revision, and bundling means the
    #: encrypted tier works on a fresh clone with no network.
    #:
    #: `runs_encrypted=True` is true of the MODEL: `clientCompute.runEncryptedTelecomStudent`
    #: evaluates these 8 dot products under real CKKS in the tab. It does NOT claim the argmax is
    #: encrypted - all 8 scores are decrypted and the maximum is taken in the clear, because CKKS
    #: has no native comparison. The weights file and the run log both say so. With these public,
    #: bundled weights that exposes nothing; in a client-server deployment, raw scores of a linear
    #: model would let the key holder recover it with about as many queries as it has parameters per
    #: class (Tramer et al., USENIX Security 2016), so a private model would need signs only or an MPC
    #: argmax.
    ModelSpec(
        usecase="telecom_ids",
        tier="t1_crypto",
        key="telecom_ids.student",
        label="CKKS-linear IDS student (8 classes)",
        architecture="multinomial logistic regression, 8x7 + 8, distilled from the forest on soft targets",
        params=64,
        dataset=DS_QKD_ATTACK_SYNTH,
        metric=M_QML_IDS,
        runs_encrypted=True,
        artifacts=(),
        hf_repo=None,
        notes=(
            "Distilled from the random forest published as telecom_ids.qml, from its PROBABILITIES, "
            "not its argmax: on 8 classes the argmax "
            "discards exactly the minority-class signal macro F1 is computed from, and hard-label "
            "distillation is a defect this project has already made once. Read n_classes_predicted "
            "before macro_f1 - a linear student that stops emitting a class scores 0 on it by "
            "construction, which reads as poor discrimination rather than an absent output."),
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
    "telecom_ids.nidd": "models/telecom_nidd",
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
            "eval_sample_reason": s.eval_sample_reason or None,
            "eval_split": s.eval_split or None,
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
                # Exported because a CC BY credit that never leaves Python discharges nothing. Four
                # of these corpora are CC BY and the page showed a licence string with no author.
                "attribution": s.dataset.attribution,
            },
            "metric": {
                "name": s.metric.name,
                "display": s.metric.display,
                "baseline": s.metric.baseline,
                "baseline_source": s.metric.baseline_source,
                "ceiling": s.metric.ceiling,
                "ceiling_source": s.metric.ceiling_source,
                "companion": s.metric.companion,
                # Hand-listed, so a new MetricSpec field reaches the browser only when someone adds
                # it here. `comparison_caveat` was added in Round 55 and the first export silently
                # omitted it - the gate caught it, but note the shape: adding a dataclass field is
                # not the same act as exporting it, and nothing about the former implies the latter.
                "comparison_caveat": s.metric.comparison_caveat,
            },
            "achieved": achieved(s.key),
            "notes": s.notes,
        })
    return out
