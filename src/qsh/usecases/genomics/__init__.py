"""Genomics confidential analysis use case — a REAL Polygenic Risk Score under CKKS.

Several biobanks pool a SNP's allele counts without revealing per-cohort counts (QSMS secure
summation), then a clinician privately evaluates a **Polygenic Risk Score (PRS)** over an *encrypted*
genotype vector. The model is a **real published PRS** (PGS Catalog PGS000004 = a 313-variant breast-cancer
score, Mavaddat et al. 2018): ``PRS = Σ effect_weight_i · dosage_i`` — the clinical-genomics standard,
which is a *linear* model and therefore runs as an **exact CKKS dot-product** (FHE-native, no
surrogate). Per-SNP SHAP shows which loci drive the risk.

Three claims in this docstring were stale as of Round 32 and are worth naming, because each was wrong
in a different way. It said **PGS000001, a 77-SNP score** — the score moved to PGS000004 (313 variants)
in Round 31 everywhere except the line below, which is the defect this round fixed. It said the score
is **CC-BY** — `models/registry.py` corrects that: PGS Catalog licences are per score and PGS000001
carries the Catalog default terms, not CC-BY. And it said dosages are **sampled under Hardy-Weinberg**
— they are real 1000 Genomes phase-3 effect-allele counts, and Hardy-Weinberg sampling is now only the
loudly-logged offline fallback. A docstring that describes the data as synthetic when it is real
understates the work; the other two overstate the licence and name the wrong model.

Maps to: QSMPC×QKD (secure summation) + QHE/FHE (private inference).

Run:  ``python -m qsh.orchestration.pipeline --usecase genomics``
"""

from __future__ import annotations

import numpy as np

from ...orchestration.pipeline import Pipeline, standard_stages
from ...orchestration.stages import Context


def build(config: dict | None = None) -> tuple[Pipeline, Context]:
    config = dict(config or {})
    seed = config.get("seed", 0)
    # No local RNG: this use case draws no random numbers. One sat here unused since the dosages
    # stopped being Hardy-Weinberg samples, which is the same staleness the docstring carried.

    from ...config import get_config
    from ...data.loaders import load_genomic_prs
    from ...fhe.ckks_inference import LinearModel

    cfg = get_config()
    d = cfg.data
    # Default comes from the config FIELD, not a literal here. See DataConfig.genomics_prs_score for
    # why: the literal that used to sit in this call is how the pipeline came to score PGS000001
    # while every other surface scored PGS000004.
    prs_name = config.get("prs", d.genomics_prs_score)
    dosages, weights, rsids, eff_alleles, _freq = load_genomic_prs(
        n_individuals=max(d.genomics_n // 3, 200), seed=seed, name=prs_name)

    # real PRS scores over the cohort → a high-risk threshold (top quartile = the clinical risk cut)
    scores = dosages @ weights
    threshold = float(np.quantile(scores, 0.75))
    # 2-class LinearModel for argmax(W·x + b): class 1 (high genetic risk) iff PRS > threshold.
    # The encrypted inference computes the REAL PRS dot-product under CKKS, then the risk class.
    w2 = np.vstack([np.zeros_like(weights), weights])
    b2 = np.array([0.0, -threshold])
    subject = dosages[int(np.argmax(scores))].astype(float)        # a high-PRS individual to score privately

    # per-biobank allele counts for SNP 0 (real dosage sums) -> secure-summed pooled count
    n_biobanks = config.get("biobanks", d.genomics_biobanks)
    splits = np.array_split(dosages[:, 0], n_biobanks)
    per_cohort = [int(s.sum()) for s in splits]

    ctx = Context(
        usecase="genomics",
        config={"qsmpc_op": "sum", "inference": "ckks", "seed": seed,
                # None => QSMS derives a prime in [n, 2n] as the construction requires.
                "prime": config.get("prime", cfg.qsmpc.prime), **config},
    )
    ctx.data["secrets"] = per_cohort
    ctx.data["linear_model"] = LinearModel(w2.astype(float), b2)
    ctx.data["features"] = subject
    # Population mean dosage as the SHAP reference. Attributions then read as "this subject against
    # the cohort", which is the clinically meaningful comparison; against a zero vector they read as
    # "against a person homozygous reference at every locus", which is not a person.
    ctx.data["explain_baseline"] = dosages.mean(axis=0).astype(float)
    ctx.data["feature_names"] = list(rsids)                         # per-SNP SHAP (which loci drive risk)
    ctx.data["_prs"] = {"name": prs_name, "n_snps": len(weights), "threshold": round(threshold, 4),
                        "subject_score": round(float(subject @ weights), 4)}
    ctx.data["_ground_truth_total"] = sum(per_cohort)
    return Pipeline(name="genomics", stages=standard_stages()), ctx


def run_demo(config: dict | None = None) -> dict:
    pipe, ctx = build(config)
    pipe.run(ctx, out_dir="results/")
    m = pipe.bus.metrics()
    return {
        "pooled_allele_count": ctx.artifacts.get("secure_sum"),
        # Reduce into the field the sum actually used, which QSMS derives from the party count.
        # Reading ctx.config["prime"] assumed a pinned modulus and returned None once the config
        # stopped pinning one -- and a pinned 4099 was outside the paper's admissible field anyway.
        "ground_truth_total": ctx.data["_ground_truth_total"] % ctx.artifacts["qsms_prime"],
        "prs": ctx.data.get("_prs"),
        # Canonical key, not one backend's - see stages.py. Reading `fhe_prediction`
        # alone meant a QCNN run computed a class and this surfaced None.
        "private_risk_class": ctx.artifacts.get("private_prediction",
                 ctx.artifacts.get("fhe_prediction")),
        "private_risk_class_backend": ctx.artifacts.get("private_prediction_backend"),
        "explanation": ctx.artifacts.get("explanation"),
        "metrics": m,
    }
