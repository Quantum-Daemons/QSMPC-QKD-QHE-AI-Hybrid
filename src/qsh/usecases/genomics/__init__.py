"""Genomics confidential analysis use case - a REAL polygenic risk score, in full, under CKKS.

Several biobanks pool a SNP's allele counts without revealing per-cohort counts (QSMS secure
summation, over a field that holds every total the counts can reach), then a clinician evaluates a
**polygenic risk score (PRS)** over an *encrypted* genotype vector. The model is a **real published
PRS** (PGS Catalog PGS000072, the 187-variant breast-cancer score of Graff et al., Nat Commun 12:970,
2021, CC BY 4.0): ``PRS = sum_i effect_weight_i * dosage_i``, over the 183 variants that have a 1000
Genomes phase 3 genotype (the other four are declared ``#unscored`` by the dosage files and left out,
never filled in). That is a *linear* model, so CKKS evaluates the whole score as one approximate dot
product, with no surrogate: a person's dosages at every scored variant in one ciphertext, multiplied by
the plaintext weights, summed by rotation, and the decrypted score compared with a high-risk cut.
Per-SNP SHAP shows which loci drive the score.

THE CUT IS STATED, NOT BORROWED. It is the ``HIGH_RISK_PERCENTILE``-th percentile of the plaintext
score over the individuals this run's loader read (numpy's default linear interpolation between order
statistics, the scored person included), so it comes from the bundled panel alone. A demo cut, not a
clinical threshold. The browser computes the same percentile the same way over the 30 individuals it
bundles.

The dosages are real 1000 Genomes phase-3 effect-allele counts; Hardy-Weinberg sampling is only the
loudly-logged offline fallback.

Maps to: QSMPC x QKD (secure summation) + QHE/FHE (private inference).

Run:  ``python -m qsh.orchestration.pipeline --usecase genomics``
"""

from __future__ import annotations

import numpy as np

from ...orchestration.pipeline import Pipeline, standard_stages
from ...orchestration.stages import Context, PrivateInferenceStage

#: The percentile of the full score this use case calls high risk. The browser's twin is
#: `PRS_HIGH_RISK_PERCENTILE` in web/src/lib/realData.ts, and both equal
#: `scripts/train/genomics_prs.py::HIGH_RISK_PERCENTILE`, the cut the Models card's
#: `global_high_risk_cut` is measured at (tests/test_r84_genomics_full_score.py pins all three). It is
#: a demo cut over the bundled panel, not a clinical threshold.
HIGH_RISK_PERCENTILE = 90.0


def high_risk_cut(scores) -> float:
    """The cut: the `HIGH_RISK_PERCENTILE`-th percentile of ``scores``, numpy's linear method."""
    return float(np.percentile(np.asarray(scores, dtype=float), HIGH_RISK_PERCENTILE))


def encrypted_scores(backend, rows, weights) -> dict:
    """Every row's FULL score under CKKS, decrypted: the panel figure's evidence (Round 84).

    A row is one person's effect-allele dosages at every variant, and it travels whole in one
    ciphertext: in a block of ``width = 2**ceil(log2(M))`` slots, as many people per ciphertext as
    ``slots // width`` allows. One multiply-plain by the weights, tiled at the same stride, and
    ``log2(width)`` rotate-and-add steps leave each person's score in the first slot of its block, which
    is what the decryption is read at. The other slots of a block hold partial sums of the same
    weighted products, so the key holder can recover each product ``w_i * d_i``; the weights are the
    published PGS weights, so that exposes nothing the page does not print.

    The packed path needs SEAL's own evaluator, which the TenSEAL backend exposes; any other backend
    scores one person per ciphertext through its ``dot_plain``, which is the same computation without
    the packing, and says so in ``packed``.
    """
    x = np.asarray(rows, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    n, m = x.shape
    width = 1 << int(np.ceil(np.log2(max(m, 2))))
    slots = int(getattr(backend, "slot_count", 0) or 0)
    sealapi = getattr(backend, "_sealapi", None)
    context = getattr(backend, "context", None)
    out: list[float] = []
    rotations = 0
    if callable(sealapi) and context is not None and hasattr(context, "galois_keys") and slots >= width:
        sa, encoder, evaluator, encryptor, decryptor = sealapi()
        galois = context.galois_keys().data
        scale = context.global_scale
        per = slots // width
        ciphertexts = 0
        for start in range(0, n, per):
            part = x[start:start + per]
            xs, ws = np.zeros(slots), np.zeros(slots)
            for r, row in enumerate(part):
                xs[r * width:r * width + m] = row
                ws[r * width:r * width + m] = w
            pt, ct, ptw, acc = sa.Plaintext(), sa.Ciphertext(), sa.Plaintext(), sa.Ciphertext()
            encoder.encode(xs.tolist(), scale, pt)
            encryptor.encrypt(pt, ct)                          # the dosages: the only thing encrypted
            encoder.encode(ws.tolist(), scale, ptw)            # the weights stay in plaintext
            evaluator.multiply_plain(ct, ptw, acc)
            step = 1
            while step < width:
                rot = sa.Ciphertext()
                evaluator.rotate_vector(acc, step, galois, rot)
                evaluator.add_inplace(acc, rot)
                rotations += 1
                step <<= 1
            res = sa.Plaintext()
            decryptor.decrypt(acc, res)
            dec = encoder.decode_double(res)
            out.extend(float(dec[r * width]) for r in range(len(part)))
            ciphertexts += 1
        packed, per_ct = True, per
    else:
        for row in x:
            out.append(float(backend.decrypt(backend.dot_plain(backend.encrypt(row), w))[0]))
        packed, per_ct, ciphertexts = False, 1, n
    plain = x @ w
    return {"encrypted": np.asarray(out), "plaintext": plain, "rows": n, "variants": m,
            "width": width, "slots": slots, "rows_per_ciphertext": per_ct,
            "ciphertexts": ciphertexts, "rotations": rotations, "packed": packed,
            "max_abs_error": float(np.max(np.abs(np.asarray(out) - plain))) if n else 0.0}


class FullScoreInferenceStage(PrivateInferenceStage):
    """`private_inference` for genomics: the parent's CKKS run, then the full score over the panel.

    Same stage name, so the pipeline's shape, its DAG and the browser's flow are unchanged. The parent
    encrypts the scored person's dosages and returns the class; this adds the verification the
    single row cannot give - every panel individual's full score under CKKS against the same score in
    the clear, and whether the high-risk flag agrees - and the log that says what is encrypted and
    what is decrypted.
    """

    def run(self, ctx, bus, reg):
        super().run(ctx, bus, reg)
        panel = ctx.data.get("_prs_panel")
        if ctx.config.get("inference", "ckks") != "ckks" or panel is None:
            return
        cut, n = float(panel["cut"]), len(panel["dosages"])
        r = encrypted_scores(reg.get("fhe", ctx.config.get("fhe"))(), panel["dosages"], panel["weights"])
        enc_flag, plain_flag = r["encrypted"] > cut, r["plaintext"] > cut
        agree = int(np.sum(enc_flag == plain_flag))
        subject = int(panel["subject_index"])
        subject_plain = int(plain_flag[subject])
        ctx.artifacts["prs_panel"] = {
            "rows": n, "ciphertexts": r["ciphertexts"], "rows_per_ciphertext": r["rows_per_ciphertext"],
            "packed": r["packed"], "max_abs_error": r["max_abs_error"], "flags_agree": agree,
            "flagged_encrypted": int(enc_flag.sum()), "flagged_plaintext": int(plain_flag.sum()),
            "cut": cut, "percentile": HIGH_RISK_PERCENTILE}
        for key, value in (("prs_high_risk_percentile", HIGH_RISK_PERCENTILE),
                           ("prs_high_risk_cut", round(cut, 6)), ("prs_cut_population", n),
                           ("prs_subject_flag_plaintext", subject_plain),
                           ("prs_subject_flags_agree", int(ctx.artifacts.get("private_prediction") == subject_plain)),
                           ("prs_panel_rows", n), ("prs_panel_ciphertexts", r["ciphertexts"]),
                           ("prs_panel_max_abs_error", r["max_abs_error"]),
                           ("prs_panel_flags_agree", agree),
                           ("prs_panel_flag_agreement", round(agree / max(1, n), 4)),
                           ("prs_panel_flagged_encrypted", int(enc_flag.sum())),
                           ("prs_panel_flagged_plaintext", int(plain_flag.sum()))):
            bus.metric(key, value, self.name)
        layout = (f"one {r['width']}-slot block of {r['slots']}, {r['rows_per_ciphertext']} people per "
                  f"ciphertext" if r["packed"] else "one person per ciphertext, through the backend's dot product")
        bus.log(
            f"full polygenic score under CKKS. Encrypted: each person's {r['variants']} effect-allele "
            f"dosages, together in one ciphertext ({layout}). In plaintext: the {r['variants']} effect "
            f"weights, applied by one multiply-plain; rotate-and-add then sums each block into its first "
            f"slot. Decrypted: the result ciphertext, read at that slot, which holds the score; the cut "
            f"{cut:.4f} is the {HIGH_RISK_PERCENTILE:.0f}th percentile of the plaintext score over the {n} "
            f"panel individuals this run loaded (linear interpolation between order statistics). Scored "
            f"person: class {ctx.artifacts.get('private_prediction')} encrypted, {subject_plain} in "
            f"plaintext. Panel: {n} people in {r['ciphertexts']} ciphertext(s), max |decrypted - "
            f"plaintext score| {r['max_abs_error']:.3e}; the high-risk flag agrees on {agree} of {n} "
            f"({int(enc_flag.sum())} above the cut encrypted, {int(plain_flag.sum())} in plaintext).",
            self.name)


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

    # The full score of every individual the loader read, and the cut over them (see the docstring):
    # the HIGH_RISK_PERCENTILE-th percentile, from this bundled panel only.
    scores = dosages @ weights
    threshold = high_risk_cut(scores)
    # 2-class LinearModel for argmax(W x + b): class 1 (high genetic risk) iff score > cut. Class 0's
    # row is zero, so the encrypted computation is the score itself, and the cut is subtracted from
    # the decrypted value in the clear.
    w2 = np.vstack([np.zeros_like(weights), weights])
    b2 = np.array([0.0, -threshold])
    subject_index = int(np.argmax(scores))
    subject = dosages[subject_index].astype(float)       # a high-PRS individual to score privately

    # per-biobank allele counts for SNP 0 (real dosage sums) -> secure-summed pooled count
    n_biobanks = config.get("biobanks", d.genomics_biobanks)
    splits = np.array_split(dosages[:, 0], n_biobanks)
    per_cohort = [int(s.sum()) for s in splits]

    # THE FIELD, FROM WHAT THE COUNTS CAN REACH (Round 84). QSMS's own default is the paper's
    # d in [n, 2n], GF(5) for four cohorts, and the pooled count wraps in it: at the score's first
    # variant (rs616488) the full panel's first 400 people split into cohorts carrying 141, 131, 138
    # and 118 effect alleles, 528 in all, which GF(5) returns as 3. Each person carries at most two
    # copies of the effect allele, so the pool can reach twice the cohort size. That bound is public;
    # the sum is not, and choosing the field from it would leak.
    from ...qsmpc.secure_summation import field_for_total
    from ...qsmpc.shamir import is_probable_prime

    max_total = 2 * int(dosages.shape[0])
    n_parties = max(len(per_cohort), 3)                 # the parties the QSMPC stage runs
    pinned = config.get("prime", cfg.qsmpc.prime)
    prime = int(pinned) if pinned else field_for_total(max_total, n_parties)
    if not is_probable_prime(prime) or prime <= max_total:
        raise ValueError(f"QSMS field {prime} cannot hold the pooled allele count, which can reach "
                         f"{max_total} for {dosages.shape[0]} people: the sum would wrap mod the field")

    ctx = Context(
        usecase="genomics",
        config={"qsmpc_op": "sum", "inference": "ckks", "seed": seed, **config, "prime": prime},
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
                        "high_risk_percentile": HIGH_RISK_PERCENTILE,
                        "cut_population": int(dosages.shape[0]),
                        "subject_score": round(float(subject @ weights), 4),
                        "subject_flag_plaintext": int(float(subject @ weights) > threshold)}
    ctx.data["_prs_panel"] = {"dosages": dosages, "weights": weights, "cut": threshold,
                              "subject_index": subject_index}
    ctx.data["_ground_truth_total"] = sum(per_cohort)
    ctx.data["_qsms_field"] = {"prime": prime, "max_possible_total": max_total, "parties": n_parties,
                               "field_within_paper_bound": bool(n_parties <= prime <= 2 * n_parties)}
    stages = [FullScoreInferenceStage() if isinstance(s, PrivateInferenceStage) else s
              for s in standard_stages()]
    return Pipeline(name="genomics", stages=stages), ctx


def run_demo(config: dict | None = None) -> dict:
    pipe, ctx = build(config)
    pipe.run(ctx, out_dir="results/")
    m = pipe.bus.metrics()
    return {
        "pooled_allele_count": ctx.artifacts.get("secure_sum"),
        # UNREDUCED (Round 84). This was reduced mod the field the sum used, so a wrapped sum
        # matched a wrapped truth: the committed slice's 153 alleles read 3 == 3 in GF(5). The field
        # now exceeds every total the counts can reach, so the two agree only if the protocol is right.
        "ground_truth_total": ctx.data["_ground_truth_total"],
        # Which field ran and why it is not the paper's: a count above 2n needs d > 2n, and d**t
        # then passes the statevector bound, so the run used the analytic sampler.
        "qsms_field": {**ctx.data["_qsms_field"],
                       "honest_statevector": ctx.artifacts.get("qsms_honest_statevector")},
        "prs": ctx.data.get("_prs"),
        # The full score over the panel under CKKS against the same score in the clear (Round 84).
        "prs_panel": ctx.artifacts.get("prs_panel"),
        # Canonical key, not one backend's - see stages.py. Reading `fhe_prediction`
        # alone meant a QCNN run computed a class and this surfaced None.
        "private_risk_class": ctx.artifacts.get("private_prediction",
                 ctx.artifacts.get("fhe_prediction")),
        "private_risk_class_backend": ctx.artifacts.get("private_prediction_backend"),
        "explanation": ctx.artifacts.get("explanation"),
        "metrics": m,
    }
