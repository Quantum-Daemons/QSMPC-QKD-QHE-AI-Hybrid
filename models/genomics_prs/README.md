# Polygenic risk score - PGS000004 (PRS313), breast cancer

> **This card documents PROVENANCE AND LICENCE, because that is what the artifact records.** The
> `weights.json` in this directory is a published effect-weight vector lifted out of a PGS Catalog
> scoring file. It was not fitted here, it has no training split, and this repository computes no
> discrimination metric for it at all - no AUROC, no C-index, no F1. What the run DOES measure is the
> distribution of the resulting score across 1000 Genomes super-populations, and that distribution is
> the finding rather than a footnote. The one performance figure anyone can quote for this score
> comes from a phenotyped cohort this project cannot reach; it is carried below as a citation, under
> its own heading, and is not a result of this run.

NOT PUBLISHED TO HUGGING FACE. `genomics.prs` carries no `hf_repo` and no pinned revision, and the
reason is stronger than "nobody got round to it": `DatasetSpec.publishable` is **False** in
`src/qsh/models/registry.py`, and `scripts/publish_hf.py` enforces that. Read the licence section
before reading the `publishable` field inside this directory's own `licence_audit` block - they do
not mean the same thing, and the one in the artifact grants nothing.

## What ships in this directory

`weights.json` is written by `scripts/train/genomics_prs.py` and holds the score and its audit:

| field | what it holds |
|---|---|
| `score_id`, `trait` | `PGS000004`, `breast cancer`. Alias PRS313_BC, GRCh37, harmonised hmPOS file. |
| `n_variants` | the variant count of the published score. |
| `rsids` | **not rsIDs.** 313 keys of the form `chr:pos:effect_allele`, e.g. `1:100880328:T`. |
| `weights` | 313 per-variant effect weights, floats, in the same order as `rsids`. |
| `high_risk_percentile`, `global_cut` | the 90th percentile of the pooled score, and the score value that sits at it. One cut, set once over everybody, which is the whole point of the measurement below. |
| `licence_audit` | what the bundled scoring file's `#` header said when it was last read. |
| `note` | effect weights only; no per-sample genotype is included and none may be published. |

**The field named `rsids` does not contain rsIDs, and that is recorded rather than tidied.**
PGS000004 publishes none - `hm_rsID` is empty for all 313 rows and `hm_source` is
`Author-reported` - so downstream tooling keys on chromosome, position and effect allele instead.
The bundled scoring file header says so, `DATA_LICENSES.md` says so, and
`scripts/data/fetch_g1k_dosages_bypos.py` exists precisely because the rsID-keyed fetcher cannot
build a panel for this score. A reader who assumes the field name is accurate will join on the wrong
key and get a silently wrong score.

## Measured on 1000 Genomes: a distribution audit, not a discrimination metric

Every figure here is recomputed on each training run and lands in
`models/metrics.json` under `genomics.prs`, which is also what `web/src/components/ModelPanel.tsx`
renders on the genomics page. Nothing below is a phenotype comparison, because there is no phenotype.

| quantity | value |
|---|---|
| `n_variants` | 313 |
| `n_samples` | 2504 |
| `score_mean` | -0.047284 |
| `score_sd` | 0.634203 |
| `global_high_risk_cut` | 0.75367 |
| `wall_clock_s` | 1.5 |

One high-risk cut, set once at the 90th percentile of the pooled score, applied to everybody:

| super-population | n | mean | sd | p90 | share above the global cut |
|---|---|---|---|---|---|
| AFR | 661 | 0.207404 | 0.554294 | 0.8989 | 0.161876 |
| EAS | 504 | 0.163061 | 0.567598 | 0.8645 | 0.14881 |
| AMR | 347 | -0.072606 | 0.65686 | 0.71352 | 0.083573 |
| SAS | 489 | -0.259791 | 0.599757 | 0.5101 | 0.05317 |
| EUR | 503 | -0.368675 | 0.592581 | 0.41544 | 0.027833 |

**Read the last column.** A single global cut flags 0.161876 of AFR samples and 0.027833 of EUR
samples - a factor of about six - and no phenotype was involved in producing that gap. It is an
allele-frequency artefact of applying a score derived largely in European cohorts to a panel that is
not, and it is the reason this use case exists in the demo. It is a **distribution shift**, and
calling it an AUROC drop would be wrong twice over: no AUROC is computed here, and the gap is in
where the score sits, not in how well it ranks.

The same pattern is visible in the score's own published record set, by a different route: across
PGS000004's Catalog performance records the lowest AUROC, 0.55, is on an African-ancestry cohort,
against a European range of 0.5877 to 0.668. That is a citation, not a measurement from this run.

## No AUROC is computed here, and none is invented

- **The metric the registry declares**: AUROC (European ancestry).
- **Published figure carried as a citation**: 0.633 - PGS Catalog PPM001647 for PGS000004 (PRS313),
  95% CI 0.624-0.641, n=400,812 Europeans (4,340 cases, 396,472 controls), from Jia et al., JNCI
  Cancer Spectrum 2020, doi:10.1093/jncics/pkaa021. Verified at the Catalog REST record 2026-08-11.
- **Published ceiling**: 0.668 - the highest of PGS000004's own 28 Catalog AUROC records (PPM012913,
  n=302,425, European), measured across all of them rather than taken from a quoted range.
- **Why neither is a target.** The cited model is fitted WITH COVARIATES - family history of cancer
  in first-degree relatives, and genotyping array - which a bare weighted dot product does not have;
  and that cohort was missing 25 of the variants, so the published figure is for 288 of the 313, not
  all of them.
- **Why it cannot be reproduced here.** 1000 Genomes is a reference panel and carries no
  breast-cancer case/control status. AUROC needs a phenotyped cohort. `genomics_prs.py` therefore
  records `auroc` as null rather than inventing one, and `not_measured.reason` in the metrics record
  states that in the record itself.

## Licence and provenance, which is the point of this card

**PGS Catalog is not blanket CC-BY-4.0.** The licence is **per score**. PGS000004 carries the
Catalog's default terms, which are a pass-through to the **EMBL-EBI Terms of Use**, and individual
scores in the Catalog can be more restrictive - the Catalog's own documentation shows a live
CC BY-NC-ND 4.0 example. The terms require citing the **original publication**, not only the
Catalog:

> Mavaddat N et al., *Am J Hum Genet* (2018). doi:10.1016/j.ajhg.2018.11.002

That citation is a condition, so it is written here rather than left to a link.

**The `licence_audit` block is an observation, not a permission**, and its `publishable` field is the
part most likely to be misread. What the four flags actually mean:

| flag | what it means |
|---|---|
| `declares_restriction` | a restrictive token (NC-ND, noncommercial) was found in the header. False here. |
| `declares_terms` | some licence-ish wording was found in the header. True here. |
| `publishable` | **only** "no restrictive token was found". It is not a grant, and it is True on any file that says nothing at all. |
| `no_restriction_found_but_terms_absent` | the silence case, separated out so absence of evidence cannot pass as evidence. False here. |

And `declares_terms` is True here for a reason worth stating: **the licence sentence in the bundled
scoring file's first `#` line was written by this project.** Exactly one line of
`data/genomics/pgs000004_breast_cancer.tsv` mentions a licence and that is the one - every other `#`
line is Catalog metadata (`pgs_id`, `trait_mapped`, `genome_build`, `citation`, harmonisation
details). A PGS000004 download from the Catalog carries no licence header at all; the licence exists
only in the REST record. So the audit is reading this repository's own annotation back to itself. It
is still worth running, because it would catch a future score that ships a real restrictive header;
it is not worth treating as a green light. Publication is gated by `publishable=False` on the
dataset spec and by `scripts/publish_hf.py`, not by anything in this file.

**1000 Genomes / IGSR is not public domain either.** The IGSR disclaimer states that "restrictions on
different pieces of data within IGSR and rights claimed on pieces of data vary" and that "it remains
the responsibility of users to ensure that their exploitation of the data does not infringe any of
the rights of third parties."

**No genotypes are in this artifact.** `note` in `weights.json` says so and it is accurate: the file
is effect weights and an audit block. The per-sample dosages this repository does redistribute are
tabulated, file by file and to the genotype cell, in `DATA_LICENSES.md` section 4; the 2504-individual
matrix behind the table above is deliberately **not** committed, and is regenerated locally by
`scripts/data/fetch_g1k_dosages_bypos.py`. What ships to a browser is the aggregate statistics plus a
30-individual payload, which is a different and much smaller exposure than what is tracked in git -
section 4 keeps those two apart on purpose.

## Reproducibility

    python scripts/train/genomics_prs.py --score breast_cancer_prs313

This rewrites `weights.json` here and the `genomics.prs` record in `models/metrics.json`. The
scoring file and the dosage file are selected together by score NAME rather than by path, so the two
cannot be mismatched into a silent wrong-variant join.

## Scope

This is a published risk score being re-scored on a reference panel, not a model this project
trained, and not a clinical instrument. No individual result on the genomics page is a medical
statement. The demonstration is the ancestry stratification of one globally-set threshold, plus the
fact that a 313-term dot product is small enough to evaluate under CKKS - and that smallness is a
property of Mavaddat et al. 2018 having 313 variants, not a limit of homomorphic evaluation: Knight
et al., *Cell Reports Methods* 6(1):101271 (2026), run polygenic scoring under HE over 110,000 SNPs
and call it practical on a single CPU.
