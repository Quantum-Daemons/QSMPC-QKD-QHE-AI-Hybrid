# Polygenic risk score - PGS000072, breast cancer

> **This card documents PROVENANCE AND LICENCE, because that is what the artifact records.** The
> `weights.json` in this directory is a published effect-weight vector lifted out of a PGS Catalog
> scoring file. It was not fitted here, it has no training split, and this repository computes no
> discrimination metric for it at all - no AUROC, no C-index, no F1. What the run DOES measure is the
> distribution of the resulting score across 1000 Genomes super-populations, and that distribution is
> the finding rather than a footnote. The one AUROC anyone can quote for this score comes from a
> phenotyped cohort this project cannot reach; it is carried below as a citation, under its own
> heading, and is not a result of this run.

NOT PUBLISHED TO HUGGING FACE. `genomics.prs` carries no `hf_repo` and no pinned revision: the
weights are a published score's own, and they ship with the page instead. Its dataset record in
`src/qsh/models/registry.py` is `publishable=True`, because the score's publication is CC BY 4.0 and
redistribution with the credit is permitted; `scripts/publish_hf.py` publishes nothing without an
`hf_repo`. Read the licence section before reading the `publishable` field inside this directory's own
`licence_audit` block - they do not mean the same thing, and the one in the artifact grants nothing.

## What ships in this directory

`weights.json` is written by `scripts/train/genomics_prs.py` and holds the score and its audit:

| field | what it holds |
|---|---|
| `score_id`, `trait` | `PGS000072`, `breast cancer`. Catalog name `CC_Breast`, GRCh37, the author-reported scoring file. |
| `n_variants` | the variants scored: those with a 1000 Genomes phase 3 genotype for every one of the 2,504 individuals. |
| `n_variants_published` | the variant count of the published score. |
| `unscored_variants` | each published variant that is not scored, with the reason the dosage file declares for it. |
| `rsids` | the scored variants' rsIDs, as the scoring file gives them. |
| `weights` | their effect weights, the score's log odds ratios, unchanged and in the same order as `rsids`. |
| `high_risk_percentile`, `global_cut` | the 90th percentile of the pooled score, and the score value that sits at it. One cut, set once over everybody, which is the whole point of the measurement below. |
| `licence_audit` | what the bundled scoring file's `#` header said when it was last read. |
| `note` | effect weights only: this file carries no per-sample genotype. Per-sample dosages are redistributed in the slices `DATA_LICENSES.md` section 4 lists (30 individuals in the browser payload), with the 1000 Genomes consortium citation. |

**Four of the 187 published variants are not scored, and they are named rather than filled in.**
`scripts/data/fetch_g1k_dosages.py` reads each variant's 1000 Genomes genotypes through the Ensembl
REST API and scores it only if every individual has one: rs8176636 does not map to GRCh38 there,
rs34207738 and rs7297051 have no 1000 Genomes genotypes there, and rs12405132 has them for 1,582 of
the 2,504. Each is declared `#unscored` in the dosage files, with that reason, and listed in
`unscored_variants`; the loader refuses any absent variant that is not declared. Filling them in
would have put a constant column into the matrix, as the retired PGS000001 dosage file shows.

## Measured on 1000 Genomes: a distribution audit, not a discrimination metric

Every figure here is recomputed on each training run and lands in
`models/metrics.json` under `genomics.prs`, which is also what `web/src/components/ModelPanel.tsx`
renders on the genomics page. Nothing below is a phenotype comparison, because there is no phenotype.

| quantity | value |
|---|---|
| `n_variants` | 183 |
| `n_samples` | 2504 |
| `score_mean` | 20.600716 |
| `score_sd` | 1.54361 |
| `global_high_risk_cut` | 22.641088 |

The scores sit around 20 because every weight is a positive log odds ratio of a risk allele. One
high-risk cut, set once at the 90th percentile of the pooled score, applied to everybody:

| super-population | n | mean | sd | p90 | share above the global cut |
|---|---|---|---|---|---|
| EAS | 504 | 22.252985 | 1.440805 | 24.103023 | 0.390873 |
| EUR | 503 | 19.816252 | 1.468938 | 21.612154 | 0.039761 |
| SAS | 489 | 20.341618 | 1.211468 | 21.870458 | 0.02863 |
| AFR | 661 | 20.374579 | 1.042049 | 21.693418 | 0.02118 |
| AMR | 347 | 20.133905 | 1.283454 | 21.711376 | 0.017291 |

**Read the last column.** A single global cut flags 0.390873 of EAS samples and 0.039761 of EUR
samples - a factor of about ten - and no phenotype was involved in producing that gap. It is an
allele-frequency artefact of applying a score whose weights come from GWAS samples that are 98.1%
European (the Catalog's ancestry distribution for PGS000072) to a panel that is not, and it is the
reason this use case exists in the demo. Most of the East Asian shift comes from one variant:
rs10979327 carries the score's largest weight, 1.3481 (an odds ratio of 3.85), and its effect allele
G has a frequency of 0.495 in the panel's East Asian individuals against 0.117 in its European ones.
It is a **distribution shift**, and calling it an AUROC drop would be wrong twice over: no AUROC is
computed here, and the gap is in where the score sits, not in how well it ranks.

The score's own published record cannot show this by another route: all three of PGS000072's Catalog
performance records are in European-ancestry samples (read 2026-10-31).

## No AUROC is computed here, and none is invented

- **The metric the registry declares**: AUROC (European ancestry).
- **Published figure carried as a citation**: 0.637 - PGS Catalog PPM002038 for PGS000072, incident
  breast cancer in n=216,273 European-ancestry women in UK Biobank (4,760 cases, 211,513 controls),
  from Kachuri et al., Nat Commun 11:6084 (2020), doi:10.1038/s41467-020-19600-4. No confidence
  interval is recorded. Read at the Catalog REST API on 2026-10-31.
- **Published ceiling**: 0.637 as well - the highest, and the only, AUROC among PGS000072's three
  Catalog performance records. The other two report effect sizes: an odds ratio per standard
  deviation of 1.26 (95% CI 1.24-1.28, PPM000192, GERA and UK Biobank, n=237,549, the score's own
  publication) and a hazard ratio of 1.26 (1.20-1.32, PPM017165, post-menopausal breast cancer in UK
  Biobank, n=1,990).
- **Why neither is a target.** The cited model is fitted WITH COVARIATES - age, family history of
  breast cancer, genotyping array, genetic principal components 1-15, reproductive and lifestyle
  factors - which a bare weighted dot product does not have; the record says it used "only 162 of the
  original 187 SNPs"; and it is an AUC at 5 years.
- **Why it cannot be reproduced here.** 1000 Genomes is a reference panel and carries no
  breast-cancer case/control status. AUROC needs a phenotyped cohort. `genomics_prs.py` therefore
  records `auroc` as null rather than inventing one, and `not_measured.reason` in the metrics record
  states that in the record itself.

## Licence and provenance, which is the point of this card

**A PGS Catalog score's terms are its publication's.** PGS000072's is Graff et al., Nat Commun
12:970 (2021), published under **CC BY 4.0** (Crossref `vor` and `tdm`, Europe PMC "cc by", read
2026-10-31). The article weights each of its 187 breast-cancer variants by the log odds ratio its
source GWAS reported, and those are the weights here. The Catalog's REST record adds a pass-through to
the **EMBL-EBI Terms of Use** and asks users to observe any licensing restrictions the authors set.
CC BY 4.0 asks for credit, a link to the licence and a note of changes, so all three travel with the
browser payload:

> Graff RE, Cavazos TB, Thai KK et al., *Cross-cancer evaluation of polygenic risk scores for 16
> cancer types in two large cohorts*, Nat Commun 12:970 (2021), doi:10.1038/s41467-021-21288-z,
> CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

The Catalog asks to be cited as Lambert, Wingfield et al., *Enhancing the Polygenic Score Catalog
with tools for score calculation and ancestry normalization*, Nat Genet 56:1989-1994 (2024),
doi:10.1038/s41588-024-01937-x. The change: 183 of the 187 variants are scored, with their weights
unchanged.

**The `licence_audit` block is an observation, not a permission**, and its `publishable` field is the
part most likely to be misread. What the four flags actually mean:

| flag | what it means |
|---|---|
| `declares_restriction` | a restrictive token (NC-ND, noncommercial) was found in the header. False here. |
| `declares_terms` | some licence-ish wording was found in the header. True here. |
| `publishable` | **only** "no restrictive token was found". It is not a grant, and it is True on any file that says nothing at all. |
| `no_restriction_found_but_terms_absent` | the silence case, separated out so absence of evidence cannot pass as evidence. False here. |

And `declares_terms` is True here for a reason worth stating: **the licence sentence in the bundled
scoring file's first `#` line was written by this project.** It records where the file came from, its
checksums, the publication's licence and both citations; every other line of
`data/genomics/pgs000072_breast_cancer.tsv` is the Catalog's file, unchanged, and none of those is a
licence (the format defines an optional `#license` line, and this file has none). Its `#citation`
line names the article's bioRxiv preprint, for which Crossref and Europe PMC record no licence; the
Catalog's publication record for the score is the Nat Commun article. So the audit reads this
repository's own annotation back to itself. It is still worth running, because it would catch a
future score that ships a real restrictive header; it is not worth treating as a green light.

**1000 Genomes / IGSR is not public domain either.** The IGSR disclaimer states that "restrictions on
different pieces of data within IGSR and rights claimed on pieces of data vary" and that "it remains
the responsibility of users to ensure that their exploitation of the data does not infringe any of
the rights of third parties." No licence instrument attaches to the genotypes; cite The 1000 Genomes
Project Consortium, *A global reference for human genetic variation*, Nature 526:68-74 (2015),
doi:10.1038/nature15393.

**No genotypes are in this artifact.** `note` in `weights.json` says so and it is accurate: the file
is effect weights and an audit block. The per-sample dosages this repository does redistribute are
tabulated, file by file and to the genotype cell, in `DATA_LICENSES.md` section 4; the 2504-individual
matrix behind the table above is deliberately **not** committed, and is regenerated locally by
`scripts/data/fetch_g1k_dosages.py`. What ships to a browser is the aggregate statistics plus a
30-individual payload, which is a different and much smaller exposure than what is tracked in git -
section 4 keeps those two apart on purpose.

## Reproducibility

    python scripts/data/fetch_g1k_dosages.py
    python scripts/train/genomics_prs.py

The first writes the 2,504-individual matrix (local) and the committed 97-individual slice; the
second rewrites `weights.json` here and the `genomics.prs` record in `models/metrics.json`. The
scoring file and the dosage file are selected together by score NAME rather than by path, so the two
cannot be mismatched into a silent wrong-variant join.

## Scope

This is a published risk score being re-scored on a reference panel, not a model this project
trained, and not a clinical instrument. No individual result on the genomics page is a medical
statement. The demonstration is the ancestry stratification of one globally-set threshold, plus the
fact that a 183-term dot product is small enough to evaluate under CKKS - and that smallness is a
property of PGS000072 having 187 variants, not a limit of homomorphic evaluation: Knight et al.,
*Cell Reports Methods* 6(1):101271 (2026), run polygenic scoring under HE over 110,000 SNPs and call
it practical on a single CPU.
