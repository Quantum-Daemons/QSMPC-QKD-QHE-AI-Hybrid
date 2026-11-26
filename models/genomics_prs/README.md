# Polygenic risk score - PGS000783, breast cancer

> **This card documents PROVENANCE AND LICENCE, because that is what the artifact records.** The
> `weights.json` in this directory is a published effect-weight vector lifted out of a PGS Catalog
> scoring file. It was not fitted here, it has no training split, and this repository computes no
> discrimination metric for it at all - no AUROC, no C-index, no F1. What the run DOES measure is the
> distribution of the resulting score across 1000 Genomes super-populations, and that distribution is
> the finding rather than a footnote. The one AUROC anyone can quote for this score comes from a
> phenotyped cohort this project cannot reach; it is carried below as a citation, under its own
> heading, and is not a result of this run.

**The score changed on 2026-11-26** (Round 86, batch 6, the owner's choice). It is PGS000783
(`CC_Breast_IV`), the 162 of PGS000072's 187 variants that reached P < 5e-8 in the
Breast Cancer Association Consortium (BCAC) meta-analysis, each weighted by beta/SE there
(Kachuri et al., Nat Commun 11:6084, 2020). PGS000072 (Graff et al. 2021), the score from Round 84, is
retired: its scoring file, its dosage slice and its old browser payload stay in `data/genomics/`, and the
page no longer serves the payload.

NOT PUBLISHED TO HUGGING FACE. `genomics.prs` carries no `hf_repo` and no pinned revision: the
weights are a published score's own, and they ship with the page instead. Its dataset record in
`src/qsh/models/registry.py` is `publishable=True`, because the score's publication is CC BY 3.0 IGO and
redistribution with the credit is permitted; `scripts/publish_hf.py` publishes nothing without an
`hf_repo`. Read the licence section before reading the `publishable` field inside this directory's own
`licence_audit` block - they do not mean the same thing, and the one in the artifact grants nothing.

## What ships in this directory

`weights.json` is written by `scripts/train/genomics_prs.py` and holds the score and its audit:

| field | what it holds |
|---|---|
| `score_id`, `trait` | `PGS000783`, `breast cancer`. Catalog name `CC_Breast_IV`, GRCh37, the author-reported scoring file. |
| `n_variants` | the variants scored: those with a 1000 Genomes phase 3 genotype for every one of the 2,504 individuals. |
| `n_variants_published` | the variant count of the published score. |
| `unscored_variants` | each published variant that is not scored, with the reason the dosage file declares for it. |
| `rsids` | the scored variants' rsIDs, as the scoring file gives them. |
| `weights` | their effect weights, beta/SE as the score publishes them, unchanged and in the same order as `rsids`. |
| `high_risk_percentile`, `global_cut` | the 90th percentile of the pooled score, and the score value that sits at it. One cut, set once over everybody, which is the measurement below; the page itself judges a person against their own super-population's 90th percentile. |
| `licence_audit` | what the bundled scoring file's `#` header said when it was last read. |
| `note` | effect weights only: this file carries no per-sample genotype. Per-sample dosages are redistributed in the slices `DATA_LICENSES.md` section 4 lists (30 individuals in the browser payload), with the 1000 Genomes consortium citation. |

**Four of the 162 published variants are not scored, and they are named rather than filled in.**
They are the four PGS000072's dosage files already declared: `scripts/data/fetch_g1k_dosages.py` read
each variant's 1000 Genomes genotypes through the Ensembl REST API and scored it only if every
individual had one, and rs8176636 does not map to GRCh38 there, rs34207738 and rs7297051 have no 1000
Genomes genotypes there, and rs12405132 has them for 1,582 of the 2,504. PGS000783's dosage files are
PGS000072's columns for its 162 variants, copied by `scripts/data/derive_g1k_dosages.py` only where the
two scoring files agree on the rsID, position, alleles and published frequency (all 162 do). Each of the
four is declared `#unscored`, with that reason, and listed in `unscored_variants`; the loader refuses any
absent variant that is not declared.

## Measured on 1000 Genomes: a distribution audit, not a discrimination metric

Every figure here is recomputed on each training run and lands in
`models/metrics.json` under `genomics.prs`, which is also what `web/src/components/ModelPanel.tsx`
renders on the genomics page. Nothing below is a phenotype comparison, because there is no phenotype.

| quantity | value |
|---|---|
| `n_variants` | 158 |
| `n_samples` | 2504 |
| `score_mean` | 1273.127448 |
| `score_sd` | 77.817328 |
| `global_high_risk_cut` | 1373.925 |

The scores sit above a thousand because every weight is a positive beta/SE, between 5.40
and 38.62, so a score is a sum of z-statistics, not of log odds. Only a score's place among others is
read, which that scale does not change. One high-risk cut, set once at the 90th percentile of the pooled
score, applied to everybody:

| super-population | n | mean | sd | p90 | share above the global cut |
|---|---|---|---|---|---|
| EAS | 504 | 1296.926567 | 73.315798 | 1391.652 | 0.154762 |
| AMR | 347 | 1279.461499 | 80.14799 | 1386.098 | 0.138329 |
| AFR | 661 | 1296.328971 | 66.369916 | 1379.89 | 0.11649 |
| SAS | 489 | 1250.056421 | 76.656506 | 1358.58 | 0.06544 |
| EUR | 503 | 1236.850815 | 75.435723 | 1333.884 | 0.031809 |

**Read the last column.** A single global cut flags 0.154762 of EAS samples and 0.031809 of EUR
samples - a factor of about five - and no phenotype was involved in producing that gap. It is an
allele-frequency artefact of applying a score whose weights come from GWAS samples that are all of
European ancestry (the Catalog's ancestry distribution for PGS000783: one GWAS, 139,274 individuals) to a
panel that is not, and it is the reason this use case exists in the demo. Here the shift is spread over
many variants: the East Asian against European mean difference, 60.1 (0.77 of the pooled standard
deviation), is the sum of contributions in both directions, the largest rs832540's +15.9 and rs4442975's
-17.0. Under PGS000072 the same cut flagged 0.390873 of EAS against 0.039761 of EUR, and 42% of that
mean difference came from one variant, rs10979327, which is not among PGS000783's 162. It is a
**distribution shift**, and calling it an AUROC drop would be wrong twice over: no AUROC is computed
here, and the gap is in where the score sits, not in how well it ranks.

The score's own published record cannot show this by another route: both of PGS000783's Catalog
performance records are in European-ancestry women in UK Biobank (read 2026-11-26).

## No AUROC is computed here, and none is invented

- **The metric the registry declares**: AUROC (European ancestry).
- **Published figure carried as a citation**: 0.626 - PGS Catalog PPM002070 for PGS000783, incident
  breast cancer in n=216,273 European-ancestry women in UK Biobank (4,760 cases, 211,513 controls),
  from Kachuri et al., Nat Commun 11:6084 (2020), doi:10.1038/s41467-020-19600-4, with age, genotyping
  array and genetic principal components 1-15 as covariates. No confidence interval is recorded (C-index
  0.62, SE 0.005). Read at the Catalog REST API on 2026-11-26.
- **Published ceiling**: 0.637 - PPM002054, the score's other record, on the same cohort with the
  reproductive, lifestyle and family-history covariates as well, beside a hazard ratio of 1.52 (95% CI
  1.47-1.56). PGS000072's PPM002038, from the same article and cohort, carries the same two figures for
  "only 162 of the original 187 SNPs", PGS000783's variants; the records do not say whether they are one
  evaluation filed twice.
- **Why neither is a target.** Both cited models are fitted WITH COVARIATES, which a bare weighted dot
  product does not have, and both are AUCs at 5 years.
- **Why it cannot be reproduced here.** 1000 Genomes is a reference panel and carries no
  breast-cancer case/control status. AUROC needs a phenotyped cohort. `genomics_prs.py` therefore
  records `auroc` as null rather than inventing one, and `not_measured.reason` in the metrics record
  states that in the record itself.

## Licence and provenance, which is the point of this card

**A PGS Catalog score's terms are its publication's.** PGS000783's is Kachuri et al., Nat Commun
11:6084 (2020), whose article page states **CC BY 3.0 IGO** (read 2026-11-26; Crossref's metadata
records CC BY 4.0, which the article page does not). The article weights each of its 162 variants by
beta/SE from the BCAC meta-analysis, publishes those numbers in its Supplementary Data 1,
and the authors deposited the score in the Catalog. The Catalog's REST record adds a pass-through to the
**EMBL-EBI Terms of Use** and asks users to observe any licensing restrictions the authors set. BCAC's
term not to repost its summary results to a third-party website binds those who download them from
BCAC, which this project did not; the meta-analysis is credited all the same (this project's reading of
BCAC's page, 2026-11-26, not legal advice). The licence asks for the credit, its URI and a credit for the
use, so all three travel with the browser payload:

> Kachuri L, Graff RE, Smith-Byrne K, Meyers TJ, Rashkin SR, Ziv E, Witte JS, Johansson M, *Pan-cancer
> analysis demonstrates that integrating polygenic risk scores with modifiable risk factors improves risk
> prediction*, Nat Commun 11:6084 (2020), doi:10.1038/s41467-020-19600-4, CC BY 3.0 IGO,
> https://creativecommons.org/licenses/by/3.0/igo/

The weights come from the BCAC meta-analysis of Michailidou K et al., *Association analysis identifies 65
new breast cancer risk loci*, Nature 551:92-94 (2017), doi:10.1038/nature24284. The Catalog asks to be
cited as Lambert, Wingfield et al., *Enhancing the Polygenic Score Catalog with tools for score
calculation and ancestry normalization*, Nat Genet 56:1989-1994 (2024), doi:10.1038/s41588-024-01937-x.
The change: 158 of the 162 variants are scored, with their weights unchanged.

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
checksums, the publication's licence and the citations; every other line of
`data/genomics/pgs000783_breast_cancer.tsv` is the Catalog's file, unchanged, and none of those is a
licence (the format defines an optional `#license` line, and this file has none). Its `#citation` line
names the Nat Commun article. So the audit reads this repository's own annotation back to itself. It is
still worth running, because it would catch a future score that ships a real restrictive header; it is
not worth treating as a green light.

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
`scripts/data/fetch_g1k_dosages.py` (PGS000072's) and `scripts/data/derive_g1k_dosages.py`
(PGS000783's, from it). What ships to a browser is the aggregate statistics plus a 30-individual
payload, which is a different and much smaller exposure than what is tracked in git - section 4 keeps
those two apart on purpose.

## Reproducibility

    python scripts/data/fetch_g1k_dosages.py --score breast_cancer_pgs000072
    python scripts/data/derive_g1k_dosages.py --from-score breast_cancer_pgs000072
    python scripts/train/genomics_prs.py

The first writes PGS000072's 2,504-individual matrix (local) and its committed 97-individual slice; the
second writes PGS000783's from them, and its gnomAD cache; the third rewrites `weights.json` here and the
`genomics.prs` record in `models/metrics.json`. The scoring file and the dosage file are selected
together by score NAME rather than by path, so the two cannot be mismatched into a silent wrong-variant
join.

## Scope

This is a published risk score being re-scored on a reference panel, not a model this project
trained, and not a clinical instrument. No individual result on the genomics page is a medical
statement. The demonstration is the ancestry stratification of one globally-set threshold, plus the
fact that a 158-term dot product is small enough to evaluate under CKKS - and that smallness is a
property of PGS000783 having 162 variants, not a limit of homomorphic evaluation: Knight et al.,
*Cell Reports Methods* 6(1):101271 (2026), run polygenic scoring under HE over 110,000 SNPs and call
it practical on a single CPU.
