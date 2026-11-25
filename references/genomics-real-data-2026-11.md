# Genomics: real reference data, ancestry-aware calibration and 2026 encrypted PRS work (read 2026-11-25)

On the owner's choice of 2026-11-25 (prefer practical large-scale REAL data to synthetic, include calibration and
preprocessing in the verification, and assess anything worth bringing to the encrypted paths), a survey read the
primary sources for the genomics page, and the agent then re-read the records and texts relied on below at their own
interfaces on 2026-11-25: the PGS Catalog calculator's documentation through `gh api`, Zenodo's records API, Crossref,
and a HEAD request for gnomAD's classifier. Nothing was downloaded. Where a figure was read by the survey and not
re-read, it says so.

## What the page uses today

- PGS000072 (`CC_Breast`, 187 variants; Graff et al., Nat Commun 12:970, 2021, CC BY 4.0) scored under CKKS over the
  183 variants with a 1000 Genomes phase 3 genotype, for one person among the **30** of 2,504 individuals the payload
  ships (`DATA_LICENSES.md` section 4 sets the slice).
- The "high risk" cut is the 90th percentile of the score over those 30 individuals, **all ancestries pooled**. The
  model card already measures why that is a poor reference: over all 2,504 individuals the pooled cut (22.641) is
  exceeded by 39.1 % of the EAS super-population against 2.1 % of AFR, 1.7 % of AMR, 4.0 % of EUR and 2.9 % of SAS
  (`models/metrics.json`, `genomics.prs`). The survey reproduced those shares from the repository's files.
- No learned model: the weights are the published score.

## What current practice says

- The PGS Catalog's own calculator, pgsc_calc (v2.3.0, 2026-06-03, Apache-2.0), produces ancestry-aware outputs
  (`Z_MostSimilarPop`, `Z_norm1`, `Z_norm2`) and its guide says to *"Use ancestry-normalised or Z-score outputs when
  applying PGS to individuals of diverse ancestries"* (`docs/explanation/interpret.rst`, line 125 at v2.3.0). The next
  line asks that groups be described by similarity to a reference, *"e.g. 1KG-EUR-like"* (NASEM 2023), not as
  ancestry labels.
- The method, as the survey read pgsc_calc's source: project the person onto a reference PCA; report the percentile
  within the most similar reference population; $`Z_{\mathrm{norm1}}`$, the residual of a regression of the score on
  four PCs; and $`Z_{\mathrm{norm2}}`$, which also models the variance as $`\exp`$ of a linear function of the PCs.
- Checks the survey ran on the repository's own files (no new data): standardising from allele frequencies alone,
  under Hardy-Weinberg and linkage equilibrium, understates the empirical spread of the score by up to about 9.6 %
  (EUR), so the empirical reference SD, or the linkage term $`w^{\top} C w`$, is needed; each super-population's 90th
  percentile carries a sampling error of about 0.07 to 0.09 SD; the 13 strand-ambiguous variants are oriented
  correctly by the allele-frequency check.

## Why it fits the CKKS path

The PCs are linear in the genotype vector, as the score is, so the server can compute the score and PCs 1 to 5 as six
encrypted dot products over one encrypted genotype vector, and $`Z_{\mathrm{norm1}} = c_0 + \sum_j c_j g_j`$ folds into
a single dot product with adjusted weights. The nonlinear parts ($`Z_{\mathrm{norm2}}`$'s variance term and the
most-similar-population assignment) run after decryption, by the key holder. No 2025-2026 paper was found that
computes an ancestry-normalised score, or ancestry similarity, under encryption (survey).

## Real data that needs no access application

| Record | Read at | Terms | What it offers |
|---|---|---|---|
| 1000 Genomes phase 3 (2,504) and the 30x panel (3,202) | IGSR | open access, no licence instrument (`DATA_LICENSES.md`); the survey reads the consent principles as covering population labels | genotypes and population labels: the only real labels here |
| gnomAD v4.0 genetic-ancestry classifier | `gcp-public-data--gnomad` (HEAD: 5,413,374 B, 2023-11-29) | gnomAD is CC0 | a random forest over PCA loadings, a cross-check for an assignment |
| *Reference distributions for the full PGS Catalog on the 1000 Genomes panel* | Zenodo 22906325 (v1.0.0, 2026-09-28) | CC BY 4.0 | per-score statistics over 3,202 individuals for 5,385 scores; a preprint whose code has no licence (survey), so a cross-check only |

Not usable: openSNP (deleted in 2025), UK Biobank, All of Us and FinnGen individual-level data (controlled); PRS313
(CC BY-NC-ND) and the PGS Browser distributions (NC-ND with a stated patent restriction); HGDP genotype slices in the
public tree (academic-use sample terms), which could be a local test target only (survey).

## Teachers and 2026 encrypted genomics

- Genomic foundation models (Evo 2, Nucleotide Transformer, DNABERT-2, AlphaGenome, Enformer, Borzoi) are not useful
  teachers here: the private computation is a weighted sum of genotypes, per-variant effects can be computed publicly,
  and full encrypted inference of a 0.1B DNA model took hours on a GPU in a 2026 preprint (survey).
- HEPRS: CKKS polygenic scores at 110,258 SNPs (Knight et al., Cell Reports Methods 6(1):101271, January 2026,
  doi:10.1016/j.crmeth.2025.101271; code MIT). The page's 183-variant score is far smaller; what it adds is the
  ancestry-aware reading, which HEPRS does not compute (survey).
- iDASH 2025 and 2026 set HE tasks on a DNA-binding CNN and a VAE decoder (survey).

## Options for the owner

1. **An ancestry-aware percentile on the existing CKKS path.** Fetch genotypes at a compact ancestry-SNP set (a few
   hundred to about 1,000 LD-pruned common SNPs) for the 2,504 phase-3 individuals, through the Ensembl API the payload
   builder already uses or range requests on IGSR's indexed VCFs (tens of MB, the survey's estimate); fit the reference
   PCA, the per-population quantiles and $`Z_{\mathrm{norm1}}`$ and $`Z_{\mathrm{norm2}}`$ in plaintext; the six
   encrypted dot products on the page; checks: CKKS against plaintext for each output, compact against full-panel
   PCs, and on held-out individuals the share above each population's 90th percentile within the binomial interval
   of 10 %. The payload stays at 30 persons; $`30 \times 1{,}000 = 30{,}000`$ cells is inside the 30,800-cell rule.
   The pooled cut stays, shown as the cautionary result, beside the ancestry-aware one, in "1KG-EAS-like" wording.
2. **A genetic-similarity classifier under CKKS** with real population labels: a multinomial logistic regression on
   the same PCs (the page's linear-layer path), checked per population and against the gnomAD classifier.
3. **Allele-frequency normalisation from the secure summation** with the linkage term, tying the MPC step to the CKKS
   step (data already in the repository).

Not recommended: a foundation-model teacher; PRS313 or the PGS Browser data; HGDP slices in the public tree; a random
forest or nearest-neighbour classifier under HE; any AUROC from 1000 Genomes (it has no case-control labels).

## Open questions

- Whether PGS000072 is among the Zenodo bundle's 5,385 scores, and whether its per-population figures match the card.
- Whether the 30x panel resolves the four variants phase 3 does not.
- Which reference to fit (phase 3 or the 30x panel), and keeping the scored person out of the reference it is read
  against.
