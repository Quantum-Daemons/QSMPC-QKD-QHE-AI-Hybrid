# Pre-registration: an encrypted predictor of 3D genome folding divergence (D1, 2026-11-26)

Written and committed before any model below is fitted or any genotype fetched, on the owner's choice of 2026-11-26 (the
genomics page's option D1): a small linear model, taught by published Akita predictions of how each 1000 Genomes
individual's chromatin folds, evaluated under CKKS on a person's encrypted genotypes. What is known when writing it: the
data's layout, its counts (2,457 individuals, 4,873 windows, 392 divergent windows), that every one of the 2,457 is in
the bundled 1000 Genomes panel with the same super-population label, and that all 30 individuals the genomics payload
already ships are among them. No figure about any individual's divergence or any variant's association has been
computed. Nothing here is changed after a figure is seen; a change is an addendum with its date and reason.

## What it is and is not

The target is a **prediction**, not a measurement: Akita (Fudenberg, Kelley and Pollard, Nature Methods 17, 1111,
2020), run by Gilbertson et al. on each individual's genome, predicts a chromatin contact map for each 1 Mb window, and
the window's **3D divergence** is one minus the Spearman correlation between that map and the map of the inferred
human-archaic hominin ancestor (Gilbertson, Brand, McArthur, Rinker, Kuang, Pollard and Capra, Mol Biol Evol 41(10),
msae209, 2024). The data are Zenodo record 13900918 (CC BY 4.0; the article itself is CC BY-NC 4.0, and nothing of its
text or figures is used). Nothing here is a disease risk, a clinical figure or a measurement of anyone's chromatin, and
the page says so.

## Data

- **Teacher.** `anc_window_spearman.csv`: for each individual $`i`$ and window $`w`$, $`y_{iw} = 1 - \rho_{iw}`$.
- **Individuals.** The 2,457 of that table, all in `data/genomics/g1k_panel.tsv`. They are split once, by individual:
  the **test** set is the 30 individuals the genomics payload ships (`sample_ids` of the active payload) and a draw of
  the rest, stratified by super-population with seed 0, to 15 percent of the 2,457 in all; **validation** another 15
  percent drawn the same way from the remainder; the rest **fit**. One split for every window.
- **Windows: twelve.** Among the 392 windows the data mark as more divergent than expected
  (`divergent_windows_exp_distributions.txt`), ranked by the standard deviation of $`y_{iw}`$ over the **fit**
  individuals, largest first, ties by position; a window is skipped if it overlaps one already chosen (the windows slide
  by half a window). The first twelve are kept.
- **Variants: ten a window.** Each chosen window's variants in `divergent_windows_10AF_variants.txt` (the data's common
  variants, each with its own effect when inserted alone into the ancestral sequence), ranked by that file's
  `1-spearman` effect, largest first, ties by position; the first ten that resolve (below) are kept, so at most 120 in
  all. No variant is chosen by its association with $`y`$.
- **Genotypes.** Read through the Ensembl REST API (GRCh38): each variant's rsID found by its position and alleles
  (`/overlap/region`), then its 1000 Genomes phase 3 genotypes (`/variation/{id}?genotypes=1`), with the resolver and
  the refusal rules of `scripts/data/fetch_g1k_dosages.py` (every individual has a genotype; the alleles match; nothing
  filled in). A variant that does not resolve is declared with its reason and the next in the ranking taken. The input
  is the **carrier indicator** $`c_{iv} = 1`$ if individual $`i`$ carries at least one alternate allele of variant
  $`v`$, else 0: Gilbertson et al. built each individual's sequence by inserting every one of their SNVs, heterozygous
  or homozygous alike, into hg38. Their genotypes are the 1000 Genomes 20190312 GRCh38 release and Ensembl's are
  phase 3's; where the two disagree the model sees a different genotype from the teacher's, which this cannot measure,
  and the record says so.

## Model

For each window, a ridge regression of $`y_{iw}`$ on that window's carrier indicators, the penalty from
$`\{10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}`$ chosen by validation $`R^2`$, refitted on fit and validation together at the
chosen penalty. The twelve models are one linear map from the at most 120 indicators to twelve outputs (block-diagonal
weights), which CKKS evaluates on a person's encrypted indicators.

## Metrics

On the test individuals, per window: $`R^2`$ and the Spearman correlation of the prediction with $`y`$; beside each, the
$`R^2`$ of the comparator that knows only the individual's super-population (its fit individuals' mean of $`y`$). Under
CKKS (TenSEAL in Python, node-seal in the browser): the largest absolute difference between the decrypted outputs and
the plaintext ones over every test individual.

## Criteria for the page

1. A window's predictor is shown only if its test $`R^2`$ is at least 0.5 and above its super-population comparator's;
   the card appears only if at least six of the twelve windows qualify. Otherwise the record says which held and the
   page shows nothing new.
2. The encrypted outputs equal the plaintext ones within $`10^{-4}`$ on every test individual, in both tiers; otherwise
   the encrypted path does not ship.
3. The card calls its figure "predicted 3D folding divergence (Akita, from Gilbertson et al.'s data), not a disease
   risk", names the genotype-release caveat above, and credits the data (CC BY 4.0), Akita and 1000 Genomes. It serves
   the 30 individuals the payload already ships, at the qualifying windows' variants: no new individual.

The record is `docs/measurements/genomics-3d-divergence-2026-11-26.json`.

## Addendum, 2026-11-26, before any figure

The first run of the input step stopped on reading the table: 171 of its 4,873 windows (134 on chromosome 18, 37 on
chromosome 22) have one or two individuals with no value, 16 of them among the 392 divergent windows, a count read
before any divergence figure was computed. An individual with no value in a window is left out of that window's
figures (its spread, fit, validation and test) and nothing is filled in; every model still scores every individual,
and the card shows no teacher value where the table has none. Nothing else changes.

## Addendum 2, 2026-11-26, before any figure

The input step stopped a second time on the variants file: 5 of its 721,722 variants give their effect as `NA`. A
variant without an effect has no rank, so it is not ranked and not chosen; the record counts how many each chosen
window had. Nothing else changes.
