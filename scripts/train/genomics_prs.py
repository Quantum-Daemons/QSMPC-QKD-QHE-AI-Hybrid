#!/usr/bin/env python
"""Evaluate the breast-cancer polygenic risk score across genetic ancestries.

There is nothing to train here, and that is the point. A PRS is a published per-SNP effect-weight
vector — PGS000004 (PRS313) is 313 variants from Mavaddat et al. 2018 — applied as a dot product.
It is the rare
model that is simultaneously the clinical standard and FHE-native, because a dot product is exactly
what CKKS evaluates well. Presenting a genomic foundation model as a PRS method would be the most
obvious credibility hole available, so this script does not.

WHAT CANNOT BE MEASURED HERE, AND WHY IT IS STILL REPORTED
    1000 Genomes is a reference panel. It has no breast-cancer case/control status, so no AUROC can
    be computed from it, and no supervised model can be trained from it either - which is why this use
    case has no QCNN and will not have one until a labelled cohort with usable terms exists. The
    published AUROC comes from an access-controlled case-control cohort (PPM001647, n=400,812). Any
    AUROC this repository printed from 1000 Genomes would be fabricated, so the published figure is
    carried as a citation from `MetricSpec.baseline` - not restated here, so the two cannot drift -
    and is labelled external, never as something this run measured.

WHAT IS MEASURED
    The score distribution per super-population. Effect weights estimated in one ancestry do not
    transport to another: allele frequencies differ, linkage disequilibrium differs, and the same
    absolute score means a different percentile in a different group. A single global "high risk"
    threshold therefore selects wildly different fractions of each population, and that is directly
    computable from real genotypes without any phenotype. Published context, now about the very score
    in use: PRS313's AUROC drops from 0.64 (European) to 0.58 (South Asian) and 0.57 (African) in UK
    Biobank (EJHG, 2026-01-13), and PGS000004's own 28 Catalog AUROC records span 0.55 for an
    African-ancestry sample against 0.5877-0.668 for European ones. The stratification measured below
    reaches the same conclusion from allele frequencies alone.

Usage:
    python scripts/train/genomics_prs.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qsh.data.loaders import _PRS_FILES, load_prs  # noqa: E402
from qsh.models import datasets, train  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402

#: The percentile a clinical PRS commonly uses as its "high risk" cut. Applied globally on purpose,
#: because applying one threshold to everyone is what actually happens and what breaks.
HIGH_RISK_PERCENTILE = 90.0


def load_dosages(path: Path, keys: list[str]) -> tuple[list[str], list[str], np.ndarray]:
    """Sample ids and the dosage matrix, selecting the score's variant columns BY NAME.

    This used to slice positionally - `r.split("\t")[1:]` - on the assumption that exactly one
    metadata column precedes the variants. The PRS313 dosage file also carries `population` and
    `super_population`, so the slice reached them and `int("GBR")` raised. That is the identical bug
    `qsh.data.loaders.load_genomic_prs` was fixed for in Round 29, reproduced here because this script
    parses the file itself instead of using the loader; the loader at least fails loudly, whereas its
    own `except Exception` fallback would have substituted Hardy-Weinberg samples.

    Selecting by name also fixes the quieter half: a positional slice silently accepts a column ORDER
    that disagrees with the scoring file, which would score each individual on the wrong variants.
    """
    rows = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    header = rows[0].split("\t")
    col = {name: j for j, name in enumerate(header)}
    missing = [k for k in keys if k not in col]
    if missing:
        raise SystemExit(
            f"{path.name} has no column for {len(missing)} of {len(keys)} score variants, e.g. "
            f"{missing[:3]}. The dosage file and the scoring file belong to different scores, or are "
            "keyed differently (rsID vs chr:pos:effect_allele).")
    idx = [col[k] for k in keys]
    samples = [r.split("\t")[0] for r in rows[1:]]
    dosages = np.asarray([[int(r.split("\t")[j]) for j in idx] for r in rows[1:]], dtype=np.float64)
    return samples, list(keys), dosages


def load_weights(score: str) -> dict[str, float]:
    """Variant key -> effect weight, using the SAME keying as the browser and the loader.

    This used to parse the scoring file here, taking column 0 as the key and column 3 as the weight.
    That is correct for the PGS000001 layout, where column 0 is `rsID`, and silently wrong for
    PGS000004, where column 0 is `chr_name` and column 3 is `other_allele`: every variant on a
    chromosome would collapse onto the key "1", and the weight lookup would try `float("A")`.

    `qsh.data.loaders.load_prs` already resolves both layouts and is tested against both, so the join
    key is built in exactly one place. Two implementations of the same keying is how the Python score
    and the browser score come to disagree, which is the failure `loaders.py` says the staged approach
    exists to avoid.
    """
    keys, _eff, weights = load_prs(score)
    return dict(zip(keys, (float(x) for x in weights), strict=True))


def load_panel(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("sample"):
            continue
        p = line.split("\t")
        if len(p) >= 3:
            out[p[0]] = p[2]
    return out


def audit_licence(prs_path: Path) -> dict:
    """Read the '#license' header of the scoring file actually in use.

    PGS Catalog is not blanket CC-BY-4.0: the licence is per score, defaults to the EMBL-EBI Terms
    of Use, and individual scores can be CC BY-NC-ND. Since the Hugging Face public-repo grant is
    irrevocable, this has to be checked from the file rather than assumed, and it gates publication.
    """
    header = [ln for ln in prs_path.read_text().splitlines() if ln.startswith("#")]
    text = " ".join(header)
    lowered = text.lower()
    restricted = any(tok in lowered for tok in ("nc-nd", "noncommercial", "non-commercial", "nd 4.0"))
    # SILENCE IS NOT PERMISSION, and this function used to conflate them.
    #
    # PGS Catalog scoring files carry no `#license` line at all - I checked a raw download of
    # PGS000004 and there is none - so on an unmodified Catalog file this returned
    # `publishable: True` purely because no restrictive token was present. The only reason it ever
    # appeared to read a licence is that the bundled PGS000001 file has a licence sentence THIS
    # PROJECT wrote into its first line. A future reader taking `publishable: True` from a fresh
    # download as a green light would be reading the absence of evidence as evidence.
    #
    # So report the two facts separately. Publication is gated by `publishable=False` on the dataset
    # spec and enforced in `scripts/publish_hf.py`; this audit informs that decision and does not
    # make it.
    declares_terms = any(tok in lowered for tok in ("licence", "license", "terms of use", "cc-by",
                                                    "cc by", "creative commons"))
    return {
        "header": header[:3],
        "declares_restriction": restricted,
        "declares_terms": declares_terms,
        # Kept for compatibility, and now meaning exactly "no restrictive token was found" - which is
        # a weaker statement than the name suggests when `declares_terms` is False.
        "publishable": not restricted,
        "no_restriction_found_but_terms_absent": (not restricted) and (not declares_terms),
        "checked": "read from the bundled scoring file at run time",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    # A SCORE NAME, not a path, so the scoring file and the dosage file cannot be mismatched. They
    # are paired in `_PRS_FILES`, and pointing --prs at one score while the dosages default to the
    # other is a KeyError at best and a silent wrong-variant join at worst.
    ap.add_argument("--score", choices=sorted(_PRS_FILES), default="breast_cancer_prs313",
                    help="which bundled PGS-Catalog score to use (PRS313 since Round 31)")
    ap.add_argument("--dosages", type=Path, default=None,
                    help="defaults to the full panel if present, else the score's bundled dosage file")
    ap.add_argument("--panel", type=Path, default=Path("data/genomics/g1k_panel.tsv"))
    ap.add_argument("--out", type=Path, default=Path("models/genomics_prs"))
    args = ap.parse_args()

    started = time.time()
    score_file, dosage_file = _PRS_FILES[args.score]
    prs_path = Path("data/genomics") / score_file
    dosage_path = args.dosages
    if dosage_path is None:
        dosage_path = Path("data/genomics") / dosage_file
        # `g1k_dosages_full.tsv` is the 2,504-sample panel for PRS77 ONLY, and it is keyed by rsID.
        # An unconditional preference for it - which is what this used to be - hands PRS313 a file
        # whose 77 column names match none of its 313 position keys. The guard below caught that, but
        # the fallback should not have needed catching: it is score-specific, so it is scoped to the
        # score it belongs to. PRS313's own dosage file already carries all 2,504 individuals.
        if args.score == "breast_cancer":
            full = Path("data/genomics/g1k_dosages_full.tsv")
            if full.exists():
                dosage_path = full
        # The committed slice, when the full matrix is not there. `qsh.data.loaders` gained exactly this
        # fallback in Round 31 and this script did not, so it raised FileNotFoundError at its own
        # default on any fresh clone or CI checkout - PRS313's full 2,504-row matrix is gitignored
        # because of the redistribution budget in DATA_LICENSES section 4.
        #
        # That mattered more than a missing convenience: this is the script that writes
        # models/metrics.json, and README points readers at its `by_super_population` rows rather than
        # at any figure quoted in prose. The generator of the numbers the documentation defers to could
        # not be run by anyone who had just cloned the repository.
        #
        # The lesson worth keeping: a fallback added to one of two readers of the same file is half a
        # fix, and the half that is missing is invisible until someone starts from a clean tree.
        if not dosage_path.exists():
            slice_path = dosage_path.with_name(f"{dosage_path.stem}_slice.tsv")
            if slice_path.exists():
                print(f"  {dosage_path.name} absent (gitignored); using the committed slice "
                      f"{slice_path.name} - REAL genotypes, fewer individuals")
                dosage_path = slice_path
    print(f"score:   {args.score} ({prs_path})")
    print(f"dosages: {dosage_path}")

    # Weights FIRST, so the dosage columns are selected in the scoring file's own variant order
    # rather than whatever order the dosage file happens to use. The alignment is then a name lookup,
    # not an assumption, and `w` below indexes the same variant as column i of `dosages` by
    # construction instead of by coincidence.
    weights_by_key = load_weights(args.score)
    keys = list(weights_by_key)
    samples, rsids, dosages = load_dosages(dosage_path, keys)
    w = np.array([weights_by_key[r] for r in rsids], dtype=np.float64)
    print(f"  {len(samples)} samples x {len(rsids)} SNPs")

    licence = audit_licence(prs_path)
    print(f"  licence audit: publishable={licence['publishable']}")

    scores = dosages @ w

    # Ancestry labels. Without them there is no stratification and therefore no finding.
    panel: dict[str, str] = {}
    if args.panel.exists():
        panel = load_panel(args.panel)
    else:
        # The IGSR panel is written next to the dosages by scripts/data/fetch_g1k_dosages.py.
        fallback = datasets.raw("igsr_panel.tsv")
        if fallback.exists():
            for line in fallback.read_text().splitlines()[1:]:
                c = line.split("\t")
                if len(c) >= 3:
                    panel[c[0]] = c[2]
    groups = np.array([panel.get(s, "UNKNOWN") for s in samples])
    present = [g for g in ("AFR", "AMR", "EAS", "EUR", "SAS") if (groups == g).sum() >= 20]
    print(f"  super-populations with >=20 samples: {present}"
          f"{'  [AFR absent - the largest published portability gap cannot be shown]' if 'AFR' not in present else ''}")

    # One global threshold, applied to everyone — the clinically realistic mistake.
    cut = float(np.percentile(scores, HIGH_RISK_PERCENTILE))
    print(f"\n  PRS distribution by ancestry (global P{HIGH_RISK_PERCENTILE:.0f} cut = {cut:.4f})")
    print(f"    {'group':8s} {'n':>5s} {'mean':>9s} {'sd':>8s} {'P90':>9s} {'%above global cut':>18s}")
    per_group = {}
    for g in present:
        m = groups == g
        s = scores[m]
        frac = float(np.mean(s >= cut))
        per_group[g] = {"n": int(m.sum()), "mean": round(float(s.mean()), 6),
                        "sd": round(float(s.std()), 6),
                        "p90": round(float(np.percentile(s, 90)), 6),
                        "frac_above_global_cut": round(frac, 6)}
        print(f"    {g:8s} {m.sum():5d} {s.mean():9.4f} {s.std():8.4f} "
              f"{np.percentile(s, 90):9.4f} {frac*100:17.1f}%")

    if len(per_group) >= 2:
        fr = {g: v["frac_above_global_cut"] for g, v in per_group.items()}
        hi, lo = max(fr, key=fr.get), min(fr, key=fr.get)
        head = (f"\n  A single global threshold flags {fr[hi]*100:.1f}% of {hi} and "
                f"{fr[lo]*100:.1f}% of {lo}")
        # A RATIO NEEDS A NON-ZERO DENOMINATOR, and `max(fr[lo], 1e-9)` is not a guard - it is a
        # division by 1e-9 wearing one. On the committed CI slice no EUR individual exceeded the cut, so
        # this line printed "a 208333000.0x difference" as though it were a measurement. The absolute
        # percentages either side of it were correct and the number between them was arithmetic on a
        # floor constant.
        #
        # The honest report when a group has nobody above the cut is that the ratio is undefined and the
        # gap is total, which is a stronger statement than any figure - so say that instead of dividing.
        if fr[lo] <= 0:
            print(head + " — no " + lo + " individual exceeded the cut at all, so the ratio is"
                  " undefined and the separation is complete. Driven entirely by ancestry, with no"
                  " phenotype involved.")
        else:
            print(head + f" — a {fr[hi]/fr[lo]:.1f}x difference driven entirely by ancestry,"
                  " with no phenotype involved.")

    spec = get_spec("genomics.prs")
    print(f"\n  published AUROC (NOT measured here): {spec.metric.baseline} — {spec.metric.baseline_source}")

    # ---- artefacts: the weight vector is the model ----------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weights.json").write_text(json.dumps({
        "score_id": "PGS000004",
        "trait": "breast cancer",
        "n_variants": len(rsids),
        "rsids": rsids,
        "weights": [float(x) for x in w],
        "high_risk_percentile": HIGH_RISK_PERCENTILE,
        "global_cut": cut,
        "licence_audit": licence,
        "note": ("Effect weights only. No per-sample genotype is included, and none may be published: "
                 "IGSR disclaims blanket openness and rights vary by data component."),
    }, indent=2) + "\n")

    train.record_metrics("genomics.prs", {
        "measured": {
            "n_samples": len(samples), "n_variants": len(rsids),
            "score_mean": round(float(scores.mean()), 6),
            "score_sd": round(float(scores.std()), 6),
            "global_high_risk_cut": round(cut, 6),
            "by_super_population": per_group,
        },
        "not_measured": {
            "auroc": None,
            # No figure is written into this sentence. It used to say "the published 0.628 comes from
            # n=200,195", and 0.628 was PGS000001's Harrell C-index -- so a field named
            # `published_auroc_eur` held a value that was not an AUROC, which is exactly the mislabel
            # `MetricSpec.baseline_source` was corrected for. Interpolating from the spec means the two
            # cannot disagree again. PGS000004 does publish a genuine AUROC, so the field name is now
            # accurate as well as consistent.
            "reason": ("1000 Genomes is a reference panel with no breast-cancer case/control status. "
                       "AUROC requires a phenotyped cohort, so the published figure is carried as a "
                       f"citation, not as a measurement from this run: {spec.metric.baseline} "
                       f"({spec.metric.display})."),
            "published_auroc_eur": spec.metric.baseline,
            "published_source": spec.metric.baseline_source,
        },
        "dataset": spec.dataset.name,
        "licence_audit": licence,
        "git_sha": train.git_sha(),
        "wall_clock_s": round(time.time() - started, 1),
    })
    print(f"\n  wrote {args.out}/weights.json and models/metrics.json in {time.time()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
