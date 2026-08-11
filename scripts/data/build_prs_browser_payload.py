#!/usr/bin/env python3
"""Generate the genomics browser payload from the scoring file and the local genotypes.

WHY THIS SCRIPT EXISTS AT ALL. ``web/public/data/genomics/pgs000001_prs.json`` had **three readers
and zero writers**: `realData.ts` fetches it, `prep_real_data_2d.py` reads it, `fetch_gnomad_af.py`
takes its rsID list, and nothing produced it. It was a hand-built committed artifact. So "regenerate
the payload for PRS313" was not a regeneration - the generator had to be written, and until it was,
the only way to change the browser's view of this use case was to edit a JSON blob by hand and hope
it stayed consistent with the Python pipeline that computes the same score.

That is not a hypothetical risk on this file. The same shape - a generated-looking artifact corrected
by hand while its producer kept emitting the old value - is what put the eight ATTACK CLASS names into
telecom's published ``feature_spec.json`` under a note reading "Generated from the source, not written
by hand".

THE REDISTRIBUTION BUDGET IS A HARD CONSTRAINT, NOT A DEFAULT.
    1000 Genomes genotypes come from IGSR, which grants no blanket licence and says so: "restrictions
    on different pieces of data within IGSR and rights claimed on pieces of data vary". What
    ``DATA_LICENSES.md`` section 4 justifies shipping is a small slice, and the PRS313 dosage matrix is
    2,504 x 313 - about 24x that. So this script subsamples to ``--individuals`` (default 30, the
    number section 4 argues for) and writes that number plus the total available into the payload's
    provenance, so the page can say what it is showing and what it is not.

    The full matrix stays local and gitignored. Nothing here uploads it.

Usage:
    .venv/bin/python scripts/data/build_prs_browser_payload.py --score breast_cancer_prs313
    .venv/bin/python scripts/data/build_prs_browser_payload.py --score breast_cancer      # PRS77
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.data.loaders import _PRS_FILES, load_genomic_prs, load_prs  # noqa: E402
from qsh.models.registry import BUNDLED_DEMO_SAMPLES, DEMO_SAMPLE_FILES  # noqa: E402


def _registry_provenance(payload_name: str) -> str:
    """The provenance text for the genomics demo sample, from the one place that owns it.

    Refuses to hand that text to a payload the registry is not describing. `BUNDLED_DEMO_SAMPLES`
    documents the CURRENT demo sample, so generating the retired score's payload and stamping this
    string on it would put a "published PGS000004 weights" claim on a file holding 77 PGS000001
    weights - a licence and provenance claim that is simply false, and `test_demo_provenance` would not
    catch it because that test only checks the file the registry points AT.

    This is the same failure the string exists to prevent, one level up: not two copies drifting, but
    one copy applied to the wrong artifact.
    """
    registered = Path(DEMO_SAMPLE_FILES["genomics"]).name
    if payload_name != registered:
        raise SystemExit(
            f"the registry's genomics demo sample is {registered}, but this run would write "
            f"{payload_name}. The provenance text in BUNDLED_DEMO_SAMPLES describes the registered "
            f"payload, so stamping it on {payload_name} would state a false provenance. Either point "
            f"DEMO_SAMPLE_FILES at {payload_name} first, or do not regenerate the retired score."
        )
    return BUNDLED_DEMO_SAMPLES["genomics"][0]

WEB = REPO / "web" / "public" / "data" / "genomics"

#: Payload filename per score. Kept explicit rather than derived so that switching the browser from
#: one score to the other is a visible one-line change in `realData.ts`, not a silent consequence of
#: renaming something here.
_PAYLOAD = {
    # PRS77 is not a browser payload any more: the file was moved to data/genomics/ in Round 32 because
    # nothing in the browser read it and Vite was serving genotype data for no demo benefit. The entry
    # stays so `--score breast_cancer` names a path rather than raising KeyError, and the provenance
    # guard below refuses to write it - which is the honest state, since the registry describes the
    # PRS313 payload and stamping its text on a 77-variant file would state a false provenance.
    "breast_cancer": "pgs000001_prs.json",
    "breast_cancer_prs313": "pgs000004_prs313.json",
}

_META = {
    "breast_cancer": {
        "name": "PGS000001 (PRS77_BC, breast cancer)",
        "score_id": "PGS000001",
        "citation": "Mavaddat et al. JNCI 2015",
        "keyed_by": "rsID",
    },
    "breast_cancer_prs313": {
        "name": "PGS000004 (PRS313_BC, breast cancer)",
        "score_id": "PGS000004",
        "citation": "Mavaddat et al. Am J Hum Genet 2018 (doi:10.1016/j.ajhg.2018.11.002)",
        # Not a stylistic choice: the Catalog publishes no rsID for any of the 313 variants
        # (hm_rsID empty, hm_source=Author-reported), 28% of the positions carry several rsIDs and
        # about 15% resolve to none, so keying on one would score a different variant's genotypes.
        "keyed_by": "chr:pos:effect_allele (GRCh37) - PGS000004 publishes no rsIDs",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", choices=sorted(_PRS_FILES), default="breast_cancer_prs313")
    ap.add_argument("--individuals", type=int, default=30,
                    help="how many genotype rows to ship; DATA_LICENSES section 4 justifies 30")
    ap.add_argument("--top", type=int, default=8, help="SNPs surfaced as SHAP features")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keys, eff_alleles, weights = load_prs(args.score)
    # Load the FULL matrix, then subsample - so the population baseline the browser shows is computed
    # over everyone available, not over the 30 rows that happen to ship. Those are different numbers
    # and the second one would be a worse estimate presented as the same thing.
    dosages, w, joined_keys, eff, _freq = load_genomic_prs(
        n_individuals=10_000, seed=args.seed, name=args.score)
    assert joined_keys == keys, "the loader reordered the variants relative to the scoring file"
    n_available = int(dosages.shape[0])

    rng = np.random.default_rng(args.seed)
    take = rng.choice(n_available, size=min(args.individuals, n_available), replace=False)
    shipped = dosages[np.sort(take)].astype(int)

    # Top-N by |weight|: the same rule `realData.ts` falls back to, computed here so the payload is
    # self-describing and the browser does not have to re-derive it.
    #
    # Order is |weight| DESCENDING and must stay that way. `realData.ts` maps this list positionally
    # onto the SHAP feature names and values, so the order is the order the bars are labelled in.
    # Sorting it numerically instead - which the first version of this script did - keeps the same set
    # and silently reorders the chart, putting the weakest of the eight first. The hand-built payload
    # this replaces was ordered by descending |weight|, and a round-trip against it is what caught the
    # difference: rsids, effect_alleles and weights matched exactly, top8_idx did not.
    order = np.argsort(-np.abs(np.asarray(w)))[: args.top]
    top_idx = [int(i) for i in order]

    meta = _META[args.score]
    payload_name = _PAYLOAD[args.score]
    payload = {
        "name": meta["name"],
        "score_id": meta["score_id"],
        "keys": list(keys),
        "keyed_by": meta["keyed_by"],
        "effect_alleles": list(eff_alleles),
        "weights": [round(float(x), 6) for x in w],
        "top8_idx": top_idx,
        "dosage_samples": [[int(v) for v in row] for row in shipped],
        "n_individuals_shipped": int(shipped.shape[0]),
        "n_individuals_available": n_available,
        # Computed over ALL available individuals, not the shipped slice. `realData.ts` uses this as
        # the SHAP reference point, and the zero vector - "an individual carrying no effect allele
        # anywhere" - is not a typical genome and at several of these loci is not present at all.
        "population_mean_dosage": [round(float(x), 6) for x in dosages.mean(axis=0)],
        # THE REGISTRY OWNS THIS STRING, and composing a second one here is the bug
        # `tests/test_demo_provenance.py` exists to catch: it asserts the shipped JSON's provenance is
        # `BUNDLED_DEMO_SAMPLES[usecase][0]` verbatim, because a licence claim on a public page that
        # disagrees with the machine-readable record has been wrong four times on this project. The
        # first version of this generator wrote its own wording and the test caught it immediately.
        "provenance": _registry_provenance(payload_name),
    }
    # PRS77 keeps an `rsids` field because two existing readers name it. PRS313 has no rsIDs to put
    # there, and inventing the field with position keys in it would let a reader treat a coordinate as
    # an identifier - which is the exact confusion this score's lack of rsIDs causes.
    if args.score == "breast_cancer":
        payload["rsids"] = list(keys)

    out = WEB / payload_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out.relative_to(REPO)}")
    print(f"  {len(keys)} variants, keyed by {meta['keyed_by']}")
    print(f"  {shipped.shape[0]} of {n_available} individuals shipped ({out.stat().st_size/1024:.1f} KB)")
    print(f"  top-{args.top} by |weight|: {top_idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
