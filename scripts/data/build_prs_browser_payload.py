#!/usr/bin/env python3
"""Generate the genomics browser payload from the scoring file and the local genotypes.

The payload is a GENERATED file with readers in `realData.ts`, `prep_real_data_2d.py` and the tests;
an earlier payload was hand-built with three readers and no writer, and a hand edit to a
generated-looking file is how a stale value outlives its source. Regenerate it, do not edit it.

THE REDISTRIBUTION BUDGET IS A HARD CONSTRAINT, NOT A DEFAULT.
    1000 Genomes phase 3 genotypes carry no licence instrument: IGSR's disclaimer says rights "vary"
    by data component, and the data are open-consent. What ``DATA_LICENSES.md`` section 4 justifies
    shipping is a small slice, so this script ships ``--individuals`` of the 2,504 (default 30) and
    writes that number plus the total available into the payload, so the page can say what it is
    showing and what it is not. The full matrix stays local and gitignored; nothing here uploads it.

THE CREDITS TRAVEL WITH THE FILE. The weights are PGS000072's, from Graff et al., Nat Commun 12:970
(2021), licensed CC BY 4.0, which asks for credit, a link to the licence and a note of changes; the
PGS Catalog asks to be cited by the paper its REST `/rest/info` names; IGSR asks for the 1000 Genomes
consortium paper. The payload carries all three, the licence link and the changes.

Usage:
    .venv/bin/python scripts/data/build_prs_browser_payload.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qsh.data.loaders import (  # noqa: E402
    _PRS_FILES,
    ACTIVE_PRS,
    load_genomic_prs,
    load_prs,
    unscored_variants,
)
from qsh.models.registry import (  # noqa: E402
    BUNDLED_DEMO_SAMPLES,
    DEMO_SAMPLE_FILES,
    DS_1000G,
    DS_PGS_CATALOG,
)


def _registry_provenance(payload_name: str) -> str:
    """The provenance text for the genomics demo sample, from the one place that owns it.

    Refuses to hand that text to a payload the registry is not describing: `BUNDLED_DEMO_SAMPLES`
    documents the CURRENT demo sample, so stamping it on another score's file would state a false
    provenance, and `test_demo_provenance` only checks the file the registry points AT.
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

#: Payload filename per score. Explicit, so switching the browser to another score is a visible change
#: in the registry's `DEMO_SAMPLE_FILES`, which this name must match.
_PAYLOAD = {
    "breast_cancer_pgs000072": "pgs000072_breast_cancer.json",
}

_META = {
    "breast_cancer_pgs000072": {
        "name": "PGS000072 (CC_Breast, breast cancer)",
        "score_id": "PGS000072",
        "keyed_by": "rsID, as the scoring file gives it",
    },
}

CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", choices=sorted(_PAYLOAD), default=ACTIVE_PRS)
    ap.add_argument("--individuals", type=int, default=30,
                    help="how many genotype rows to ship; DATA_LICENSES section 4 justifies 30")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    published_keys, _eff, _w = load_prs(args.score)
    # Load the FULL matrix, then subsample, so the population baseline the browser shows is computed
    # over everyone available, not over the 30 rows that happen to ship.
    dosages, w, keys, eff, _freq = load_genomic_prs(n_individuals=10_000, seed=args.seed, name=args.score)
    dosage_path = REPO / "data" / "genomics" / _PRS_FILES[args.score][1]
    if not dosage_path.exists():
        raise SystemExit(f"{dosage_path.name} is absent: run scripts/data/fetch_g1k_dosages.py first. "
                         "The committed slice is not a population, so it cannot stand in here.")
    samples = [ln.split("\t")[0] for ln in dosage_path.read_text().splitlines()
               if ln.strip() and not ln.startswith("#")][1:]
    unscored = unscored_variants(dosage_path)
    assert [k for k in published_keys if k not in unscored] == list(keys), \
        "the loader's variants are not the scoring file's minus the declared #unscored ones"
    n_available = int(dosages.shape[0])
    assert len(samples) == n_available

    # The same draw over the same sorted 2,504-sample panel selects the same individuals for any
    # score, and `sample_ids` lets a reader check each row against IGSR.
    rng = np.random.default_rng(args.seed)
    take = np.sort(rng.choice(n_available, size=min(args.individuals, n_available), replace=False))
    shipped = dosages[take].astype(int)

    meta = _META[args.score]
    payload_name = _PAYLOAD[args.score]
    payload = {
        "name": meta["name"],
        "score_id": meta["score_id"],
        "keys": list(keys),
        "keyed_by": meta["keyed_by"],
        "effect_alleles": list(eff),
        # Full precision, as the scoring file publishes them: rounding would be a change to the work
        # that CC BY 4.0 asks to be noted, and the Python tier scores the file's own values.
        "weights": [float(x) for x in w],
        "variants_published": len(published_keys),
        # Score variants with no 1000 Genomes genotype column, each with the fetcher's reason. They are
        # left out of the score, never filled in.
        "variants_unscored": [{"key": k, "reason": unscored[k]} for k in published_keys if k in unscored],
        "dosage_samples": [[int(v) for v in row] for row in shipped],
        "sample_ids": [samples[i] for i in take],
        "n_individuals_shipped": int(shipped.shape[0]),
        "n_individuals_available": n_available,
        # Over ALL available individuals, not the shipped slice: `realData.ts` uses it as the SHAP
        # reference point, and the zero vector is not a typical genome.
        "population_mean_dosage": [round(float(x), 6) for x in dosages.mean(axis=0)],
        "licence": "CC-BY-4.0 (the effect weights); none (the 1000 Genomes genotypes)",
        "licence_url": CC_BY_4,
        "attribution": DS_PGS_CATALOG.attribution,
        "genotypes_attribution": DS_1000G.attribution,
        "modifications": (f"The {len(keys)} of PGS000072's {len(published_keys)} published variants that "
                          f"Ensembl's 1000 Genomes phase 3 records resolve are scored, with their weights "
                          f"unchanged; the other {len(unscored)} are listed in variants_unscored and left "
                          f"out. The dosages are effect-allele counts from 1000 Genomes phase 3 genotypes "
                          f"read through the Ensembl REST API; {int(shipped.shape[0])} of {n_available} "
                          f"individuals are included."),
        # THE REGISTRY OWNS THIS STRING: `tests/test_demo_provenance.py` asserts the shipped JSON's
        # provenance is `BUNDLED_DEMO_SAMPLES[usecase][0]` verbatim.
        "provenance": _registry_provenance(payload_name),
    }

    out = WEB / payload_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out.relative_to(REPO)}")
    print(f"  {len(keys)} of {len(published_keys)} variants scored, keyed by {meta['keyed_by']}")
    print(f"  {shipped.shape[0]} of {n_available} individuals shipped ({out.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
