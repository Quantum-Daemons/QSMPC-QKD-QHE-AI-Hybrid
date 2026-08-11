#!/usr/bin/env python
"""Evaluate the breast-cancer polygenic risk score across genetic ancestries.

There is nothing to train here, and that is the point. A PRS is a published per-SNP effect-weight
vector — PGS000001 is 77 variants from Mavaddat et al. — applied as a dot product. It is the rare
model that is simultaneously the clinical standard and FHE-native, because a dot product is exactly
what CKKS evaluates well. Presenting a genomic foundation model as a PRS method would be the most
obvious credibility hole available, so this script does not.

WHAT CANNOT BE MEASURED HERE, AND WHY IT IS STILL REPORTED
    1000 Genomes is a reference panel. It has no breast-cancer case/control status, so no AUROC can
    be computed from it. The published AUROC of 0.628 comes from a case-control cohort of n=200,195,
    which is access-controlled. Any AUROC this repository printed from 1000 Genomes would be
    fabricated. The published figure is therefore carried as a citation and labelled as external,
    never as something this run measured.

WHAT IS MEASURED
    The score distribution per super-population. Effect weights estimated in one ancestry do not
    transport to another: allele frequencies differ, linkage disequilibrium differs, and the same
    absolute score means a different percentile in a different group. A single global "high risk"
    threshold therefore selects wildly different fractions of each population, and that is directly
    computable from real genotypes without any phenotype. Published context: PGS313 AUROC drops from
    0.64 (European) to 0.58 (South Asian) and 0.57 (African) in UK Biobank (EJHG, 2026-01-13).

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

from qsh.models import datasets, train  # noqa: E402
from qsh.models.registry import get as get_spec  # noqa: E402

#: The percentile a clinical PRS commonly uses as its "high risk" cut. Applied globally on purpose,
#: because applying one threshold to everyone is what actually happens and what breaks.
HIGH_RISK_PERCENTILE = 90.0


def load_dosages(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    rows = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    rsids = rows[0].split("\t")[1:]
    samples = [r.split("\t")[0] for r in rows[1:]]
    dosages = np.asarray([[int(v) for v in r.split("\t")[1:]] for r in rows[1:]], dtype=np.float64)
    return samples, rsids, dosages


def load_weights(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("rsID"):
            continue
        p = line.split("\t")
        out[p[0]] = float(p[3])
    return out


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
    ap.add_argument("--prs", type=Path, default=Path("data/genomics/pgs000001_breast_cancer.tsv"))
    ap.add_argument("--dosages", type=Path, default=None,
                    help="defaults to the full panel if present, else the bundled 400-sample file")
    ap.add_argument("--panel", type=Path, default=Path("data/genomics/g1k_panel.tsv"))
    ap.add_argument("--out", type=Path, default=Path("models/genomics_prs"))
    args = ap.parse_args()

    started = time.time()
    dosage_path = args.dosages
    if dosage_path is None:
        full = Path("data/genomics/g1k_dosages_full.tsv")
        dosage_path = full if full.exists() else Path("data/genomics/g1k_dosages.tsv")
    print(f"dosages: {dosage_path}")

    samples, rsids, dosages = load_dosages(dosage_path)
    weights_by_rsid = load_weights(args.prs)
    w = np.array([weights_by_rsid[r] for r in rsids], dtype=np.float64)
    print(f"  {len(samples)} samples x {len(rsids)} SNPs")

    licence = audit_licence(args.prs)
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
        print(f"\n  A single global threshold flags {fr[hi]*100:.1f}% of {hi} and {fr[lo]*100:.1f}% of {lo}"
              f" — a {fr[hi]/max(fr[lo], 1e-9):.1f}x difference driven entirely by ancestry,"
              " with no phenotype involved.")

    spec = get_spec("genomics.prs")
    print(f"\n  published AUROC (NOT measured here): {spec.metric.baseline} — {spec.metric.baseline_source}")

    # ---- artefacts: the weight vector is the model ----------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weights.json").write_text(json.dumps({
        "score_id": "PGS000001",
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
            "reason": ("1000 Genomes is a reference panel with no breast-cancer case/control status. "
                       "AUROC requires a phenotyped cohort; the published 0.628 comes from n=200,195 "
                       "and is carried as a citation, not as a measurement from this run."),
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
