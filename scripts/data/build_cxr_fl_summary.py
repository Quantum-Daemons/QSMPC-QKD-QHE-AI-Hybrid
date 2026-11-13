#!/usr/bin/env python3
"""Summarise the chest X-ray federation's runs (Round 86): `scripts/train/medical_cxr_fl.py`'s report.json per training
seed into the measurement record and the medical page's card data.

  docs/measurements/cxr-fl-2026-11-14.json     every run's report, and the summary over seeds
  web/public/data/medical/cxr_fl.json           the summary, the hospitals and the attribution: what the card shows

Aggregate figures only: no image, no row of the index and no model weight. The hospitals' split is the same in every run
(one split seed), so it is read from the first; a figure is summarised over the seeds that ran its arm.

Usage: python -I scripts/data/build_cxr_fl_summary.py <run dir> [<run dir> ...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "docs" / "measurements" / "cxr-fl-2026-11-14.json"
CARD = REPO / "web" / "public" / "data" / "medical" / "cxr_fl.json"
ATTRIBUTION = {
    "provider": "NIH Clinical Center",
    "dataset": "ChestX-ray14",
    "download": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
    "citation": ("Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and "
                 "Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. CVPR 2017."),
    "terms": "no restrictions on use; a link to the download site, the citation and an acknowledgement of the provider are asked",
}


def _stats(vals: list[float]) -> dict:
    out = {"mean": statistics.fmean(vals), "per_seed": vals}
    out["sd"] = statistics.stdev(vals) if len(vals) > 1 else None
    return out


CAM_KEYS = ("pointing_hit_rate", "pointing_hit_rate_15px", "mass_in_box", "box_area_share")


def _cam_stats(cams: list[dict]) -> dict:
    return {part: {k: _stats([c[part][k] for c in cams]) for k in CAM_KEYS} | {"n": cams[0][part]["n"]} for part in ("all", "PA", "AP")}


def _public_environment(env: dict) -> dict:
    """The environment as recorded, with library paths cut to their file names: no local directory reaches the record."""
    out = dict(env)
    out["cublas_loaded"] = [Path(p).name for p in env.get("cublas_loaded", [])]
    return out


def summarise(reports: list[dict]) -> dict:
    arms: dict[str, list[dict]] = {}
    for r in reports:
        for a in r["arms"]:
            arms.setdefault(a["arm"], []).append(a)
    out = {}
    for name, runs in arms.items():
        s = {"seeds": len(runs)}
        for part in ("overall", "PA", "AP"):
            s[f"auroc_{part}"] = _stats([x[part]["auroc"] for x in runs])
            s[f"interval_{part}_first_seed"] = [runs[0][part]["lo"], runs[0][part]["hi"]]
            s[f"specificity_{part}"] = _stats([x["at_threshold"][part]["specificity"] for x in runs])
            s[f"sensitivity_{part}"] = _stats([x["at_threshold"][part]["sensitivity"] for x in runs])
        s["score_view_auroc_negatives"] = _stats([x["score_view_auroc_negatives"] for x in runs])
        s["score_view_auroc_positives"] = _stats([x["score_view_auroc_positives"] for x in runs])
        if all("view_probe_auroc" in x for x in runs):
            s["view_probe_auroc"] = _stats([x["view_probe_auroc"] for x in runs])
        if all("cam" in x for x in runs):
            s["cam"] = _cam_stats([x["cam"] for x in runs])
        if "params" in runs[0]:
            s["params"] = runs[0]["params"]
        s["train_s"] = _stats([x["train_s"] for x in runs])
        out[name] = s
    return out


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        raise SystemExit(__doc__)
    reports = []
    for d in dirs:
        with open(d / "report.json") as f:
            r = json.load(f)
        r["environment"] = _public_environment(r["environment"])
        reports.append(r)
    first = reports[0]
    for r in reports[1:]:
        if r["hospitals"] != first["hospitals"] or r["split_seed"] != first["split_seed"]:
            raise SystemExit("the runs do not share one split of the hospitals")
    summary = summarise(reports)
    data = {"test": first["test"], "hospitals": first["hospitals"], "dropped_duplicates": first["dropped_duplicates"],
            "view_only_auroc": first["view_only_auroc"], "split_seed": first["split_seed"], "sharpness": first["sharpness"],
            "boxes": first["boxes"]}
    if any("untrained" not in r for r in reports):
        raise SystemExit("every run needs its untrained baseline (--arms untrained --update)")
    untrained = {"view_probe_auroc": _stats([r["untrained"]["view_probe_auroc"] for r in reports]),
                 "cam": _cam_stats([r["untrained"]["cam"] for r in reports])}
    recipe = {k: first[k] for k in ("epochs", "batch", "lr", "normalisation")}
    recipe["seeds"] = [r["seed"] for r in reports]
    record = {"_about": "The chest X-ray federation on NIH ChestX-ray14, Effusion (Round 86). Written by scripts/data/build_cxr_fl_summary.py "
                        "from scripts/train/medical_cxr_fl.py's reports; aggregate figures only.",
              "attribution": ATTRIBUTION, "data": data, "recipe": recipe, "environment": first["environment"],
              "untrained_view_probe_auroc": untrained, "summary": summary, "runs": reports}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORD, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    card = {"_about": "What the medical page's chest X-ray card shows; written by scripts/data/build_cxr_fl_summary.py from "
                      "docs/measurements/cxr-fl-2026-11-14.json. Aggregate figures only.",
            "attribution": ATTRIBUTION, "data": data, "recipe": recipe, "untrained_view_probe_auroc": untrained, "summary": summary,
            "environment": {k: first["environment"].get(k) for k in ("gpu", "driver", "torch", "torch_cuda", "cublas_version", "kernel")}}
    CARD.parent.mkdir(parents=True, exist_ok=True)
    with open(CARD, "w") as f:
        json.dump(card, f, indent=1)
        f.write("\n")
    print(f"wrote {RECORD.relative_to(REPO)} and {CARD.relative_to(REPO)} from {len(reports)} runs")


if __name__ == "__main__":
    main()
