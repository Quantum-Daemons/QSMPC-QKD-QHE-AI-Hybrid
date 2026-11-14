#!/usr/bin/env python3
"""Summarise the heart-sound federation's runs (Round 86): `scripts/train/medical_pcg_fl.py`'s report.json into the
measurement record and the medical page's card data.

  docs/measurements/pcg-fl-2026-11-14.json     the report, and the summary over seeds
  web/public/data/medical/pcg_fl.json           the summary, the sources and the attribution: what the card shows

Aggregate figures only: no recording, no window, no record name and no model weight.

Usage: python -I scripts/data/build_pcg_fl_summary.py <run dir> [--band-mean <run dir>]

The second run is the same study with `--band-mean` (each mel band's mean removed over the window): the record keeps its
runs and summary under `band_mean`, and the card its summary, beside the main run's.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "docs" / "measurements" / "pcg-fl-2026-11-14.json"
CARD = REPO / "web" / "public" / "data" / "medical" / "pcg_fl.json"
ATTRIBUTION = {
    "provider": "PhysioNet",
    "dataset": "Classification of Heart Sound Recordings: the PhysioNet/Computing in Cardiology Challenge 2016, version 1.0.0",
    "download": "https://physionet.org/content/challenge-2016/1.0.0/",
    "citation": ("Liu C, Springer D, Li Q, Moody B, Juan RA, et al. An open access database for the evaluation of heart sound "
                 "algorithms. Physiol Meas. 2016;37(12):2181-2213. doi:10.1088/0967-3334/37/12/2181"),
    "platform_citation": "Pollard T, Moody BE, Lehman L, Gow B, et al. PhysioNet as a global platform for biomedical research. Nature Health. 2026",
    "licence": "Open Data Commons Attribution License v1.0",
    "licence_url": "https://opendatacommons.org/licenses/by/1-0/",
}


def _stats(vals: list[float]) -> dict:
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else None, "per_seed": vals}


def summarise(runs: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["held_out"], r["arm"]), []).append(r)
    out = {}
    for (held, arm), rs in groups.items():
        out.setdefault(held, {})[arm] = {
            "seeds": len(rs), "auroc": _stats([r["auroc"]["auroc"] for r in rs]),
            "interval_first_seed": [rs[0]["auroc"]["lo"], rs[0]["auroc"]["hi"]],
            "auroc_good_quality": _stats([r["auroc_good_quality"]["auroc"] for r in rs if r["auroc_good_quality"]]),
            "macc": _stats([r["macc"] for r in rs]), "sensitivity": _stats([r["sensitivity"] for r in rs]),
            "specificity": _stats([r["specificity"] for r in rs]), "train_s": _stats([r["train_s"] for r in rs]),
        }
    return out


def _recipe(report: dict) -> dict:
    recipe = {k: report[k] for k in ("epochs", "batch", "lr", "window_s", "features", "params")}
    recipe["seeds"] = sorted({r["seed"] for r in report["runs"]})
    return recipe


def main() -> None:
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--band-mean", type=Path, help="the same study run with --band-mean")
    a = ap.parse_args()
    with open(a.run / "report.json") as f:
        report = json.load(f)
    summary = summarise(report["runs"])
    recipe = _recipe(report)
    band = None
    if a.band_mean:
        with open(a.band_mean / "report.json") as f:
            other = json.load(f)
        if not other["features"].get("band_mean_removed") or report["features"].get("band_mean_removed"):
            raise SystemExit("--band-mean must name the run with band means removed, and the main run the one without")
        if other["sources"] != report["sources"]:
            raise SystemExit("the two runs did not read the same recordings")
        band = {"recipe": _recipe(other), "summary": summarise(other["runs"]), "environment": other["environment"], "runs": other["runs"]}
    counts = {"released_zip": sum(s["recordings"] for s in report["sources"].values()), "physionet_page_introduction": 3126,
              "physionet_page_announcement": 3125, "cinc_2016_paper_table_1": 3153}
    record = {"_about": "The heart-sound federation on the PhysioNet/CinC Challenge 2016 data, each source held out in turn (Round 86). "
                        "Written by scripts/data/build_pcg_fl_summary.py from scripts/train/medical_pcg_fl.py's report; aggregate figures only.",
              "attribution": ATTRIBUTION, "sources": report["sources"], "training_recording_counts": counts, "recipe": recipe,
              "environment": report["environment"], "summary": summary, "runs": report["runs"]}
    if band:
        record["band_mean"] = band
    for path, body in ((RECORD, record),
                       (CARD, {"_about": "What the medical page's heart-sound card shows; written by scripts/data/build_pcg_fl_summary.py from "
                                         "docs/measurements/pcg-fl-2026-11-14.json. Aggregate figures only.",
                               "attribution": ATTRIBUTION, "sources": report["sources"], "training_recording_counts": counts,
                               "recipe": recipe, "summary": summary,
                               **({"band_mean": {"recipe": band["recipe"], "summary": band["summary"]}} if band else {}),
                               "environment": {k: report["environment"].get(k) for k in ("gpu", "driver", "torch", "torch_cuda", "cublas_version", "kernel")}})):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(body, f, indent=1)
            f.write("\n")
    extra = f" and {len(band['runs'])} with band means removed" if band else ""
    print(f"wrote {RECORD.relative_to(REPO)} and {CARD.relative_to(REPO)} from {len(report['runs'])} runs{extra}")


if __name__ == "__main__":
    main()
