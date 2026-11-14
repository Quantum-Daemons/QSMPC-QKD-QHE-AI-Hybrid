#!/usr/bin/env python3
"""Summarise the murmur study under differential privacy (Round 86): every tuning route side by side, into the
measurement record and the medical page's card data.

  docs/measurements/murmur-dp-2026-11-14.json     each route's report and summary (the search's K and losing
                                                  candidates are left out: Papernot and Steinke's guarantee covers the
                                                  best candidate only)
  web/public/data/medical/murmur_dp.json           the summaries, the counts and the attribution: what the card shows

Routes: `circor` (rates chosen on CirCor validation people; the stated epsilon leaves the choice out), `public` (rates
and thresholds chosen on the PhysioNet 2016 stand-in from databases the feature CNN never saw: the headline), `naive`
(a stand-in from the feature CNN's own databases, which failed), `search` (the accounted random search; one directory a
model) and `objpert` (objective perturbation). Aggregate figures only: no recording, window, ID or weight.

Usage: python -I scripts/data/build_murmur_dp_summary.py --route public=<dir> --route circor=<dir> --route naive=<dir>
       --route search=<dir> --route search=<dir> --route objpert=<dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "docs" / "measurements" / "murmur-dp-2026-11-14.json"
CARD = REPO / "web" / "public" / "data" / "medical" / "murmur_dp.json"
ATTRIBUTION = {
    "circor": {
        "provider": "PhysioNet", "dataset": "The CirCor DigiScope Phonocardiogram Dataset, version 1.0.3",
        "download": "https://physionet.org/content/circor-heart-sound/1.0.3/", "dataset_doi": "10.13026/tshs-mw03",
        "citation": ("Oliveira J, Renna F, Costa P, Nogueira D, Oliveira C, Ferreira C, Jorge A, Mattos S, Hatem T, Tavares T, Elola A, "
                     "Rad A, Sameni R, Clifford GD, Coimbra MT. The CirCor DigiScope Dataset: From Murmur Detection to Murmur "
                     "Classification. IEEE Journal of Biomedical and Health Informatics. doi:10.1109/JBHI.2021.3137048"),
        "licence": "Open Data Commons Attribution License v1.0", "licence_url": "https://opendatacommons.org/licenses/by/1-0/"},
    "physionet_2016": {
        "dataset": "Classification of Heart Sound Recordings: the PhysioNet/Computing in Cardiology Challenge 2016, version 1.0.0",
        "download": "https://physionet.org/content/challenge-2016/1.0.0/",
        "citation": ("Liu C, Springer D, Li Q, Moody B, Juan RA, et al. An open access database for the evaluation of heart sound "
                     "algorithms. Physiol Meas. 2016;37(12):2181-2213. doi:10.1088/0967-3334/37/12/2181"),
        "licence": "Open Data Commons Attribution License v1.0"},
    "panns": {
        "model": "PANNs CNN14 (Cnn14_mAP=0.431.pth), trained on AudioSet", "download": "https://zenodo.org/records/3987831",
        "citation": ("Kong Q, Cao Y, Iqbal T, Wang Y, Wang W, Plumbley MD. PANNs: Large-Scale Pretrained Audio Neural Networks for "
                     "Audio Pattern Recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020). "
                     "Weights doi:10.5281/zenodo.3987831"),
        "licence": "Creative Commons Attribution 4.0 International", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "use": "frozen feature extractor, offline; the weights are not redistributed"},
    "platform_citation": ("Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T. PhysioNet as a global "
                          "platform for biomedical research. Nature Health. 2026;1(8):792-795. doi:10.1038/s44360-026-00096-z"),
}
ROUTES = ("public", "circor", "naive", "search", "objpert")


def _stats(vals: list[float]) -> dict:
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else None}


def _thresholds(rs: list[dict]) -> dict:
    out = {}
    for name in ("zero", "prevalence", "balanced"):
        got = [r["at_thresholds"][name] for r in rs if name in r.get("at_thresholds", {})]
        if got:
            out[name] = {"balanced_accuracy": _stats([g["balanced_accuracy"] for g in got]),
                         "share_predicted_positive": _stats([g["share_predicted_positive"] for g in got])}
    return out


def summarise_study(report: dict) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in report["runs"]:
        groups.setdefault((r["model"], r["arm"], r["kind"], r["epsilon"]), []).append(r)
    chosen = {(s["model"], s["arm"], s["kind"], s["epsilon"]): s for s in report["selection"]}
    out = []
    for (model, arm, kind, eps), rs in groups.items():
        sel = chosen[(model, arm, kind, eps)]
        row = {"model": model, "arm": arm, "kind": kind, "epsilon": eps, "seeds": len(rs),
               "auroc": _stats([r["auroc"]["auroc"] for r in rs]), "lr": sel["chosen_lr"],
               "lr_grid": [t["lr"] for t in sel["tried"]], "thresholds_from": sel.get("thresholds"),
               "at_thresholds": _thresholds(rs)}
        for k in ("sigma", "variance_per_site", "noise_sd_over_gamma_c", "epsilon_accounted"):
            if k in rs[0]:
                row[k] = rs[0][k]
        out.append(row)
    return out


def summarise_search(reports: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for rep in reports:
        for r in rep["runs"]:
            groups.setdefault((r["model"], r["arm"], r["epsilon"]), []).append(r)
    return [{"model": m, "arm": a, "kind": "DP", "epsilon": e, "searches": len(rs),
             "auroc": _stats([r["auroc"]["auroc"] for r in rs]), "grid": next(rep["grid"] for rep in reports if any(x in rep["runs"] for x in rs)),
             "noise": rs[0]["noise"], "score_sigma": rs[0]["score_sigma"], "epsilon_accounted": rs[0]["epsilon_accounted"],
             "at_thresholds": {"zero": {"balanced_accuracy": _stats([r["balanced_accuracy"] for r in rs]),
                                        "share_predicted_positive": _stats([r["share_predicted_positive"] for r in rs])}}}
            for (m, a, e), rs in groups.items()]


def summarise_objpert(report: dict) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in report["runs"]:
        groups.setdefault((r["features"], r["epsilon"]), []).append(r)
    return [{"model": f, "arm": "central", "kind": "DP" if e else "non-private", "epsilon": e, "seeds": len(rs),
             "auroc": _stats([r["auroc"]["auroc"] for r in rs]), "sigma": rs[0]["sigma"], "epsilon_accounted": rs[0]["epsilon_accounted"],
             "at_thresholds": _thresholds(rs)} for (f, e), rs in groups.items()]


def rounded(x, digits: int):
    """Floats to `digits` significant figures, recursively: the record keeps what a reader can check, not float noise."""
    if isinstance(x, float):
        return float(f"{x:.{digits}g}")
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v, digits) for v in x]
    return x


def strip_local(report: dict) -> dict:
    """A run's report without the search's K and losing candidates, and without the environment the record states once."""
    out = {k: v for k, v in report.items() if k not in ("environment", "runs")}
    out["runs"] = [{k: v for k, v in r.items() if k not in ("local_only", "tried", "k", "train_s")} for r in report["runs"]]
    return rounded(out, 6)


def card_rows(summary: dict) -> dict:
    """What the card draws: every arm of the headline route, the central arm of the others."""
    return {name: [r for r in rows if name == "public" or r["arm"] == "central"] for name, rows in summary.items()}


def main() -> None:
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("--route", action="append", required=True, metavar="NAME=DIR")
    a = ap.parse_args()
    dirs: dict[str, list[Path]] = {}
    for item in a.route:
        name, _, path = item.partition("=")
        if name not in ROUTES:
            raise SystemExit(f"unknown route {name}; one of {ROUTES}")
        dirs.setdefault(name, []).append(Path(path))
    reports = {name: [json.loads((p / "report.json").read_text()) for p in ps] for name, ps in dirs.items()}
    head = reports["public"][0]
    for name in ("circor", "naive"):
        for r in reports.get(name, []):
            if r["data"] != head["data"]:
                raise SystemExit(f"route {name} did not use the headline's split")
    summary = {"public": summarise_study(head)}
    for name in ("circor", "naive"):
        if name in reports:
            summary[name] = summarise_study(reports[name][0])
    if "search" in reports:
        summary["search"] = summarise_search(reports["search"])
    if "objpert" in reports:
        summary["objpert"] = summarise_objpert(reports["objpert"][0])
    recipe = {k: head[k] for k in ("delta", "clip", "windows_per_record", "pos_weight", "expected_batch", "epochs", "sites",
                                   "scale_bits", "params")}
    recipe["public_cnn"] = head.get("public_cnn")
    recipe["stand_in"] = head.get("tuned_on")
    meta = {"search": {k: reports["search"][0][k] for k in ("eta", "expected_k", "score_sensitivity", "val_records")}} if "search" in reports else {}
    if "objpert" in reports:
        meta["objpert"] = {k: reports["objpert"][0][k] for k in ("lambda", "tau", "sigma_out", "beta", "C")}
    record = {"_about": "Murmur detection on CirCor under differential privacy, the person as the unit, every tuning route side by "
                        "side (Round 86). Written by scripts/data/build_murmur_dp_summary.py; aggregate figures only.",
              "attribution": ATTRIBUTION, "data": head["data"], "recipe": recipe, "routes": meta, "environment": head["environment"],
              "summary": summary,
              "reports": {name: [strip_local(r) for r in rs] for name, rs in reports.items()}}
    card = {"_about": "What the medical page's murmur card shows; written by scripts/data/build_murmur_dp_summary.py from "
                      "docs/measurements/murmur-dp-2026-11-14.json. Aggregate figures only.",
            "attribution": ATTRIBUTION, "data": head["data"], "recipe": recipe, "routes": meta, "summary": summary,
            "environment": {k: head["environment"].get(k) for k in ("gpu", "driver", "torch", "torch_cuda", "cublas_version", "kernel")}}
    card["summary"] = rounded(card_rows(summary), 4)
    record["summary"] = rounded(summary, 6)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with open(RECORD, "w") as f:
        head_part = {k: v for k, v in record.items() if k != "reports"}
        text = json.dumps(head_part, indent=1)[:-2]
        lines = [f'"{name}": [' + ", ".join(json.dumps({k: v for k, v in r.items() if k != "runs"}) [:-1] + ', "runs": [\n'
                 + ",\n".join(json.dumps(x, separators=(",", ":")) for x in r["runs"]) + "]}" for r in reps) + "]"
                 for name, reps in record["reports"].items()]
        f.write(text + ',\n "reports": {\n' + ",\n".join(lines) + "\n }\n}\n")
    json.loads(RECORD.read_text())
    CARD.parent.mkdir(parents=True, exist_ok=True)
    with open(CARD, "w") as f:
        json.dump(card, f, indent=1)
        f.write("\n")
    print(f"wrote {RECORD.relative_to(REPO)} and {CARD.relative_to(REPO)} from routes {', '.join(summary)}")


if __name__ == "__main__":
    main()
