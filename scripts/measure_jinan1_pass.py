#!/usr/bin/env python3
"""The satellite downlink against Jinan-1's measured pass, as pre-registered (Round 86, batch 6).

references/satellite-jinan1-pass-preregistration.md (and its addendum) fixed everything below before this ran, on the
owner's approval of the download (2026-11-25): the seconds compared (those with a QBER in the paper's Fig. 4c), the
background, the residual in dB, the one constant refitted by bisection, the falls with elevation, and which QBER and
which key the model's are. The model is the page's `jinan1_2022_jinan` profile, through the Python twin of its link
budget and, for the pass key, the page's own TypeScript in node.

Reads `$QSH_DATA_ROOT/raw/jinan1-qkd/` (Zenodo 14732295, CC BY 4.0), each file checked against the MD5 its record
gives. The workbooks are read with the standard library as data: no macro, formula or external link is evaluated.
Writes docs/measurements/satellite-jinan1-pass-2026-11-26.json, and a page file only where `--site-out` names one.
Plaintext only.

`--post-hoc` runs the pre-registration's addendum 2 instead, which the run showed was needed and which was chosen
after its figures were seen: the background from the file's first and last 30 seconds, and the link seconds by the
counts. It writes docs/measurements/satellite-jinan1-pass-posthoc-2026-11-26.json, and the page file only to the path
`--site-out` names: whether the page shows a post-hoc comparison is the owner's choice.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/measure_jinan1_pass.py [--post-hoc [--site-out PATH]]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets as D  # noqa: E402
from qsh.orchestration.stages import Context  # noqa: E402
from qsh.qkd.bb84 import signal_qber  # noqa: E402
from qsh.qkd.channel import air_mass, elevation_deg, intrinsic_qber, link_elevation_deg  # noqa: E402
from qsh.qkd.finite_key import finite_key_over  # noqa: E402
from qsh.qkd.satellite_pass import simulate_pass  # noqa: E402
from qsh.usecases.satellite_comms import _sat_channel, _sat_finite_key_params, _sat_source  # noqa: E402

PREREG = REPO / "references" / "satellite-jinan1-pass-preregistration.md"
PREREG_SHA256 = "11665e2b800dd3f921d9b4bf5122252354e95d85fe773e64130c4ae284ae1984"
RECORD = REPO / "docs" / "measurements" / "satellite-jinan1-pass-2026-11-26.json"
POST_HOC_RECORD = REPO / "docs" / "measurements" / "satellite-jinan1-pass-posthoc-2026-11-26.json"
#: Addendum 2 (post hoc): the seconds at each end of the file the background is taken from, and the multiple of it above
#: which a second's counts mark the link.
POST_HOC_END_SECONDS = 30
POST_HOC_LINK_MULTIPLE = 5.0
WEB = REPO / "web"
PROFILE = "jinan1_2022_jinan"
PROFILE_EFFICIENCY = 0.1683
EDGE_SECONDS = 10
RULE_1_DB = 0.1
RULE_2_DB = 0.5
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXCEL_EPOCH = dt.datetime(1899, 12, 30)
CITATION = ("Li et al., 'Microsatellite-based real-time quantum key distribution', Nature 640, 47-54 (2025), "
            "arXiv:2408.10994. Data: its source data, Zenodo doi:10.5281/zenodo.14732295, CC BY 4.0.")


def sheets(path: Path) -> dict[str, list[list[str | None]]]:
    """Every sheet's cells as text, by the workbook's own XML; nothing in the file is run."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", NS)]
    out = {}
    for i, name in enumerate(names, 1):
        rows = []
        for r in ET.fromstring(z.read(f"xl/worksheets/sheet{i}.xml")).iter("{%s}row" % NS["m"]):
            cells = {}
            for c in r.findall("m:c", NS):
                v = c.find("m:v", NS)
                if v is None:
                    continue
                col = 0
                for ch in re.match(r"[A-Z]+", c.get("r")).group(0):
                    col = col * 26 + ord(ch) - 64
                cells[col - 1] = shared[int(v.text)] if c.get("t") == "s" else v.text
            if cells:
                rows.append([cells.get(j) for j in range(max(cells) + 1)])
        out[name] = rows
    return out


def second_of_serial(v: str) -> int:
    return round(float(v) * 86400)


def second_of_text(v: str) -> int:
    t = dt.datetime.strptime(v.strip().rstrip("'"), "%Y/%m/%d %H:%M:%S")
    return round((t - EXCEL_EPOCH).total_seconds())


def first_by_second(pairs) -> dict[int, tuple]:
    """The first row of each second, and how many rows repeated a second."""
    out: dict[int, tuple] = {}
    for s, *vals in pairs:
        out.setdefault(s, tuple(vals))
    return out


def link(km: float, efficiency: float | None = None):
    ctx = Context(usecase="satellite_comms", config={"sat_profile": PROFILE, "sat_distance_km": km,
                                                      "sat_min_elevation_deg": 0.0})
    ch, src = _sat_channel(ctx), _sat_source(ctx)
    if efficiency is not None:
        ch.free_space.receiver_optics_efficiency = efficiency
    return ch, src


_LINKS: dict[float, tuple] = {}


def detections(km: float, efficiency: float) -> float:
    """Detections per second at one range, as tests/test_the_model_sits_among_micius_nights.py computes them.

    One channel per range, built once; only the station efficiency changes between calls."""
    if km not in _LINKS:
        _LINKS[km] = link(km)
    ch, src = _LINKS[km]
    ch.free_space.receiver_optics_efficiency = efficiency
    eta, y0 = ch.transmittance(), 2 * ch.dark_count_prob
    q = lambda mu: y0 + 1 - math.exp(-eta * mu)  # noqa: E731
    return (src.p_signal * q(src.mu_signal) + src.p_decoy * q(src.mu_decoy)
            + (1 - src.p_signal - src.p_decoy) * y0) * ch.source_rate_hz


def signal_detections(km: float, efficiency: float) -> float:
    """`detections` without the model's own background yield (its dark-count term, about 325 per second here).

    Added after the run (addendum 2's checks): the measured side has the whole background taken off, so the comparison
    like for like is with the model's signal alone; the pre-registered comparison keeps the test's formula."""
    if km not in _LINKS:
        _LINKS[km] = link(km)
    ch, src = _LINKS[km]
    ch.free_space.receiver_optics_efficiency = efficiency
    eta = ch.transmittance()
    q = lambda mu: 1 - math.exp(-eta * mu)  # noqa: E731
    return (src.p_signal * q(src.mu_signal) + src.p_decoy * q(src.mu_decoy)) * ch.source_rate_hz


def db(x: float) -> float:
    return 10 * math.log10(x)


def page_pass_key(top: float) -> float:
    """The page's own pass key: `simulatePass(jinan, top).keyBits` from satNetwork.ts, run in node."""
    esbuild = WEB / "node_modules" / ".bin" / "esbuild"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/satNetwork'))};\n"
                         f"export {{ satChannel }} from {json.dumps(str(WEB / 'src/lib/satLink'))};\n")
        subprocess.run([str(esbuild), str(entry), "--bundle", "--format=esm", "--platform=node", f"--outfile={bundle}",
                        "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n"
                          f"const j = m.satChannel({{ sat_profile: {json.dumps(PROFILE)} }});\n"
                          f"console.log(JSON.stringify(m.simulatePass(j, {top!r}).keyBits));\n")
        return float(subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True).stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--post-hoc", action="store_true", help="addendum 2's windows, chosen after the run")
    ap.add_argument("--site-out", type=Path, help="where to write the page file (none without it)")
    args = ap.parse_args()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    folder = D.raw("jinan1-qkd")
    prov = json.loads((folder / "PROVENANCE.json").read_text())
    for f in prov["files"]:
        md5 = hashlib.md5((folder / f["name"]).read_bytes()).hexdigest()
        if f"md5:{md5}" != f["checksum"]:
            raise SystemExit(f"{f['name']}: MD5 {md5} is not the record's {f['checksum']}: run nothing")

    fig4 = sheets(folder / "Data_Figure4.xlsx")
    b, c = fig4["4b"][1:], fig4["4c"][1:]
    counts_rows = [(second_of_serial(r[0]), float(r[1])) for r in b if r and r[0] and r[1] is not None]
    range_rows = [(second_of_text(r[4]), float(r[5])) for r in b if len(r) > 5 and r[4] and r[5] is not None]
    qkd_rows = [(second_of_serial(r[0]), float(r[1]), float(r[2])) for r in c if r and r[0] and r[1] is not None]
    counts, rng, qkd = first_by_second(counts_rows), first_by_second(range_rows), first_by_second(qkd_rows)
    seconds = sorted(set(counts) & set(rng) & set(qkd))
    dropped = {"counts_rows": len(counts_rows), "range_rows": len(range_rows), "qkd_rows": len(qkd_rows),
               "repeated_seconds": {"counts": len(counts_rows) - len(counts), "range": len(range_rows) - len(rng),
                                    "qkd": len(qkd_rows) - len(qkd)},
               "seconds_joined": len(seconds),
               "seconds_without_a_match": len((set(counts) | set(rng) | set(qkd)) - set(seconds))}
    if args.post_hoc:
        outside = seconds[:POST_HOC_END_SECONDS] + seconds[-POST_HOC_END_SECONDS:]
        background = statistics.median(counts[s][0] for s in outside)
        link_s = [s for s in seconds if counts[s][0] > POST_HOC_LINK_MULTIPLE * background]
        first, last = link_s[0], link_s[-1]
    else:
        link_s = [s for s in seconds if qkd[s][0] > 0]
        first, last = link_s[0], link_s[-1]
        outside = [s for s in seconds if s < first - EDGE_SECONDS or s > last + EDGE_SECONDS]
        background = statistics.median(counts[s][0] for s in outside)
    measured = {s: counts[s][0] - background for s in link_s}
    used = [s for s in link_s if measured[s] > 0]
    km = {s: rng[s][0] for s in used}

    def residuals(eff: float) -> list[float]:
        return [db(measured[s] / detections(km[s], eff)) for s in used]

    def summary(r: list[float]) -> dict:
        return {"mean_db": statistics.fmean(r), "sd_db": statistics.pstdev(r),
                "rms_db": math.sqrt(statistics.fmean(x * x for x in r))}

    before = residuals(PROFILE_EFFICIENCY)
    lo, hi = 1e-4, 1.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if statistics.fmean(residuals(mid)) > 0:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + 1e-12:
            break
    fit = math.sqrt(lo * hi)
    after = residuals(fit)
    fit_db = db(fit / PROFILE_EFFICIENCY)

    ch0, src0 = link(rng[used[0]][0])
    fs = ch0.free_space
    elev = {s: elevation_deg(km[s], fs.orbit_altitude_km, fs.station_altitude_km) for s in used}
    am = [air_mass(elev[s]) for s in used]
    am_mean, r_mean = statistics.fmean(am), statistics.fmean(after)
    slope = sum((a - am_mean) * (r - r_mean) for a, r in zip(am, after, strict=True)) / sum((a - am_mean) ** 2 for a in am)

    by_range = sorted(used, key=lambda s: km[s])[:EDGE_SECONDS]
    falls = {}
    for name, edge in (("first", used[:EDGE_SECONDS]), ("last", used[-EDGE_SECONDS:])):
        meas = statistics.fmean(db(measured[s]) for s in by_range) - statistics.fmean(db(measured[s]) for s in edge)
        model = statistics.fmean(db(detections(km[s], PROFILE_EFFICIENCY)) for s in by_range) - statistics.fmean(
            db(detections(km[s], PROFILE_EFFICIENCY)) for s in edge)
        falls[name] = {"seconds": [edge[0] - first, edge[-1] - first], "mean_range_km": statistics.fmean(km[s] for s in edge),
                       "measured_db": meas, "model_db": model, "measured_minus_model_db": meas - model,
                       "within_0_5_db": abs(meas - model) <= RULE_2_DB}

    def qber_at(s: int) -> tuple[float, float]:
        """The profile as it stands, at that second's range: the signal state's QBER and the analytic baseline."""
        ch, src = link(km[s], PROFILE_EFFICIENCY)
        return signal_qber(ch, src.mu_signal), intrinsic_qber(ch)

    q_both = {s: qber_at(s) for s in used}
    q_model = {s: v[0] for s, v in q_both.items()}
    q_base = {s: v[1] for s, v in q_both.items()}
    # The QBER is one value per key block, held over its seconds (found by the run): the comparison takes the link
    # seconds that carry one, and names the blocks.
    qs = [s for s in used if qkd[s][0] > 0]
    blocks: list[dict] = []
    for s in qs:
        if not blocks or blocks[-1]["qber"] != qkd[s][0]:
            blocks.append({"qber": qkd[s][0], "seconds": 0, "model_signal_mean": 0.0})
        blocks[-1]["seconds"] += 1
        blocks[-1]["model_signal_mean"] += q_model[s]
    for blk in blocks:
        blk["model_signal_mean"] /= blk["seconds"]
    qber = {"seconds_with_a_block_qber": len(qs), "blocks": blocks,
            "measured_mean": statistics.fmean(qkd[s][0] for s in qs),
            "model_signal_mean": statistics.fmean(q_model[s] for s in qs),
            "model_baseline_mean": statistics.fmean(q_base[s] for s in qs)}
    qber["model_signal_minus_measured"] = qber["model_signal_mean"] - qber["measured_mean"]
    ext = sheets(folder / "Data_Extented_Figure2.xlsx")
    noise = [float(r[2]) for r in next(iter(ext.values()))[1:] if len(r) > 2 and r[2] is not None]
    qber["noise_share_extended_data_fig_2"] = {"rows": len(noise), "mean": statistics.fmean(noise),
                                               "min": min(noise), "max": max(noise)}

    measured_key = max(qkd[s][1] for s in seconds)
    ctx = Context(usecase="satellite_comms", config={"sat_profile": PROFILE})
    chp, srcp = _sat_channel(ctx), _sat_source(ctx)
    top = link_elevation_deg(chp.free_space, chp.distance_km)
    asym = page_pass_key(top)
    fin = finite_key_over(simulate_pass(chp, top).segments, _sat_finite_key_params(ctx, srcp)).key_bits
    key = {"measured_bits": measured_key, "model_pass_key_bits": asym, "model_finite_key_bits": fin,
           "model_over_measured": asym / measured_key, "finite_over_measured": fin / measured_key,
           "top_elevation_deg": top}

    # Post hoc only: how far the figures move with the two things addendum 2 chose and with the model's background term.
    robustness = None
    if args.post_hoc:
        def stats_for(bg: float, mult: float, fn) -> dict:
            ls = [s for s in seconds if counts[s][0] > mult * bg and counts[s][0] - bg > 0]
            r = [db((counts[s][0] - bg) / fn(rng[s][0], PROFILE_EFFICIENCY)) for s in ls]
            pk = sorted(ls, key=lambda s: rng[s][0])[:EDGE_SECONDS]

            def fall(edge: list[int]) -> float:
                m = statistics.fmean(db(counts[s][0] - bg) for s in pk) - statistics.fmean(db(counts[s][0] - bg) for s in edge)
                return m - (statistics.fmean(db(fn(rng[s][0], PROFILE_EFFICIENCY)) for s in pk)
                            - statistics.fmean(db(fn(rng[s][0], PROFILE_EFFICIENCY)) for s in edge))
            return {"background_per_s": bg, "link_seconds": len(ls), "mean_db": statistics.fmean(r),
                    "sd_db": statistics.pstdev(r), "fall_first_minus_model_db": fall(ls[:EDGE_SECONDS]),
                    "fall_last_minus_model_db": fall(ls[-EDGE_SECONDS:])}
        grid = []
        for n_end in (20, 30, 40):
            bg = statistics.median(counts[s][0] for s in seconds[:n_end] + seconds[-n_end:])
            for mult in (3.0, 5.0, 10.0):
                grid.append({"end_seconds": n_end, "link_multiple": mult,
                             "model_with_its_background_term": stats_for(bg, mult, detections),
                             "model_signal_only": stats_for(bg, mult, signal_detections)})
        robustness = {"note": ("post hoc: the background's end seconds and the link threshold varied, and the model with "
                               "and without its own background term (the measured side has the whole background off)"),
                      "grid": grid}

    rule_1 = abs(fit_db) > RULE_1_DB
    record = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
        "windows": ("addendum 2, POST HOC: background from the file's first and last 30 seconds, link seconds by the "
                    "counts; chosen after the pre-registered windows' figures were seen" if args.post_hoc else
                    "as pre-registered: link seconds by the QBER column, background from the seconds outside them"),
        "data": {"record": prov["doi"], "licence": "CC-BY-4.0", "files_md5_checked": [f["name"] for f in prov["files"]],
                 "alignment": dropped},
        "link_seconds": {"count": len(link_s), "used": len(used), "duration_s": last - first + 1,
                         "range_km": [min(km.values()), max(km.values())]},
        "background_per_s": background, "background_seconds": len(outside),
        "profile_efficiency": PROFILE_EFFICIENCY, "fitted_efficiency": fit, "fit_minus_profile_db": fit_db,
        "residuals_before": summary(before), "residuals_after": summary(after),
        "trend_after_db_per_air_mass": slope,
        "falls": falls, "qber": qber, "key": key,
        "rules": {"1_profile_takes_the_fit": rule_1, "2_falls_within_0_5_db": all(f["within_0_5_db"] for f in falls.values())},
        **({"robustness": robustness} if robustness else {}),
    }
    (POST_HOC_RECORD if args.post_hoc else RECORD).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("link_seconds", "background_per_s", "fitted_efficiency", "fit_minus_profile_db",
                                             "residuals_before", "residuals_after", "trend_after_db_per_air_mass",
                                             "falls", "qber", "key", "rules")}, indent=1))
    t0 = used[0]
    # The page file only where `--site-out` names one: the pre-registered windows answer nothing (addendum 2), and
    # whether the page shows the post-hoc comparison is the owner's choice.
    site = args.site_out
    if site is None:
        print("wrote the record; no page file")
        return 0
    site.write_text(json.dumps({
        "_generated_by": "scripts/measure_jinan1_pass.py; do not edit by hand",
        "preregistration": record["preregistration"], "windows": record["windows"], "post_hoc": bool(args.post_hoc),
        "record": str((POST_HOC_RECORD if args.post_hoc else RECORD).relative_to(REPO)),
        "pass": "Jinan-1 to Jinan, 25 September 2022, from the paper's Fig. 4b and 4c",
        "seconds": [s - t0 for s in used], "range_km": [round(km[s], 3) for s in used],
        "measured_detections_per_s": [round(measured[s], 1) for s in used],
        "model_detections_per_s": [round(detections(km[s], PROFILE_EFFICIENCY), 1) for s in used],
        "model_detections_per_s_refitted": [round(detections(km[s], fit), 1) for s in used],
        "measured_qber": [qkd[s][0] for s in used], "model_qber": [round(q_model[s], 6) for s in used],
        "summary": {**{k: record[k] for k in ("background_per_s", "profile_efficiency", "fitted_efficiency",
                                              "fit_minus_profile_db", "residuals_before", "residuals_after",
                                              "trend_after_db_per_air_mass", "falls", "qber", "key", "rules")},
                    # The grid in brief, for the card: the spread of the mean residual and the largest fall difference.
                    **({"robustness": {
                        "choices": 2 * len(robustness["grid"]),
                        "mean_residual_db": [min(g[m]["mean_db"] for g in robustness["grid"] for m in g if m.startswith("model_")),
                                             max(g[m]["mean_db"] for g in robustness["grid"] for m in g if m.startswith("model_"))],
                        "largest_fall_difference_db": max(abs(g[m][f]) for g in robustness["grid"] for m in g
                                                          if m.startswith("model_") for f in ("fall_first_minus_model_db",
                                                                                              "fall_last_minus_model_db"))}}
                       if robustness else {})},
        "licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/", "attribution": CITATION,
        "modifications": (f"Counts and slant ranges per second from Fig. 4b and QBER from Fig. 4c, joined on the second "
                          f"(the first row of a repeated second kept); the {len(used)} seconds with a QBER and counts above "
                          f"the background; the background, {background:g} per second, subtracted; the model's figures "
                          "beside them are this project's."),
    }, indent=1) + "\n")
    print(f"wrote the record and {site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
