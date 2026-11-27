#!/usr/bin/env python3
"""The Jinan-1 pass, second analysis, as pre-registered (`references/satellite-jinan1-pass-v2-preregistration.md`;
Round 86, batch 8).

The first analysis's reading of the source data, windows (its addendum 2's, fixed in advance here), model and helpers
are imported from `scripts/measure_jinan1_pass.py`, not typed again. The four questions: the station efficiency and
the zenith atmospheric loss fitted together; the model's sifted bits against the 20 packets, with the lag that aligns
them; the finite key of each 500-kbit block; and the three other passes of the preprint's Table 1. Writes
docs/measurements/satellite-jinan1-pass-v2-2026-11-27.json and web/public/data/satellite/jinan1_pass_v2.json.

Usage: QSH_DATA_ROOT=<data root> python -I scripts/measure_jinan1_pass_v2.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location("measure_jinan1_pass", REPO / "scripts" / "measure_jinan1_pass.py")
J = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(J)

PREREG = REPO / "references" / "satellite-jinan1-pass-v2-preregistration.md"
PREREG_SHA256 = "ddaea49f1f277ef457af9d41c1a775b4134daf0c8aec0dec73ee40b8b338292e"
RECORD = REPO / "docs" / "measurements" / "satellite-jinan1-pass-v2-2026-11-27.json"
SITE = REPO / "web" / "public" / "data" / "satellite" / "jinan1_pass_v2.json"
GATE, SAMPLED_OUT, BASIS = 0.8, 0.9, 0.5
PACKET_BITS, PACKETS_MEASURED = 100_000, 20
#: The preprint's Table 1 (arXiv:2408.10994v1): the other three passes' maximum elevation, packets and secure key bits.
OTHER_PASSES = (("2022-09-20", 41.8, 17, 330_240), ("2022-09-28", 77.7, 31, 571_392), ("2022-09-29", 52.7, 20, 436_096))
RULE_1_SD_DB, RULE_1_ZENITH_DB, RULE_2_FACTOR = 0.05, (0.0, 3.0), 1.5


def main() -> int:
    import numpy as np
    from scipy.optimize import least_squares

    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got}, not the committed {PREREG_SHA256}: run nothing")
    folder = J.D.raw("jinan1-qkd")
    prov = json.loads((folder / "PROVENANCE.json").read_text())
    for f in prov["files"]:
        md5 = hashlib.md5((folder / f["name"]).read_bytes()).hexdigest()
        if f"md5:{md5}" != f["checksum"]:
            raise SystemExit(f"{f['name']}: MD5 {md5} is not the record's {f['checksum']}: run nothing")
    first = json.loads(J.POST_HOC_RECORD.read_text())

    # ---- the source data, read and joined exactly as the first analysis reads it; its addendum 2's windows
    fig4 = J.sheets(folder / "Data_Figure4.xlsx")
    b, c = fig4["4b"][1:], fig4["4c"][1:]
    counts = J.first_by_second([(J.second_of_serial(r[0]), float(r[1])) for r in b if r and r[0] and r[1] is not None])
    rng = J.first_by_second([(J.second_of_text(r[4]), float(r[5])) for r in b if len(r) > 5 and r[4] and r[5] is not None])
    qkd = J.first_by_second([(J.second_of_serial(r[0]), float(r[1]), float(r[2])) for r in c if r and r[0] and r[1] is not None])
    seconds = sorted(set(counts) & set(rng) & set(qkd))
    outside = seconds[:J.POST_HOC_END_SECONDS] + seconds[-J.POST_HOC_END_SECONDS:]
    background = statistics.median(counts[s][0] for s in outside)
    link_s = [s for s in seconds if counts[s][0] > J.POST_HOC_LINK_MULTIPLE * background]
    measured = {s: counts[s][0] - background for s in link_s}
    used = [s for s in link_s if measured[s] > 0]
    km = {s: rng[s][0] for s in used}

    # ---- the model with both constants free: the station efficiency, and the zenith loss scaled by the air mass
    ch0, src = J.link(km[used[0]])
    fs0 = ch0.free_space
    am_ref = J.air_mass(J.elevation_deg(fs0.atmospheric_ref_km, fs0.orbit_altitude_km, fs0.station_altitude_km))
    zenith0 = fs0.atmospheric_db / am_ref

    def detections(k: float, eff: float, zenith: float) -> float:
        if k not in J._LINKS:
            J._LINKS[k] = J.link(k)
        ch, s_ = J._LINKS[k]
        ch.free_space.receiver_optics_efficiency = eff
        ch.free_space.atmospheric_db = zenith * am_ref
        eta, y0 = ch.transmittance(), 2 * ch.dark_count_prob
        q = lambda mu: y0 + 1 - math.exp(-eta * mu)  # noqa: E731
        return (s_.p_signal * q(s_.mu_signal) + s_.p_decoy * q(s_.mu_decoy)
                + (1 - s_.p_signal - s_.p_decoy) * y0) * ch.source_rate_hz

    def signal_state(k: float, eff: float, zenith: float) -> float:
        """Signal-state detections per second, the dark-count yield included as the page's key model counts it."""
        detections(k, eff, zenith)
        ch, s_ = J._LINKS[k]
        eta, y0 = ch.transmittance(), 2 * ch.dark_count_prob
        return s_.p_signal * (y0 + 1 - math.exp(-eta * s_.mu_signal)) * ch.source_rate_hz

    def resid(p) -> np.ndarray:
        eff, zen = 10.0 ** p[0], p[1]
        return np.array([J.db(measured[s] / detections(km[s], eff, zen)) for s in used])

    fit = least_squares(resid, x0=[math.log10(J.PROFILE_EFFICIENCY), zenith0], method="lm")
    r = fit.fun
    n, k_params = len(r), 2
    s2 = float(r @ r) / (n - k_params)
    cov = s2 * np.linalg.inv(fit.jac.T @ fit.jac)
    eff_fit, zen_fit = 10.0 ** fit.x[0], float(fit.x[1])
    se_log, se_zen = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
    q1 = {"fitted_efficiency": eff_fit, "efficiency_se": eff_fit * math.log(10) * se_log,
          "fitted_zenith_loss_db": zen_fit, "zenith_loss_se_db": se_zen,
          "correlation": float(cov[0, 1] / (se_log * se_zen)),
          "profile_efficiency": J.PROFILE_EFFICIENCY, "profile_zenith_loss_db": zenith0,
          "residual_sd_db": statistics.pstdev(r.tolist()), "residual_mean_db": statistics.fmean(r.tolist()),
          "first_analysis_one_parameter_sd_db": first["residuals_after"]["sd_db"],
          "link_seconds_used": len(used)}
    q1["sd_change_db"] = q1["residual_sd_db"] - q1["first_analysis_one_parameter_sd_db"]
    sets = {"profile": (J.PROFILE_EFFICIENCY, zenith0), "fitted": (eff_fit, zen_fit)}

    # ---- question 2: the sifted bits against the 20 packets, and the lag that aligns their boundaries
    factor = GATE * SAMPLED_OUT * BASIS
    q4c = [s for s in seconds if qkd[s][0] > 0]
    boundaries, last_q = [], None
    for s in q4c:
        if qkd[s][0] != last_q:
            boundaries.append(s)
            last_q = qkd[s][0]
    q2: dict = {"measured_sifted_bits": PACKETS_MEASURED * PACKET_BITS, "measured_packet_boundaries": len(boundaries),
                "factors": {"gate": GATE, "not_sampled": SAMPLED_OUT, "basis": BASIS}}
    for name, (eff, zen) in sets.items():
        cum, total, model_bounds = [], 0.0, []
        for s in used:
            total += signal_state(km[s], eff, zen) * factor
            cum.append((s, total))
        for kk in range(1, PACKETS_MEASURED + 1):
            hit = next((s for s, v in cum if v >= kk * PACKET_BITS), None)
            if hit is not None:
                model_bounds.append(hit)
        pairs = list(zip(model_bounds, boundaries, strict=False))
        lag = statistics.fmean(m - mo for mo, m in pairs) if pairs else None
        rms = math.sqrt(statistics.fmean((mo + lag - m) ** 2 for mo, m in pairs)) if pairs else None
        q2[name] = {"model_sifted_bits": total, "model_over_measured": total / (PACKETS_MEASURED * PACKET_BITS),
                    "model_packets_reached": len(model_bounds), "lag_s": lag, "rms_after_lag_s": rms}

    # ---- question 3: the key of each 500-kbit block, against the page's finite bound over its seconds
    keys = [qkd[s][1] for s in seconds]
    steps = [(s, v - p) for s, v, p in zip(seconds[1:], keys[1:], keys[:-1], strict=True) if v > p]
    params = J._sat_finite_key_params(J.Context(usecase="satellite_comms", config={"sat_profile": J.PROFILE}), src)
    q3: dict = {"measured_key_steps": [round(v, 3) for _, v in steps]}
    per_block = 5
    edges = [used[0] - 1] + [boundaries[i * per_block + per_block - 1] for i in range(len(boundaries) // per_block)]
    for name, (eff, zen) in sets.items():
        ratios = []
        for j, (_, key_bits) in enumerate(steps[: len(edges) - 1]):
            secs = [s for s in used if edges[j] < s <= edges[j + 1]]
            segs = []
            for s in secs:
                detections(km[s], eff, zen)
                ch, _ = J._LINKS[km[s]]
                segs.append((ch.source_rate_hz * GATE * SAMPLED_OUT, replace(ch, free_space=replace(ch.free_space))))
            model_key = J.finite_key_over(segs, params).key_bits if segs else 0.0
            ratios.append({"block": j + 1, "seconds": len(secs), "measured_bits": key_bits, "model_bits": model_key,
                           "measured_over_model": key_bits / model_key if model_key > 0 else None})
        good = [x["measured_over_model"] for x in ratios if x["measured_over_model"] is not None]
        q3[name] = {"blocks": ratios, "mean_measured_over_model": statistics.fmean(good) if good else None}

    # ---- question 4: the three other passes, at their maximum elevations, with either set of constants
    ctx = J.Context(usecase="satellite_comms", config={"sat_profile": J.PROFILE})
    chp = J._sat_channel(ctx)
    q4: dict = {"passes": []}
    for date, elev, packets, bits in OTHER_PASSES:
        row = {"date": date, "max_elevation_deg": elev, "published_packets": packets, "published_key_bits": bits}
        for name, (eff, zen) in sets.items():
            fs = replace(chp.free_space, receiver_optics_efficiency=eff, atmospheric_db=zen * am_ref)
            p = J.simulate_pass(replace(chp, free_space=fs), elev)
            sifted = sum(npulses * src.p_signal * (2 * seg.dark_count_prob + 1 - math.exp(-seg.transmittance() * src.mu_signal))
                         for npulses, seg in p.segments) * factor
            scaled = [(npulses * GATE * SAMPLED_OUT, seg) for npulses, seg in p.segments]
            key = J.finite_key_over(scaled, params).key_bits if scaled else 0.0
            row[name] = {"model_packets": sifted / PACKET_BITS, "packets_ratio": (sifted / PACKET_BITS) / packets,
                         "model_key_bits": key, "key_ratio": key / bits if bits else None, "duration_s": p.duration_s}
        q4["passes"].append(row)

    # ---- the rules
    zlo, zhi = RULE_1_ZENITH_DB
    rule1 = bool(q1["sd_change_db"] <= -RULE_1_SD_DB and zlo <= zen_fit <= zhi)
    within = lambda ratio: (1 / RULE_2_FACTOR) <= ratio <= RULE_2_FACTOR  # noqa: E731
    which = "fitted" if rule1 else "profile"
    sep25 = q2[which]["model_over_measured"]
    others = [row[which]["packets_ratio"] for row in q4["passes"]]
    rule2 = bool(within(sep25) and all(within(x) for x in others))
    record = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": got},
              "windows": "addendum 2 of the first analysis, fixed in advance: first and last 30 seconds, five times the background",
              "data": {"record": prov["doi"], "licence": "CC-BY-4.0", "files_md5_checked": [f["name"] for f in prov["files"]]},
              "background_per_s": background, "q1_atmosphere": q1, "q2_sifted_bits": q2, "q3_keys_by_block": q3,
              "q4_other_passes": q4,
              "rules": {"1_profile_takes_the_fitted_pair": rule1, "2_key_model_gains_the_factors": rule2,
                        "2_judged_with": which, "2_ratios": {"2022-09-25": sep25, **{r_["date"]: x for r_, x in zip(q4["passes"], others, strict=True)}}},
              "first_analysis": {"record": str(J.POST_HOC_RECORD.relative_to(REPO)), "fitted_efficiency": first["fitted_efficiency"],
                                 "residuals_after": first["residuals_after"]},
              "attribution": J.CITATION}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    SITE.write_text(json.dumps({"_generated_by": "scripts/measure_jinan1_pass_v2.py; do not edit by hand",
                                "preregistration": record["preregistration"]["file"], "record": str(RECORD.relative_to(REPO)),
                                **{k: record[k] for k in ("q1_atmosphere", "q2_sifted_bits", "q3_keys_by_block", "q4_other_passes",
                                                          "rules", "first_analysis")},
                                "licence": "CC-BY-4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
                                "attribution": {"dataset": "Jinan-1 source data (Li et al., Nature 640, 47-54, 2025)",
                                                "download": "https://doi.org/10.5281/zenodo.14732295",
                                                "citation": J.CITATION, "licence": "CC-BY-4.0"},
                                "modifications": ("Figures computed here from the source data's counts, ranges, QBER and key "
                                                  "columns and from this project's model; no per-second row is shipped.")},
                               indent=1) + "\n")
    print(json.dumps(record["rules"], indent=1))
    print(json.dumps({"q1": q1, "q2": {k: v for k, v in q2.items() if k in sets}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
