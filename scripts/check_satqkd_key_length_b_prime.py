#!/usr/bin/env python3
"""Check B' of the SatQKD check, chosen after Check B's result, as its addendum states
(`references/satellite-satqkd-key-length-preregistration-addendum-loss.md`, Round 86, batch 12).

Check B as pre-registered (`scripts/check_satqkd_key_length.py`) with one change, the loss: Fig7's total loss shifted
so its value at the peak is the row's `SysLoss`, the link's total loss at zenith. Writes
`docs/measurements/satellite-satqkd-key-length-b-prime-2026-11-28.json`.

    QSH_DATA_ROOT=... python scripts/check_satqkd_key_length_b_prime.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADDENDUM = REPO / "references" / "satellite-satqkd-key-length-preregistration-addendum-loss.md"
ADDENDUM_SHA256 = "2186bb42575ff2e876c39a4895b86cf72bb35659e59787a68733153686f9f76f"
RECORD = REPO / "docs" / "measurements" / "satellite-satqkd-key-length-b-prime-2026-11-28.json"

_spec = importlib.util.spec_from_file_location("check_satqkd_key_length", REPO / "scripts" / "check_satqkd_key_length.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def check_b_prime(z: zipfile.ZipFile, members: dict[str, str]) -> dict:
    fig7 = sorted(S.rows_of(z, members["Fig7"]), key=lambda p: p["Time"])
    total_ok = all(abs(p["TotLoss"] - (p["DiffLoss"] + p["AtmLoss"] + p["SysLoss"])) <= S.TOTAL_LOSS_TOLERANCE_DB for p in fig7)
    elev = [(p["Time"], p["ElevationDeg"]) for p in fig7]
    total = [(p["Time"], p["TotLoss"]) for p in fig7]
    at_peak = fig7[0]["TotLoss"]
    times = [p["Time"] for p in fig7]
    results = []
    for name in S.FIG5A:
        for i, r in enumerate(S.rows_of(z, members[name])):
            if r["xi"] != 0.0:
                continue
            dt = r["dt"]
            edge = S._interp(elev, dt)
            entry = {"file": name, "row": i, "dt_s": dt, "sys_loss_db": r["SysLoss"], "fig7_total_loss_at_peak_db": at_peak,
                     "fig7_elevation_at_dt_deg": edge, "row_start_elevation_deg": r["startElev"],
                     "satquma": {"key": r["SKL"], "n_key_basis": r["nX"], "n_test_basis": r["nZ"], "s1": r["sX1"],
                                 "s0": r["sX0"], "phase_error": r["phiX"], "lambda_ec": r["lambdaEC"]}}
            if abs(edge - r["startElev"]) > S.EDGE_TOLERANCE_DEG or not total_ok:
                entry["computed"] = False
                results.append(entry)
                continue
            cuts = [0.0] + [t for t in times if 0.0 < t < dt] + [dt]
            segs = []
            for t0, t1 in zip(cuts, cuts[1:], strict=False):
                loss = S._interp(total, 0.5 * (t0 + t1)) - at_peak + r["SysLoss"]
                ch = S.ChannelParams(distance_km=1.0, fiber_loss_db_per_km=loss, detector_efficiency=1.0,
                                     dark_count_prob=r["Pec"], misalignment=r["QBERI"], source_rate_hz=r["fs"],
                                     dead_time_s=0.0)
                segs.append((2.0 * r["fs"] * (t1 - t0), ch))
            params = S.VacuumDecoyParams(mu_signal=r["mu1"], mu_decoy=r["mu2"], mu_vacuum=r["mu3"], p_signal=r["P1"],
                                         p_decoy=r["P2"], p_z=r["Px"], eps_sec=r["eps_s"], eps_cor=r["eps_c"], f_ec=S.F_EC)
            ours = S.finite_key_over(segs, params)
            entry.update({"computed": True, "segments_per_half": len(segs), "deviation": params.deviation,
                          "this_project": {"key": ours.key_bits, "n_key_basis": ours.n_z, "n_test_basis": ours.n_x,
                                           "s1": ours.s_z1, "s0": ours.s_z0, "phase_error": ours.phi_z,
                                           "lambda_ec": ours.lambda_ec},
                          "key_ratio": ours.key_bits / r["SKL"] if r["SKL"] else None,
                          "key_basis_count_ratio": ours.n_z / r["nX"] if r["nX"] else None})
            results.append(entry)
    return {"rows": results, "chosen_after_check_b": True,
            "loss": "Fig7's TotLoss at the segment's middle time, minus its value at the peak, plus the row's SysLoss",
            "bound": "this project's vacuum_decoy_key_length (Lim et al., three intensities)",
            "differences_named": ["error-correction term: f_EC h(E) n with f_EC 1.16 here, SatQuMA's lambda_EC as it records it",
                                  "statistics: this project's configured deviation",
                                  "after-pulsing: none here; dead time: none here"]}


def main() -> int:
    got = hashlib.sha256(ADDENDUM.read_bytes()).hexdigest()
    if got != ADDENDUM_SHA256:
        raise SystemExit(f"{ADDENDUM.name} hashes to {got[:12]}, not the committed {ADDENDUM_SHA256[:12]}: run nothing")
    path = Path(S.D.data_root()) / "raw" / "satqkd-sim-data" / "satqkd_sim_data.zip"
    with zipfile.ZipFile(path) as z:
        members = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")}
        record = {"addendum": {"file": str(ADDENDUM.relative_to(REPO)), "sha256": ADDENDUM_SHA256},
                  "preregistration": {"file": str(S.PREREG.relative_to(REPO)), "sha256": S.PREREG_SHA256},
                  "deposit": {"doi": "10.5281/zenodo.8101679", "licence": "CC-BY-4.0",
                              "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                  "check_b_prime": check_b_prime(z, members)}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps([(x["file"], x["dt_s"], x["computed"], x["satquma"]["key"], (x.get("this_project") or {}).get("key"),
                       x.get("key_basis_count_ratio")) for x in record["check_b_prime"]["rows"]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
