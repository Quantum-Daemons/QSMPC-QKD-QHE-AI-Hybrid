#!/usr/bin/env python3
"""SatQuMA's published key lengths recomputed from their own terms, and this project's bound on the same pass, as
pre-registered (Round 86, batch 12): references/satellite-satqkd-key-length-preregistration.md.

Reads the deposit's archive in place (no file is extracted), checks the pre-registration's SHA-256 first, and writes
docs/measurements/satellite-satqkd-key-length-<date>.json.

    QSH_DATA_ROOT=... python scripts/check_satqkd_key_length.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets as D  # noqa: E402
from qsh.qkd.channel import ChannelParams  # noqa: E402
from qsh.qkd.finite_key import VacuumDecoyParams, finite_key_over  # noqa: E402

PREREG = REPO / "references" / "satellite-satqkd-key-length-preregistration.md"
PREREG_SHA256 = "fba1775109a21ba2e9f76bde55758995938cc79f1927eae0a3263b7c97ca70d7"
FIG5A = ("Fig5a_LeftAxis_4GB", "Fig5a_LeftAxis_8GB", "Fig5a_LeftAxis_16GB", "Fig5a_RightAxis")
F_EC = 1.16                    # this project's error-correction factor
EDGE_TOLERANCE_DEG = 0.01
TOTAL_LOSS_TOLERANCE_DB = 1e-6


def h2(x: float) -> float:
    return 0.0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def _name(n: str) -> str:
    """The right-axis file writes `xi (deg)` and `shift dt (s)` where the left-axis files write `xi` and `shift_dt`."""
    n = re.sub(r"\s*\((deg|s)\)$", "", n.strip().strip('"'))
    return {"mean photon no.": "mpn", "shift dt": "shift_dt"}.get(n, n)


def rows_of(z: zipfile.ZipFile, member: str) -> list[dict]:
    lines = [ln for ln in z.read(member).decode("utf-8").splitlines() if ln.strip()]
    names = [_name(c) for c in lines[0].lstrip("# ").split(",")]
    out = []
    for row in csv.reader(lines[1:]):
        if len(row) < len(names):
            continue
        out.append({n: (float(v) if v.strip() not in ("", '""') else math.nan) for n, v in zip(names, row, strict=False)})
    return out


def eq2(r: dict) -> int:
    return math.floor(r["sX0"] + r["sX1"] * (1 - h2(r["phiX"])) - r["lambdaEC"] - 6 * math.log2(21 / r["eps_s"])
                      - math.log2(2 / r["eps_c"]))


def check_a(z: zipfile.ZipFile, members: dict[str, str]) -> dict:
    out = {}
    for name in FIG5A:
        rows = rows_of(z, members[name])
        keyed, unkeyed = [], []
        for i, r in enumerate(rows):
            ell = eq2(r)
            (keyed if ell > 0 else unkeyed).append((i, ell, r["SKL"]))
        off = [(i, ell - skl) for i, ell, skl in keyed if abs(ell - skl) > 1]
        off += [(i, ell - skl) for i, ell, skl in unkeyed if abs(ell - skl) > 1 and abs(skl) > 1]
        out[name] = {
            "rows": len(rows),
            "rows_with_a_key": len(keyed),
            "keyed_equal": sum(ell == skl for _, ell, skl in keyed),
            "keyed_within_one_bit": sum(abs(ell - skl) <= 1 for _, ell, skl in keyed),
            "largest_abs_difference_keyed": max((abs(ell - skl) for _, ell, skl in keyed), default=None),
            "rows_without_a_key": len(unkeyed),
            "unkeyed_recorded_as_zero": sum(abs(skl) <= 1 for _, _, skl in unkeyed),
            "unkeyed_within_one_bit_of_l": sum(abs(ell - skl) <= 1 for _, ell, skl in unkeyed),
            "rows_off_by_more_than_one": [{"row": i, "difference": d} for i, d in off],
        }
    out["meets"] = all(not out[k]["rows_off_by_more_than_one"] for k in FIG5A)
    return out


def _interp(points: list[tuple[float, float]], t: float) -> float:
    for (t0, v0), (t1, v1) in zip(points, points[1:], strict=False):
        if t0 <= t <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    raise ValueError(f"time {t} is outside Fig7's {points[0][0]} to {points[-1][0]}")


def check_b(z: zipfile.ZipFile, members: dict[str, str]) -> dict:
    fig7 = sorted(rows_of(z, members["Fig7"]), key=lambda p: p["Time"])
    total_ok = all(abs(p["TotLoss"] - (p["DiffLoss"] + p["AtmLoss"] + p["SysLoss"])) <= TOTAL_LOSS_TOLERANCE_DB for p in fig7)
    elev = [(p["Time"], p["ElevationDeg"]) for p in fig7]
    path_loss = [(p["Time"], p["DiffLoss"] + p["AtmLoss"]) for p in fig7]
    times = [p["Time"] for p in fig7]
    results = []
    for name in FIG5A:
        for i, r in enumerate(rows_of(z, members[name])):
            if r["xi"] != 0.0:
                continue
            dt = r["dt"]
            edge = _interp(elev, dt)
            entry = {"file": name, "row": i, "dt_s": dt, "max_elevation_deg": r["maxElev"], "sys_loss_db": r["SysLoss"],
                     "fig7_elevation_at_dt_deg": edge, "row_start_elevation_deg": r["startElev"],
                     "fig7_total_loss_is_the_sum": total_ok, "satquma_Pap": r["Pap"],
                     "satquma": {"key": r["SKL"], "n_key_basis": r["nX"], "n_test_basis": r["nZ"], "s1": r["sX1"],
                                 "s0": r["sX0"], "phase_error": r["phiX"], "lambda_ec": r["lambdaEC"]}}
            if abs(edge - r["startElev"]) > EDGE_TOLERANCE_DEG or not total_ok:
                entry["computed"] = False
                results.append(entry)
                continue
            cuts = [0.0] + [t for t in times if 0.0 < t < dt] + [dt]
            segs = []
            for t0, t1 in zip(cuts, cuts[1:], strict=False):
                loss = _interp(path_loss, 0.5 * (t0 + t1)) + r["SysLoss"]
                ch = ChannelParams(distance_km=1.0, fiber_loss_db_per_km=loss, detector_efficiency=1.0,
                                   dark_count_prob=r["Pec"], misalignment=r["QBERI"], source_rate_hz=r["fs"],
                                   dead_time_s=0.0)
                segs.append((2.0 * r["fs"] * (t1 - t0), ch))        # both halves of the pass
            params = VacuumDecoyParams(mu_signal=r["mu1"], mu_decoy=r["mu2"], mu_vacuum=r["mu3"], p_signal=r["P1"],
                                       p_decoy=r["P2"], p_z=r["Px"], eps_sec=r["eps_s"], eps_cor=r["eps_c"], f_ec=F_EC)
            ours = finite_key_over(segs, params)
            entry.update({"computed": True, "segments_per_half": len(segs), "deviation": params.deviation,
                          "this_project": {"key": ours.key_bits, "n_key_basis": ours.n_z, "n_test_basis": ours.n_x,
                                           "s1": ours.s_z1, "s0": ours.s_z0, "phase_error": ours.phi_z,
                                           "lambda_ec": ours.lambda_ec},
                          "key_ratio": ours.key_bits / r["SKL"] if r["SKL"] else None})
            results.append(entry)
    return {"rows": results, "bound": "this project's vacuum_decoy_key_length (Lim et al., three intensities)",
            "differences_named": ["error-correction term: f_EC h(E) n with f_EC 1.16 here, SatQuMA's lambda_EC as it records it",
                                  "statistics: this project's configured deviation",
                                  "after-pulsing: none here; dead time: none here"]}


def main() -> int:
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got[:12]}, not the committed {PREREG_SHA256[:12]}: run nothing")
    path = Path(D.data_root()) / "raw" / "satqkd-sim-data" / "satqkd_sim_data.zip"
    with zipfile.ZipFile(path) as z:
        members = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")}
        record = {"preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
                  "deposit": {"doi": "10.5281/zenodo.8101679", "licence": "CC-BY-4.0",
                              "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                  "paper": "Sidhu et al., arXiv:2301.13209v2, Eq. (2)",
                  "check_a": check_a(z, members), "check_b": check_b(z, members)}
    out = REPO / "docs" / "measurements" / f"satellite-satqkd-key-length-{date.today().isoformat()}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"check_a_meets": record["check_a"]["meets"],
                      "check_b": [(x["file"], x["dt_s"], x["computed"], x["satquma"]["key"],
                                   (x.get("this_project") or {}).get("key")) for x in record["check_b"]["rows"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
