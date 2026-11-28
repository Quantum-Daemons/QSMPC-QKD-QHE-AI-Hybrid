#!/usr/bin/env python3
"""Check B'' of the SatQKD check, chosen after Checks B and B' and an exploration of their results, as its addendum
states (`references/satellite-satqkd-key-length-preregistration-addendum-cap.md`, Round 86, batch 13).

Check B' (`scripts/check_satqkd_key_length_b_prime.py`) in three variants: B''a with this project's bound's named
option `cap_single_photon_errors` on; B''b also with the satellite page's statistics; B''c also on SatQuMA's window of
one-second slots at the whole seconds either side of the peak. Recomputes Check B' first and stops if it differs from
its record. Writes `docs/measurements/satellite-satqkd-key-length-b-double-prime-2026-11-28.json`.

    QSH_DATA_ROOT=... python scripts/check_satqkd_key_length_b_double_prime.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADDENDUM = REPO / "references" / "satellite-satqkd-key-length-preregistration-addendum-cap.md"
ADDENDUM_SHA256 = "f5bc1f1366d24353adf0a1ba8a6b712a134263a02b2a4d35a1400e3e6c35b1e5"
RECORD = REPO / "docs" / "measurements" / "satellite-satqkd-key-length-b-double-prime-2026-11-28.json"
B_PRIME_RECORD = REPO / "docs" / "measurements" / "satellite-satqkd-key-length-b-prime-2026-11-28.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("check_satqkd_key_length")

from qsh.config import get_config  # noqa: E402  (S put src/ on the path)

VARIANTS = {
    "b_prime": {"cap": False, "statistics": "check_b_prime", "window": "continuous"},
    "a": {"cap": True, "statistics": "check_b_prime", "window": "continuous"},
    "b": {"cap": True, "statistics": "satellite_page", "window": "continuous"},
    "c": {"cap": True, "statistics": "satellite_page", "window": "satquma_slots"},
}


def _channel(r: dict, loss_db: float):
    return S.ChannelParams(distance_km=1.0, fiber_loss_db_per_km=loss_db, detector_efficiency=1.0,
                           dark_count_prob=r["Pec"], misalignment=r["QBERI"], source_rate_hz=r["fs"], dead_time_s=0.0)


def _segments(r: dict, window: str, total: list[tuple[float, float]], at_peak: float, times: list[float]):
    dt = r["dt"]

    def loss(t: float) -> float:
        return S._interp(total, t) - at_peak + r["SysLoss"]

    if window == "continuous":                       # Check B''s: 2 dt seconds, cut at Fig7's own times
        cuts = [0.0] + [t for t in times if 0.0 < t < dt] + [dt]
        return [(2.0 * r["fs"] * (t1 - t0), _channel(r, loss(0.5 * (t0 + t1)))) for t0, t1 in zip(cuts, cuts[1:], strict=False)]
    if window == "satquma_slots":                    # SatQuMA's: one second at each whole second -dt..dt
        if dt != int(dt):
            raise ValueError(f"SatQuMA's window needs a whole number of seconds, not dt = {dt}")
        return [(r["fs"], _channel(r, loss(0.0)))] + [(2.0 * r["fs"], _channel(r, loss(float(k)))) for k in range(1, int(dt) + 1)]
    raise ValueError(window)


def check_b_double_prime(z: zipfile.ZipFile, members: dict[str, str]) -> dict:
    fig7 = sorted(S.rows_of(z, members["Fig7"]), key=lambda p: p["Time"])
    total_ok = all(abs(p["TotLoss"] - (p["DiffLoss"] + p["AtmLoss"] + p["SysLoss"])) <= S.TOTAL_LOSS_TOLERANCE_DB for p in fig7)
    if not total_ok:
        raise SystemExit("Fig7's total loss is not the sum of its parts: run nothing")
    elev = [(p["Time"], p["ElevationDeg"]) for p in fig7]
    total = [(p["Time"], p["TotLoss"]) for p in fig7]
    at_peak = fig7[0]["TotLoss"]
    times = [p["Time"] for p in fig7]
    satellite = get_config().qkd.free_space_finite_key_deviation
    rows = []
    for name in S.FIG5A:
        for i, r in enumerate(S.rows_of(z, members[name])):
            if r["xi"] != 0.0:
                continue
            edge = S._interp(elev, r["dt"])
            if abs(edge - r["startElev"]) > S.EDGE_TOLERANCE_DEG:
                raise SystemExit(f"{name} row {i}: Fig7's elevation at dt is {edge}, the row's {r['startElev']}: run nothing")
            entry = {"file": name, "row": i, "dt_s": r["dt"],
                     "satquma": {"key": r["SKL"], "n_key_basis": r["nX"], "n_test_basis": r["nZ"],
                                 "phase_error": r["phiX"], "after_pulsing": r["Pap"]},
                     "variants": {}}
            for key, v in VARIANTS.items():
                kw = {} if v["statistics"] == "check_b_prime" else {"deviation": satellite}
                params = S.VacuumDecoyParams(mu_signal=r["mu1"], mu_decoy=r["mu2"], mu_vacuum=r["mu3"], p_signal=r["P1"],
                                             p_decoy=r["P2"], p_z=r["Px"], eps_sec=r["eps_s"], eps_cor=r["eps_c"],
                                             f_ec=S.F_EC, cap_single_photon_errors=v["cap"], **kw)
                segs = _segments(r, v["window"], total, at_peak, times)
                ours = S.finite_key_over(segs, params)
                entry["variants"][key] = {
                    "cap_single_photon_errors": v["cap"], "deviation": params.deviation, "window": v["window"],
                    "segments": len(segs), "seconds": sum(n for n, _ in segs) / r["fs"],
                    "key": ours.key_bits, "key_ratio": ours.key_bits / r["SKL"] if r["SKL"] else None,
                    "n_key_basis": ours.n_z, "key_basis_count_ratio": ours.n_z / r["nX"] if r["nX"] else None,
                    "n_test_basis": ours.n_x, "test_basis_count_ratio": ours.n_x / r["nZ"] if r["nZ"] else None,
                    "phase_error": ours.phi_z}
            rows.append(entry)
    return {"rows": rows, "chosen_after_checks_b_and_b_prime_and_an_exploration": True,
            "statistics": {"check_b_prime": "the configured default, finite_key_deviation",
                           "satellite_page": f"free_space_finite_key_deviation = {satellite!r}"},
            "bound": "this project's vacuum_decoy_key_length (Lim et al., three intensities)",
            "differences_still_named": [
                "after-pulsing: SatQuMA's Pap on these rows, none here",
                "error-correction term: SatQuMA's lambda_EC as it records it, f_EC h(E) n with f_EC 1.16 here",
                "errors per intensity: SatQuMA's at the basis's average error rate, each intensity's own rate here",
                "Chernoff lower bounds: SatQuMA's not floored at zero, floored here"]}


def main() -> int:
    got = hashlib.sha256(ADDENDUM.read_bytes()).hexdigest()
    if got != ADDENDUM_SHA256:
        raise SystemExit(f"{ADDENDUM.name} hashes to {got[:12]}, not the committed {ADDENDUM_SHA256[:12]}: run nothing")
    path = Path(S.D.data_root()) / "raw" / "satqkd-sim-data" / "satqkd_sim_data.zip"
    with zipfile.ZipFile(path) as z:
        members = {Path(n).stem: n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")}
        result = check_b_double_prime(z, members)
    recorded = {(x["file"], x["dt_s"]): x["this_project"]["key"] for x in json.loads(B_PRIME_RECORD.read_text())["check_b_prime"]["rows"]}
    for row in result["rows"]:
        want, got_key = recorded[(row["file"], row["dt_s"])], row["variants"]["b_prime"]["key"]
        if not math.isclose(got_key, want, rel_tol=1e-9, abs_tol=1e-6):
            raise SystemExit(f"{row['file']}: Check B' recomputes to {got_key}, its record says {want}: write nothing")
    record = {"addendum": {"file": str(ADDENDUM.relative_to(REPO)), "sha256": ADDENDUM_SHA256},
              "preregistration": {"file": str(S.PREREG.relative_to(REPO)), "sha256": S.PREREG_SHA256},
              "check_b_prime_record": str(B_PRIME_RECORD.relative_to(REPO)), "check_b_prime_reproduced": True,
              "deposit": {"doi": "10.5281/zenodo.8101679", "licence": "CC-BY-4.0",
                          "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
              "check_b_double_prime": result}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps([(x["file"], x["dt_s"], {k: round(v["key_ratio"], 4) for k, v in x["variants"].items()})
                      for x in result["rows"]]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
