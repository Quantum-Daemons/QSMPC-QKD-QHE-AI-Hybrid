"""Link profiles: one JSON file per published link, every value carrying its source.

WHY THIS EXISTS (Round 77, the user's request). The satellite page modelled exactly one link,
Micius to Xinglong, with its values spread across `QKDConfig.free_space_*`. Adding a second link -
another satellite, another ground station, a different reading of the same paper - meant editing
the config, both languages' channel code and the tests. A profile is a file instead: drop a JSON
file into `downlink/` or `isl/` and it is loaded here, exported to the browser by
`frontend_config()`, offered by the page's profile selector and checked by
`tests/test_link_profiles_are_sourced.py`, with no code change.

THE FORMAT. Each file holds one link::

    {"id": "...", "kind": "downlink" | "isl", "label": "...", "source": "...",
     "values": {"<name>": {"value": <number or boolean>, "quote": "<the source's words>"}}}

Every value has exactly one of "quote" (the source states it) or "assumption" (it does not, and
this says what was assumed and why), and may add a "note". A value with neither is refused at load,
so a profile cannot carry a number nobody can trace. Quotes are the source's words, with Greek
letters, degree signs and primes spelled out in ASCII (mu_s, deg, ').

`downlink` profiles describe a satellite-to-ground link and mirror `QKDConfig.free_space_<name>`;
the default one (`QKDConfig.free_space_profile`) must equal those config values, which the test
asserts. `isl` profiles describe an inter-satellite link terminal pair for the constellation model
in `web/src/lib/satNetwork.ts`.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent

#: The values every profile of a kind must state, in the units their names carry.
REQUIRED: dict[str, tuple[str, ...]] = {
    "downlink": (
        "orbit_altitude_km", "station_altitude_km", "beam_divergence_urad", "rx_aperture_m",
        "receiver_optics_efficiency", "atmospheric_db", "atmospheric_ref_km", "pointing_db",
        "detector_efficiency", "dark_count_prob", "misalignment", "source_rate_hz",
        "detector_dead_time_s", "mu_signal", "mu_decoy", "p_signal", "p_decoy", "vacuum_decoy",
        "min_elevation_deg", "distance_km",
    ),
    "isl": (
        "tx_divergence_urad", "rx_aperture_m", "receiver_optics_efficiency", "pointing_db",
        "detector_efficiency", "dark_count_prob", "misalignment", "source_rate_hz",
        "detector_dead_time_s", "mu_signal", "mu_decoy", "p_signal", "p_decoy", "vacuum_decoy",
        "clearance_km", "eclipse_only",
    ),
}
#: Values a profile MAY state; absent, the config's default applies (Round 78: the optional
#: turbulence coupling, off unless a profile sets a field stop).
OPTIONAL: dict[str, tuple[str, ...]] = {
    "downlink": ("field_stop_urad", "seeing_arcsec", "wavelength_nm"),
    "isl": (),
}
#: Values that are switches rather than quantities.
BOOLEAN = frozenset({"vacuum_decoy", "eclipse_only"})


class ProfileError(ValueError):
    """A profile file that does not meet the format above."""


def _check(path: Path, data: dict, kind: str) -> dict:
    where = f"{path.parent.name}/{path.name}"
    for key in ("id", "kind", "label", "source", "values"):
        if key not in data:
            raise ProfileError(f"{where}: no {key!r}")
    if data["kind"] != kind:
        raise ProfileError(f"{where}: kind {data['kind']!r} in the {kind!r} folder")
    if data["id"] != path.stem:
        raise ProfileError(f"{where}: id {data['id']!r} is not the file name")
    values = data["values"]
    missing = [k for k in REQUIRED[kind] if k not in values]
    if missing:
        raise ProfileError(f"{where}: missing {', '.join(missing)}")
    unknown = [k for k in values if k not in REQUIRED[kind] and k not in OPTIONAL[kind]]
    if unknown:
        raise ProfileError(f"{where}: unknown {', '.join(unknown)}")
    for name, entry in values.items():
        if ("quote" in entry) == ("assumption" in entry):
            raise ProfileError(f"{where}: {name} needs exactly one of 'quote' or 'assumption'")
        v = entry.get("value")
        if name in BOOLEAN:
            if not isinstance(v, bool):
                raise ProfileError(f"{where}: {name} must be true or false")
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ProfileError(f"{where}: {name} must be a number")
    p_s, p_d = float(values["p_signal"]["value"]), float(values["p_decoy"]["value"])
    p_vac = 1.0 - p_s - p_d
    if not (0.0 < p_s <= 1.0 and 0.0 < p_d < 1.0 and p_vac > -1e-12):
        raise ProfileError(f"{where}: state probabilities {p_s}, {p_d} do not leave a vacuum share >= 0")
    if values["vacuum_decoy"]["value"] and p_vac <= 1e-12:
        raise ProfileError(f"{where}: vacuum_decoy is set but no pulses are left for the vacuum state")
    if not (0.0 < float(values["mu_decoy"]["value"]) < float(values["mu_signal"]["value"])):
        raise ProfileError(f"{where}: the decoy must be weaker than the signal")
    return data


@cache
def load_profiles(kind: str) -> dict[str, dict]:
    """Every profile of `kind`, by id, in file-name order. Raises ProfileError on a malformed file."""
    if kind not in REQUIRED:
        raise ProfileError(f"unknown profile kind {kind!r}")
    out: dict[str, dict] = {}
    for path in sorted((_DIR / kind).glob("*.json")):
        out[path.stem] = _check(path, json.loads(path.read_text()), kind)
    return out


def profile_values(kind: str, profile_id: str) -> dict[str, float | bool]:
    """The bare values of one profile, `{name: value}`."""
    profiles = load_profiles(kind)
    if profile_id not in profiles:
        raise ProfileError(f"no {kind} profile {profile_id!r}; have {', '.join(profiles)}")
    return {k: e["value"] for k, e in profiles[profile_id]["values"].items()}


def frontend_profiles() -> dict[str, dict[str, dict]]:
    """Both kinds, whole (values with their quotes), for the browser."""
    return {kind: load_profiles(kind) for kind in REQUIRED}
