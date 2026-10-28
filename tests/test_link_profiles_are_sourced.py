"""Every link profile is sourced value by value, the default one is the config, and both languages
build the same link from each.

WHY THIS EXISTS (Round 77). The satellite page modelled one link, Micius to Xinglong, with its
values spread across `QKDConfig.free_space_*`, so a second link meant editing the config, two
languages' channel code and the tests. By the user's request a link is now a JSON file under
`src/qsh/config/profiles/` - downlinks and inter-satellite terminal pairs - loaded, exported to the
browser and offered on the page with no code change. A file is only worth loading if its numbers can
be traced, so the loader refuses a value with neither a quote nor a stated assumption, and this test
holds the files to that and to the code that reads them.

WHAT IS ASSERTED

- Every profile loads; its id is its file name; every value carries exactly one of "quote" or
  "assumption"; the loader refuses one that carries neither.
- The default downlink profile equals `QKDConfig.free_space_*` value for value, and the default
  inter-satellite profile exists.
- The browser's baked config carries the profiles exactly as the loader reads them, so a new file
  cannot reach Python and not the page.
- For every downlink profile, the browser's `satChannel` and the Python stage's `_sat_channel` /
  `_sat_source` build the same link, and both BB84 models give the same rate at 800 km.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from qsh.config import get_config
from qsh.config.profiles import REQUIRED, ProfileError, _check, frontend_profiles, load_profiles

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
ALL = [(kind, pid) for kind in REQUIRED for pid in load_profiles(kind)]


@pytest.mark.parametrize("kind,pid", ALL, ids=[f"{k}/{p}" for k, p in ALL])
def test_every_value_is_sourced(kind: str, pid: str) -> None:
    prof = load_profiles(kind)[pid]
    assert prof["id"] == pid and prof["kind"] == kind and prof["label"] and prof["source"]
    for name, entry in prof["values"].items():
        has_quote, has_assumption = bool(entry.get("quote")), bool(entry.get("assumption"))
        assert has_quote != has_assumption, f"{pid}.{name} needs exactly one of quote or assumption"


def test_the_loader_refuses_an_untraceable_value(tmp_path: Path) -> None:
    good = load_profiles("downlink")[get_config().qkd.free_space_profile]
    bad = json.loads(json.dumps(good))
    bad["values"]["rx_aperture_m"] = {"value": 2.0}
    path = tmp_path / "downlink" / f"{bad['id']}.json"
    path.parent.mkdir()
    with pytest.raises(ProfileError, match="exactly one of"):
        _check(path, bad, "downlink")
    del bad["values"]["rx_aperture_m"]
    with pytest.raises(ProfileError, match="missing rx_aperture_m"):
        _check(path, bad, "downlink")


def test_the_default_profiles_are_the_config() -> None:
    q = get_config().qkd
    values = load_profiles("downlink")[q.free_space_profile]["values"]
    for name in REQUIRED["downlink"]:
        assert values[name]["value"] == pytest.approx(getattr(q, f"free_space_{name}")), name
    assert q.isl_profile in load_profiles("isl")


def test_the_browser_carries_the_profiles_as_loaded() -> None:
    baked = json.loads((WEB / "src" / "lib" / "config.defaults.json").read_text())
    assert baked["link_profiles"] == json.loads(json.dumps(frontend_profiles())), (
        "config.defaults.json is stale: run scripts/export_config.py")


def _bundle(entry_src: str, runner_src: str) -> object:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(entry_src)
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + runner_src)
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def test_both_languages_build_the_same_link_from_every_profile() -> None:
    from qsh.orchestration.stages import Context
    from qsh.qkd.bb84 import run_bb84
    from qsh.usecases.satellite_comms import _sat_channel, _sat_source

    ids = list(load_profiles("downlink"))
    rows = _bundle(
        f"export {{ satChannel }} from {json.dumps(str(WEB / 'src/lib/satLink'))};\n"
        f"export {{ bb84, channelLossDb }} from {json.dumps(str(WEB / 'src/lib/quantumSim'))};\n",
        f"const ids = {json.dumps(ids)};\n"
        "console.log(JSON.stringify(ids.flatMap((id) => [0, 1].map((vac) => {\n"
        "  const ch = m.satChannel({ sat_profile: id, sat_distance_km: 800, sat_vacuum_decoy: vac });\n"
        "  return { id, vac, loss: m.channelLossDb(ch), rate: m.bb84(200000, ch, 0).secretKeyRate,\n"
        "           mu: ch.muSignal, nu: ch.muDecoy, pS: ch.pSignal, pD: ch.pDecoy, vacuum: ch.vacuumDecoy };\n"
        "}))));\n")
    for row in rows:
        ctx = Context(usecase="satellite_comms", config={"sat_profile": row["id"], "sat_distance_km": 800,
                                                         "sat_vacuum_decoy": row["vac"]})
        ch, src = _sat_channel(ctx), _sat_source(ctx)
        label = f"{row['id']} vacuum={row['vac']}"
        assert row["loss"] == pytest.approx(ch.loss_db(), rel=1e-12), label
        assert (row["mu"], row["nu"], row["vacuum"]) == (src.mu_signal, src.mu_decoy, src.vacuum_decoy), label
        assert (row["pS"], row["pD"]) == pytest.approx((src.p_signal, src.p_decoy)), label
        py = run_bb84(200_000, ch, seed=0, qkd_params=src).secret_key_rate
        assert row["rate"] == pytest.approx(py, rel=1e-9), label
