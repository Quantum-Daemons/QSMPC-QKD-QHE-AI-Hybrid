"""The satellite pages cite the versions of record, and their numbers are the model's (Round 84).

WHY THIS EXISTS. The Round 83 sweep of the satellite pages found preprints cited where journal
versions exist (Khmelev et al., Opt. Express 32(7), 11964 (2024); Vergoossen et al., Acta
Astronautica 173, 164 (2020)), Jinan-1's per-pass keys from arXiv:2408.10994v1 labelled as the
Nature paper's (whose version of record reports "up to 1.07 million bits" in a single pass where v1
said 0.59 million), the relay's 7,600 km given as the reach of a satellite relay after Jinan-1's
12,900 km, a profile tooltip quoting 3,517,343 detections where the model gives 3,519,418, a QSDC
quotation attributed to the published paper that only its preprint contains, and comments carrying
figures from budgets three rounds old.

WHAT IS ASSERTED
  - The profiles name the journal versions, and Jinan-1's says its pass figures are the preprint's.
  - Each Micius profile's atmosphere note quotes the detections the model gives over Liao's pass on
    that profile, computed here.
  - The Network tab's relay names Jinan-1's 12,900 km, and its constellation cites the journal.
  - Both languages attribute the QSDC termination quote to arXiv:1810.11806v1.
  - The satellite comments carry today's figures, not the old ones.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
PROFILES = REPO / "src" / "qsh" / "config" / "profiles"
LIAO_DETECTIONS = 3_551_136


def _profile(kind: str, pid: str) -> dict:
    return json.loads((PROFILES / kind / f"{pid}.json").read_text())


def test_the_profiles_cite_the_versions_of_record() -> None:
    z = _profile("downlink", "micius_2022_zvenigorod")
    assert "Opt. Express 32(7), 11964 (2024)" in z["source"] and "(Khmelev 2024)" in z["label"]
    v = _profile("isl", "micius_type_2019")
    assert "Acta Astronautica 173, 164 (2020)" in v["source"] and "(Vergoossen 2020)" in v["label"]
    j = _profile("downlink", "jinan1_2022_jinan")["source"]
    assert "Nature 640, 47 (2025)" in j and "arXiv:2408.10994v1" in j and "1.07 million bits" in j


@pytest.fixture(scope="module")
def detections() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        bundle, runner = d / "bundle.mjs", d / "run.mjs"
        entry = (f"export {{ liaoPass }} from {json.dumps(str(WEB / 'src/lib/satNetwork'))};\n"
                 f"export {{ satChannel }} from {json.dumps(str(WEB / 'src/lib/satLink'))};\n")
        subprocess.run([str(ESBUILD), "--bundle", "--loader=ts", "--format=esm", "--platform=node",
                        '--define:import.meta.env={"BASE_URL":"/"}', f"--outfile={bundle}", "--log-level=error"],
                       cwd=WEB, check=True, capture_output=True, input=entry.encode())
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + """
const ids = ["micius_2017", "micius_2017_one_decoy", "micius_2017_field_stop"];
console.log(JSON.stringify(Object.fromEntries(ids.map((id) =>
  [id, m.liaoPass(m.satChannel({ sat_profile: id, sat_eve: 0 })).detections]))));
""")
        return json.loads(subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True).stdout)


def test_each_profile_note_quotes_the_models_own_detections(detections) -> None:
    default = round(detections["micius_2017"])
    for pid, det in detections.items():
        note = _profile("downlink", pid)["values"]["atmospheric_db"]["note"]
        assert "3,517,343" not in note, pid
        assert f"({default:,} against {LIAO_DETECTIONS:,})" in note, (pid, note)
        pct = 100 * (1 - default / LIAO_DETECTIONS)
        assert f"{pct:.1f} percent fewer detections" in note, (pid, pct)
        if pid != "micius_2017":
            assert f"counts {round(det):,}" in note, (pid, det)


def test_the_network_tab_names_the_longest_relay_and_the_journals() -> None:
    panel = (WEB / "src" / "components" / "SatNetworkPanel.tsx").read_text()
    assert "separated by over 12,900 kilometres on Earth" in panel and "Nature 640, 47, 2025" in panel
    assert "Acta Astronautica 173," in panel and "(Vergoossen et al. 2019)" not in panel
    assert "Khmelev et al., Opt. Express 32, 11964 (2024)" in panel
    net = (WEB / "src" / "lib" / "satNetwork.ts").read_text()
    assert "12,900 kilometres" in net and "Acta Astronautica 173, 164 (2020)" in net


def test_the_qsdc_quote_is_the_preprints_in_both_languages() -> None:
    py = (REPO / "src" / "qsh" / "qkd" / "qsdc.py").read_text()
    ts = (WEB / "src" / "lib" / "clientCompute.ts").read_text()
    for src in (py, ts):
        assert "arXiv:1810.11806v1" in src
        quote = "they conclude the channel is insecure and terminate the process".split()
        assert re.search(r"(?:\s|//)+".join(map(re.escape, quote)), src)       # across wrapped lines
        assert "correcting capability" in src
    assert "Light Sci. Appl. 8, 22 (2019), protocol step 2" not in py


def _satellite_region() -> str:
    ts = (WEB / "src" / "lib" / "clientCompute.ts").read_text()
    return ts[ts.index('case "satellite_qkd": {'):ts.index('case "qsmpc_compute": {')]


def test_the_satellite_comments_carry_todays_figures() -> None:
    region = _satellite_region()
    for stale in ("5.5e-6", "37.0 dB with the detector", "g 280.43 -> 701.06", "g = 12.6 even at zero range",
                  "g\n        // here is about 808", "about 808", "-51.58 bit/carrier", "(24.5 dB, g=280)"):
        assert stale not in region, stale
    for today in ("6.5e-5", "g = 532.66", "C_s -80.82", "g = 6.14 even at zero range"):
        assert today in region, today
    stage = (REPO / "src" / "qsh" / "usecases" / "satellite_comms" / "__init__.py").read_text()
    for stale in ("24.48 dB", "~27 dB", "floor total loss at 8 dB"):
        assert stale not in stage, stale
    assert "32.21 dB at the default 800 km" in stage
