"""The trusted relay's "Key with A" is the pass section's figure at the defaults (QK49, Round 86).

The Network tab's pass section simulates Liao et al. 2017's pass at its highest elevation to a tenth of a degree, and
the relay section below it started at that elevation rounded to a whole degree: 49 against 48.9, so the same Xinglong
pass read 494.5 kbit in one row and 493.2 kbit in the other, and the relay's note had to say they could differ. The
relay now starts at the pass section's own elevation, and the default link's station is Xinglong at the altitude the
relay gives Xinglong, so the two figures are one simulation.

Two halves: the browser's model, executed, gives the same key at the same elevation; and the panel starts both
sections at the same elevation, which is the part a rounding change would break without any model changing.
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
PANEL = WEB / "src" / "components" / "SatNetworkPanel.tsx"


def _run() -> dict:
    if not ESBUILD.exists():
        pytest.skip("web/node_modules/.bin/esbuild is absent; run `npm ci` in web/")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
        entry.write_text(
            f"export * from {json.dumps(str(WEB / 'src/lib/satNetwork'))};\n"
            f"export {{ satChannel }} from {json.dumps(str(WEB / 'src/lib/satLink'))};\n"
            f"export {{ qkdCfg }} from {json.dumps(str(WEB / 'src/lib/config'))};\n")
        subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node",
                        f"--outfile={bundle}", "--log-level=error"], cwd=WEB, check=True, capture_output=True)
        runner.write_text(f"import * as m from {json.dumps(bundle.as_uri())};\n" + r"""
const link = m.satChannel({});
const liao = m.liaoPass(link);
const top = Math.round(liao.maxElevationDeg * 10) / 10;
const pass = m.simulatePass(link, top, { stepS: 1 });
const [X, , G] = m.LIAO_2018_STATIONS;
const relay = m.trustedRelay(link, X, G, top, top);
const whole = m.trustedRelay(link, X, G, Math.round(liao.maxElevationDeg), Math.round(liao.maxElevationDeg));
console.log(JSON.stringify({ top, pass: pass.keyBits, relayA: relay.a.keyBits, wholeA: whole.a.keyBits,
                             station: X.id, stationKm: X.altitudeKm,
                             // The pass section's station altitude: the link's own, or the config default it falls to.
                             linkKm: link.freeSpace?.stationAltitudeKm ?? m.qkdCfg().free_space_station_altitude_km }));
""")
        out = subprocess.run(["node", str(runner)], check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


def test_the_relay_at_the_pass_elevation_is_the_pass():
    r = _run()
    assert r["station"] == "xinglong" and r["stationKm"] == r["linkKm"], (
        "the relay's station A and the pass section's link are no longer the same station, so the figures need not agree")
    assert r["relayA"] == r["pass"], f"same pass, same elevation {r['top']}: relay {r['relayA']} against {r['pass']}"
    assert r["wholeA"] != r["pass"], "the whole-degree start this test guards against no longer differs; re-read QK49"


def test_both_sections_start_at_the_same_elevation():
    src = PANEL.read_text()
    assert "useState(Math.round(liao.maxElevationDeg * 10) / 10)" in src, "the pass section's start changed"
    assert "defaultElevation={liao.maxElevationDeg}" in src
    starts = re.findall(r"useState\((Math\.round\(defaultElevation[^;]*)\);", src)
    assert starts and all(s == "Math.round(defaultElevation * 10) / 10" for s in starts), (
        f"the relay starts at {starts}, not at the pass section's elevation to a tenth of a degree")
