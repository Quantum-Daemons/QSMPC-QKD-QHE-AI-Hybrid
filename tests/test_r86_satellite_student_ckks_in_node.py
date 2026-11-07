"""The satellite forecaster student under the page's real CKKS, end to end, in node (Round 86).

The page runs every forecast of the student's published stretch through `encryptedLinearBatch` (node-seal, the
module the poster cites, unchanged), decrypts it and thresholds it with Telemanom's procedure. This bundles
`forecastStudent.ts` with esbuild, as tests/test_r84_c3_encrypted_heldout_in_node.py bundles the held-out check, and
runs that against the REAL node-seal, so the claim the page makes is checked where it can be: the decrypted forecast
moves by CKKS noise only, and the events it gives are the plaintext forecast's. About twenty seconds of CKKS.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from qsh.models import registry

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"

HARNESS = r"""
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3], MODEL = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const wasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { evaluateForecastStudent } from ${JSON.stringify(resolve(WEB, "src/lib/forecastStudent"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  // The one Vite-only specifier: the SEAL wasm's `?url` import becomes its path on disk.
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(wasm)};`, loader: "js" }));
  } }],
});
const m = await import(`file://${OUT}`);
const sample = JSON.parse(readFileSync(MODEL + "/eval_sample.json", "utf8"));
const weights = JSON.parse(readFileSync(MODEL + "/weights.json", "utf8"));
const ev = await m.evaluateForecastStudent(sample, weights, { ckks: "all" });
const c = ev.ckks;
console.log(JSON.stringify({ digest: ev.flagsDigestMatches, score: ev.scoreMatches, rows: c.rows,
  offered: sample.stretch.rows, maxAbs: c.maxAbsError, changed: c.flagsChanged,
  plain: [ev.plain.anomalies.tp, ev.plain.anomalies.fp, ev.plain.anomalies.fn, ev.plain.anomalies.f05],
  enc: [c.encrypted.anomalies.tp, c.encrypted.anomalies.fp, c.encrypted.anomalies.fn, c.encrypted.anomalies.f05] }));
"""


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", (WEB / "node_modules" / ".bin" / "esbuild").exists()),
                                     ("web/node_modules/node-seal", (WEB / "node_modules" / "node-seal").exists()))
               if not ok]
    if not missing:
        return
    msg = f"{', '.join(missing)} absent, so the student's CKKS path cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg + " QSH_REQUIRE_BROWSER_GATES=1 is set, so this is a failure.")
    pytest.skip(msg)


def test_the_decrypted_forecast_gives_the_plaintext_forecasts_events() -> None:
    _require_node_or_skip()
    model = REPO / registry.ARTIFACT_DIRS["satellite_comms.student"]
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "harness.mjs").write_text(HARNESS)
        out = subprocess.run(["node", str(d / "harness.mjs"), str(WEB), str(d / "bundle.mjs"), str(model)],
                             check=True, capture_output=True, text=True, cwd=WEB, timeout=600)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["digest"] and r["score"], "the plaintext path no longer reproduces the published stretch"
    assert r["rows"] == r["offered"], "the CKKS batch dropped rows"
    # CKKS noise at this ring and scale is about 1e-9 on these forecasts; a thousandth would be a broken encoding.
    assert r["maxAbs"] < 1e-6, r["maxAbs"]
    assert r["changed"] == 0 and r["enc"] == r["plain"], "the decrypted forecast gives other events"
