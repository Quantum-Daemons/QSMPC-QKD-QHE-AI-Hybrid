"""The fibre's two intensity pairs are both selectable, and a run at either is one run in both languages.

WHY THIS EXISTS (Round 84, the user's choice: make both selectable)

Round 82 found that the shipped pair, mu = 0.3416 and nu = 0.1417, is not the pair that needs the
fewest pulses before any key exists: 0.399 / 0.098 needs 3,312,035 pulses on the shipped 50 km fibre
where the shipped pair needs 3,595,158. On the default 1e7-bit block the shipped pair carries more key,
5,214,327 bits against 4,977,636, which is why it stayed the default. The two optimise different
things, so the user chose to offer both. `QKDConfig.intensity_preset` names the default, a run names
its own by `qkd_intensities`, and `qkd.bb84.fibre_source` / `fibreIntensities` resolve the name in
Python and in the browser.

A control of this kind has failed here before by reaching one surface: the stage would run the pair
while the bound, OptiQKD's baseline or the chart beside it stayed on the shipped one. So the pair is
put on the channel (browser) and on the one source the stage builds (Python), and both stages are run.

WHAT IS ASSERTED

- The two names resolve to the configured fields in both languages, the default is the shipped
  pair, and an unknown name is refused rather than run at a default.
- The trade that justifies offering both still holds, at the figures quoted in `QKDConfig`.
- The Python stage runs the pair it is given: the key, the block, the reported intensities and
  OptiQKD's baseline all move with it.
- The browser stage, run in node, reports the same key, block and intensities as the Python stage
  at both pairs, and its OptiQKD baseline is the run's rate.
- The Parameters panel offers both pairs from config and the run's default comes from config.

On the base commit (de1111c) every test fails: the names, the fields and the metrics do not exist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from qsh.config import get_config
from qsh.qkd.bb84 import INTENSITY_PRESETS, fibre_source
from qsh.qkd.channel import ChannelParams
from qsh.qkd.finite_key import (
    block_counts,
    fibre_block_pulses,
    key_length,
    minimum_pulses_for_positive_key,
    params_for_source,
)

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"

#: preset -> (pulses to a first key, key bits on the default block), on the shipped 50 km fibre under
#: the default statistics. The figures `QKDConfig`'s comments quote.
TRADE = {"block_key": (3_595_158, 5_214_327), "first_key": (3_312_035, 4_977_636)}


def _telecom(cfg: dict) -> dict:
    from qsh.usecases import telecom_ids

    return telecom_ids.run_demo({"seed": 1, **cfg})["metrics"]


def test_the_names_resolve_to_the_configured_pairs() -> None:
    q = get_config().qkd
    assert INTENSITY_PRESETS == ("block_key", "first_key")
    assert q.intensity_preset == "block_key"
    shipped, first = fibre_source("block_key"), fibre_source("first_key")
    assert (shipped.mu_signal, shipped.mu_decoy) == (q.mu_signal, q.mu_decoy) == (0.3416, 0.1417)
    assert (first.mu_signal, first.mu_decoy) == (q.first_key_mu_signal, q.first_key_mu_decoy) == (0.399, 0.098)
    assert fibre_source() == fibre_source("block_key")
    # Everything else is the bare source every fibre caller ran before the choice existed.
    from qsh.qkd.bb84 import QKDParams

    assert shipped == QKDParams()
    with pytest.raises(ValueError, match="one of"):
        fibre_source("bogus")
    # The browser's twin names the same two, in the same order.
    ts = (WEB / "src" / "lib" / "quantumSim.ts").read_text(encoding="utf-8")
    m = re.search(r'export const INTENSITY_PRESETS: IntensityPreset\[\] = \[([^\]]*)\];', ts)
    assert m and [s.strip().strip('"') for s in m.group(1).split(",")] == list(INTENSITY_PRESETS)


@pytest.mark.parametrize("preset", INTENSITY_PRESETS)
def test_the_trade_that_justifies_two_pairs_still_holds(preset: str) -> None:
    q = get_config().qkd
    ch = ChannelParams()
    fk = params_for_source(fibre_source(preset), eps_sec=q.finite_key_eps_sec)
    block = fibre_block_pulses(ch, fk, kind="pa_bits", run_pulses=q.pulses,
                               seconds=q.finite_key_block_seconds, pa_bits=q.finite_key_block_pa_bits)
    first_key_pulses, block_bits = TRADE[preset]
    assert minimum_pulses_for_positive_key(ch, fk) == first_key_pulses
    assert int(key_length(*block_counts([(block, ch)], fk), fk).key_bits) == block_bits
    # Each pair wins on its own objective and loses on the other's.
    assert TRADE["first_key"][0] < TRADE["block_key"][0] and TRADE["block_key"][1] > TRADE["first_key"][1]


def test_the_python_stage_runs_the_pair_it_is_given() -> None:
    shipped, first = _telecom({}), _telecom({"qkd_intensities": "first_key"})
    assert shipped["qkd_intensities"] == "block_key" and first["qkd_intensities"] == "first_key"
    assert (shipped["qkd_mu_signal"], shipped["qkd_mu_decoy"]) == (0.3416, 0.1417)
    assert (first["qkd_mu_signal"], first["qkd_mu_decoy"]) == (0.399, 0.098)
    assert shipped["secure_key_bits"] == shipped["qkd_finite_key_bits"] == TRADE["block_key"][1]
    assert first["secure_key_bits"] == first["qkd_finite_key_bits"] == TRADE["first_key"][1]
    assert first["qkd_block_pulses"] < shipped["qkd_block_pulses"]       # 5.25e9 against 5.33e9
    assert first["skr_per_pulse"] > shipped["skr_per_pulse"]              # the asymptotic rate moves too
    # OptiQKD's baseline is the run's own source, not the shipped pair.
    tuned = _telecom({"qkd_intensities": "first_key", "qkd_optimize": True})
    assert tuned["qkd_baseline_skr_per_pulse"] == round(first["skr_per_pulse"], 6)
    with pytest.raises(ValueError, match="one of"):
        _telecom({"qkd_intensities": "bogus"})


HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-mu-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    b.onResolve({ filter: /\.wasm\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
globalThis.fetch = async (url) => {
  const p = String(url);
  const miss = { ok: false, status: 404, headers: { get: () => null },
                 json: async () => { throw new Error("404"); }, text: async () => "" };
  if (/^[a-z]+:/i.test(p)) return miss;
  let text;
  try { text = readFileSync(join(WEB, "public", p.replace(/^\/+/, "")), "utf8"); } catch { return miss; }
  return { ok: true, status: 200, headers: { get: () => "application/json" },
           json: async () => JSON.parse(text), text: async () => text };
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const result = {};
for (const run of [{ name: "block_key", config: { seed: 1 } },
                   { name: "first_key", config: { seed: 1, qkd_intensities: "first_key", qkd_optimize: true } }]) {
  // Only the QKD stage is read, so the run is left as soon as it ends.
  result[run.name] = await new Promise((done) => {
    const metrics = {};
    const timer = setTimeout(() => done({ metrics, timedOut: true }), 300_000);
    m.runClientPipeline("telecom_ids", run.config, {
      onEvent: (e) => {
        if (e.type === "metric" && e.stage === "qkd_key_agreement") metrics[e.data.name] = e.data.value;
        if (e.type === "stage_end" && e.stage === "qkd_key_agreement") { clearTimeout(timer); done({ metrics }); }
      },
      onEnd: () => { clearTimeout(timer); done({ metrics }); },
    });
  });
}
console.log(JSON.stringify(result));
process.exit(0);
"""


def _require_node_or_skip() -> None:
    missing = [name for name, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()),
                                     ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = f"{' and '.join(missing)} absent, so the browser pipeline cannot run. This is a SKIP, not a pass."
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def browser(tmp_path_factory) -> dict:
    _require_node_or_skip()
    script = tmp_path_factory.mktemp("r84mu") / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB)], cwd=WEB, capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0, f"the browser pipeline did not run:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("preset", INTENSITY_PRESETS)
def test_the_browser_stage_runs_the_same_pair_as_python(browser, preset: str) -> None:
    js = browser[preset]
    assert not js.get("timedOut"), "the browser's QKD stage did not end"
    js = js["metrics"]
    py = _telecom({} if preset == "block_key" else {"qkd_intensities": preset})
    for key in ("qkd_intensities", "qkd_mu_signal", "qkd_mu_decoy", "secure_key_bits", "qkd_finite_key_bits"):
        assert js[key] == py[key], (key, js[key], py[key])
    assert js["qkd_block_pulses"] == pytest.approx(py["qkd_block_pulses"], rel=1e-9)
    assert js["skr_per_pulse"] == pytest.approx(py["skr_per_pulse"], rel=1e-9)
    if preset == "first_key":
        # OptiQKD's "default" is this run's rate at this run's pair (the browser rounds it to 6 places).
        assert js["qkd_surrogate_skr_per_pulse_default"] == round(js["skr_per_pulse"], 6)


def test_the_panel_offers_both_pairs_from_config() -> None:
    panel = (WEB / "src" / "components" / "ParameterPanel.tsx").read_text(encoding="utf-8")
    assert 'name="qkd_intensities"' in panel
    assert 'fibreIntensities("block_key")' in panel and 'fibreIntensities("first_key")' in panel
    assert '<option value="block_key">' in panel and '<option value="first_key">' in panel
    # No pair is typed into the page: both come from config through `fibreIntensities`.
    assert not re.search(r"0\.3416|0\.1417|0\.399|0\.098", panel)
    run = (WEB / "src" / "lib" / "usePipelineRun.ts").read_text(encoding="utf-8")
    assert "qkd_intensities: qkdCfg().intensity_preset" in run
    cfg = json.loads((WEB / "src" / "lib" / "config.defaults.json").read_text(encoding="utf-8"))["qkd"]
    assert (cfg["intensity_preset"], cfg["first_key_mu_signal"], cfg["first_key_mu_decoy"]) == ("block_key", 0.399, 0.098)
    # The chart beside the run draws the run's pair.
    page = (WEB / "src" / "pages" / "UseCasePage.tsx").read_text(encoding="utf-8")
    assert "intensities={run.params.qkd_intensities" in page
