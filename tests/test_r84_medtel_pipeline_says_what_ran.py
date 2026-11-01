"""The browser pipeline's medical and QKD stages say what ran, end to end (Round 84).

WHY THIS EXISTS (scratchpad audit83 page-medical-telecom.md rows 3, 4, 6 and 7)

- MEDICAL, NO WEIGHTS. When the published student's weights could not be fetched, the run evaluated
  the student's SHAPE with a fixed kernel pattern, ((ky * 3 + kx + k) % 5 - 2) * 0.1 with zero biases,
  while its log said "none were invented" and the page promised the run would not substitute any. A
  fixed pattern is invented weights. Now the encrypted student does not run without its weights, and
  the run says so.
- MEDICAL, WITH WEIGHTS. The log said the client decrypts after each layer and that the reported class
  comes from the garbled-circuit argmax, without saying that the biases were added after decryption
  (they are now added under encryption, with every other slot masked) or that the argmax receives logits
  the client has already decrypted. Its dense layers now take 12 rotations, not 136.
- OPTIQKD. The log called the optimiser's model "NOT the decoy-state run above ... a different model",
  while it calls the same bound and its default equals `skr_per_pulse`. It said nothing of what the
  search does not do: nothing it finds is applied, most of its gain is a basis bias the finite key does
  not model, and at its tuned intensities the run's block would carry LESS composable key per pulse.

WHAT IS ASSERTED, BY RUNNING THE PIPELINE

`runClientPipeline` bundled with esbuild and run in node, as `test_r84_integrator_wiring.py` does: the
real node-seal CKKS, a plaintext stand-in for the gitignored garbled-circuit wasm, and every fetch
answered from web/public, a remote one with a 404 - except, in one run, the student's weights, served
from models/ at the Hub address the page asks for. The OptiQKD figures are checked against the
browser's own finite key, recomputed here through the same module.

Each assertion fails on the baseline: the offline run reported `student shape, stated kernel pattern`,
the logs carried the old sentences, and no tuned finite key was computed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
STUDENT = REPO / "models" / "medical_breastmnist_student" / "weights.json"

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const RUNS = JSON.parse(readFileSync(process.argv[3], "utf8"));
const STUDENT = process.argv[4];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r84-medtel-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};
export { bb84, FIBER } from ${JSON.stringify(resolve(WEB, "src/lib/quantumSim"))};
export { optimizeQKD } from ${JSON.stringify(resolve(WEB, "src/lib/clientOptiQKD"))};
export { fibreBlockPulses, finiteKeySourceOf } from ${JSON.stringify(resolve(WEB, "src/lib/finiteKey"))};
export { qkdCfg } from ${JSON.stringify(resolve(WEB, "src/lib/config"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}', "process.env.NODE_ENV": '"production"' },
  plugins: [{ name: "node-stand-ins", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    b.onResolve({ filter: /(^|\/)clientMPC$/ }, () => ({ path: "fakempc", namespace: "fakempc" }));
    b.onLoad({ filter: /.*/, namespace: "fakempc" }, () => ({ loader: "js", contents: `
      export async function warmMPC() {}
      export async function secureArgmax(l) {
        let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i;
        return { out: [b], compares: Math.max(0, l.length - 1), commBytes: 0, mpcMs: 0, messages: 0, clearOut: [b] };
      }` }));
    b.onResolve({ filter: /\.wasm\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
    b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
  } }],
});
let serveStudent = false;
globalThis.fetch = async (url) => {
  const p = String(url);
  const miss = { ok: false, status: 404, headers: { get: () => null },
                 json: async () => { throw new Error("404"); }, text: async () => "" };
  const ok = (text) => ({ ok: true, status: 200, headers: { get: () => "application/json" },
                          json: async () => JSON.parse(text), text: async () => text });
  if (/^[a-z]+:/i.test(p)) {
    if (serveStudent && /qsh-medical-breastmnist-ckks-student\/resolve\/[0-9a-f]{40}\/weights\.json$/.test(p)) {
      return ok(readFileSync(STUDENT, "utf8"));
    }
    return miss;
  }
  try { return ok(readFileSync(join(WEB, "public", p.replace(/^\/+/, "")), "utf8")); } catch { return miss; }
};
const m = await import(pathToFileURL(out).href);
rmSync(dir, { recursive: true, force: true });
const result = {};
for (const run of RUNS) {
  serveStudent = !!run.serveStudent;
  const logs = [], metrics = {}, seen = [];
  await new Promise((done) => {
    let ctl = null, finished = false;
    const finish = () => { if (!finished) { finished = true; clearTimeout(timer); ctl?.stop?.(); done(); } };
    const timer = setTimeout(finish, 240_000);
    ctl = m.runClientPipeline(run.usecase, { seed: 0, ...run.config }, {
      onEvent: (e) => {
        if (finished) return;
        if (e.stage && !seen.includes(e.stage)) {
          if (seen.length && seen[seen.length - 1] === run.stopAfter) { finish(); return; }
          seen.push(e.stage);
        }
        if (e.type === "log") logs.push([e.stage, String(e.message ?? "")]);
        if (e.type === "metric") (metrics[e.stage] ??= {})[e.data.name] = e.data.value;
      },
      onEnd: finish,
    });
  });
  result[run.name] = { stages: seen, logs, metrics };
}
// The browser's own finite key at the optimiser's tuned intensities, recomputed outside the pipeline.
{
  const q = m.qkdCfg();
  const chan = { ...m.FIBER, distanceKm: Number(q.distance_km), lossDbPerKm: Number(q.fiber_loss_db_per_km),
    detectorEff: Number(q.detector_efficiency), darkCount: Number(q.dark_count_prob),
    misalignment: Number(q.misalignment), eveFraction: Number(q.eve_intercept_fraction),
    sourceRateHz: Number(q.source_rate_hz) };
  const opt = m.optimizeQKD(chan, "bb84", Number(q.optimize_n_samples));
  const spec = { kind: String(q.finite_key_block), runPulses: Number(q.pulses),
                 seconds: Number(q.finite_key_block_seconds), paBits: Number(q.finite_key_block_pa_bits) };
  const at = (ch) => {
    const src = m.finiteKeySourceOf(ch, Number(q.finite_key_eps_sec), undefined, q.finite_key_deviation);
    const blk = m.fibreBlockPulses(ch, src, spec);
    return { bits: m.bb84(Number(q.pulses), ch, 0, undefined, { blockPulses: blk, epsSec: src.epsSec,
                                                               deviation: src.deviation }).secureKeyBits, pulses: blk };
  };
  result.reference = { shipped: at(chan),
                       tuned: at({ ...chan, muSignal: opt.bestParams.muSignal, muDecoy: opt.bestParams.muDecoy }),
                       basisProb: opt.bestParams.basisProb, gainPct: opt.improvementPct };
}
console.log(JSON.stringify(result));
process.exit(0);
"""

#: One process per run: the student's weights and the bundled payloads are cached for a tab's life.
GROUPS = {
    "offline": [{"name": "medical_offline", "usecase": "medical_fl", "config": {},
                 "stopAfter": "private_inference"}],
    "weights": [{"name": "medical_weights", "usecase": "medical_fl", "config": {},
                 "stopAfter": "private_inference", "serveStudent": True}],
    "optiqkd": [{"name": "telecom_optiqkd", "usecase": "telecom_ids", "config": {"qkd_optimize": True},
                 "stopAfter": "qkd_key_agreement"}],
}


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
    tmp = tmp_path_factory.mktemp("r84medtel")
    script = tmp / "harness.mjs"
    script.write_text(HARNESS)
    out: dict = {}
    for group, runs in GROUPS.items():
        spec = tmp / f"{group}.json"
        spec.write_text(json.dumps(runs))
        proc = subprocess.run(["node", str(script), str(WEB), str(spec), str(STUDENT)], cwd=WEB,
                              capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, f"the browser pipeline did not run ({group}):\n{proc.stderr[-3000:]}"
        res = json.loads(proc.stdout.strip().splitlines()[-1])
        out.setdefault("reference", res.pop("reference"))
        out.update(res)
    return out


def _stage(run: dict, stage: str) -> tuple[dict, list[str]]:
    assert stage in run["stages"], f"the run never reached {stage}: {run['stages']}"
    return run["metrics"].get(stage, {}), [m for s, m in run["logs"] if s == stage]


def test_without_its_weights_the_student_does_not_run_and_the_run_says_so(browser) -> None:
    m, logs = _stage(browser["medical_offline"], "private_inference")
    assert m.get("fhe_weights") == "not fetched: the encrypted student did not run", (
        f"fhe_weights = {m.get('fhe_weights')!r}: a stand-in kernel ran in place of the published weights")
    for key in ("fhe_conv_rotations", "fhe_conv_ms", "fhe_total_rotations", "fhe_student_class"):
        assert key not in m, f"{key} was reported for a student that did not run"
    assert any("encrypted inference did NOT run" in x and "No kernel pattern stands in" in x for x in logs), logs
    assert not any("none were invented" in x or "stated kernel pattern" in x for x in logs)
    # The page's own regression still ran under CKKS, and nothing wrote a key twice.
    assert m.get("encrypted_tier_measured_ms", 0) > 0
    assert "metric_overwritten" not in m, m.get("metric_overwritten")


def test_with_its_weights_the_student_runs_and_the_log_says_what_the_client_decrypts(browser) -> None:
    m, logs = _stage(browser["medical_weights"], "private_inference")
    assert m.get("fhe_weights") == "published student (HF, pinned revision)"
    assert (m["fhe_conv_rotations"], m["fhe_student_dense_rotations"], m["fhe_total_rotations"]) == (16, 12, 28)
    assert m["fhe_student_logit_max_abs_error"] < 1e-6
    line = next(x for x in logs if x.startswith("private inference (REAL in-browser FHE, the PUBLISHED"))
    assert "already decrypted" in line, "the log does not say the argmax receives decrypted logits"
    assert "bias added under encryption" in line and "outputs and nothing else" in line, line
    assert "decrypts after every layer" in line and "ReLU and average pooling in plaintext" in line, line
    assert "metric_overwritten" not in m, m.get("metric_overwritten")


def test_optiqkd_says_it_scores_the_runs_own_bound_and_applies_nothing(browser) -> None:
    m, logs = _stage(browser["telecom_optiqkd"], "qkd_key_agreement")
    line = next(x for x in logs if x.startswith("OptiQKD (coordinate ascent"))
    assert "a different model" not in line and "NOT the decoy-state run" not in line, line
    assert "the asymptotic decoy-state bound this run's skr_per_pulse comes from" in line, line
    assert "nothing it finds is applied to this run" in line, line
    assert "basis bias" in line and "pZ = 0.5" in line, line
    # Its default IS the run's per-pulse rate (both rounded as emitted: 1e-6 and 1e-9).
    assert abs(m["qkd_surrogate_skr_default"] - m["skr_per_pulse"]) < 1e-6


def test_optiqkd_reports_the_finite_key_its_tuned_intensities_would_give(browser) -> None:
    m, logs = _stage(browser["telecom_optiqkd"], "qkd_key_agreement")
    ref = browser["reference"]
    assert m["secure_key_bits"] == ref["shipped"]["bits"]
    assert m["qkd_tuned_finite_key_bits"] == ref["tuned"]["bits"]
    change = 100 * ((ref["tuned"]["bits"] / ref["tuned"]["pulses"])
                    / (ref["shipped"]["bits"] / ref["shipped"]["pulses"]) - 1)
    assert m["qkd_tuned_finite_key_change_pct"] == pytest.approx(change, abs=0.051)
    # On the default fibre the asymptotic search gains while the composable key per pulse falls: the
    # sentence the log and the tooltip rest on.
    assert ref["gainPct"] > 0 > change, (ref["gainPct"], change)
    line = next(x for x in logs if x.startswith("OptiQKD (coordinate ascent"))
    assert f"{ref['tuned']['bits']:,}" in line and "less key per pulse sent" in line, line
