"""The genomics page evaluates the FULL polygenic score under CKKS, against a cut it states.

WHY THIS EXISTS (Round 84, scratchpad audit83/page-finance-genomics.md, findings 1, 2, 8)

The page's CKKS step was a sign test on 8 of the then score's (PGS000004's) 313 variants (W = [-w/2, +w/2], b = 0). The
four positive-weight variants among those eight carry dosage 0 in all 30 bundled individuals, so the
partial score was at most 0 on every row: 23 negative, 7 exactly 0, and the encrypted "class" was
"not high risk" or a tie that CKKS noise decided. Over the 2,504-sample panel it agreed with the
full-score P90 flag less often than a constant "not high risk" would. Beside it, the benchmark card
priced a "genomic risk MLP" (8 -> 64 -> 2, 706 parameters, x33) that nothing trains, and a requested
QSMS prime below the pooled total wrapped the count while Python refused the same field.

WHAT IS ASSERTED, BY RUNNING THE PAGE'S OWN CODE

`runClientPipeline("genomics")` is bundled with esbuild and run in node against the real node-seal
CKKS and the real Polytune wasm where it is built, with `fetch` answering from web/public. The wasm is
gitignored and CI's Python job does not build it, so there a plaintext argmax stands in for the
garbled circuit, as in tests/test_r84_finance_browser_run.py; nothing below but the MPC agreement
depends on which ran, and that check is real only with the module:

  - the row CKKS encrypts is the person's dosage at every variant of the payload, and the model is
    W = [0, w], b = [0, -cut] with the cut the 90th percentile of the payload's own scores, as numpy
    computes it (`percentileLinear` is checked against numpy directly);
  - every bundled individual is scored (all 30, in as many ciphertexts as they need), the decrypted
    score agrees with the plaintext one, the high-risk flag agrees on every row, and the flag comes
    out both ways - the count above the cut is numpy's own count;
  - the log says what is encrypted and what is decrypted, and nothing says "sign test";
  - the benchmark row prices the weighted sum it runs at scale 1, not an MLP;
  - a requested field that cannot hold the pooled count is refused, without moving the row the
    private-inference stage scores.

On the base commit this fails at the first assertion: the row is 8 values wide.

BATCH 6 (2026-11-25): the cut is the person's own super-population's. A pooled 90th percentile over mixed ancestries is
not current practice (pgsc_calc v2.3.0: "Use ancestry-normalised or Z-score outputs"; references/genomics-real-data-2026-11.md),
so the model's bias is the 90th percentile of the person's super-population over all 2,504 individuals (aggregate
statistics the payload carries and the Models card measured), every panel row is judged against its own, and the pooled
cut over the bundled rows is reported beside it as `prs_pooled_cut`. Below, `_cut_and_flags` therefore counts each row
against its own cut, and the pooled cut keeps its own check.

THE CUT AFTER DECRYPTION (2026-11-26, the owner's choice). A person's cut names their super-population, so the page adds it
to the decrypted logits instead of handing it to the evaluator, and web/src/lib/clientFHE.ts stays as the FHE.org poster
cites it. The harness routes clientFHE through a wrapper that records every call the page makes into it;
`test_the_evaluator_is_never_given_a_persons_cut` asserts that no run enters an evaluation other than the two linear ones,
that each of those is handed a zero bias and no per-row bias, and, on seed 0's run, the published weights and the bundled
rows, so the cut is folded into nothing the evaluator sees.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
PAYLOAD = WEB / "public" / "data" / "genomics" / "pgs000783_breast_cancer.json"

HARNESS = r"""
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
const WEB = process.argv[2], OUT = process.argv[3];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
const mpcWasm = resolve(WEB, "src/wasm/qsh_mpc/qsh_mpc_bg.wasm");
await build({
  stdin: { contents: `export { runClientPipeline, assembleBenchmark } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};
export * as realData from ${JSON.stringify(resolve(WEB, "src/lib/realData"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: OUT, logLevel: "error",
  define: { "import.meta.env": '{"BASE_URL":"/"}' },
  // The one Vite-only specifier: the SEAL wasm's `?url` import becomes its path on disk. Without the
  // built Polytune module, a plaintext argmax stands in for it (CI's Python job).
  plugins: [{ name: "wasm-url", setup(b) {
    b.onResolve({ filter: /seal_throws\.wasm\?url$/ }, () => ({ path: "sealwasm", namespace: "wasmurl" }));
    b.onLoad({ filter: /.*/, namespace: "wasmurl" },
             () => ({ contents: `export default ${JSON.stringify(sealWasm)};`, loader: "js" }));
    if (!existsSync(mpcWasm)) {
      b.onResolve({ filter: /(^|\/)clientMPC$/ }, () => ({ path: "fakempc", namespace: "fakempc" }));
      b.onLoad({ filter: /.*/, namespace: "fakempc" }, () => ({ loader: "js", contents: `
        export async function warmMPC() {}
        export async function secureArgmax(l) {
          let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i;
          return { out: [b], compares: Math.max(0, l.length - 1), commBytes: 0, mpcMs: 0, messages: 0, clearOut: [b] };
        }` }));
      b.onResolve({ filter: /\.wasm\?url$|qsh_mpc\.js$/ }, (a) => ({ path: a.path, namespace: "stub" }));
      b.onLoad({ filter: /.*/, namespace: "stub" }, () => ({ contents: "export default '';", loader: "js" }));
    }
    // Any other `?url` import, ONNX Runtime's files among them (web/src/lib/ortSameOrigin.ts, Round 86): its path on
    // disk, as for the SEAL wasm. With the MPC module built the stubs above are not registered, and esbuild had no loader
    // for those .wasm files (2026-11-20, a gate run in a worktree whose wasm an earlier run had built).
    b.onResolve({ filter: /\?url$/ }, (a) => ({ path: resolve(a.resolveDir, a.path.slice(0, -"?url".length)), namespace: "fileurl" }));
    b.onLoad({ filter: /.*/, namespace: "fileurl" }, (a) => ({ contents: `export default ${JSON.stringify(a.path)};`, loader: "js" }));
    // WHAT THE EVALUATOR IS HANDED (2026-11-26). Every import of the CKKS module resolves to a wrapper that re-exports it
    // and records each call into each of its exported functions, and whether the call completed; for the two linear
    // evaluations it also keeps the bias, the per-row bias and, on a detailed run, the weights and rows. The wrapper's own
    // import of the module names the file, which this filter does not match. Calls clientFHE makes inside itself are not
    // seen, which is why the test also requires that the run enters it through no other evaluation.
    const realFhe = resolve(WEB, "src/lib/clientFHE.ts");
    const fheFns = [...readFileSync(realFhe, "utf8").matchAll(/^export (?:async )?function (\w+)/gm)].map((x) => x[1]);
    b.onResolve({ filter: /(^|\/)clientFHE$/ }, () => ({ path: "fhespy", namespace: "fhespy" }));
    b.onLoad({ filter: /.*/, namespace: "fhespy" }, () => ({ loader: "js", resolveDir: WEB, contents: `
      import * as real from ${JSON.stringify(realFhe)};
      export * from ${JSON.stringify(realFhe)};
      const rec = (e) => { (globalThis.__ckksCalls ??= []).push(e); return e; };
      const linear = (fn, [W, b, x, opts]) => {
        const d = { b: Array.from(b ?? []), rowBias: fn === "encryptedLinearBatch" ? (opts?.rowBias ?? null) : null };
        if (globalThis.__ckksDetail) {
          d.W = W.map((r) => Array.from(r));
          d.x = fn === "encryptedLinear" ? Array.from(x) : x.map((r) => Array.from(r));
        }
        return d;
      };
      ${fheFns.map((f) => `export function ${f}(...args) {
        const e = rec({ fn: ${JSON.stringify(f)}, ok: false,
                        ...(${JSON.stringify(f)} === "encryptedLinear" || ${JSON.stringify(f)} === "encryptedLinearBatch"
                            ? linear(${JSON.stringify(f)}, args) : {}) });
        const r = real.${f}(...args);
        if (r && typeof r.then === "function") return r.then((v) => { e.ok = true; return v; });
        e.ok = true;
        return r;
      }`).join("\n")}` }));
  } }],
});
globalThis.fetch = async (url) => {
  const u = String(url);
  if (u.endsWith("qsh_mpc_bg.wasm")) {
    return new Response(readFileSync(mpcWasm), { headers: { "Content-Type": "application/wasm" } });
  }
  const p = join(WEB, "public", u.replace(/^\/+/, ""));
  // The genomics payload's parsed object is kept: realData caches that same object, so a check below can edit one value.
  const keep = (j) => { if (u.endsWith("data/genomics/pgs000783_breast_cancer.json")) globalThis.__prsPayload = j; return j; };
  try { const t = readFileSync(p, "utf8"); return { ok: true, status: 200, json: async () => keep(JSON.parse(t)) }; }
  catch { return { ok: false, status: 404, json: async () => { throw new Error("404 " + u); } }; }
};
const m = await import(`file://${OUT}`);
const run = (config, detail = false) => new Promise((done) => {
  const logs = [];
  globalThis.__ckksCalls = [];
  globalThis.__ckksDetail = detail;
  m.runClientPipeline("genomics", config, {
    onEvent: (e) => { if (e.type === "log") logs.push({ stage: e.stage, message: e.message }); },
    onEnd: (r) => done({ metrics: r.metrics, logs, plot: r.artifacts.encrypted_agreement_plot ?? null,
                         explanation: r.artifacts.explanation ?? null, ckks: globalThis.__ckksCalls.slice(),
                         bench: m.assembleBenchmark("genomics", config, { metrics: r.metrics, artifacts: r.artifacts }) }),
  });
});
const out = { mpcReal: existsSync(mpcWasm), auto: await run({ seed: 0 }, true), small: await run({ seed: 0, prime: 11 }) };
// The scored person changes with the seed; over enough seeds the encrypted flag must come out both ways. Since batch 6
// each person meets their own super-population's cut, which few of the 30 exceed, so the loop runs until it has seen both.
out.seeds = [];
for (let seed = 1; seed <= 120; seed++) {
  if (seed > 12 && new Set(out.seeds.map((r) => r.enc)).size === 2) break;
  const r = await run({ seed });
  out.seeds.push({ seed, score: r.metrics.polygenic_risk_score, cut: r.metrics.prs_high_risk_cut,
                   enc: r.metrics.prs_high_risk_flag, plain: r.metrics.prs_high_risk_flag_plaintext,
                   mpc: r.metrics.mpc_matches_clear, ckks: r.ckks });
}
// numpy's default percentile, reproduced: random vectors and percentiles, checked in Python.
const rnd = (() => { let s = 84; return () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; }; })();
// Guarded, so a tree without these functions fails at an assertion that names them, not here.
const has = (f) => typeof m.realData[f] === "function";
out.percentile = has("percentileLinear") ? [] : null;
for (let k = 0; out.percentile && k < 60; k++) {
  const n = 1 + Math.floor(rnd() * 40), q = [0, 10, 25, 50, 75, 89.9, 90, 95, 99, 100][k % 10];
  const xs = Array.from({ length: n }, () => (rnd() - 0.5) * 4);
  out.percentile.push({ xs, q, v: m.realData.percentileLinear(xs, q) });
}
const panel = has("prsPanel") ? await m.realData.prsPanel() : null;
out.panel = panel && { cut: panel.cut, percentile: panel.percentile, rows: panel.rows.length,
                       variants: panel.weights.length, rowCut: panel.rowCut, superPop: panel.superPop,
                       pooledAll: panel.pooledAll };
out.corpusWidth = ((await m.realData.realInputCorpus("genomics")) ?? []).map((r) => r.length);
// A bundled row's reference cut that is not a number is refused when the panel is read (2026-11-26): added after
// decryption it would be NaN, which every comparison fails. The value is restored after.
const pay = globalThis.__prsPayload;
if (pay && pay.sample_super_population?.length) {
  const g = pay.sample_super_population[0], ref = pay.reference_by_super_population[g], cut = ref.cut;
  ref.cut = null;
  out.nonFiniteCut = await m.realData.prsPanel().then(() => "accepted", (e) => String(e?.message ?? e));
  ref.cut = cut;
} else {
  out.nonFiniteCut = null;
}
console.log(JSON.stringify(out));
"""


def _require_node_or_skip() -> None:
    """Skip without the toolchain, unless the environment says a skip is a failure (CI does)."""
    missing = [name for name, ok in (
        ("web/node_modules/.bin/esbuild", (WEB / "node_modules" / ".bin" / "esbuild").exists()),
        ("node", shutil.which("node") is not None)) if not ok]
    if not missing:
        return
    msg = (f"{' and '.join(missing)} absent, so the genomics page cannot run here. Run `npm ci` in "
           "web/. This is a SKIP, not a pass.")
    if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
        pytest.fail(msg)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:
    _require_node_or_skip()
    tmp = tmp_path_factory.mktemp("genomics_page")
    script = tmp / "harness.mjs"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(tmp / "bundle.mjs")], cwd=WEB,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, "the genomics page did not run in node:\n" + proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _payload() -> tuple[np.ndarray, np.ndarray, dict]:
    d = json.loads(PAYLOAD.read_text())
    return np.asarray(d["dosage_samples"], dtype=float), np.asarray(d["weights"], dtype=float), d


def _pooled_cut() -> float:
    from qsh.usecases.genomics import HIGH_RISK_PERCENTILE

    x, w, _ = _payload()
    return float(np.percentile(x @ w, HIGH_RISK_PERCENTILE))


def _cut_and_flags() -> tuple[np.ndarray, int]:
    """Each bundled row's own cut, its super-population's, and how many rows are above theirs (batch 6)."""
    x, w, d = _payload()
    ref = d["reference_by_super_population"]
    cuts = np.asarray([ref[g]["cut"] for g in d["sample_super_population"]], dtype=float)
    return cuts, int(np.sum(x @ w > cuts))


def test_ckks_evaluates_the_full_score_not_eight_variants(run: dict) -> None:
    x, w, d = _payload()
    m = run["auto"]["metrics"]
    assert m.get("prs_variants") == len(w) == len(d["keys"]), (
        f"the encrypted row is {m.get('prs_variants')} values wide; the score has {len(w)} variants")
    assert run["corpusWidth"] and set(run["corpusWidth"]) == {len(w)}, (
        f"the panel rows are {sorted(set(run['corpusWidth']))} wide, not every variant ({len(w)})")
    assert run["panel"], "realData.ts exports no prsPanel: the cut has no stated source"
    assert run["panel"]["rows"] == len(x) == m["prs_pooled_cut_population"]
    ref = d["reference_by_super_population"][m["prs_reference_super_population"]]
    assert m["prs_cut_population"] == ref["n"] and m["prs_pooled_cut_all_population"] == d["reference_pooled"]["n"] == 2504
    # The explanation is of the same model: every variant is a candidate driver, not eight of them.
    assert run["auto"]["explanation"]["top_features"][0]["feature"] in set(d["keys"])
    assert abs(run["auto"]["explanation"]["efficiency_gap"]) < 1e-9


def test_the_cut_is_the_stated_percentile_of_the_bundled_scores(run: dict) -> None:
    from qsh.usecases.genomics import HIGH_RISK_PERCENTILE

    pooled = _pooled_cut()
    _, _, d = _payload()
    m = run["auto"]["metrics"]
    assert run["panel"], "realData.ts exports no prsPanel"
    assert m["prs_high_risk_percentile"] == HIGH_RISK_PERCENTILE == run["panel"]["percentile"] == d["reference_percentile"]
    # The pooled cut over the bundled rows, as before, now the cautionary one beside the decision.
    assert math.isclose(run["panel"]["cut"], pooled, rel_tol=0, abs_tol=1e-12), (run["panel"]["cut"], pooled)
    assert math.isclose(m["prs_pooled_cut"], round(pooled, 6), abs_tol=1e-12)
    # The decision's cut is the person's super-population's, and each super-population's is the Models card's own
    # measurement over all 2,504 (the full matrix is local only, so the card is the independent record of it).
    card = json.loads((REPO / "models" / "metrics.json").read_text())["genomics.prs"]["measured"]
    g = m["prs_reference_super_population"]
    assert math.isclose(m["prs_high_risk_cut"], d["reference_by_super_population"][g]["cut"], abs_tol=1e-12)
    for grp, ref in d["reference_by_super_population"].items():
        c = card["by_super_population"][grp]
        assert (ref["n"], ref["cut"], ref["mean"], ref["sd"]) == (c["n"], c["p90"], c["mean"], c["sd"]), grp
    assert d["reference_pooled"]["cut"] == card["global_high_risk_cut"]
    assert run["panel"]["rowCut"] == [d["reference_by_super_population"][x]["cut"] for x in d["sample_super_population"]]


def test_the_browser_percentile_is_numpys(run: dict) -> None:
    assert run["percentile"] is not None, "realData.ts exports no percentileLinear"
    for case in run["percentile"]:
        expected = float(np.percentile(np.asarray(case["xs"]), case["q"]))
        assert case["v"] == expected or math.isclose(case["v"], expected, rel_tol=0, abs_tol=1e-15), case


def test_every_bundled_individual_is_scored_and_the_flag_agrees(run: dict) -> None:
    x, w, _ = _payload()
    _, flagged = _cut_and_flags()
    m, plot = run["auto"]["metrics"], run["auto"]["plot"]
    assert m["encrypted_rows_scored"] == m["encrypted_rows_available"] == len(x), (
        f"{m['encrypted_rows_scored']} of {m['encrypted_rows_available']} rows scored")
    per_ct = m["encrypted_rows_per_ciphertext"]
    assert m["encrypted_ciphertexts"] == math.ceil(len(x) / per_ct) == plot["ciphertexts"]
    assert 0 < m["encrypted_batch_max_abs_error"] < 1e-6 and 0 < m["prs_score_abs_error"] < 1e-6
    assert m["encrypted_agreement"] == 1 and m["encrypted_agreement_rows"] == len(x)
    # The test can say yes AND no, and says it where numpy does.
    assert 0 < flagged < len(x), f"numpy flags {flagged} of {len(x)}: the cut no longer separates anyone"
    assert m["prs_panel_flagged_encrypted"] == m["prs_panel_flagged_plaintext"] == flagged
    assert m["prs_high_risk_flag"] == m["prs_high_risk_flag_plaintext"] and m["prs_flags_agree"] == 1
    assert m["encrypted_undecidable_rows"] == 0, "a full-score row sits exactly on the cut"
    # The figure plots the decision logit, score minus cut, so a flagged row sits above zero.
    assert plot["scoreKind"] == "logit" and plot["flipBoundFactor"] == 1
    assert sum(1 for p in plot["points"] if p["plain"] > 0) == flagged
    assert "full score" in plot["rowsAre"] and "its own super-population" in plot["rowsAre"]


def test_the_log_says_what_is_encrypted_and_what_is_decrypted(run: dict) -> None:
    x, w, _ = _payload()
    logs = run["auto"]["logs"]
    text = " ".join(e["message"] for e in logs)
    line = next((e["message"] for e in logs if e["message"].startswith("full polygenic score under CKKS")), None)
    assert line, "no log line describes the full-score evaluation"
    for part in (f"Encrypted: this person's {len(w)} effect-allele dosages", "In plaintext:",
                 "Decrypted: the result ciphertext, read at slot 0", "90th percentile of the plaintext score",
                 "No bias goes in under encryption: this tab subtracts the cut after decryption",
                 f"the flag agrees on {len(x)} of {len(x)}"):
        assert part in line, f"{part!r} missing from: {line}"
    names = " ".join(run["auto"]["metrics"])
    for stale in ("sign test", "8 of the 313", "partial score", "top8", "top-8"):
        assert stale not in text and stale not in names, f"{stale!r} is still in the genomics run"


def test_the_benchmark_prices_the_weighted_sum_it_runs(run: dict) -> None:
    _, w, _ = _payload()
    m = run["auto"]["metrics"]
    assert m["fhe_bench_scale"] == 1, f"the measured dot product is multiplied by {m['fhe_bench_scale']}"
    row = next(r for r in run["auto"]["bench"]["results"] if r["path"] == "MPC-FHE")
    assert "MLP" not in row["model_arch"] and "weighted sum" in row["model_arch"], row["model_arch"]
    assert row["model_params"] == len(w) and row["projected"] is False


def test_a_field_too_small_for_the_pooled_count_is_refused(run: dict) -> None:
    auto, small = run["auto"], run["small"]
    x, _, _ = _payload()
    total = int(x[:, 0].sum())
    assert auto["metrics"]["secure_sum"] == total and auto["metrics"]["qsms_matches_plain_total"] == 1
    assert auto["metrics"]["qsms_prime"] > 2 * len(x)
    err = next((e["message"] for e in small["logs"] if e["stage"] == "qsmpc_compute"
                and "cannot hold the pooled allele count" in e["message"]), None)
    assert err, "a requested GF(11) was used for counts that can reach 60 instead of being refused"
    assert "secure_sum" not in small["metrics"], (
        f"a wrapped count was published: {small['metrics'].get('secure_sum')} for {total}")
    # The refusal advances the run's stream as a completed stage does: the same person is scored.
    assert small["metrics"]["polygenic_risk_score"] == auto["metrics"]["polygenic_risk_score"]
    for e in auto["logs"]:
        if e["stage"] == "qsmpc_compute":
            assert "classical equivalent" not in e["message"], e["message"]


def test_the_scored_person_is_flagged_both_ways(run: dict) -> None:
    """The sign test could only answer "not high risk"; the full score answers both ways, encrypted."""
    seeds = run["seeds"]
    assert seeds and all(r["enc"] is not None for r in seeds), seeds
    for r in seeds:
        assert r["enc"] == r["plain"] == int(r["score"] > r["cut"]) and r["mpc"] == 1, r
    assert {r["enc"] for r in seeds} == {0, 1}, f"over {len(seeds)} seeds the flag never changed: {seeds}"


#: The calls into clientFHE a genomics run may make: its setup, its plaintext argmax and the two linear evaluations whose
#: inputs the test reads. Any other entry point is an evaluation this test cannot see inside, so it fails until checked.
GENOMICS_FHE_CALLS = {"warmFHE", "argmax", "encryptedLinear", "encryptedLinearBatch"}


def test_the_evaluator_is_never_given_a_persons_cut(run: dict) -> None:
    """Each person's cut is their super-population's, so it is added after decryption (the owner's choice of 2026-11-26).

    The harness records every call the page makes into clientFHE. On the run of seed 0, which scores the person and the
    whole panel, on the GF(11) run and on every seed's run, each call completes, none enters an evaluation other than the
    two linear ones, and each linear one is handed a zero bias and no per-row bias. On seed 0's run the cut is folded into
    nothing else the evaluator sees either: the weights are [0, w] as published and the rows are the bundled dosages. The
    decisions still meet numpy's against each person's own cut (the tests above), so the cut did arrive, after decryption.
    """
    x, w, _ = _payload()
    auto = run["auto"]["ckks"]
    assert {c["fn"] for c in auto} >= {"encryptedLinear", "encryptedLinearBatch"}, (
        f"the run asked for {sorted({c['fn'] for c in auto})}: one person and the panel should both be scored under CKKS")
    runs = [("seed 0", auto), ("the GF(11) run", run["small"]["ckks"])]
    runs += [(f"seed {r['seed']}", r["ckks"]) for r in run["seeds"]]
    for label, calls in runs:
        assert any(c["fn"] == "encryptedLinear" for c in calls), f"{label}: the person was not scored under CKKS"
        for c in calls:
            assert c["ok"], f"{label}: clientFHE.{c['fn']} did not complete"
            assert c["fn"] in GENOMICS_FHE_CALLS, (
                f"{label}: the genomics run called clientFHE.{c['fn']}; check what it hands the evaluator, then allow it")
            if c["fn"] in ("encryptedLinear", "encryptedLinearBatch"):
                assert all(v == 0 for v in c["b"]), f"{label}: {c['fn']} was handed the bias {c['b']}"
                assert c["rowBias"] is None, f"{label}: {c['fn']} was handed a per-row bias"
    model = np.vstack([np.zeros_like(w), w])
    for c in (c for c in auto if c["fn"] in ("encryptedLinear", "encryptedLinearBatch")):
        assert np.array_equal(np.asarray(c["W"], dtype=float), model), f"{c['fn']} was handed weights other than [0, w]"
    single = next(c for c in auto if c["fn"] == "encryptedLinear")
    assert any(np.array_equal(np.asarray(single["x"], dtype=float), row) for row in x), (
        "the row encrypted for the person is not one of the bundled individuals' dosages")
    batch = next(c for c in auto if c["fn"] == "encryptedLinearBatch")
    assert np.array_equal(np.asarray(batch["x"], dtype=float), x), "the panel's encrypted rows are not the bundled dosages"


def test_a_reference_cut_that_is_not_a_number_is_refused(run: dict) -> None:
    """Under encryption the encoder refused a non-finite bias by name; added after decryption, a NaN cut would pass as a
    row that agrees and is never flagged. So the panel refuses it when read (the harness nulls one bundled row's cut)."""
    msg = run["nonFiniteCut"]
    assert msg is not None, "the harness did not capture the genomics payload, so the refusal was not exercised"
    assert msg != "accepted" and "finite cut" in msg, f"a null reference cut was not refused: {msg}"
