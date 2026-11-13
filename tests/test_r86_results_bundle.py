"""One run's results bundle carries what a reader needs and nothing a party keeps (Round 86).

The use-case page's "Results (JSON)" button (web/src/lib/resultsBundle.ts) writes one run's parameters, metrics, log,
each card's rows, every published model's re-scoring and the signed record. Held here, on runs of the browser pipeline
in node built as the client-only demo (the harness of tests/test_r84_signed_record_names_what_ran.py), for finance and
genomics:

1. the signature verifies from the bundle alone, after a JSON round trip, and fails on a changed metric. Until this
   round the record kept only the signature's first 12 bytes, so the public key and payload it recorded "so a reader
   can re-check it" could not be checked;
2. nothing a party keeps is in it: no key named like a secret anywhere, no parameter beginning with an underscore
   (a `_sessionKey` is passed in to be sure), none of the genomics cohorts' own noisy contributions, and none of the
   PSI's member IDs;
3. each page's cards are there: finance's count as published and every model's re-scoring, genomics' release.
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

HARNESS = r"""
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
const WEB = process.argv[2];
const { build } = createRequire(WEB + "/package.json")("esbuild");
const dir = mkdtempSync(join(tmpdir(), "qsh-r86-bundle-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};
export { buildResultsBundle } from ${JSON.stringify(resolve(WEB, "src/lib/resultsBundle"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
  // The client-only demo's build flag: the branch of the record this test is about.
  define: { "import.meta.env": '{"BASE_URL":"/","VITE_CLIENT_ONLY":"1"}', "process.env.NODE_ENV": '"production"' },
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
const { ml_dsa65 } = await import(pathToFileURL(resolve(WEB, "node_modules/@noble/post-quantum/ml-dsa.js")).href);
rmSync(dir, { recursive: true, force: true });
const hex = (h) => Uint8Array.from(h.match(/../g).map((b) => parseInt(b, 16)));
const result = {};
for (const usecase of ["finance_aml", "genomics"]) {
  const metrics = {}, events = [];
  const artifacts = await new Promise((done) => {
    const timer = setTimeout(() => done(null), 600_000);
    m.runClientPipeline(usecase, { seed: 0, usecase }, {
      onEvent: (e) => { events.push(e); if (e.type === "metric") metrics[e.data.name] = e.data.value; },
      onEnd: (s) => { clearTimeout(timer); done(s.artifacts); },
    });
  });
  const bundle = m.buildResultsBundle({ usecase, metrics, artifacts, events, bundleAsset: "assets/index-test.js",
    params: { seed: 0, kem: "ml-kem-768+x25519", _sessionKey: new Uint8Array(32).fill(7), nested: { a: 1 } } });
  const text = JSON.stringify(bundle);
  const b = JSON.parse(text);
  const s = b.signature;
  const msg = new TextEncoder().encode(JSON.stringify(s.signed_payload));
  const verifies = ml_dsa65.verify(hex(s.signature_hex), msg, hex(s.public_key_hex));
  const changed = structuredClone(s.signed_payload);
  changed.metrics.seed_tampered = 1;
  const tamperedVerifies = ml_dsa65.verify(hex(s.signature_hex), new TextEncoder().encode(JSON.stringify(changed)),
                                           hex(s.public_key_hex));
  result[usecase] = {
    algorithm: s.algorithm, verifies, tamperedVerifies, signatureBytes: s.signature_hex.length / 2,
    text, params: b.params, cards: Object.keys(b.cards),
    contributions: artifacts.genomicsDp?.contributions ?? null,
    psiMembers: artifacts.psi_intersection ?? null,
  };
}
// To a file, not stdout: a pipe is written asynchronously and process.exit does not wait for it, so a result this
// size (each bundle carries its run's whole log) arrived cut off near 47 kB.
writeFileSync(process.argv[3], JSON.stringify(result));
process.exit(0);
"""

SECRET_KEYS = ("secretkey", "privatekey", "secret_key", "private_key", "sessionkey", "_sessionkey", "seeds", "pairseeds")


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
def runs(tmp_path_factory) -> dict:
    _require_node_or_skip()
    d = tmp_path_factory.mktemp("r86bundle")
    script, out = d / "harness.mjs", d / "result.json"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(out)], cwd=WEB, capture_output=True, text=True,
                          timeout=1500)
    assert proc.returncode == 0, f"the browser pipeline did not run:\n{proc.stderr[-3000:]}"
    return json.loads(out.read_text())


def _keys(o, out: set[str]) -> set[str]:
    if isinstance(o, dict):
        for k, v in o.items():
            out.add(k.lower())
            _keys(v, out)
    elif isinstance(o, list):
        for v in o:
            _keys(v, out)
    return out


@pytest.mark.parametrize("usecase", ["finance_aml", "genomics"])
def test_the_signature_verifies_from_the_bundle_alone(runs, usecase) -> None:
    r = runs[usecase]
    assert r["algorithm"].startswith("ML-DSA"), r["algorithm"]
    assert r["signatureBytes"] == 3309, "ML-DSA-65's signature is 3,309 bytes: the bundle must carry all of it"
    assert r["verifies"] is True
    assert r["tamperedVerifies"] is False


@pytest.mark.parametrize("usecase", ["finance_aml", "genomics"])
def test_nothing_a_party_keeps_is_in_the_bundle(runs, usecase) -> None:
    r = runs[usecase]
    bundle = json.loads(r["text"])
    assert set(r["params"]) == {"seed", "kem"}, r["params"]   # no underscore key, nothing that is not a plain value
    bad = [k for k in _keys(bundle, set()) if k in SECRET_KEYS or k == "contributions"]
    assert not bad, f"keys that hold what a party keeps: {bad}"
    if r["contributions"]:
        assert json.dumps(r["contributions"]) not in r["text"]
    for member in r["psiMembers"] or []:
        assert member not in r["text"], f"a PSI member reached the bundle: {member}"


def test_each_page_carries_its_cards(runs) -> None:
    assert {"differential_privacy", "verifications"} <= set(runs["finance_aml"]["cards"]), runs["finance_aml"]["cards"]
    assert "differential_privacy" in runs["genomics"]["cards"], runs["genomics"]["cards"]
    fin = json.loads(runs["finance_aml"]["text"])["cards"]["differential_privacy"]["rows"][0]
    assert fin["epsilon_asked"] == 1 and fin["delta"] == 1e-8 and fin["variance"] == 38
    gen = json.loads(runs["genomics"]["text"])["cards"]["differential_privacy"]["rows"][0]
    assert gen["released_count"] is not None and "contributions" not in gen
