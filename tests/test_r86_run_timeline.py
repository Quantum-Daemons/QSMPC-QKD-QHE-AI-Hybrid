"""The run's log replays to what the run showed, and its per-party events say only what may be said (Round 86).

The timeline (`web/src/lib/runTimeline.ts`, the owner's choice of 2026-11-14) draws the canvas, the log, the metrics table
and "Who holds what" from the recorded log at a cursor, and draws a finished run's animation by replaying it. Held here
on runs of the browser pipeline in node, built as the client-only demo (the harness of
tests/test_r84_signed_record_names_what_ran.py), for finance, genomics and the satellite page:

1. the replay is the run: the log folded to its end gives every stage done, none active, and exactly the metrics the
   run ended with, so nothing a page shows at the end is missing from the log; at each stage boundary the fold names
   the stage that is running;
2. every `message` names two parties, a protection from the vocabulary and a size that is a whole number of bytes or
   none, and carries hex only for what is sent in the clear; every `holds` names a party;
3. nothing a party keeps is in the log: not the session key, in hex or as numbers, and no key named like a secret;
4. the sizes are the run's own: the handshake's two messages are the ML-KEM-768 key and ciphertext plus 32 bytes of
   X25519, the signed record carries the ML-DSA-65 signature and key, and the secure sum and the federation count
   their parties;
5. the replay helpers on a log written by hand: the marks, back and forward, and the stage at a position.
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
const dir = mkdtempSync(join(tmpdir(), "qsh-r86-timeline-"));
const out = join(dir, "bundle.mjs");
const sealWasm = resolve(WEB, "node_modules/node-seal/dist/seal_throws.wasm");
await build({
  stdin: { contents: `export { runClientPipeline } from ${JSON.stringify(resolve(WEB, "src/lib/clientCompute"))};
export { foldRun, stageMarks, prevMark, nextMark, stageAt, ledgerAt, hexPrefix } from ${JSON.stringify(resolve(WEB, "src/lib/runTimeline"))};`,
           resolveDir: WEB, loader: "ts" },
  bundle: true, format: "esm", platform: "node", outfile: out, logLevel: "error",
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
rmSync(dir, { recursive: true, force: true });
const toHex = (u) => [...u].map((b) => b.toString(16).padStart(2, "0")).join("");
const result = { runs: {} };
for (const usecase of ["finance_aml", "genomics", "satellite_comms"]) {
  // `qfl` as the page sets it on finance (usePipelineRun's buildConfig), so the federated stage runs there.
  const config = { seed: 0, usecase, ...(usecase === "finance_aml" ? { qfl: true } : {}) };
  const events = [];
  const end = await new Promise((done) => {
    const timer = setTimeout(() => done(null), 600_000);
    m.runClientPipeline(usecase, config, {
      onEvent: (e) => events.push(e),
      onEnd: (s) => { clearTimeout(timer); done(s); },
    });
  });
  const marks = m.stageMarks(events);
  const atMarks = marks.map((k) => { const f = m.foldRun(events, k); return { k, active: f.active, done: [...f.done] }; });
  const last = m.foldRun(events);
  const ledger = m.ledgerAt(events);
  const session = config._sessionKey;
  result.runs[usecase] = {
    state: end?.state ?? null, n: events.length, marks, atMarks,
    stageEvents: events.map((e, i) => ({ i, type: e.type, stage: e.stage })).filter((e) => e.type === "stage_start" || e.type === "stage_end"),
    lastActive: last.active, lastDone: [...last.done], foldedMetrics: last.metrics, endMetrics: end?.metrics ?? null,
    messages: ledger.messages, holdings: ledger.holdings,
    perParty: events.filter((e) => e.type === "message" || e.type === "holds"),
    logText: JSON.stringify(events),
    sessionHex: session ? toHex(session) : null,
    sessionCsv: session ? [...session].join(",") : null,
  };
}
// The helpers on a log written by hand.
const hand = [
  { type: "stage_start", stage: "s1", message: "", data: {}, t: 0 },
  { type: "metric", stage: "s1", message: "", data: { name: "x", value: 1 }, t: 0.1 },
  { type: "stage_end", stage: "s1", message: "", data: {}, t: 0.2 },
  { type: "stage_start", stage: "s2", message: "", data: {}, t: 0.3 },
  { type: "metric", stage: "s2", message: "", data: { name: "x", value: 2 }, t: 0.4 },
  { type: "stage_end", stage: "s2", message: "", data: {}, t: 0.5 },
];
const hm = m.stageMarks(hand);
result.hand = {
  marks: hm, prevOf3: m.prevMark(hm, 3), nextOf3: m.nextMark(hm, 3), prevOf0: m.prevMark(hm, 0), nextOfEnd: m.nextMark(hm, 6),
  at2: (() => { const f = m.foldRun(hand, 2); return { active: f.active, done: [...f.done], x: f.metrics.x }; })(),
  at5: (() => { const f = m.foldRun(hand, 5); return { active: f.active, done: [...f.done], x: f.metrics.x }; })(),
  stageAt0: m.stageAt(hand, 0), stageAt4: m.stageAt(hand, 4),
  hex: m.hexPrefix(Uint8Array.from([0, 1, 2, 255, 16, 32, 64, 128, 7]), 8), hexShort: m.hexPrefix(Uint8Array.from([10, 11]), 8),
};
writeFileSync(process.argv[3], JSON.stringify(result));
process.exit(0);
"""

PROTECTIONS = {"public", "session key", "CKKS", "BGV", "garbled circuit", "additive shares", "pairwise masks",
               "photons (simulated)", "classical (simulated)"}
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
def result(tmp_path_factory) -> dict:
    _require_node_or_skip()
    d = tmp_path_factory.mktemp("r86timeline")
    script, out = d / "harness.mjs", d / "result.json"
    script.write_text(HARNESS)
    proc = subprocess.run(["node", str(script), str(WEB), str(out)], cwd=WEB, capture_output=True, text=True,
                          timeout=1800)
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


@pytest.mark.parametrize("usecase", ["finance_aml", "genomics", "satellite_comms"])
def test_the_replay_ends_where_the_run_ended(result, usecase) -> None:
    r = result["runs"][usecase]
    assert r["state"] == "done", r["state"]
    ended = sorted({e["stage"] for e in r["stageEvents"] if e["type"] == "stage_end"})
    assert r["lastActive"] is None and sorted(r["lastDone"]) == ended and len(ended) >= 5
    # Every metric the run ended with is in the log, at the value it ended with.
    assert r["foldedMetrics"] == r["endMetrics"], (
        sorted(set(r["endMetrics"]) ^ set(r["foldedMetrics"])) or
        [k for k in r["endMetrics"] if r["endMetrics"][k] != r["foldedMetrics"].get(k)])


@pytest.mark.parametrize("usecase", ["finance_aml", "genomics", "satellite_comms"])
def test_each_stage_boundary_replays_its_stage(result, usecase) -> None:
    r = result["runs"][usecase]
    assert r["marks"][0] == 0 and r["marks"][-1] == r["n"]
    by_pos = {a["k"]: a for a in r["atMarks"]}
    for e in r["stageEvents"]:
        at = by_pos[e["i"] + 1]
        if e["type"] == "stage_start":
            assert at["active"] == e["stage"], (e, at)
        else:
            assert at["active"] is None and e["stage"] in at["done"], (e, at)


@pytest.mark.parametrize("usecase", ["finance_aml", "genomics", "satellite_comms"])
def test_every_per_party_event_says_only_what_may_be_said(result, usecase) -> None:
    r = result["runs"][usecase]
    assert r["messages"], "the run reported no message between its parties"
    for m in r["messages"]:
        assert m["from"] and m["to"] and m["what"], m
        assert m["protection"] in PROTECTIONS, m
        assert m["bytes"] is None or (isinstance(m["bytes"], int) and m["bytes"] > 0), m
        if "hex" in m:
            assert m["protection"] == "public", f"hex of something not sent in the clear: {m}"
            assert len(m["hex"].rstrip(".")) <= 16 and all(c in "0123456789abcdef" for c in m["hex"].rstrip(".")), m
    for party, held in r["holdings"].items():
        assert party and all(h["what"] and isinstance(h["kept"], bool) for h in held), (party, held)
    bad = [k for k in _keys(r["perParty"], set()) if k in SECRET_KEYS]
    assert not bad, f"per-party events carry keys named like a secret: {bad}"


@pytest.mark.parametrize("usecase", ["finance_aml", "genomics", "satellite_comms"])
def test_the_session_key_is_nowhere_in_the_log(result, usecase) -> None:
    r = result["runs"][usecase]
    assert r["sessionHex"] and len(r["sessionHex"]) == 64, "the run made no session key to look for"
    assert r["sessionHex"] not in r["logText"]
    assert r["sessionHex"][:16] not in r["logText"], "the session key's first 8 bytes are in the log"
    assert r["sessionCsv"] not in r["logText"]


def test_the_sizes_are_the_run_s_own(result) -> None:
    fin = result["runs"]["finance_aml"]
    hs = [m for m in fin["messages"] if m["stage"] == "pqc_handshake"]
    assert [(m["from"], m["to"], m["bytes"]) for m in hs] == [("b", "a", 1184 + 32), ("a", "b", 1088 + 32)], hs
    assert fin["endMetrics"]["kem_ciphertext_bytes"] + fin["endMetrics"]["classical_public_key_bytes"] == hs[1]["bytes"]
    sig = [m for m in fin["messages"] if m["stage"] == "sign_result"]
    assert len(sig) == 1 and sig[0]["bytes"] > 3309 + 1952 and sig[0]["hex"], sig
    psi = [m for m in fin["messages"] if m["stage"] == "qsmpc_compute"]
    assert [(m["from"], m["to"], m["protection"]) for m in psi] == [("a", "b", "session key"), ("b", "a", "session key")], psi
    assert all(m["bytes"] and m["bytes"] > 0 for m in psi)
    gen = result["runs"]["genomics"]
    shares = [m for m in gen["messages"] if m["protection"] == "additive shares"]
    assert len(shares) == 1 and shares[0]["from"] == "many" and shares[0]["count"] == gen["endMetrics"]["qsms_parties"], shares
    fed = [m for m in fin["messages"] if m["stage"] == "federated_training"]
    assert fed and all(m["from"] in ("many", "b") for m in fed), fed
    sat = result["runs"]["satellite_comms"]
    photons = [m for m in sat["messages"] if m["stage"] == "satellite_qkd" and m["protection"] == "photons (simulated)"]
    assert len(photons) == 1 and (photons[0]["from"], photons[0]["to"]) == ("b", "a"), photons
    assert all(h["kept"] for p in ("a", "b") for h in sat["holdings"][p] if "session key" in h["what"])


def test_the_helpers_on_a_log_written_by_hand(result) -> None:
    h = result["hand"]
    assert h["marks"] == [0, 1, 3, 4, 6]
    assert (h["prevOf3"], h["nextOf3"], h["prevOf0"], h["nextOfEnd"]) == (1, 4, 0, 6)
    assert h["at2"] == {"active": "s1", "done": [], "x": 1}
    assert h["at5"] == {"active": "s2", "done": ["s1"], "x": 2}
    assert (h["stageAt0"], h["stageAt4"]) == (None, "s2")
    assert h["hex"] == "000102ff10204080..." and h["hexShort"] == "0a0b"
