"""The CBOM graph's run of the checks, its replay, its years and its broken copies (Round 86, 2026-11-23).

WHY THIS EXISTS

The owner asked on 2026-11-23 that the CBOM graph be visualised and verified dynamically by the page's own controls, as
the E2E simulator's is, before reaching for WebGL or WebGPU. `web/src/components/CbomLive.tsx` does it in SVG: a run
that fetches the file again, hashes it, checks it against the schema and reads every asset against NIST IR 8547 one
event at a time; a replay slider; the years from 2026 to 2037 played on the panel's date; a selected asset's details;
and broken copies the in-tab schema check must refuse. The logic is pure (`web/src/lib/cbomRun.ts`), so this file runs it
in node on the published CBOM and holds:

1. a run's events: the fetch, the digest against the build's, the schema check, one reading per asset in the document's
   order with the standing `ir8547.ts` gives, the combiner's parts from the document's own dependencies, and a summary
   that passes only when every check did; a wrong digest or an invalid document fails the run;
2. the replay: after k events exactly the assets read so far are lit, and the combiner's edges appear only with the
   event that reads its parts;
3. the years: each asset's standing year by year equals the Python twin's, X25519 and ECDH P-256 turning deprecated in
   2031 and disallowed in 2036;
4. every broken copy refused by the page's validator and by Python's jsonschema, the damage in the asset it names;
5. the panel using all of it, in SVG, with keyboard selection.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

from qsh.pqc import ir8547 as py

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web"
ESBUILD = WEB / "node_modules" / ".bin" / "esbuild"
CBOM = WEB / "public" / "cbom" / "qsh.cdx.json"
SCHEMAS = REPO / "schemas" / "cyclonedx"
DATE = "2026-11-23"

RUNNER = r"""
import { readFileSync } from "node:fs";
const m = await import(%(bundle)s);
const bytes = readFileSync(%(cbom)s);
const doc = JSON.parse(new TextDecoder().decode(bytes));
const sha = await m.sha256Hex(bytes);
const schema = await m.checkCbom(doc);
const base = { doc, bytes: bytes.length, sha256: sha, builtSha256: sha, schema, isoDate: %(date)s };
const out = {};
out.events = m.cbomRunEvents(base);
out.staleDigest = m.cbomRunEvents({ ...base, builtSha256: "0".repeat(64) });
out.invalid = m.cbomRunEvents({ ...base, schema: { valid: false, error: "/x: wrong" } });
const n = out.events.length;
out.states = [0, 3, 4, 10, 27, n - 2, n - 1, n].map((k) => { const s = m.graphStateAt(out.events, k, %(date)s);
  return { k, read: [...s.read.entries()], edges: [...s.edges], current: s.current, date: s.date }; });
out.years = m.standingsByYear(doc.components.map((c) => c["bom-ref"]), 2026, 2037);
out.broken = [];
for (const b of m.BROKEN_COPIES) {
  const copy = b.apply(doc);
  const r = await m.checkCbom(copy);
  out.broken.push({ id: b.id, ref: b.ref, label: b.label, valid: r.valid, error: r.error ?? null, copy });
}
out.untouched = JSON.stringify(doc) === new TextDecoder().decode(bytes).trim() || JSON.stringify(doc) === JSON.stringify(JSON.parse(new TextDecoder().decode(bytes)));
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> dict:
    missing = [n for n, ok in (("web/node_modules/.bin/esbuild", ESBUILD.exists()), ("node", shutil.which("node"))) if not ok]
    if missing:
        msg = f"{' and '.join(missing)} absent, so the browser code cannot run. This is a SKIP, not a pass."
        if os.environ.get("QSH_REQUIRE_BROWSER_GATES") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    d = tmp_path_factory.mktemp("cbomrun")
    entry, bundle, runner = d / "entry.ts", d / "bundle.mjs", d / "run.mjs"
    entry.write_text(f"export * from {json.dumps(str(WEB / 'src/lib/cbomRun'))};\n"
                     f"export {{ sha256Hex }} from {json.dumps(str(WEB / 'src/lib/cbomReport'))};\n"
                     f"export {{ checkCbom }} from {json.dumps(str(WEB / 'src/lib/cbomSchemaCheck'))};\n")
    subprocess.run([str(ESBUILD), str(entry), "--bundle", "--format=esm", "--platform=node", f"--outfile={bundle}",
                    "--log-level=error"], cwd=WEB, check=True, capture_output=True)
    runner.write_text(RUNNER % {"bundle": json.dumps(bundle.as_uri()), "cbom": json.dumps(str(CBOM)), "date": json.dumps(DATE)})
    proc = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=300, cwd=WEB)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


def _doc() -> dict:
    return json.loads(CBOM.read_text(encoding="utf-8"))


def test_a_run_emits_every_check_and_every_asset_in_order(page: dict) -> None:
    ev, doc = page["events"], _doc()
    kinds = [e["kind"] for e in ev]
    n = len(doc["components"])
    assert kinds == ["fetch", "digest", "schema"] + ["read"] * n + ["parts", "summary"], kinds
    assert [e["ok"] for e in ev[:3]] == [True, True, True]
    assert f"{CBOM.stat().st_size:,} bytes" in ev[0]["text"]
    reads = ev[3:3 + n]
    assert [e["ref"] for e in reads] == [c["bom-ref"] for c in doc["components"]]
    for e in reads:
        _, label = py.standing_on(py.BY_REF[e["ref"]], DATE)
        assert e["text"].endswith(f"on {DATE}, {label}"), e
    parts = ev[-2]
    assert parts["ref"] == "crypto/hybrid-combiner" and parts["ok"] is True
    assert parts["to"] == ["crypto/ml-kem-768", "crypto/ml-kem-1024", "crypto/x25519", "crypto/hkdf-sha256"]
    assert ev[-1]["ok"] is True and ev[-1]["text"] == f"Done: every check passed, {n} assets read on {DATE}"


def test_a_wrong_digest_or_an_invalid_document_fails_the_run(page: dict) -> None:
    stale, invalid = page["staleDigest"], page["invalid"]
    assert stale[1]["ok"] is False and "NOT the file this build was made from" in stale[1]["text"]
    assert stale[-1]["ok"] is False and stale[-1]["text"] == "Done: 1 check failed"
    assert invalid[2]["ok"] is False and "/x: wrong" in invalid[2]["text"]
    assert invalid[-1]["ok"] is False


def test_the_replay_lights_exactly_what_has_been_read(page: dict) -> None:
    refs = [c["bom-ref"] for c in _doc()["components"]]
    states = {s["k"]: s for s in page["states"]}
    n = len(page["events"])
    assert states[0]["read"] == [] and states[3]["read"] == [], "nothing is read before the three checks are done"
    assert [r for r, _ in states[4]["read"]] == refs[:1] and states[4]["current"] == refs[0]
    assert [r for r, _ in states[10]["read"]] == refs[:7] and states[10]["current"] == refs[6]
    done = 3 + len(refs)   # the three checks, then one state per asset read
    assert [r for r, _ in states[done]["read"]] == refs and states[done]["edges"] == []
    assert states[n - 1]["edges"] == [f"crypto/hybrid-combiner>{t}" for t in
                                      ("crypto/ml-kem-768", "crypto/ml-kem-1024", "crypto/x25519", "crypto/hkdf-sha256")]
    assert states[n]["edges"] == states[n - 1]["edges"] and states[n]["date"] == DATE
    for ref, standing in states[n]["read"]:
        assert standing == py.standing_on(py.BY_REF[ref], DATE)[0], ref


def test_the_years_are_ir_8547_s(page: dict) -> None:
    years = {y["year"]: y["standings"] for y in page["years"]}
    assert sorted(years) == list(range(2026, 2038))
    for year, standings in years.items():
        for ref, standing in standings.items():
            assert standing == py.standing_on(py.BY_REF[ref], f"{year}-07-01")[0], (year, ref)
    for ref in ("crypto/x25519", "crypto/ecdh-p256"):
        seq = [years[y][ref] for y in range(2026, 2038)]
        assert seq[:5] == [seq[0]] * 5 and seq[0] in ("quantum-vulnerable", "approved-quantum-vulnerable")
        assert seq[5:10] == ["deprecated"] * 5 and seq[10:] == ["disallowed"] * 2, (ref, seq)
    assert {years[y]["crypto/ml-kem-768"] for y in years} == {"post-quantum"}


def _python_valid(doc: dict) -> bool:
    registry = Registry()
    for name in ("bom-1.7.schema.json", "spdx.schema.json", "jsf-0.82.schema.json", "cryptography-defs.schema.json"):
        data = json.loads((SCHEMAS / name).read_text())
        resource = Resource.from_contents(data, default_specification=DRAFT7)
        registry = registry.with_resource(uri=data.get("$id", name), resource=resource).with_resource(uri=name, resource=resource)
    root = json.loads((SCHEMAS / "bom-1.7.schema.json").read_text())
    return not list(jsonschema.validators.validator_for(root)(root, registry=registry).iter_errors(doc))


def test_every_broken_copy_is_refused_where_it_is_broken(page: dict) -> None:
    doc = _doc()
    names = {c["bom-ref"]: c["name"] for c in doc["components"]}
    index = {c["bom-ref"]: i for i, c in enumerate(doc["components"])}
    assert len(page["broken"]) == 5
    for b in page["broken"]:
        assert b["valid"] is False, f"{b['id']}: the page's validator accepted a broken copy"
        assert not _python_valid(b["copy"]), f"{b['id']}: Python's jsonschema accepts the copy, so it is not broken"
        if b["ref"] is None:
            assert "bomFormat" in b["error"]
            continue
        assert names[b["ref"]].split(" ")[0] in b["label"], b
        where = f"/components/{index[b['ref']]}/" if b["id"] != "duplicate-dependency" else "/dependencies/1/dependsOn"
        assert b["error"].startswith(where), b["error"]
    assert page["untouched"], "a broken copy changed the document it was made from"


def test_the_panel_runs_it_in_svg_with_keyboard_selection() -> None:
    live = (WEB / "src" / "components" / "CbomLive.tsx").read_text(encoding="utf-8")
    graph = (WEB / "src" / "components" / "CbomGraph.tsx").read_text(encoding="utf-8")
    panel = (WEB / "src" / "components" / "CbomPanel.tsx").read_text(encoding="utf-8")
    for needed in ("cbomRunEvents(", "graphStateAt(", "standingsByYear(", "BROKEN_COPIES", 'cache: "no-store"', "checkCbom(", "sha256Hex(",
                   'type="range"', "prefers-reduced-motion"):
        assert needed in live, needed
    assert "<CbomLive" in panel
    assert "<svg" in graph and "<canvas" not in graph and "WebGL" in graph, "SVG, and the reason it is not WebGL written beside it"
    for needed in ('tabIndex={onSelect ? 0 : undefined}', 'role={onSelect ? "button" : undefined}', 'e.key === "Enter"', "aria-pressed"):
        assert needed in graph, needed
