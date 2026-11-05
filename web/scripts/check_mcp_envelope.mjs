// The browser's MCP request must be the Python channel's text, and the AI-generated record must read
// back in both languages.
//
// WHY THIS EXISTS (Round 86)
//
// The channel was described as MCP while its messages had no id and no `_meta`, and the browser's
// comment said it signed "the exact same bytes" as Python, which wrote spaces after every separator.
// Comments cannot hold two implementations together. `scripts/export_mcp_vectors.py` commits the
// Python functions' text and this reproduces it through the PRODUCTION module, bundled the way
// `check_hybrid_combiner.mjs` bundles it.
//
// The record gets a shared artifact too: `tests/fixtures/ai_provenance_sample.json` is a file this
// script produced with the production signer. Every run checks it here, and the Python test checks the
// same bytes with `scripts/verify_ai_provenance.py`, so neither side's format can move alone.
// Regenerate with: node web/scripts/check_mcp_envelope.mjs --write-fixture
//
// Run: node web/scripts/check_mcp_envelope.mjs   (npm run check:mcp-envelope; CI gate)

import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = resolve(WEB, "..");
const FIXTURE = join(ROOT, "tests/fixtures/ai_provenance_sample.json");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const dir = mkdtempSync(join(tmpdir(), "qsh-mcp-envelope-"));
const entry = join(dir, "entry.ts");
writeFileSync(entry, `
export { mcpRequest, stableStringify, clientMcpDemo, clientSecureWrap, MCP_PROTOCOL_VERSION } from ${JSON.stringify(join(WEB, "src/lib/clientMcp"))};
export { signProvenance, checkProvenance, PROVENANCE_SPEC } from ${JSON.stringify(join(WEB, "src/lib/aiProvenance"))};
`);
const out = join(dir, "bundle.mjs");
execFileSync("npx", ["esbuild", entry, "--bundle", "--format=esm", "--platform=node",
                     "--external:/node_modules/*", `--outfile=${out}`, "--log-level=error"],
             { cwd: WEB, stdio: "pipe" });
const m = await import(`file://${out}`);

// ---------------------------------------------------------------------------------------------
// 1. Every committed request, reproduced byte for byte.
// ---------------------------------------------------------------------------------------------
const committed = JSON.parse(readFileSync(join(WEB, "scripts/vectors/mcp_envelope_vectors.json"), "utf8"));
check("vectors present", committed.vectors.length >= 2, `only ${committed.vectors.length} vectors`);
check("one revision", committed.protocol_version === m.MCP_PROTOCOL_VERSION,
      `Python follows ${committed.protocol_version}, the browser ${m.MCP_PROTOCOL_VERSION}`);
for (const v of committed.vectors) {
  const got = m.stableStringify(m.mcpRequest(v.client, v.method, v.params, committed.version));
  check(`vector ${v.client}`, got === v.body,
        `the browser wrote\n  ${got}\nand Python committed\n  ${v.body}`);
}

// ---------------------------------------------------------------------------------------------
// 2. The real channel reports the request it signed and sealed.
// ---------------------------------------------------------------------------------------------
for (const qkd of [false, true]) {
  const r = await m.clientMcpDemo(qkd);
  check(`demo qkd=${qkd} revision`, r.protocol_version === committed.protocol_version, r.protocol_version);
  check(`demo qkd=${qkd} id`, r.request_id === 1, r.request_id);
  check(`demo qkd=${qkd} verified`, r.authenticated === true && r.tamper_rejected === true,
        `authenticated ${r.authenticated}, tamper rejected ${r.tamper_rejected}`);
}
const wrapped = await m.clientSecureWrap({ prompt: "p", answer: "a" }, undefined, "score_transaction");
check("wrap keeps the payload", wrapped.echoedPayload?.answer === "a", JSON.stringify(wrapped.echoedPayload));

// ---------------------------------------------------------------------------------------------
// 3. The AI-generated record: a good file passes, and each edit fails the check it should.
// ---------------------------------------------------------------------------------------------
const TEXT = "A short answer from the model, with non-ASCII: Ωmega é.";
const input = { text: TEXT, engine: "transformers", model: "Xenova/distilgpt2",
                generatedAt: "2026-11-05T00:00:00.000Z", signedAt: new Date("2026-11-05T00:00:01.000Z") };
const record = await m.signProvenance(input);
const good = { run: { text: TEXT, engine: input.engine, model: input.model },
               mcp: { echoedPayload: { answer: TEXT } }, ai_provenance: record };

if (process.argv.includes("--write-fixture")) {
  writeFileSync(FIXTURE, JSON.stringify(good, null, 2) + "\n");
  console.log(`wrote ${FIXTURE}`);
}

const ok = await m.checkProvenance(good);
check("good file", ok.signature_valid && ok.text_matches && ok.copy_matches === true && ok.ai_generated === true,
      JSON.stringify(ok));
check("good file names its spec", ok.record?.spec === m.PROVENANCE_SPEC, JSON.stringify(ok.record));

const editedText = await m.checkProvenance({ ...good, run: { ...good.run, text: TEXT + " edited" } });
check("edited text", editedText.signature_valid && !editedText.text_matches && editedText.problem,
      JSON.stringify(editedText));

const editedBody = await m.checkProvenance({ ...good,
  ai_provenance: { ...record, body: record.body.replace('"ai_generated":true', '"ai_generated":false') } });
check("edited record", !editedBody.signature_valid && editedBody.found, JSON.stringify(editedBody));

const flipped = record.sig.slice(0, -1) + (record.sig.endsWith("0") ? "1" : "0");
const editedSig = await m.checkProvenance({ ...good, ai_provenance: { ...record, sig: flipped } });
check("edited signature", !editedSig.signature_valid, JSON.stringify(editedSig));

// Short keys are the shape of GHSA-w696-4vrr-x6x3 in the Python binding; the check must not accept one.
const shortKey = await m.checkProvenance({ ...good, ai_provenance: { ...record, public_key: "00".repeat(32) } });
check("short key", !shortKey.signature_valid, JSON.stringify(shortKey));

const notHex = await m.checkProvenance({ ...good, ai_provenance: { ...record, sig: "zz" } });
check("signature not hex", !notHex.signature_valid && notHex.problem, JSON.stringify(notHex));

const otherAlg = await m.checkProvenance({ ...good, ai_provenance: { ...record, alg: "none" } });
check("another algorithm", !otherAlg.signature_valid && /names none/.test(otherAlg.problem ?? ""), JSON.stringify(otherAlg));

const none = await m.checkProvenance({ run: { text: TEXT } });
check("no record", !none.found && none.problem, JSON.stringify(none));
for (const junk of [null, 7, "text", [], { ai_provenance: 3 }, { ai_provenance: { body: 1 } }]) {
  const r = await m.checkProvenance(junk);
  check(`junk ${JSON.stringify(junk)}`, !r.signature_valid && r.problem, JSON.stringify(r));
}

const copyDiffers = await m.checkProvenance({ ...good, mcp: { echoedPayload: { answer: "other" } } });
check("copy differs", copyDiffers.copy_matches === false, JSON.stringify(copyDiffers));

// ---------------------------------------------------------------------------------------------
// 4. The file the Python test also reads.
// ---------------------------------------------------------------------------------------------
const fixture = await m.checkProvenance(JSON.parse(readFileSync(FIXTURE, "utf8")));
check("committed sample", fixture.signature_valid && fixture.text_matches && fixture.copy_matches === true,
      `${JSON.stringify(fixture)}: regenerate with --write-fixture if the format moved on purpose`);

rmSync(dir, { recursive: true, force: true });
if (failures.length) {
  console.error(`check:mcp-envelope FAILED (${failures.length})`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`check:mcp-envelope OK (${committed.vectors.length} requests reproduced from Python; the record `
  + "signs, reads back and refuses each edit)");
