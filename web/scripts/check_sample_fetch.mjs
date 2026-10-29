/**
 * A failed request for a model's evaluation sample is never reported as the model having none.
 *
 * WHY THIS EXISTS
 *
 * The published models' evaluation samples are fetched from the Hugging Face Hub. Until Round 80 the
 * fetch returned `null` for a 404 and for every failure alike: offline, a timeout, an outage, a rate
 * limit. The page then stated that the published model "ships no eval_sample.json at that
 * revision", which is a claim about the artefact made on no evidence about it, and cached the null,
 * so it kept saying so for the rest of the session. Found by scanning the pages with the network
 * cut (web/scripts/check_text_overlap.mjs, SCAN_OFFLINE=1).
 *
 * What this asserts, with `fetch` replaced, through the real module bundled by esbuild:
 *   1. a network error, a timeout, an HTTP 503 and a body that does not parse are each reported as a
 *      fetch that failed, with the reason, by both readers (the one-row run and the verification),
 *      and never as a sample the model lacks;
 *   2. a failed fetch is asked again on the next run;
 *   3. a 404 is still the Hub's answer: "ships no eval_sample.json", no verification error, and it
 *      is not asked again.
 *
 * Run: npm run check:sample-fetch   (CHECK_MODULE=<path> checks another copy of the module)
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MODULE = process.env.CHECK_MODULE ?? join(WEB, "src/lib/onnxInference");
const USECASE = "finance_aml";            // a published browser model with a sample at its revision
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const dir = mkdtempSync(join(tmpdir(), "qsh-sample-fetch-"));
const entry = join(dir, "entry.ts");
writeFileSync(entry, `export { runPublishedModel, evaluatePublishedModel } from ${JSON.stringify(MODULE)};\n`);
const out = join(dir, "bundle.mjs");
execFileSync("npx", ["esbuild", entry, "--bundle", "--format=esm", "--platform=node",
                     `--outfile=${out}`, "--log-level=error"], { cwd: WEB, stdio: "pipe" });

/** A fresh copy of the module per case, so one case's cache cannot answer for another's. */
let instance = 0;
const fresh = () => import(`${pathToFileURL(out).href}?case=${++instance}`);

const timeout = () => { const e = new Error("signal timed out"); e.name = "TimeoutError"; return e; };
const FAILURES = {
  "a network error": { fetch: async () => { throw new TypeError("fetch failed"); }, says: "failed" },
  "a timeout": { fetch: async () => { throw timeout(); }, says: "did not answer within" },
  "an HTTP 503": { fetch: async () => new Response("busy", { status: 503 }), says: "HTTP 503" },
  "a body that does not parse": { fetch: async () => new Response("<html>", { status: 200 }), says: "did not parse" },
};
const ABSENT = "ships no eval_sample.json";

try {
  for (const [what, { fetch: impl, says }] of Object.entries(FAILURES)) {
    const m = await fresh();
    let calls = 0;
    globalThis.fetch = async (...a) => { calls++; return impl(...a); };
    const run = await m.runPublishedModel(USECASE);
    check(what, run.kind === "failed" && run.error.includes("could not be fetched") && run.error.includes(says),
      `the one-row run said ${JSON.stringify(run.error ?? run.kind)}`);
    check(what, !String(run.error).includes(ABSENT),
      "the one-row run reported a failed request as the model shipping no sample");
    const v = await m.evaluatePublishedModel(USECASE);
    check(what, v !== null && String(v.error).includes("could not be fetched"),
      `the verification read it as ${v === null ? "no sample published" : JSON.stringify(v.error)}`);
    check(what, calls === 2, `${calls} requests for two readers: a failed fetch was cached`);
  }

  {
    const m = await fresh();
    let calls = 0;
    globalThis.fetch = async () => { calls++; return new Response("Entry not found", { status: 404 }); };
    const run = await m.runPublishedModel(USECASE);
    check("a 404", run.kind === "failed" && String(run.error).includes(ABSENT),
      `the Hub's own answer was reported as ${JSON.stringify(run.error ?? run.kind)}`);
    const v = await m.evaluatePublishedModel(USECASE);
    check("a 404", v === null, `the verification reported ${JSON.stringify(v?.error)} for an absent sample`);
    check("a 404", calls === 1, `${calls} requests: the Hub's answer was asked for again`);
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}

if (failures.length) {
  console.error("check:sample-fetch FAILED");
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`check:sample-fetch OK  ${Object.keys(FAILURES).length} kinds of failed request reported as `
  + "failed requests by both readers and asked again; a 404 still reads as no sample, and is kept");
