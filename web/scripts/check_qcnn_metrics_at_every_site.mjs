// Every site that reports `qcnn-trained` must report its cost the same way.
//
// WHY THIS EXISTS
//
// `qcnn_inference_ms` and `qcnn_qubits` were added to the `private_inference` call site of
// `qcnnInferTrained` and not to the `qml_ids` one. Telecom has no `private_inference` stage - the
// second site exists precisely because of that - so the telecom page reported
// `primary_inference_backend = qcnn-trained` with no latency beside it, while checklist rows PF9 and
// PF10 asserted the latency renders on every page whose primary inference is qcnn-trained. Rows that
// can only fail, added by the round whose stated job was removing rows that can only fail.
//
// Nothing could catch it. tsc was green, the Python suite was green, all thirteen local gates were
// green, and both branches are individually well-formed - the defect is the RELATION between them.
// It was found by running the deployed telecom page and reading "inference not run".
//
// This is the fourth time this repository has shipped one fix to one of two twins: the combiner
// length tag (Python, not the browser), the ITS combiner (SHAKE in Python, SHA-256 chain in the
// browser), the federated mask round index (TypeScript, not Python), and now this. So the check is
// structural rather than another careful read: find every place that claims the backend, and require
// the same metric set at each.
//
// Run: node web/scripts/check_qcnn_metrics_at_every_site.mjs   (npm run check:qcnn-metrics; CI gate)

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(WEB, "src/lib/clientCompute.ts");
const src = readFileSync(SRC, "utf8");
const lines = src.split("\n");

// A site is a line that declares this run's primary inference to be the trained QCNN. That claim is
// what obliges the cost metrics, so it is what the check keys on - not the name of the function,
// which could be called for a comparison that is never surfaced.
const CLAIM = /metric\(\s*stage\s*,\s*"primary_inference_backend"\s*,\s*"qcnn-trained"/;
//: Metrics every claiming site must also emit, within the same block.
const REQUIRED = ["qcnn_inference_ms", "qcnn_qubits"];
//: How far to look for them. Blocks here are tens of lines including comments; 60 covers the longer
//: of the two comfortably without running into the next case of the switch.
const WINDOW = 60;

const sites = [];
lines.forEach((line, i) => { if (CLAIM.test(line)) sites.push(i); });

const failures = [];

if (sites.length < 2) {
  failures.push(
    `only ${sites.length} site(s) claim primary_inference_backend="qcnn-trained"; there were two ` +
    "(private_inference and qml_ids). If one was genuinely removed, update this check with the " +
    "reason - a shrinking site count is how a check quietly stops covering anything.");
}

for (const at of sites) {
  const block = lines.slice(at, at + WINDOW).join("\n");
  const missing = REQUIRED.filter((m) => !block.includes(m));
  if (missing.length) {
    failures.push(
      `clientCompute.ts:${at + 1} claims qcnn-trained but does not emit ${missing.join(", ")} ` +
      `within ${WINDOW} lines. Every site that reports this backend must report its cost the same ` +
      "way, or a page renders the claim with no latency and PF9/PF10 become unfalsifiable there.");
  }
}

// The detector must be able to see a site at all; a regex that matches nothing passes forever.
if (!CLAIM.test(src)) {
  failures.push("the primary_inference_backend claim pattern matched nothing - this check is inert");
}

// Every site must also REQUEST the selected width. `loadQcnnWeights` gained a width argument in
// Round 40 so the "QCNN qubits" control could reach the trained branch; passing it at one of the two
// sites would leave that control inert on exactly one page, which is the same one-of-two shape this
// file was written for. Keyed on the call, not on a comment.
{
  const calls = [...src.matchAll(/loadQcnnWeights\(([^)]*)\)/g)]
    .map((m) => m[1].trim())
    .filter((args) => !args.startsWith("usecase: string"));   // drop the definition itself
  if (calls.length < 2) {
    failures.push(`only ${calls.length} loadQcnnWeights call site(s); there were two`);
  }
  // `qubitsOf(config` without the closing paren: the capture above stops at the FIRST `)`, so the
  // argument list of a nested call arrives truncated. Testing for the full `qubitsOf(config)` made
  // this gate fail on correct code - a false positive in the check written to catch false negatives.
  const bare = calls.filter((args) => !args.includes("qubitsOf(config"));
  if (bare.length) {
    failures.push(
      `loadQcnnWeights called without the selected width at ${bare.length} site(s): ` +
      `${JSON.stringify(bare)}. A site that omits it silently ignores the QCNN-qubits control, ` +
      "which is the inert-control defect this repository has found six times.");
  }
}

if (failures.length) {
  console.error("check:qcnn-metrics FAILED");
  for (const f of failures) console.error("  - " + f);
  process.exit(1);
}
console.log(`check:qcnn-metrics OK (${sites.length} qcnn-trained sites, each emitting ${REQUIRED.join(" + ")})`);
