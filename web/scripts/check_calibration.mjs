/**
 * A model's output is turned into a probability before anything calls it one.
 *
 * WHY THIS EXISTS
 *
 * The reliability chart binned column 1 of the raw output as P(positive). That held for the
 * scikit-learn exports, whose output is named `probabilities`, and not for the three medical PyTorch
 * exports, whose output is named `logits`. Every medical page drew a calibration chart of logits,
 * captioned "mean p(positive)", reporting "predicted -5.471". tsc, the build and the
 * verification verdict were all green, because the verdict is an argmax, which logits answer
 * correctly. Found on the deployed page in Round 74, by reading a caption.
 *
 * What this asserts:
 *   1. the graph's output NAME decides the conversion, and an unknown name is refused;
 *   2. softmax and sigmoid are right, keep the argmax, and stay finite at a logit of 114 (the
 *      ResNet-50 reaches 114.4 on its own sample);
 *   3. the binning refuses a value outside [0, 1], so a logit can never be drawn as a probability
 *      again, whatever the caller does;
 *   4. both readers of a published graph - the whole-sample evaluation and the one-row run - derive
 *      the kind from the output name, so the decision threshold means the same in both. One fix to
 *      one of two twins is this repository's most repeated defect.
 *
 * HOW IT LOADS THE TS: esbuild bundles the module to a temp ESM file, as check_qkd.mjs does.
 *
 * Run: npm run check:calibration
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => { if (!ok) failures.push(`${name}: ${detail}`); };

const dir = mkdtempSync(join(tmpdir(), "qsh-calibration-"));
const entry = join(dir, "entry.ts");
writeFileSync(entry,
  `export { outputKind, toProbabilities, reliabilityBins, calibrationWithheld, MIN_CALIBRATION_EVENTS } from ${JSON.stringify(join(WEB, "src/lib/calibration"))};\n`);
const out = join(dir, "bundle.mjs");
execFileSync("npx", ["esbuild", entry, "--bundle", "--format=esm", "--platform=node",
                     `--outfile=${out}`, "--log-level=error"], { cwd: WEB, stdio: "pipe" });
const m = await import(`file://${out}`);
rmSync(dir, { recursive: true, force: true });

// 1. The name decides, and an unknown name is refused rather than guessed.
check("kind", m.outputKind("probabilities") === "probabilities", "`probabilities` not recognised");
check("kind", m.outputKind("logits") === "logits", "`logits` not recognised");
check("kind", m.outputKind("output") === null && m.outputKind("label") === null,
  "an output name that says nothing about its meaning was given a meaning");

// 2. Probabilities pass through unchanged; logits become a distribution with the same argmax.
{
  const probs = [0.25, 0.75, 0.9, 0.1];
  const p = m.toProbabilities(probs, 2, 2, "probabilities");
  check("identity", p.every((v, i) => v === probs[i]), `probabilities were altered: ${[...p]}`);

  // Rows from the medical models' own range, including the ResNet-50's extremes.
  const logits = [-11.719, 14.115, 114.381, -104.209, 0, 0, -5.471, 0.23];
  const q = m.toProbabilities(logits, 4, 2, "logits");
  for (let i = 0; i < 4; i++) {
    const a = q[2 * i], b = q[2 * i + 1];
    check("softmax", Number.isFinite(a) && Number.isFinite(b), `row ${i} not finite: ${a}, ${b}`);
    check("softmax", Math.abs(a + b - 1) < 1e-12, `row ${i} sums to ${a + b}`);
    check("softmax", a >= 0 && b >= 0 && a <= 1 && b <= 1, `row ${i} outside [0, 1]: ${a}, ${b}`);
    const raw = logits[2 * i + 1] > logits[2 * i] ? 1 : logits[2 * i + 1] < logits[2 * i] ? 0 : -1;
    const now = b > a ? 1 : b < a ? 0 : -1;
    check("softmax", raw === now, `row ${i} changed its argmax`);
  }
  check("softmax", q[4] === 0.5 && q[5] === 0.5, `equal logits did not give 0.5: ${q[4]}, ${q[5]}`);
  // Two-class softmax is the sigmoid of the logit difference: 0.23 - (-5.471) = 5.701.
  check("softmax", Math.abs(q[7] - 1 / (1 + Math.exp(-5.701))) < 1e-12, `row 3 is ${q[7]}`);
  const s = m.toProbabilities([0, 2], 2, 1, "logits");
  check("sigmoid", Math.abs(s[0] - 0.5) < 1e-12 && Math.abs(s[1] - 1 / (1 + Math.exp(-2))) < 1e-12,
    `single-column logits not passed through a sigmoid: ${[...s]}`);
}

// 3. The binning refuses what is not a probability: the exact defect, replayed.
{
  const rawLogitColumn = [14.115, -5.471, 0.23];
  let threw = false;
  try { m.reliabilityBins(3, (i) => rawLogitColumn[i], () => true); } catch { threw = true; }
  check("refuse", threw, "reliabilityBins accepted a logit as a probability");
  const ok = m.reliabilityBins(3, (i) => [0.02, 0.5, 0.97][i], (i) => i === 2);
  check("bins", ok.every((b) => b.p >= 0 && b.p <= 1) && ok.reduce((a, b) => a + b.n, 0) === 3,
    `bins of valid probabilities are wrong: ${JSON.stringify(ok)}`);
}

// 4. PP8: no curve below 200 events or 200 non-events, at the counts the published samples have.
{
  check("pp8", m.MIN_CALIBRATION_EVENTS === 200, `the floor is ${m.MIN_CALIBRATION_EVENTS}, not 200`);
  for (const [ev, non, label] of [[164, 8028, "finance"], [42, 114, "medical"], [113, 416, "satellite"],
                                  [8, 592, "telecom QML"]]) {
    const why = m.calibrationWithheld(ev, non, "positives", "negatives");
    check("pp8", typeof why === "string" && why.includes(String(ev)), `${label} (${ev} events) was not withheld`);
  }
  check("pp8", m.calibrationWithheld(250, 16129, "wrong", "right") === null, "telecom NIDD (250 / 16,129) was withheld");
  check("pp8", m.calibrationWithheld(200, 200, "a", "b") === null, "exactly 200 of each was withheld");
  check("pp8", m.calibrationWithheld(199, 5000, "a", "b") !== null && m.calibrationWithheld(5000, 199, "a", "b") !== null,
    "199 events or 199 non-events was drawn");
}

// 5. Both readers derive the kind from the output name and bin only converted values.
{
  const src = readFileSync(join(WEB, "src/lib/onnxInference.ts"), "utf8");
  const kinds = (src.match(/outputKind\(/g) ?? []).length;
  check("twins", kinds >= 2, `outputKind is called ${kinds} time(s); the evaluation and the one-row run both need it`);
  check("twins", !/function reliabilityBins\(/.test(src), "a second reliabilityBins is defined in onnxInference.ts");
  const binned = [...src.matchAll(/reliabilityBins\(([\s\S]*?)\),\s*\n/g)].map((x) => x[1]);
  check("binned", binned.length >= 2, `found ${binned.length} reliability call(s), expected the binary and multi-class ones`);
  for (const b of binned) {
    check("binned", !/\bdata\[/.test(b), `a reliability call reads the raw output: ${b.replace(/\s+/g, " ").slice(0, 100)}`);
  }
  const guards = (src.match(/calibrationWithheld\(/g) ?? []).length;
  check("pp8", guards >= 2, `calibrationWithheld guards ${guards} reliability site(s); the binary and multi-class ones both need it`);
}

if (failures.length) {
  console.error("calibration check FAILED:");
  for (const f of failures) console.error(`  ${f}`);
  process.exit(1);
}
console.log("PASS  output kind from the graph's name; softmax/sigmoid exact and finite; non-probabilities refused; PP8 floor applied; both readers wired");
