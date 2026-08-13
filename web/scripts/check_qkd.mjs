#!/usr/bin/env node
/**
 * Assert the browser QKD model's PLUMBING and its stated identities.
 *
 * WHAT THIS IS FOR. The checklist's Level-1 rule forbids re-deriving a formula in the same language
 * as the implementation - a mirror written in TypeScript would copy a TypeScript bug. So this does
 * not re-derive the key rate. It asserts three things a mirror cannot:
 *
 *   1. CONFIG REACHES THE MODEL. `error_correction_eff` lived only in `config.defaults.json` with no
 *      reader: `bb84`/`e91` took f_EC as a TypeScript *default parameter* of 1.16 and every call site
 *      omitted it, so editing config moved the Python model and left the browser on the literal.
 *      Checked by changing the value and requiring the output to move to a predicted number - a diff
 *      alone is passed perfectly by a reader that reads nothing.
 *   2. THE STATED IDENTITIES HOLD, at full precision. The E91 relation the checklist published was
 *      not the one implemented; it passed only because both values displayed as "2.7".
 *   3. THE OPTIMISER IS DETERMINISTIC in the ways its own structure promises. `optimizeQKD` takes no
 *      seed and calls no RNG, so bootstrap intervals and seed sweeps are the wrong instrument -
 *      monotonicity, an exact evaluation count and bit-identical reruns are sharper and cheaper.
 *
 * HOW IT LOADS THE TS. The modules use extensionless imports, which node's ESM resolver will not
 * take, so esbuild (already a vite dependency - no new package) bundles them to a temp ESM file
 * first. Bundling also means this exercises the same module graph the browser gets.
 *
 * Run: npm run check:qkd
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];
const check = (name, ok, detail) => {
  if (ok) return;
  failures.push(`${name}: ${detail}`);
};
/** Exact within double-rounding, not "close enough" - everything here is deterministic. */
const near = (a, b, tol = 1e-12) => Math.abs(a - b) <= tol;

function load() {
  const dir = mkdtempSync(join(tmpdir(), "qsh-qkd-"));
  const entry = join(dir, "entry.ts");
  writeFileSync(entry, `
export { bb84, e91, FIBER, FREE_SPACE, intrinsicQber, h2, transmittance } from ${JSON.stringify(join(WEB, "src/lib/quantumSim"))};
export { optimizeQKD } from ${JSON.stringify(join(WEB, "src/lib/clientOptiQKD"))};
export { repetitionCountCutoff, adaptiveProportionCutoff, generateQuantumBits } from ${JSON.stringify(join(WEB, "src/lib/clientQRNG"))};
export { qkdCfg, qrngCfg } from ${JSON.stringify(join(WEB, "src/lib/config"))};
`);
  const out = join(dir, "bundle.mjs");
  execFileSync("npx", ["esbuild", entry, "--bundle", "--format=esm", "--platform=node",
                       `--outfile=${out}`, "--log-level=error"], { cwd: WEB, stdio: "pipe" });
  return { out, dir };
}

const { out, dir } = load();
const m = await import(`file://${out}`);
const cfg = m.qkdCfg();
const cfg_qrng = m.qrngCfg();

// --------------------------------------------------------------------------------------------- //
//  1. Config reaches the model
// --------------------------------------------------------------------------------------------- //
// The GLLP secure fraction is 1 - h2(q) - f_EC*h2(q). Raising f_EC must LOWER it, at every QBER
// where a key exists. If the model ignores config, the two runs are identical and this fails.
{
  const chan = { ...m.FIBER, distanceKm: 50 };
  const base = m.bb84(200_000, chan, 1);
  const raised = m.bb84(200_000, chan, 1, 1.22);
  check("fEC-is-a-real-input",
    raised.secretKeyRate < base.secretKeyRate,
    `f_EC 1.16 -> 1.22 left skr unchanged (${base.secretKeyRate} vs ${raised.secretKeyRate}); the ` +
    `parameter is not reaching the rate`);

  // THE DISCRIMINATING TEST, and the reason a before/after diff is not enough. While config and the
  // literal both read 1.16, "the default reads config" and "the default is a literal that happens to
  // match" produce identical output. Mutating the live config object separates them: a default
  // expression is evaluated at call time, so a reader follows the mutation and a literal does not.
  // Measured before the fix: skr stayed at 0.009576592335210885 with config moved to 1.22.
  const beforeMutation = m.bb84(200_000, chan, 1).secretKeyRate;
  const original = cfg.error_correction_eff;
  cfg.error_correction_eff = 1.22;
  const afterMutation = m.bb84(200_000, chan, 1).secretKeyRate;
  const eAfter = m.e91(200_000, chan, 1, cfg.e91_visibility).secretKeyRate;
  const optAfter = m.optimizeQKD(chan, undefined, 48).bestSkr;
  cfg.error_correction_eff = original;
  const restored = m.bb84(200_000, chan, 1).secretKeyRate;

  check("fEC-default-reads-config",
    afterMutation < beforeMutation,
    `moving config's error_correction_eff 1.16 -> 1.22 left bb84's skr at ${afterMutation}. The ` +
    `default is a literal, so a config edit moves the Python model and leaves the browser behind.`);
  check("fEC-default-reads-config-e91",
    eAfter < m.e91(200_000, chan, 1, cfg.e91_visibility, 1.16).secretKeyRate,
    `e91 ignored the config change; its f_EC default is a literal`);
  check("fEC-reaches-the-optimiser",
    optAfter !== m.optimizeQKD(chan, undefined, 48).bestSkr,
    `optimizeQKD's surrogate ignored the config change. A HALF-WIRE is worse than none: the ` +
    `reported skr and the optimiser's baseline would use different f_EC, silently corrupting ` +
    `qkd_skr_improvement_pct while config happens to equal the literal.`);
  check("fEC-mutation-is-reversible",
    near(restored, beforeMutation),
    `restoring config did not restore the rate (${restored} vs ${beforeMutation})`);

  // AND IT MOVES BY THE PREDICTED AMOUNT, not merely in the right direction. Sifting does not depend
  // on f_EC, so the rate scales exactly with the GLLP secure fraction 1 - h2(q) - f*h2(q). Asserting
  // the ratio is what separates "config is read" from "config is read and used correctly".
  const q0 = m.bb84(200_000, chan, 1).qber;
  const frac = (f) => Math.max(0, 1 - m.h2(q0) - f * m.h2(q0));
  const predicted = frac(1.22) / frac(1.16);
  check("fEC-moves-by-the-predicted-ratio",
    near(afterMutation / beforeMutation, predicted, 1e-6),
    `f_EC 1.16 -> 1.22 changed skr by a factor ${afterMutation / beforeMutation}, but the GLLP ` +
    `fraction at QBER ${q0} predicts ${predicted}`);
}

// --------------------------------------------------------------------------------------------- //
//  2. Stated identities, at full precision
// --------------------------------------------------------------------------------------------- //
{
  const chan = { ...m.FIBER, distanceKm: 50 };
  const V = cfg.e91_visibility;
  const e = m.e91(200_000, chan, 1, V);
  const eIntr = m.intrinsicQber(chan);

  // The relation the code implements. Asserted exactly, because both sides are deterministic.
  check("chsh-werner-identity",
    near(e.chshS, 2 * Math.SQRT2 * V * (1 - eIntr), 1e-9),
    `S=${e.chshS} but 2*sqrt(2)*V*(1-e_intrinsic)=${2 * Math.SQRT2 * V * (1 - eIntr)}`);

  // S must never exceed the Tsirelson bound. An S above it is unphysical, not a good result.
  check("chsh-tsirelson", e.chshS <= 2 * Math.SQRT2 + 1e-12,
    `S=${e.chshS} exceeds the Tsirelson bound ${2 * Math.SQRT2}`);

  // skr is a per-pulse RATE and secureKeyBits a TOTAL. Dimensional tie, to the floor applied.
  const b = m.bb84(200_000, chan, 1);
  check("skr-times-pulses-is-bits",
    Math.abs(b.secretKeyRate * 200_000 - b.secureKeyBits) <= 1,
    `skr*pulses=${b.secretKeyRate * 200_000} against secureKeyBits=${b.secureKeyBits}`);

  // h2 at its maximum. Cheap, and catches a broken entropy helper before anything downstream.
  check("h2-max", near(m.h2(0.5), 1), `h2(0.5)=${m.h2(0.5)}`);
}

// --------------------------------------------------------------------------------------------- //
//  3. The optimiser is deterministic in the ways its structure promises
// --------------------------------------------------------------------------------------------- //
{
  const chan = { ...m.FIBER, distanceKm: 50 };

  // Bit-identical reruns. `optimizeQKD` takes no seed and calls no RNG; anything else means
  // something stochastic entered a model documented as analytic.
  const a = m.optimizeQKD(chan, undefined, 48);
  const b = m.optimizeQKD(chan, undefined, 48);
  check("optiqkd-reproducible",
    a.improvementPct === b.improvementPct && a.nEvals === b.nEvals,
    `two identical calls differed: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`);

  // Budget monotonicity. rounds = max(1, round(budget/8)) and the search path is prefix-identical,
  // so more budget can never find a worse optimum.
  let prev = -Infinity;
  for (const budget of [8, 16, 24, 32, 48, 64]) {
    const r = m.optimizeQKD(chan, undefined, budget);
    check(`optiqkd-monotone-${budget}`, r.bestSkr >= prev - 1e-15,
      `budget ${budget} gave bestSkr ${r.bestSkr}, below the previous ${prev}`);
    prev = Math.max(prev, r.bestSkr);
  }

  // Evaluation-count identity: 1 for the baseline b0, 1 for the midpoint, then 4 keys x 2 directions
  // per round, with an early break when a round improves nothing. So nEvals <= 2 + 8*rounds exactly.
  //
  // NOTE THE SIGNATURE. It is optimizeQKD(channel, protocol?, nSamples?) - the budget is THIRD. The
  // first version of this file passed it second, where it became a `protocol` string, so every call
  // silently used the config default and "budget monotonicity" compared six identical runs. JS
  // calling a TS API gets no type check; asserting a predicted count is what surfaced it.
  for (const budget of [16, 48]) {
    const r = m.optimizeQKD(chan, undefined, budget);
    const rounds = Math.max(1, Math.round(budget / 8));
    check(`optiqkd-evals-${budget}`, r.nEvals <= 2 + 8 * rounds && r.convergenceEvals <= r.nEvals,
      `budget ${budget}: nEvals=${r.nEvals} against the ceiling ${2 + 8 * rounds}, ` +
      `convergenceEvals=${r.convergenceEvals}`);
  }
}

// --------------------------------------------------------------------------------------------- //
//  4. QRNG: the standard the code cites, asserted rather than mentioned
// --------------------------------------------------------------------------------------------- //
// The checklist named SP 800-90B zero times while the code implements section 4.4 with clause
// numbers. These are the two numbers in that section that are closed-form and checkable.
{
  // SP 800-90B section 4.4.1's own worked example: H = 2 bits/sample at alpha = 2^-20 gives C = 11.
  check("sp80090b-repetition-cutoff",
    m.repetitionCountCutoff(2.0, 2 ** -20) === 11,
    `repetitionCountCutoff(2.0, 2^-20) = ${m.repetitionCountCutoff(2.0, 2 ** -20)}, section 4.4.1 gives 11`);

  // The LHL ceiling, absolute rather than as a delta. A delta check cancels the "+2" term, the
  // min-entropy factor and the raw over-provision - three of the four config inputs to the number it
  // claims to verify - and this file's stated form was missing the "+2" as a result.
  const perRaw = cfg_qrng.min_entropy_per_raw_bit;
  for (const [bits, eps] of [[4096, -40], [4096, -80], [1024, -40]]) {
    const r = m.generateQuantumBits(bits, { postProcessing: "toeplitz", epsilonLog2: eps });
    if (r.lhlMaxBits == null) continue;
    const predicted = Math.max(1, Math.floor(perRaw * (2 * bits + 16) - 2 * Math.abs(eps) + 2));
    check(`lhl-absolute-${bits}-${eps}`, r.lhlMaxBits === predicted,
      `lhlMaxBits=${r.lhlMaxBits} but the closed form gives ${predicted} ` +
      `(bits=${bits}, eps=${eps}, min_entropy_per_raw_bit=${perRaw})`);
  }
}

rmSync(dir, { recursive: true, force: true });

if (failures.length) {
  console.error(`check:qkd FAILED (${failures.length})`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`check:qkd OK - config reaches the model, the stated identities hold at full precision, ` +
            `and the optimiser is deterministic (f_EC=${cfg.error_correction_eff}, V=${cfg.e91_visibility})`);
