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
export { bb84, e91, FIBER, FREE_SPACE, intrinsicQber, h2, transmittance, taggedFraction, channelLossDb, decoySecureFraction } from ${JSON.stringify(join(WEB, "src/lib/quantumSim"))};
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
// Raising f_EC must LOWER the secure fraction, at every QBER where a key exists. (This line read
// "The GLLP secure fraction is 1 - h2(q) - f_EC*h2(q)" until Round 46 - the premise of the
// RETIRED model, contradicting this file's own repoint 45 lines below, which already says the
// runtime is the two-intensity decoy estimator. One comment of a pair.)
// If the model ignores config, the two runs are identical and this fails.
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

  // AND IT MOVES BY THE PREDICTED AMOUNT, not merely in the right direction. Asserting the ratio is
  // what separates "config is read" from "config is read and used correctly".
  //
  // ROUND 45 REPOINTED THIS PREDICTION, AND THE OLD ONE GOING RED IS WHY. It predicted the ratio
  // from the GLLP secure fraction 1 - h2(q) - f*h2(q), which was the model until this round; `bb84()`
  // now runs the two-intensity decoy-state estimator, where f_EC enters as the error-correction term
  // of R = Q1(1 - h2(e1)) - f Q_mu h2(E_mu). The check caught its own staleness the moment the
  // physics moved, which is the behaviour a gate is for - so it is repointed rather than relaxed.
  //
  // Two changes ride along, both required by the new model:
  //   - the prediction reads the INTRINSIC error rate, not `bb84().qber`. The bound is a function of
  //     the channel and no longer of one session's sampling draw, so predicting from a sampled QBER
  //     would compare a deterministic quantity against a noisy one and fail intermittently.
  //   - it calls `decoySecureFraction` with EXPLICIT arguments. That keeps this a plumbing check -
  //     does the config value reach the model - rather than a second TypeScript derivation of the
  //     same algebra, which this file's header forbids for good reason.
  const eta = m.transmittance(chan);
  const eDet = m.intrinsicQber(chan);
  const y0 = 2 * chan.darkCount;
  const mu = Math.max(1e-3, cfg.mu_signal);
  const nu = Math.min(Math.max(1e-4, cfg.mu_decoy), mu * 0.9);
  const frac = (f) => m.decoySecureFraction(eta, mu, nu, eDet, y0, f, 0.5).perPulse;
  const predicted = frac(1.22) / frac(1.16);
  check("fEC-moves-by-the-predicted-ratio",
    near(afterMutation / beforeMutation, predicted, 1e-6),
    `f_EC 1.16 -> 1.22 changed skr by a factor ${afterMutation / beforeMutation}, but the ` +
    `two-intensity decoy bound at intrinsic QBER ${eDet} predicts ${predicted}`);
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

// --------------------------------------------------------------------------------------------- //
//  6. GLLP tagged fraction, and the free-space loss law
// --------------------------------------------------------------------------------------------- //
// MOVED HERE FROM THE BROWSER, and the reason is the point. QK6 was written as a browser check -
// "tagged fraction rises with distance and with mu" - and could never be run: `mu` is
// `qkdCfg().mu_signal`, a config constant with NO control on any page, so one of its two axes was
// not exercisable by a person at all. That is the sibling of a check that cannot fail: a check that
// cannot be RUN. `taggedFraction` is a pure function, so the assertion belongs where both axes move.
//
// Delta = P(n >= 2) / Q_mu, and this is not a formula re-derivation: the ORDERING is asserted, not
// the value. Distance lowers the transmittance and so Q_mu, raising the ratio; mu raises the
// multiphoton numerator quadratically against a denominator near-linear in the lossy regime.
{
  const at = (km, mu) => m.taggedFraction(mu, { ...m.FIBER, distanceKm: km });
  const MU = cfg.mu_signal;
  const KM = [0, 10, 25, 50, 75, 100, 150];
  for (let i = 1; i < KM.length; i++) {
    check(`tagged-monotone-in-distance-${KM[i - 1]}-${KM[i]}`,
      at(KM[i], MU) > at(KM[i - 1], MU),
      `Delta at ${KM[i]} km is ${at(KM[i], MU)}, not above ${at(KM[i - 1], MU)} at ${KM[i - 1]} km`);
  }
  const MUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9];
  for (let i = 1; i < MUS.length; i++) {
    check(`tagged-monotone-in-mu-${MUS[i - 1]}-${MUS[i]}`,
      at(50, MUS[i]) > at(50, MUS[i - 1]),
      `Delta at mu=${MUS[i]} is ${at(50, MUS[i])}, not above ${at(50, MUS[i - 1])} at mu=${MUS[i - 1]}`);
  }
  // The sharp reference point `decoy_reference.py`'s docstring names: with the GYS set and mu = 0.1
  // the tagged fraction exceeds 1 at ZERO distance - no detection can be certified single-photon and
  // the key rate is zero before any loss at all. An ordering check alone would pass a model that was
  // monotone and wrong by a factor.
  const gys = { ...m.FIBER, distanceKm: 0, lossDbPerKm: 0.21, detectorEff: 0.045,
                darkCount: 1.7e-6, misalignment: 0.033 };
  check("tagged-exceeds-one-on-the-GYS-set-at-zero-distance",
    m.taggedFraction(0.1, gys) > 1,
    `Delta = ${m.taggedFraction(0.1, gys)} at mu=0.1, zero distance, GYS parameters; ` +
    `decoy_reference.py records this as above 1`);

  // Free-space loss is diffraction plus a fixed budget, not a per-km law. A module-level counterpart
  // to QK7's browser reading, and the assertion the Python side lacked while it was still applying
  // `distance_km * 0.035` to the same nominal link.
  const fsAt = (km) => m.channelLossDb({ ...m.FREE_SPACE, distanceKm: km });
  check("free-space-800km-is-24.5dB", Math.abs(fsAt(800) - 24.48) < 0.01,
    `800 km gives ${fsAt(800)} dB, not the 24.48 the diffraction model predicts`);
  check("free-space-has-a-fixed-floor",
    near(fsAt(1), cfg.free_space_atmospheric_db + cfg.free_space_pointing_db, 1e-9),
    `at 1 km the loss is ${fsAt(1)} dB, not the ${cfg.free_space_atmospheric_db} + ` +
    `${cfg.free_space_pointing_db} dB budget floor - the near-field cap is not being applied`);
  check("free-space-is-not-linear-in-distance",
    fsAt(2000) - fsAt(1200) < fsAt(1200) - fsAt(400),
    `loss increments are not decreasing with range, so this is a linear law rather than an ` +
    `inverse-square one: 400->1200 gives ${fsAt(1200) - fsAt(400)}, 1200->2000 gives ` +
    `${fsAt(2000) - fsAt(1200)}`);
}

rmSync(dir, { recursive: true, force: true });

if (failures.length) {
  console.error(`check:qkd FAILED (${failures.length})`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`check:qkd OK - config reaches the model, the stated identities hold at full precision, ` +
            `and the optimiser is deterministic (f_EC=${cfg.error_correction_eff}, V=${cfg.e91_visibility})`);
