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
import { mkdtempSync, writeFileSync, rmSync, readFileSync } from "node:fs";
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
export { bb84, e91, FIBER, FREE_SPACE, intrinsicQber, eDetForBound, h2, transmittance, taggedFraction, channelLossDb, decoySecureFraction, qsdcLink } from ${JSON.stringify(join(WEB, "src/lib/quantumSim"))};
export { optimizeQKD } from ${JSON.stringify(join(WEB, "src/lib/clientOptiQKD"))};
export { repetitionCountCutoff, adaptiveProportionCutoff, generateQuantumBits, minEntropyMcv, toeplitzExtract } from ${JSON.stringify(join(WEB, "src/lib/clientQRNG"))};
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
  // `chan.misalignment`, NOT `intrinsicQber(chan)`. In Ma et al. the `e_det` argument is the
  // probability a detected SIGNAL photon lands in the wrong detector; `decoySecureFraction` adds
  // the dark-count contribution back itself as `e0 * y0`. Passing the observed QBER charged the
  // background twice, and it was then corrected to the bare misalignment, which dropped the
  // eavesdropper. FOUR surfaces implement this algebra - `bb84.py`, `quantumSim.ts`,
  // `clientOptiQKD.ts` and this file - and the commit that claimed to fix it named three.
  // `eDetForBound(chan)`, the SAME helper the model uses, rather than a local re-derivation.
  // This line has now been wrong twice in one round: first as `intrinsicQber(chan)`, which charged
  // dark counts twice, then as the bare misalignment, which dropped the eavesdropper. A check that
  // re-derives the model's argument can be wrong in a way the model is not, which is the opposite
  // of what a check is for.
  const eDet = m.eDetForBound(chan);
  const y0 = 2 * chan.darkCount;
  const mu = Math.max(1e-3, cfg.mu_signal);
  const nu = Math.min(Math.max(1e-4, cfg.mu_decoy), mu * 0.9);
  const frac = (f) => m.decoySecureFraction(eta, mu, nu, eDet, y0, f, 0.5).perPulse;
  const predicted = frac(1.22) / frac(1.16);
  check("fEC-moves-by-the-predicted-ratio",
    near(afterMutation / beforeMutation, predicted, 1e-6),
    `f_EC 1.16 -> 1.22 changed skr by a factor ${afterMutation / beforeMutation}, but the ` +
    `two-intensity decoy bound at misalignment ${eDet} predicts ${predicted}`);
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
  //
  // (1 - 2e), NOT (1 - e). This checker carried the SAME units confusion as the three
  // implementations it exists to check - a channel flipping a bit with probability e has visibility
  // (1 - 2e) - so it was asserting the bug rather than catching it, and it went green for as long as
  // every implementation was wrong in the same way. Round 54: the fourth and last copy.
  check("chsh-werner-identity",
    near(e.chshS, 2 * Math.SQRT2 * V * (1 - 2 * eIntr), 1e-9),
    `S=${e.chshS} but 2*sqrt(2)*V*(1-2*e_intrinsic)=${2 * Math.SQRT2 * V * (1 - 2 * eIntr)}`);

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

  // AND THE WHOLE GRID, AGAINST PYTHON, BECAUSE THE SPOT CHECK ABOVE COULD NOT SEE THE DEFECT.
  //
  // The check above is correct, is the standard's own worked example, and is on the SIBLING
  // function. `adaptiveProportionCutoff` meanwhile returned `1 + c` where the loop already held the
  // answer - the `1 +` belongs to the Python copy, where `binom.isf` returns one BELOW the cutoff -
  // so it was uniformly one too high at EVERY entropy: 991/990, 793/792, 626/625, 590/589, 325/324,
  // 106/105. Permissive, because a higher cutoff fails fewer windows, and visitor-facing, because
  // this is the client-side-only demo.
  //
  // One point on one of two functions cannot detect a uniform offset in the other. The vectors are
  // generated by `scripts/export_qrng_cutoffs.py` and regenerated in CI, so the Python side stays
  // the authority rather than this file becoming a second one.
  const cutoffs = JSON.parse(
    readFileSync(join(WEB, "scripts/vectors/qrng_cutoffs.json"), "utf8"));
  const mismatches = [];
  for (const [h, want] of Object.entries(cutoffs.adaptive_proportion)) {
    const got = m.adaptiveProportionCutoff(Number(h), cutoffs.w, cutoffs.alpha);
    if (got !== want) mismatches.push(`adaptive h=${h}: browser ${got}, python ${want}`);
  }
  for (const [h, want] of Object.entries(cutoffs.repetition_count)) {
    const got = m.repetitionCountCutoff(Number(h), cutoffs.alpha);
    if (got !== want) mismatches.push(`repetition h=${h}: browser ${got}, python ${want}`);
  }
  // THE ENTROPY FIGURE MUST DESCRIBE THE SOURCE, NOT THE EXTRACTOR.
  //
  // SP 800-90B section 4.4 runs its continuous tests on raw samples BEFORE conditioning, and the
  // min-entropy figure is a statement about the noise source. `generateQuantumBits` computed all of
  // it from the post-extractor output. `src/qsh/qrng/health.py` takes `raw=` and
  // `tests/test_health_tests_measure_the_noise_source.py` pins it - in PYTHON ONLY, so the Round 51
  // fix reached the reference implementation and not the one every visitor runs.
  //
  // Checked on a BIASED stream because the shipped source is near-fair, which is exactly why this
  // survived: at P(1) = 0.5 the two readings differ in the third decimal. At P(1) = 0.90 they are
  // 0.151 against 0.975.
  {
    const n = 200000, p1 = 0.90;
    let st = 12345;
    const u = () => (st = (st * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const raw = new Uint8Array(n);
    for (let i = 0; i < n; i++) raw[i] = u() < p1 ? 1 : 0;
    const cond = m.toeplitzExtract(raw, Math.floor(n / 4), u);
    const hRaw = m.minEntropyMcv(raw), hCond = m.minEntropyMcv(cond);
    check("sp80090b-entropy-separates-source-from-extractor",
      hRaw < 0.30 && hCond > 0.90 && hCond - hRaw > 0.5,
      `raw ${hRaw.toFixed(4)} vs conditioned ${hCond.toFixed(4)} - these must SEPARATE, or the `
      + "fixture is no longer biased and this check proves nothing");
    // The property that matters: what the module REPORTS tracks the source. Asserted through the
    // module's own estimator on the raw stream, which is what the call site now passes.
    // AND THE CALL SITE, because the two checks above pass whether or not `generateQuantumBits` uses
    // `raw` - they exercise the ESTIMATOR on hand-built streams, and the estimator was never the
    // thing that was wrong. Seeded by reverting the call site: both stayed green. That is the
    // mechanism-versus-outcome defect, committed inside the gate written to catch it.
    //
    // No output-based check can discriminate here. The shipped source is fair by construction, so
    // raw and conditioned agree to the third decimal on every stream the public API can produce -
    // which is why `tests/test_health_tests_measure_the_noise_source.py` asserts on the CALL too,
    // with an `ast` parse. Same reasoning, same remedy.
    //
    // COMMENTS ARE STRIPPED FIRST. The comment block explaining why `raw` must be used contains the
    // string `minEntropyMcv(raw)`, so a plain grep matches its own documentation - the exact trap
    // the QRNG register gate hit once already.
    {
      const src = readFileSync(join(WEB, "src/lib/clientQRNG.ts"), "utf8")
        .replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
      const body = src.slice(src.indexOf("export function generateQuantumBits"));
      check("sp80090b-call-site-passes-the-raw-stream",
        /minEntropyMcv\(\s*raw\s*\)/.test(body)
        && /repetitionCountTest\(\s*raw\s*,/.test(body)
        && /adaptiveProportionTest\(\s*raw\s*,/.test(body)
        && !/minEntropyMcv\(\s*bits\s*\)/.test(body),
        "generateQuantumBits computes its SP 800-90B figures from `bits` (post-conditioning) rather "
        + "than `raw`. An extractor makes any source look uniform, so the reported min-entropy would "
        + "describe the Toeplitz stage under a label that says entropy source - measured 0.9749 "
        + "against a true 0.1510 at P(1)=0.90.");
    }

    check("sp80090b-cutoffs-follow-the-raw-stream",
      m.repetitionCountCutoff(hRaw) > m.repetitionCountCutoff(hCond)
      && m.adaptiveProportionCutoff(hRaw) > m.adaptiveProportionCutoff(hCond),
      `repetition ${m.repetitionCountCutoff(hRaw)}/${m.repetitionCountCutoff(hCond)}, adaptive `
      + `${m.adaptiveProportionCutoff(hRaw)}/${m.adaptiveProportionCutoff(hCond)} - a biased source `
      + "must widen both cutoffs relative to its conditioned output");
  }

  check("sp80090b-cutoffs-match-python",
    mismatches.length === 0,
    mismatches.length
      ? `${mismatches.length} cutoff(s) differ from src/qsh/qrng/health.py - ${mismatches.join("; ")}`
      : `${Object.keys(cutoffs.adaptive_proportion).length * 2} cutoffs match Python across the grid`);

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
  // Since Round 75 the budget carries the ground station's optics (Liao 2017: ~16%, 7.96 dB) and
  // Xinglong's 1 m aperture: 18.06 dB of diffraction at 800 km + 5 + 3 + 7.96 = 34.02 dB. It read
  // 24.48 with a 1.2 m aperture and no optics term.
  const opticsDb = -10 * Math.log10(cfg.free_space_receiver_optics_efficiency);
  check("free-space-800km-is-34.0dB", Math.abs(fsAt(800) - 34.02) < 0.01,
    `800 km gives ${fsAt(800)} dB, not the 34.02 the diffraction model and the Micius budget predict`);
  check("free-space-has-a-fixed-floor",
    near(fsAt(1), cfg.free_space_atmospheric_db + cfg.free_space_pointing_db + opticsDb, 1e-9),
    `at 1 km the loss is ${fsAt(1)} dB, not the ${cfg.free_space_atmospheric_db} + ` +
    `${cfg.free_space_pointing_db} + ${opticsDb.toFixed(2)} dB budget floor - the near-field cap ` +
    `is not being applied`);
  // The station's optics remove signal photons but are not Eve's: QSDC's g must leave them out.
  {
    const link = m.qsdcLink({ ...m.FREE_SPACE, distanceKm: 800 });
    const expected = fsAt(800) - opticsDb - 10 * Math.log10(cfg.free_space_detector_efficiency);
    check("qsdc-g-leaves-out-the-station-optics", near(link.lossDb, expected, 1e-9),
      `QSDC's g is built from ${link.lossDb} dB, not the ${expected} dB of channel and detector`);
  }
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
