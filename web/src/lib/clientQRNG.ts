// Client-side QRNG mirror of src/qsh/qrng/ — a genuine state-vector quantum-measurement simulation
// entirely in the browser, with the same SP 800-22-style health checks and MEASURED throughput.
// Honest label: a SIMULATED quantum measurement (state-vector), not quantum hardware. The optional ANU
// hardware source is opt-in + online (never the default), matching the backend.
import { rng } from "./quantumSim";
import { qrngCfg } from "./config";

const SR2 = Math.SQRT1_2;
// |+> = H|0> = [1/sqrt2, 1/sqrt2]; Born rule => P(measure |1>) = |amp_1|^2 = 0.5.
const PROB_ONE = SR2 * SR2;

export interface QRNGResult {
  bits: Uint8Array;
  nBits: number;
  source: string;
  postProcessing: string;
  healthPassed: boolean;
  monobitP: number;
  runsP: number;
  blockFreqP: number;
  throughputBitsPerSec: number;
  simulated: boolean;
  minEntropyPerBit: number;   // SP 800-90B most-common-value estimate (bits of min-entropy per output bit)
  extractionRate: number;     // output bits / raw bits consumed
  /** Bits the leftover hash lemma permits at this eps and this much raw min-entropy, or null for
   *  the von Neumann extractor, which does not consume eps at all.
   *
   *  Surfaced because it is the ONLY place the LHL eps parameter is observable. Delivered output is
   *  min(this, requested), and requested is normally the smaller of the two - so at every setting the
   *  spinner can reach, changing eps from -1 to -128 produced byte-identical output and the control
   *  looked inert. It is not inert; it moves this bound, and the bound was never shown. */
  lhlMaxBits: number | null;
  /** What the caller asked for, so a shortfall against `nBits` is visible rather than inferred. */
  requestedBits: number;
  /** Whether there were enough bits for the health tests to mean anything.
   *
   *  Both SP 800-90B continuous tests return `passed: true` when handed too little data - the
   *  adaptive-proportion test short-circuits below its 1024-bit window, and the repetition-count
   *  cutoff derived from a zero min-entropy estimate is ~2e10, which no run can exceed. So a 1-bit
   *  output reported health PASS with min-entropy 0: a check that passed because there was nothing
   *  left to test. `healthPassed` now requires this to be true, and the run says which it was. */
  healthConclusive: boolean;
  /** SP 800-90B section 4.4 continuous health tests — these gate `healthPassed`. */
  repetitionCountPassed: boolean;
  repetitionCountCutoff: number;
  repetitionCountLongest: number;
  adaptiveProportionPassed: boolean;
  adaptiveProportionCutoff: number;
  adaptiveProportionMax: number;
  /** SP 800-22 output-stream p-values verdict: reported, deliberately NOT gating (see section 4.1). */
  sp80022Passed: boolean;
}

// erfc via the Abramowitz-Stegun 7.1.26 rational approximation (pure JS, no dep).
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const y = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277))))))))); // erfc(z) for z>=0
  return x >= 0 ? y : 2 - y;
}

// Prepare n |+> qubits and measure each in the Z basis (Born-rule sampling); genuine 50/50 outcomes.
function measurePlus(n: number, u: () => number): Uint8Array {
  const b = new Uint8Array(n);
  for (let i = 0; i < n; i++) b[i] = u() < PROB_ONE ? 1 : 0;
  return b;
}

function vonNeumann(bits: Uint8Array): Uint8Array {
  const out: number[] = [];
  for (let i = 0; i + 1 < bits.length; i += 2) if (bits[i] !== bits[i + 1]) out.push(bits[i]);
  return Uint8Array.from(out);
}

// SP 800-22 subset: monobit + runs + block-frequency (p-values).
function monobit(b: Uint8Array): number {
  if (!b.length) return 0;
  let s = 0;
  for (const x of b) s += 2 * x - 1;
  return erfc(Math.abs(s) / Math.sqrt(b.length) / Math.SQRT2);
}
function runs(b: Uint8Array): number {
  const n = b.length;
  if (n < 2) return 0;
  let ones = 0;
  for (const x of b) ones += x;
  const pi = ones / n;
  if (Math.abs(pi - 0.5) >= 2 / Math.sqrt(n)) return 0;
  let v = 1;
  for (let i = 1; i < n; i++) if (b[i] !== b[i - 1]) v++;
  const den = 2 * Math.sqrt(2 * n) * pi * (1 - pi);
  return den > 0 ? erfc(Math.abs(v - 2 * n * pi * (1 - pi)) / den) : 0;
}
function blockFrequency(b: Uint8Array, m = 128): number {
  const nb = Math.floor(b.length / m);
  if (nb === 0) return 1;
  let chi = 0;
  for (let k = 0; k < nb; k++) {
    let ones = 0;
    for (let i = 0; i < m; i++) ones += b[k * m + i];
    const pi = ones / m;
    chi += (pi - 0.5) * (pi - 0.5);
  }
  chi *= 4 * m;
  // P = igamc(nb/2, chi/2); series/continued-fraction upper incomplete gamma (regularized).
  return gammaincc(nb / 2, chi / 2);
}
// Regularized upper incomplete gamma Q(a,x) (Numerical-Recipes style), pure JS.
function gammaincc(a: number, x: number): number {
  if (x <= 0) return 1;
  const gln = lgamma(a);
  if (x < a + 1) {
    let ap = a, sum = 1 / a, del = sum;
    for (let n = 0; n < 200; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-12) break; }
    return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 200; i++) {
    const an = -i * (i - a);
    b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-12) break;
  }
  return Math.exp(-x + a * Math.log(x) - gln) * h;
}
function lgamma(x: number): number {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

// 99% one-sided confidence Z for the SP 800-90B most-common-value bound (intentionally fixed spec constant).
const MCV_Z = 2.5758293035489004;

// SP 800-90B (6.3.1) most-common-value min-entropy estimate: bits of min-entropy per bit.
// EXPORTED so a cross-language gate can reach it. The shipped source is near-fair, so the
// raw-versus-conditioned defect is invisible through the public `generateQuantumBits` API - it
// only shows on a BIASED stream, and the internal PROB_ONE cannot be injected. What is not
// exported cannot be checked, which is the same reason the default MCP combiner went unpinned.
export function minEntropyMcv(bits: Uint8Array): number {
  const n = bits.length;
  if (n === 0) return 0;
  let ones = 0;
  for (const x of bits) ones += x;
  const pHat = Math.max(ones, n - ones) / n;
  const pUpper = Math.min(1, pHat + MCV_Z * Math.sqrt((pHat * (1 - pHat)) / n));
  return pUpper > 0 ? -Math.log2(pUpper) : 0;
}

// Safe extractor output length by the (Quantum) Leftover-Hash-Lemma: out <= H_min - 2*log2(1/eps) + 2.
function lhlOutputLength(minEntropyBits: number, epsilonLog2: number): number {
  return Math.max(1, Math.floor(minEntropyBits - 2 * -epsilonLog2 + 2));
}

// Toeplitz-hash (universal-2) extractor: out = T @ bits (mod 2). T's defining bits come from an
// independent second source (toeplitzBits -> the seedless two-source construction) or a PRG (u).
export function toeplitzExtract(bits: Uint8Array, outLen: number, u: () => number, toeplitzBits?: Uint8Array): Uint8Array {
  const m = bits.length;
  outLen = Math.max(1, Math.min(outLen, m));
  const needT = outLen + m - 1;
  let tvec: Uint8Array;
  if (toeplitzBits) {
    if (toeplitzBits.length < needT) throw new Error("toeplitzBits too short");
    tvec = toeplitzBits;
  } else {
    tvec = new Uint8Array(needT);
    for (let i = 0; i < needT; i++) tvec[i] = u() < 0.5 ? 1 : 0;
  }
  const out = new Uint8Array(outLen);
  for (let i = 0; i < outLen; i++) {
    let acc = 0;
    for (let j = 0; j < m; j++) acc ^= tvec[i + j] & bits[j];
    out[i] = acc & 1;
  }
  return out;
}

// Generate nBits post-processed quantum-random bits + a health report (incl. a min-entropy estimate) +
// MEASURED throughput + extraction rate. The information-theoretic extractor ("toeplitz"/"seedless") sizes
// its output by the Leftover-Hash-Lemma bound (no fixed rate); eps + min-entropy come from config.

// ---------------------------------------------------------------------------------------------
// SP 800-90B section 4.4 continuous health tests.
//
// These GATE the health verdict; the SP 800-22 p-values below are reported beside them but do not.
// That ordering is the standard's, not a preference: SP 800-90B section 4.1 says in terms that
// "the randomness tests described in NIST SP 800-22 ... are not useful for monitoring the noise
// source". The browser previously computed healthPassed from monobit/runs/block-frequency alone,
// i.e. exactly the tests the standard disqualifies for this purpose, while src/qsh/qrng/health.py
// had already been corrected. This is the browser catching up to the backend.
const HEALTH_ALPHA = Math.pow(2, -20);
const ADAPTIVE_PROPORTION_W = 1024;

/** section 4.4.1: C = 1 + ceil(-log2(alpha) / H). The standard's worked example, H = 2.0 at
 *  alpha = 2^-20, gives C = 11 — asserted in tests/test_qrng.py against this same formula. */
export function repetitionCountCutoff(h: number, alpha = HEALTH_ALPHA): number {
  const hh = h > 0 ? h : 1e-9;
  return 1 + Math.ceil(-Math.log2(alpha) / hh);
}

/** Fail if any value repeats C or more times consecutively — a stuck noise source. */
function repetitionCountTest(bits: Uint8Array, h: number) {
  const cutoff = repetitionCountCutoff(h);
  if (bits.length === 0) return { passed: true, cutoff, longest: 0 };
  let longest = 1, run = 1;
  for (let i = 1; i < bits.length; i++) {
    run = bits[i] === bits[i - 1] ? run + 1 : 1;
    if (run > longest) longest = run;
  }
  return { passed: longest < cutoff, cutoff, longest };
}

/** section 4.4.2 cutoff: smallest C whose Binom(W-1, 2^-H) upper tail is <= alpha. Computed with an
 *  exact binomial survival function — the normal approximation is unusable this far into the tail. */
export function adaptiveProportionCutoff(h: number, w = ADAPTIVE_PROPORTION_W, alpha = HEALTH_ALPHA): number {
  const p = Math.pow(2, -Math.max(h, 1e-9));
  const n = w - 1;
  // survival: P[X >= c] computed by summing the pmf downward from c.
  let logPmf = n * Math.log1p(-Math.min(p, 1)); // pmf(0)
  let tail = 1.0, c = 0;
  const lp = Math.log(Math.min(p, 1)), lq = Math.log1p(-Math.min(p, 1));
  for (c = 0; c <= n; c++) {
    if (tail <= alpha) break;
    tail -= Math.exp(logPmf);
    logPmf += Math.log((n - c) / (c + 1)) + lp - lq;
  }
  // `c`, NOT `1 + c`. At the top of each iteration `tail` holds P[X >= c] exactly - it starts at
  // P[X >= 0] = 1 and loses one pmf term per step - so the loop exits holding the SMALLEST c with
  // P[X >= c] <= alpha, which is the cutoff by definition.
  //
  // The `1 +` was transcribed from the Python twin, where it is correct for a reason that does not
  // transfer: `binom.isf(alpha, n, p)` returns the largest x with P[X > x] <= alpha, i.e. one BELOW
  // the cutoff, so that copy has to add one back. Two implementations of one quantity, and the
  // correction term belonged to only one of them.
  //
  // The result was a uniform off-by-one at EVERY entropy - measured 991/990, 793/792, 626/625,
  // 590/589, 325/324, 106/105 - in the PERMISSIVE direction, because a higher cutoff fails fewer
  // windows. This is the client-side-only demo, so the wrong value is the one a visitor is shown
  // and the one `healthPassed` is computed from.
  //
  // Nothing asserted either cutoff. `check_qkd.mjs` spot-checked `repetitionCountCutoff(2.0)` alone,
  // and one spot check on a sibling function cannot see a uniform offset in this one.
  return Math.min(w, c);
}

/** Fail if one value dominates any non-overlapping window of W samples — a collapsed source. */
function adaptiveProportionTest(bits: Uint8Array, h: number, w = ADAPTIVE_PROPORTION_W) {
  const cutoff = adaptiveProportionCutoff(h, w);
  if (bits.length < w) return { passed: true, cutoff, worst: 0 };
  let worst = 0;
  for (let start = 0; start + w <= bits.length; start += w) {
    const ref = bits[start];
    let count = 0;
    for (let i = start; i < start + w; i++) if (bits[i] === ref) count++;
    if (count > worst) worst = count;
  }
  return { passed: worst < cutoff, cutoff, worst };
}

export function generateQuantumBits(nBits: number, opts: { source?: string; postProcessing?: string;
  seed?: number; alpha?: number; epsilonLog2?: number; minEntropyPerRawBit?: number; blockSize?: number } = {}): QRNGResult {
  const source = opts.source ?? "sim";
  if (source === "anu") throw new Error("ANU QRNG is opt-in + online; the offline 'sim' source is the demo default");
  const q = qrngCfg();
  const post = opts.postProcessing ?? q.post_processing ?? "von_neumann";
  const alpha = opts.alpha ?? q.health_alpha ?? 0.01;
  const epsLog2 = opts.epsilonLog2 ?? q.security_epsilon_log2 ?? -40;
  const minH = opts.minEntropyPerRawBit ?? q.min_entropy_per_raw_bit ?? 0.9;
  const blockSize = opts.blockSize ?? q.health_block_size ?? 128;
  const u = opts.seed === undefined ? Math.random : rng(opts.seed);
  const t0 = performance.now();
  const isExtract = post === "toeplitz" || post === "seedless";
  let lhlMaxBits: number | null = null;
  const need = post === "von_neumann" ? 5 * nBits + 16 : isExtract ? 2 * nBits + 16 : nBits;
  const raw = measurePlus(need, u);
  let proc: Uint8Array;
  let rawConsumed = raw.length;
  if (post === "von_neumann") {
    proc = vonNeumann(raw);
  } else if (isExtract) {
    const outLen = lhlOutputLength(minH * raw.length, epsLog2);
    lhlMaxBits = outLen;
    if (post === "seedless") {
      const u2 = opts.seed === undefined ? Math.random : rng(opts.seed + 1);
      const raw2 = measurePlus(outLen + raw.length - 1, u2);   // independent second source defines T
      proc = toeplitzExtract(raw, outLen, u, raw2);
      rawConsumed += raw2.length;
    } else {
      proc = toeplitzExtract(raw, outLen, u);
    }
  } else {
    proc = raw;
  }
  const bits = proc.slice(0, nBits);
  const dt = Math.max(performance.now() - t0, 1e-6);
  // SP 800-22 STAYS ON THE OUTPUT: monobit, runs and block-frequency are statements about the stream
  // a consumer receives, and running them on raw would be the mirror-image error.
  const mp = monobit(bits), rp = runs(bits), bp = blockFrequency(bits, blockSize);

  // SP 800-90B SECTION 4.4 RUNS ON `raw`, NOT ON `bits`, AND THIS COPY DID NOT.
  //
  // Section 4.4's continuous tests run on raw samples BEFORE any conditioning, and the min-entropy
  // figure is a statement about the NOISE SOURCE. This function bound `raw` at the top and then
  // computed `hMin`, both continuous tests and the reported `minEntropyPerBit` from `bits` - the
  // post-extractor output. An extractor produces near-uniform output by construction, so the number
  // described the Toeplitz stage under a label that says entropy source.
  //
  // Measured in this module on a p(1) = 0.90 stream: raw min-entropy **0.1510**, conditioned
  // **0.9749**. A ruined source and a healthy one are a hundredth of a bit apart. The cutoffs move
  // too - repetition count 134 against 22, adaptive proportion 965 against 597.
  //
  // `src/qsh/qrng/health.py` takes `raw=` and its docstring states this clause; `simulator.py`
  // passes it; `tests/test_health_tests_measure_the_noise_source.py` pins it - IN PYTHON ONLY. The
  // Round 51 fix reached the reference implementation and not the one every visitor executes, and
  // the demo is client-side only, so the wrong figure is the only figure anyone sees.
  //
  // NOT changed: `healthPassed`. Sections 4.4.1 and 4.4.2 catch a source that has gone STUCK, not
  // one that is merely skewed, so the verdict is correct either way and asserting otherwise would
  // claim the standard says something it does not. What was wrong is the reported entropy.
  const hMin = minEntropyMcv(raw);
  const rct = repetitionCountTest(raw, hMin);
  const apt = adaptiveProportionTest(raw, hMin);
  const sp80022Passed = mp >= alpha && rp >= alpha && bp >= alpha;
  // The adaptive-proportion window is the binding requirement: below it that test cannot run at all.
  // Judged on `raw`, because that is now the stream the adaptive-proportion test consumes. Judging
  // it on `bits` would let a short raw stream report a conclusive verdict from a test that could not
  // run - the same mismatch, one field over.
  const healthConclusive = raw.length >= ADAPTIVE_PROPORTION_W;
  return {
    bits, nBits: bits.length, source, postProcessing: post,
    healthPassed: healthConclusive && rct.passed && apt.passed,
    healthConclusive,
    repetitionCountPassed: rct.passed, repetitionCountCutoff: rct.cutoff, repetitionCountLongest: rct.longest,
    adaptiveProportionPassed: apt.passed, adaptiveProportionCutoff: apt.cutoff, adaptiveProportionMax: apt.worst,
    sp80022Passed,
    monobitP: mp, runsP: rp, blockFreqP: bp,
    throughputBitsPerSec: (bits.length / dt) * 1000, simulated: true,
    minEntropyPerBit: hMin, extractionRate: bits.length / Math.max(1, rawConsumed),
    lhlMaxBits, requestedBits: nBits,
  };
}
