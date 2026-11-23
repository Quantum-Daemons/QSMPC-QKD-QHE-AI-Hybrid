/**
 * SP 800-90B section 4.4's two continuous health tests, run sample by sample (Round 86, the owner's choice of
 * 2026-11-22: a live QRNG view with fault injection).
 *
 * `clientQRNG.ts` runs both tests over a finished block; a noise source runs them on every sample as it arrives, and
 * stops at the first failure. This is that form, so the page can show the counters moving and a fault tripping them:
 *
 * - Repetition count (4.4.1): A is the current value and B how many times in a row it has come; when B reaches C, the
 *   test fails. C = 1 + ceil(-log2(alpha) / H).
 * - Adaptive proportion (4.4.2): in each non-overlapping window of W samples, A is the window's first value and B how
 *   many of the window's samples equal it, the first included; when B reaches C, the test fails. C is the binomial
 *   cutoff `adaptiveProportionCutoff`.
 *
 * Both cutoffs are `clientQRNG.ts`'s, at the source's ASSESSED min-entropy (`qrng.min_entropy_per_raw_bit`) and false
 * alarm rate alpha = 2^-20, and `tests/test_r86_qrng_live.py` holds this stream to the Python tests
 * (`qsh.qrng.health`) on the same samples. The samples are simulated: a fair coin standing in for measuring |+>.
 */
import { adaptiveProportionCutoff, repetitionCountCutoff } from "./clientQRNG";

export const APT_WINDOW = 1024;

/** What the source does: work, stick at 1, or come up 1 three times in four. */
export type Fault = "none" | "stuck" | "biased";
export const BIASED_ONES = 0.75;

export interface HealthState {
  samples: number;
  ones: number;
  rctCutoff: number;
  /** The value of the current run and its length. */
  runValue: number | null;
  run: number;
  longestRun: number;
  aptCutoff: number;
  window: number;
  /** Samples seen in the current window, its first value, and how many equal it so far. */
  inWindow: number;
  windowValue: number | null;
  count: number;
  worstCount: number;
  /** The sample (1-based) at which each test first failed, or null. */
  rctFailedAt: number | null;
  aptFailedAt: number | null;
}

export function newHealthState(h: number, w = APT_WINDOW): HealthState {
  return {
    samples: 0, ones: 0, rctCutoff: repetitionCountCutoff(h), runValue: null, run: 0, longestRun: 0,
    aptCutoff: adaptiveProportionCutoff(h, w), window: w, inWindow: 0, windowValue: null, count: 0, worstCount: 0,
    rctFailedAt: null, aptFailedAt: null,
  };
}

/** One sample through both tests, in place. */
export function feed(s: HealthState, bit: number): void {
  s.samples += 1;
  if (bit) s.ones += 1;
  if (bit === s.runValue) s.run += 1;
  else { s.runValue = bit; s.run = 1; }
  if (s.run > s.longestRun) s.longestRun = s.run;
  if (s.rctFailedAt === null && s.run >= s.rctCutoff) s.rctFailedAt = s.samples;
  if (s.inWindow === 0) { s.windowValue = bit; s.count = 1; }
  else if (bit === s.windowValue) s.count += 1;
  s.inWindow += 1;
  if (s.count > s.worstCount) s.worstCount = s.count;
  if (s.aptFailedAt === null && s.count >= s.aptCutoff) s.aptFailedAt = s.samples;
  if (s.inWindow === s.window) s.inWindow = 0;
}

export const failed = (s: HealthState) => s.rctFailedAt !== null || s.aptFailedAt !== null;

/** The next raw sample from a uniform draw `u` in [0, 1): the simulated measurement, or the injected fault. */
export function sampleBit(u: number, fault: Fault): number {
  if (fault === "stuck") return 1;
  if (fault === "biased") return u < BIASED_ONES ? 1 : 0;
  return u < 0.5 ? 1 : 0;
}
