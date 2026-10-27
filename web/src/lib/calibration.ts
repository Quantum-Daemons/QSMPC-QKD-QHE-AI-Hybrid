/**
 * What a published graph's output MEANS, and the reliability bins drawn from it.
 *
 * WHY THIS EXISTS
 *
 * The reliability chart binned column 1 of a two-column output as P(positive). That holds for the
 * scikit-learn exports, whose output is named `probabilities`, and not for the PyTorch exports,
 * whose output is named `logits`. All three medical models are PyTorch exports, and their logits run
 * from -11.7 to 14.1 (ResNet-18), -104 to 114 (ResNet-50) and -3.5 to 4.5 (the student), measured on
 * their own evaluation samples in Round 74. So every medical page drew a calibration chart of
 * logits, captioned "mean p(positive)", with "Widest deviation 5.701 at predicted -5.471". A
 * probability of -5.471 was on the deployed page.
 *
 * The graph says which it is, so read the graph, not the values: every published graph names its
 * score output either `probabilities` or `logits`. Any other name is refused rather than guessed,
 * the same rule as `applyPreprocessing`.
 *
 * Pure, with no onnxruntime import, so `web/scripts/check_calibration.mjs` can test it directly.
 */

export type OutputKind = "probabilities" | "logits";

/** The meaning of a score output, from its name; null when the name does not say. */
export function outputKind(name: string): OutputKind | null {
  if (name === "probabilities") return "probabilities";
  if (name === "logits") return "logits";
  return null;
}

/** Class probabilities, `n` rows of `width`. Logits pass through a softmax, or a sigmoid when the
 *  graph emits a single column; probabilities are returned as they are. */
export function toProbabilities(
  data: ArrayLike<number>, n: number, width: number, kind: OutputKind,
): Float64Array {
  const prob = new Float64Array(n * width);
  if (kind === "probabilities") {
    for (let k = 0; k < n * width; k++) prob[k] = data[k];
    return prob;
  }
  if (width === 1) {
    for (let i = 0; i < n; i++) prob[i] = 1 / (1 + Math.exp(-data[i]));
    return prob;
  }
  for (let i = 0; i < n; i++) {
    let max = -Infinity;
    for (let c = 0; c < width; c++) max = Math.max(max, data[i * width + c]);
    let z = 0;
    for (let c = 0; c < width; c++) {
      const e = Math.exp(data[i * width + c] - max);     // shifted, so a logit of 114 cannot overflow
      prob[i * width + c] = e;
      z += e;
    }
    for (let c = 0; c < width; c++) prob[i * width + c] /= z;
  }
  return prob;
}

/** At least this many events AND non-events before a calibration curve is drawn. Checklist row PP8.
 *
 *  The figure is Van Calster et al. (J Clin Epidemiol 2016), "a minimum of 200 events and 200
 *  non-events" for a flexible calibration curve, and Collins et al. (Stat Med 2016). PP8 cited Riley
 *  et al. (Stat Med 2021) for it, which is the paper that argues such rules of thumb can be too SMALL,
 *  so this is a floor, not a target. Until Round 74 nothing applied it: the samples the browser bins
 *  have 164 events (finance), 42 (medical) and 113 (satellite), and each drew a curve. */
export const MIN_CALIBRATION_EVENTS = 200;

/** Why no calibration curve is drawn for these counts, or null when one may be. */
export function calibrationWithheld(
  events: number, nonEvents: number, eventsName: string, nonEventsName: string,
): string | null {
  if (events >= MIN_CALIBRATION_EVENTS && nonEvents >= MIN_CALIBRATION_EVENTS) return null;
  const f = (v: number) => v.toLocaleString("en-US");
  return `this sample has ${f(events)} ${eventsName} and ${f(nonEvents)} ${nonEventsName}, and a `
    + `calibration curve needs at least ${MIN_CALIBRATION_EVENTS} of each (Van Calster et al., `
    + "J Clin Epidemiol 2016); a binned curve from fewer is noise";
}

/** Equal-WIDTH probability bins: mean predicted probability against observed frequency.
 *
 * Equal-width rather than equal-frequency (deciles) on purpose. Equal-frequency bins hide exactly
 * the failure this plot exists to show: when a calibrator collapses most of its output onto one
 * value - which is what Round 50 found `fit_platt` doing, returning 0.00090148 for nine of finance's
 * ten deciles - equal-frequency binning spreads that constant across ten x-positions and draws a
 * plausible curve. Equal-width binning puts every one of those rows in a single bin and the collapse
 * is the thing you see.
 *
 * Empty bins are dropped rather than plotted at zero: a bin with no rows has no observed frequency,
 * and drawing 0 there would read as "predicted high, observed never".
 *
 * A value outside [0, 1] is not a probability, and binning it would draw the defect this module
 * exists for, so it throws instead of clamping it into the first or last bin.
 */
export function reliabilityBins(
  n: number, prob: (i: number) => number, hit: (i: number) => boolean, bins = 12,
): Array<{ p: number; observed: number; n: number }> {
  const sum = new Float64Array(bins), obs = new Float64Array(bins), cnt = new Int32Array(bins);
  for (let i = 0; i < n; i++) {
    const p = prob(i);
    if (!Number.isFinite(p)) continue;
    if (p < -1e-6 || p > 1 + 1e-6) {
      throw new Error(`reliability: row ${i} has p = ${p}, which is not a probability`);
    }
    const b = Math.min(bins - 1, Math.max(0, Math.floor(p * bins)));
    sum[b] += p; cnt[b] += 1; if (hit(i)) obs[b] += 1;
  }
  const out: Array<{ p: number; observed: number; n: number }> = [];
  for (let b = 0; b < bins; b++) {
    if (cnt[b] > 0) out.push({ p: sum[b] / cnt[b], observed: obs[b] / cnt[b], n: cnt[b] });
  }
  return out;
}
