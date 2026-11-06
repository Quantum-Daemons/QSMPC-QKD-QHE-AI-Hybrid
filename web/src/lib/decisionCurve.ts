/**
 * A decision curve from a published evaluation sample, at the prevalence of the window it was drawn from (Round 86).
 *
 * Net benefit at a threshold probability t is the true positives net of the false positives, each false one priced
 * at the odds t / (1 - t) that acting at t declares (Vickers and Elkin, Medical Decision Making 26(6):565-574, 2006):
 *
 *     NB(t) = mu * TPR(t) - (1 - mu) * FPR(t) * t / (1 - t)
 *
 * The 2006 contribution is the curve across t, so a single net benefit is read against its threshold or not at all.
 *
 * WHY mu IS NOT THE SAMPLE'S RATE. Finance's sample is 164 laundering rows of 8,192, drawn separately from each class
 * of the test window, whose own rate is 2,321 of 1,523,504. Drawn that way the true- and false-positive rates are
 * unbiased and the rate is not, so the estimator puts the window's rate back: Pfeiffer and Gail, Biometrical Journal
 * 62(3):764-776 (2020), eq. 11, the case-control form. Weighted by the sample's 2 percent instead, every caught case
 * would count about thirteen times. `metrics.net_benefit_at_prevalence` is the Python twin, and
 * tests/test_r86_decision_curve.py holds the two together.
 *
 * WHY THE SCORE IS CALIBRATED FIRST. t is a risk, so the number compared with it must be one. The published graph
 * emits XGBoost's probability, fitted with scale_pos_weight, and its record publishes a Platt map fitted on the
 * validation window, which this applies: the trainer's own operating point, 0.3869 on the graph's output, is a
 * calibrated risk near 0.13.
 *
 * The interval is pointwise, from the two independent binomial proportions the sample estimates; the window's rate is
 * a count over the whole window and carries no sampling error of its own.
 */
import { decisionCurveCfg, type ModelSpec } from "./config";

/** A fitted Platt map as `calibration.Platt.as_dict()` publishes it, `p = 1 / (1 + exp(a * s + b))`. */
export interface PlattMap {
  method?: string;
  a: number;
  b: number;
  /** When true, `s` is `logit(clip(score, eps, 1 - eps))`; when false, the score itself. */
  on_logit: boolean;
  logit_clip_eps: number;
}

export interface DecisionCurvePoint {
  /** Threshold probability: act when the calibrated risk is at least this. */
  t: number;
  /** Net benefit of acting on the model at `t`, and its pointwise interval. */
  model: number;
  lo: number;
  hi: number;
  /** Net benefit of acting on every row. Acting on none is 0 at every `t`. */
  treatAll: number;
  tpr: number;
  fpr: number;
}

export interface DecisionCurve {
  points: DecisionCurvePoint[];
  /** The thresholds the config declares; the curve evaluates each exactly. */
  thresholds: number[];
  /** The window's rate, and the counts it is computed from. */
  prevalence: number;
  windowEvents: number;
  windowRows: number;
  /** The sample's rows of each class. */
  events: number;
  nonEvents: number;
  coverage: number;
  /** At the declared thresholds' median: this curve's value, and what weighting by the sample's own rate reads. */
  atMedian: { t: number; corrected: number; naive: number };
  /** The page's classification threshold on the graph's output, and the calibrated risk it is. */
  operating: { raw: number; risk: number } | null;
}

/** `logit(clip(q, eps, 1 - eps))`, as `qsh.models.calibration.logit` computes it. */
function clippedLogit(q: number, eps: number): number {
  const c = Math.min(Math.max(q, eps), 1 - eps);
  return Math.log(c / (1 - c));
}

/** The calibrated risk of one raw score under a published Platt map. */
export function plattRisk(score: number, map: PlattMap): number {
  const s = map.on_logit ? clippedLogit(score, map.logit_clip_eps) : score;
  return 1 / (1 + Math.exp(map.a * s + map.b));
}

/** The thresholds a curve is drawn at: `points` log-spaced from the first declared threshold to the last, with every
 *  declared one added exactly. An interior point within a part in 10^9 of a declared one is the declared one. */
export function decisionGrid(thresholds: number[], points: number): number[] {
  const declared = [...thresholds].sort((x, y) => x - y);
  const lo = declared[0], hi = declared[declared.length - 1];
  const grid = [...declared];
  for (let k = 1; k < points - 1; k++) {
    const t = lo * Math.pow(hi / lo, k / (points - 1));
    if (!declared.some((d) => Math.abs(t - d) <= 1e-9 * d)) grid.push(t);
  }
  return grid.sort((x, y) => x - y);
}

/** First index whose value is at least `t`, in an ascending array. */
function lowerBound(a: Float64Array, t: number): number {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/** The exchange rate a threshold declares, `t / (1 - t)`, with `t` held off 0 and 1 as `metrics.net_benefit` does. */
const odds = (t: number) => {
  const q = Math.min(Math.max(t, 1e-9), 1 - 1e-9);
  return q / (1 - q);
};

/**
 * Net benefit across `grid` from risks and labels drawn separately from each class, at the population's `prevalence`.
 * `null` when the sample has no event or no non-event, since one of the two rates is then undefined.
 */
export function decisionCurve(
  risk: ArrayLike<number>, isEvent: ArrayLike<boolean>, prevalence: number, grid: number[], z: number,
): DecisionCurvePoint[] | null {
  const ev: number[] = [], ne: number[] = [];
  for (let i = 0; i < risk.length; i++) (isEvent[i] ? ev : ne).push(risk[i]);
  if (!ev.length || !ne.length) return null;
  const events = Float64Array.from(ev).sort(), nonEvents = Float64Array.from(ne).sort();
  const m = events.length, n = nonEvents.length, mu = prevalence;
  return grid.map((t) => {
    const tpr = (m - lowerBound(events, t)) / m;
    const fpr = (n - lowerBound(nonEvents, t)) / n;
    const w = odds(t);
    const model = mu * tpr - (1 - mu) * fpr * w;
    const se = Math.sqrt(mu * mu * tpr * (1 - tpr) / m + (1 - mu) ** 2 * w * w * fpr * (1 - fpr) / n);
    return { t, model, lo: model - z * se, hi: model + z * se, treatAll: mu - (1 - mu) * w, tpr, fpr };
  });
}

const median = (xs: number[]) => {
  const s = [...xs].sort((x, y) => x - y);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

/**
 * The decision curve of a published binary model on its own evaluation sample, or why the page cannot draw one.
 *
 * `null` when the config declares no curve for this model: only finance's does (the owner's decision, Round 29).
 * `probability(i)` is the graph's P(positive) for row `i`, or `null` when the graph's output is neither probabilities
 * nor logits, and `operatingRaw` is the threshold the page classifies at, on the same scale.
 */
export function publishedDecisionCurve(
  usecase: string, spec: ModelSpec | undefined, probability: ((i: number) => number) | null,
  labels: ArrayLike<number>, n: number, operatingRaw: number | null,
): { curve: DecisionCurve } | { withheld: string } | null {
  const cfg = decisionCurveCfg();
  const entry = cfg.curves[usecase];
  if (!entry || !spec || entry.model_key !== spec.key) return null;
  if (!probability) return { withheld: "the graph's output says neither probabilities nor logits, so it has no risk to compare with a threshold" };
  const rec = (spec.achieved ?? {}) as Record<string, unknown>;
  const cal = rec.calibration as (PlattMap & { corp_calibrated?: { positives?: number; n?: number } }) | undefined;
  if (!cal || cal.method !== "platt" || typeof cal.a !== "number" || typeof cal.b !== "number") {
    return { withheld: "its record publishes no calibration map, so its scores are not risks a threshold can be read against" };
  }
  // The window's rate as two counts, from the record's CORP block, which scores the calibrated map over that window.
  const windowEvents = cal.corp_calibrated?.positives, windowRows = cal.corp_calibrated?.n;
  if (typeof windowEvents !== "number" || typeof windowRows !== "number" || windowRows !== rec.n_test) {
    return { withheld: "its record does not state the event count of the window its sample was drawn from, and the curve needs that rate" };
  }
  const prevalence = windowEvents / windowRows;
  const risk = Float64Array.from({ length: n }, (_, i) => plattRisk(probability(i), cal));
  const isEvent = Array.from({ length: n }, (_, i) => labels[i] === 1);
  const grid = decisionGrid(entry.thresholds, cfg.grid_points);
  const points = decisionCurve(risk, isEvent, prevalence, grid, cfg.interval_z);
  if (!points) return { withheld: "its sample holds rows of one class only, so one of the two rates is undefined" };
  const events = isEvent.filter(Boolean).length, nonEvents = n - events;
  const tMed = median(entry.thresholds);
  const atMed = points.find((p) => p.t === tMed)!;
  const sampleRate = events / n;
  const naive = sampleRate * atMed.tpr - (1 - sampleRate) * atMed.fpr * odds(tMed);
  return {
    curve: {
      points, thresholds: [...entry.thresholds].sort((x, y) => x - y), prevalence, windowEvents, windowRows,
      events, nonEvents, coverage: cfg.interval_coverage,
      atMedian: { t: tMed, corrected: atMed.model, naive },
      operating: operatingRaw != null ? { raw: operatingRaw, risk: plattRisk(operatingRaw, cal) } : null,
    },
  };
}
