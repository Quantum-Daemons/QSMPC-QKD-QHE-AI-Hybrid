/**
 * ESA-ADB's corrected event-wise F-score, as its `ESAScores` computes it (Round 86). A port of
 * `src/qsh/models/esa_scores.py`, which reimplements ESAScores from kplabs-pl/ESA-ADB (MIT, commit 67194d3) and is
 * checked against that repository's own test cases; a test runs both on the same inputs and requires equal results.
 *
 * Times are integer MILLISECONDS here where the Python uses nanoseconds: a nanosecond epoch does not fit a double
 * exactly, a millisecond one does, and ESA-ADB's labels carry milliseconds. Interval lengths are summed as integers and
 * divided once, in the Python's order, so the true-negative rate is the same double.
 */

export interface EsaLabel {
  id: string;
  startMs: number;
  endMs: number;
  category: string;
}

export interface EsaScore {
  tp: number;
  fp: number;
  fn: number;
  redundantDetections: number;
  alarmingPrecision: number;
  tnr: number | null;
  precision: number;
  recall: number;
  f05: number;
  nSelectedIds: number;
}

type Event = [number, number, boolean];   // [lower, upper, upperClosed]

/** Runs of 1s as [t_first, t_next) events, or [t_first, t_last] when the run reaches the end. */
export function eventsFromSeries(timesMs: ArrayLike<number>, flags: ArrayLike<number>): Event[] {
  const n = flags.length;
  const out: Event[] = [];
  let i = 0;
  while (i < n) {
    if (!flags[i]) { i++; continue; }
    let j = i;
    while (j < n && flags[j]) j++;
    out.push(j === n ? [timesMs[i], timesMs[n - 1], true] : [timesMs[i], timesMs[j], false]);
    i = j;
  }
  return out;
}

/** Does the event intersect the CLOSED interval [lo, hi]? */
function intersects(ev: Event, lo: number, hi: number): boolean {
  const [a, b, closed] = ev;
  return a <= hi && (closed ? lo <= b : lo < b);
}

function union(intervals: [number, number][]): [number, number][] {
  const out: [number, number][] = [];
  for (const [lo, hi] of [...intervals].sort((x, y) => x[0] - y[0] || x[1] - y[1])) {
    if (out.length && lo <= out[out.length - 1][1]) out[out.length - 1][1] = Math.max(out[out.length - 1][1], hi);
    else out.push([lo, hi]);
  }
  return out;
}

function overlapMs(a: [number, number][], b: [number, number][]): number {
  let i = 0, j = 0, total = 0;
  while (i < a.length && j < b.length) {
    const lo = Math.max(a[i][0], b[j][0]), hi = Math.min(a[i][1], b[j][1]);
    if (hi > lo) total += hi - lo;
    if (a[i][1] < b[j][1]) i++; else j++;
  }
  return total;
}

/** ESA-ADB's corrected event-wise scores for one prediction series; `select` lists the categories that count. */
export function esaEventWise(timesMs: number[], flags: ArrayLike<number>, labels: EsaLabel[], select: string[],
                             fullRange: [number, number], beta = 0.5): EsaScore {
  let t = [...timesMs];
  let f = Array.from(flags, (v) => (v ? 1 : 0));
  const [fullLo, fullHi] = fullRange;
  if (t[0] > fullLo) { t = [fullLo, ...t]; f = [f[0], ...f]; }
  if (t[t.length - 1] < fullHi) { t = [...t, fullHi]; f = [...f, f[f.length - 1]]; }
  const events = eventsFromSeries(t, f);

  const selected = labels.filter((l) => select.includes(l.category));
  const byId = new Map<string, [number, number][]>();
  for (const l of selected) {
    if (!byId.has(l.id)) byId.set(l.id, []);
    byId.get(l.id)!.push([l.startMs, l.endMs]);
  }
  let tp = 0, fn = 0, redundant = 0;
  const matched = new Array<boolean>(events.length).fill(false);
  for (const ivs of byId.values()) {
    const loAll = Math.min(...ivs.map(([a]) => a)), hiAll = Math.max(...ivs.map(([, b]) => b));
    const hits = new Array<number>(ivs.length).fill(0);
    let found = false;
    events.forEach((ev, k) => {
      if (ev[1] < loAll || ev[0] > hiAll) return;
      const inter = ivs.map(([a, b]) => intersects(ev, a, b));
      if (!inter.some(Boolean)) return;
      matched[k] = true;
      found = true;
      inter.forEach((v, i) => { if (v) hits[i]++; });
    });
    redundant += hits.reduce((s, h) => s + (h > 1 ? h - 1 : 0), 0);
    if (found) tp++; else fn++;
  }
  const all = labels.map((l) => [l.startMs, l.endMs] as [number, number]);
  const fp = events.filter((ev, k) => !matched[k] && !all.some(([a, b]) => intersects(ev, a, b))).length;
  let precision = tp + fp ? tp / (tp + fp) : 0;
  const alarming = tp + redundant ? tp / (tp + redundant) : 0;
  let tnr: number | null = null;
  if (precision > 0) {
    const gt = union(all);
    const nominal: [number, number][] = [];
    let cur = fullLo;
    for (const [a, b] of gt) {
      if (a > cur) nominal.push([cur, Math.min(a, fullHi)]);
      cur = Math.max(cur, b);
    }
    if (cur < fullHi) nominal.push([cur, fullHi]);
    const pred = union(events.map(([a, b]) => [a, b] as [number, number]));
    const nominalS = nominal.reduce((s, [a, b]) => s + (b - a), 0) / 1e3;
    tnr = nominalS ? 1 - (overlapMs(nominal, pred) / 1e3) / nominalS : 0;
    precision *= tnr;
  }
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const d = beta ** 2 * precision + recall;
  const fb = d ? ((1 + beta ** 2) * precision * recall) / d : 0;
  return { tp, fp, fn, redundantDetections: redundant, alarmingPrecision: alarming, tnr, precision, recall, f05: fb,
           nSelectedIds: tp + fn };
}
