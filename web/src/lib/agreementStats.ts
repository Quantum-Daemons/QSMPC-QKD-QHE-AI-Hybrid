/**
 * How far the encrypted scores sit from the plaintext ones, two ways (Round 86, the owner's choice of 2026-11-14: the
 * evidence plots under the private-inference scatter).
 *
 * BLAND-ALTMAN. Bland and Altman, "Statistical methods for assessing agreement between two methods of clinical
 * measurement", Lancet 327(8476):307-310, 1986, doi:10.1016/S0140-6736(86)90837-8: plot each pair's difference against
 * its mean, and give the mean difference (the bias) and the limits of agreement, the bias plus and minus 1.96 standard
 * deviations of the differences. Here the two "methods" are the same model evaluated in the clear and under CKKS, so the
 * bias says whether the encryption shifts the scores and the limits how far one row's score moves.
 *
 * PRECISION IN BITS. Lattigo's `ckks.GetPrecisionStats` (schemes/ckks/precision.go, read 2026-11-14) gives each value
 * the precision -log2 |decrypted - reference|, of the absolute error; a value equal to its reference has no finite
 * precision and is counted here as exact, where Lattigo substitutes log2 of the scale, which this page does not know.
 * The sources: references/evidence-plots.md.
 */

export interface AgreementStats {
  n: number;
  /** Mean of enc - plain. */
  bias: number;
  /** Sample standard deviation of enc - plain (n - 1); 0 for one pair. */
  sd: number;
  /** bias - 1.96 sd and bias + 1.96 sd. */
  limits: [number, number];
  /** Pairs whose difference lies within the limits, inclusive. */
  within: number;
  /** The Bland-Altman points: x the pair's mean, y its difference. */
  points: { x: number; y: number }[];
  /** Pairs with no difference at all, which have no finite precision. */
  exact: number;
  /** floor(-log2 |enc - plain|) -> how many pairs, over the pairs with a difference, ascending by bits. */
  bits: { bits: number; count: number }[];
  /** Median and minimum of -log2 |enc - plain| over the pairs with a difference; null when every pair is exact. */
  medianBits: number | null;
  minBits: number | null;
}

export function agreementStats(pairs: readonly { enc: number; plain: number }[]): AgreementStats | null {
  const ok = pairs.filter((p) => Number.isFinite(p.enc) && Number.isFinite(p.plain));
  const n = ok.length;
  if (n === 0) return null;
  const d = ok.map((p) => p.enc - p.plain);
  let sum = 0;
  for (const v of d) sum += v;
  const bias = sum / n;
  let ss = 0;
  for (const v of d) ss += (v - bias) * (v - bias);
  const sd = n > 1 ? Math.sqrt(ss / (n - 1)) : 0;
  const limits: [number, number] = [bias - 1.96 * sd, bias + 1.96 * sd];
  let within = 0;
  for (const v of d) if (v >= limits[0] && v <= limits[1]) within += 1;
  const points = ok.map((p, i) => ({ x: (p.enc + p.plain) / 2, y: d[i] }));
  const prec: number[] = [];
  let exact = 0;
  for (const v of d) {
    if (v === 0) exact += 1;
    else prec.push(-Math.log2(Math.abs(v)));
  }
  const counts = new Map<number, number>();
  for (const b of prec) {
    const k = Math.floor(b);
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  const bits = [...counts.entries()].sort((a, b) => a[0] - b[0]).map(([k, c]) => ({ bits: k, count: c }));
  const sorted = [...prec].sort((a, b) => a - b);
  const median = sorted.length === 0 ? null
    : sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  return { n, bias, sd, limits, within, points, exact, bits, medianBits: median, minBits: sorted.length ? sorted[0] : null };
}
