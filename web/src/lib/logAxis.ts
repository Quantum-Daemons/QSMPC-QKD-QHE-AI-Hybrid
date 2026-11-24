/** The decade at or below v (up = false), or at or above it (up = true). */
const decade = (v: number, up: boolean) => 10 ** (up ? Math.ceil(Math.log10(v)) : Math.floor(Math.log10(v)));

/** A point this close to a decade line, as a ratio, would be drawn on the axis: its mark cut in half and its name
 *  struck through by the axis line. On the deployed site (2026-11-24) a tab run's 1.05e-5 sat on a 1e-5 floor. */
export const EDGE = 1.5;

/** The decades a log error axis needs to hold every point and the limit, at least 1e-5 to 1e-2, with each point at
 *  least a factor of EDGE inside its ends (the square student's parameter-set plot). */
export function errorDomain(errors: readonly number[], limit: number): [number, number] {
  const finite = errors.filter((e) => Number.isFinite(e) && e > 0);
  const below = (v: number) => { const d = decade(v, false); return v / d < EDGE ? d / 10 : d; };
  const above = (v: number) => { const d = decade(v, true); return d / v < EDGE ? d * 10 : d; };
  return [Math.min(1e-5, decade(limit, false), ...finite.map(below)),
          Math.max(1e-2, decade(limit, true), ...finite.map(above))];
}
