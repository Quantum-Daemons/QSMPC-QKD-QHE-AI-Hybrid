// Marching squares for one level of a grid of values (Round 86, the satellite boundary card).

/** Segments of the level-`level` contour of a grid of values at points (xs[i], ys[j]), by marching squares; the two
 *  saddle cases are split by the cell's mean. */
export function contour(xs: number[], ys: number[], v: number[][], level = 0.5): [number, number, number, number][] {
  const out: [number, number, number, number][] = [];
  const lerp = (a: number, b: number, va: number, vb: number) => a + (b - a) * (level - va) / (vb - va);
  for (let j = 0; j + 1 < ys.length; j++) {
    for (let i = 0; i + 1 < xs.length; i++) {
      const a = v[j][i], b = v[j][i + 1], c = v[j + 1][i + 1], d = v[j + 1][i];
      const code = (a >= level ? 8 : 0) | (b >= level ? 4 : 0) | (c >= level ? 2 : 0) | (d >= level ? 1 : 0);
      if (code === 0 || code === 15) continue;
      const bottom: [number, number] = [lerp(xs[i], xs[i + 1], a, b), ys[j]];
      const right: [number, number] = [xs[i + 1], lerp(ys[j], ys[j + 1], b, c)];
      const top: [number, number] = [lerp(xs[i], xs[i + 1], d, c), ys[j + 1]];
      const left: [number, number] = [xs[i], lerp(ys[j], ys[j + 1], a, d)];
      const seg = (p: [number, number], q: [number, number]) => out.push([p[0], p[1], q[0], q[1]]);
      const high = (a + b + c + d) / 4 >= level;
      switch (code) {
        case 1: case 14: seg(left, top); break;
        case 2: case 13: seg(top, right); break;
        case 3: case 12: seg(left, right); break;
        case 4: case 11: seg(bottom, right); break;
        case 6: case 9: seg(bottom, top); break;
        case 7: case 8: seg(left, bottom); break;
        case 5: if (high) { seg(left, bottom); seg(top, right); } else { seg(left, top); seg(bottom, right); } break;
        case 10: if (high) { seg(left, top); seg(bottom, right); } else { seg(left, bottom); seg(top, right); } break;
      }
    }
  }
  return out;
}
