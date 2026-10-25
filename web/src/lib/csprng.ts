// Randomness for secret-sharing masks, from the browser's CSPRNG (`crypto.getRandomValues`).
//
// A mask is what hides a share, so the other party must not be able to predict it. The run's seeded
// generator (`rng` in quantumSim.ts: mulberry32, keyed on a seed the page shows) is the right source
// for reproducible demo inputs and the wrong one for masks: anyone who knows the seed can recompute
// every mask, and 32 bits of state cannot be uniform over every range a mask needs. Until Round 66
// both of the demo's mask sites drew from it - the secure argmax in clientMPC.ts and the secure sum in
// clientCompute.ts. Neither result depends on the mask values, and no seeded figure moved: the secure
// sum still advances the run's generator once per secret, and the argmax's draws were the run's last.

/** `n` independent words, each uniform over [0, 2^32). */
export function randomWords(n: number): Uint32Array {
  const out = new Uint32Array(n);
  // getRandomValues fills at most 65536 bytes per call: 16384 words.
  for (let i = 0; i < n; i += 16384) crypto.getRandomValues(out.subarray(i, Math.min(n, i + 16384)));
  return out;
}

/** A uniform integer in [0, p) for an integer 1 <= p <= 2^32, by rejection, so with no modulo bias. */
export function randomBelow(p: number): number {
  if (!Number.isInteger(p) || p < 1 || p > 2 ** 32) {
    throw new RangeError(`randomBelow needs an integer in [1, 2^32], got ${p}`);
  }
  // The largest multiple of p that is at most 2^32. Words at or above it are redrawn. That happens
  // with probability below one half for any p, so it takes fewer than two draws on average.
  const limit = 2 ** 32 - (2 ** 32 % p);
  const w = new Uint32Array(1);
  for (;;) {
    crypto.getRandomValues(w);
    if (w[0] < limit) return w[0] % p;
  }
}
