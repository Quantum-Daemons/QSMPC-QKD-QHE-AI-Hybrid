#!/usr/bin/env node
/**
 * The lending simulator against closed forms and its own books (Round 86).
 *
 * The page draws what `web/src/lib/lending/model.ts` returns, so the model is checked here: the health factor on Aave's own worked figures, the two-slope rate at its
 * knots, the close factor at its boundary, a run's accounting, and that a run is what its seed says.
 *
 * Run: npm run check:lending
 */
import { createRequire } from "node:module";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const { build } = createRequire(join(WEB, "package.json"))("esbuild");
const tmp = mkdtempSync(join(tmpdir(), "qsh-lending-"));
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failed++; };
const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

try {
  await build({
    stdin: { contents: `export * from ${JSON.stringify(join(WEB, "src/lib/lending/model"))};`, resolveDir: WEB, loader: "ts" },
    bundle: true, format: "esm", platform: "node", outfile: join(tmp, "b.mjs"), logLevel: "error",
  });
  const m = await import(pathToFileURL(join(tmp, "b.mjs")).href);
  const c = JSON.parse(readFileSync(join(WEB, "src/lib/config.defaults.json"), "utf8")).lending;

  // Aave's worked example: 10,000 of collateral at an 80% threshold against 6,000 of debt.
  check(near(m.healthFactor([1], [10000], [0.8], 6000), 8000 / 6000), "the health factor on the worked example is 1.333");
  check(m.healthFactor([1], [10000], [0.8], 0) === Infinity, "no debt, no liquidation");

  check(near(m.twoSlopeRate(0, c), c.base_rate), "the rate at no utilisation is the base");
  check(near(m.twoSlopeRate(c.optimal_utilisation, c), c.base_rate + c.slope1), "the rate at the optimum is base plus slope 1");
  check(near(m.twoSlopeRate(1, c), c.base_rate + c.slope1 + c.slope2), "the rate at full utilisation adds slope 2");
  const grid = Array.from({ length: 101 }, (_, i) => m.twoSlopeRate(i / 100, c));
  check(grid.every((r, i) => i === 0 || r >= grid[i - 1]), "the rate never falls as utilisation rises");
  let threw = false;
  try { m.twoSlopeRate(1.01, c); } catch { threw = true; }
  check(threw, "a utilisation above 1 is refused");

  check(m.closeFactor(0.97, c) === c.close_factor && m.closeFactor(0.95, c) === 1 && m.closeFactor(0.94, c) === 1, "the close factor is 50% above 0.95 and 100% at or below it, as in LiquidationLogic.sol");
  const params = m.withRunParams(c, { lending_borrowers: 12, lending_bob_ltv_pct: 50, lending_shock_hour: 100, lending_shock_pct: 20, lending_vol_mult: 2, lending_checks: 4 });
  check(params.borrowers === 12 && params.supply === c.supply / 2 && params.bob_initial_ltv === 0.5 && params.shock_hour === 100 && params.shock_fraction === 0.2
    && params.annual_vol.every((v, i) => v === c.annual_vol[i] * 2) && params.check_points === 4, "every control on the page reaches the simulation");
  check(JSON.stringify(m.withRunParams(c, {})) === JSON.stringify(c), "with no controls set the simulation runs on the configured values");

  let sameSign = true;
  for (let i = 0; i < 2000; i++) {
    const amounts = [Math.random() * 10, Math.random() * 3], prices = [Math.random() * 4000, Math.random() * 70000], debt = Math.random() * 100000 + 1;
    const hf = m.healthFactor(amounts, prices, c.liquidation_thresholds, debt), mg = m.margin(amounts, prices, c.liquidation_thresholds, debt);
    if ((hf < 1) !== (mg < 0)) sameSign = false;
  }
  check(sameSign, "the margin is negative exactly when the health factor is below 1");

  const seed = 7, a = m.simulate(c, seed), b = m.simulate(c, seed), other = m.simulate(c, seed + 1);
  check(JSON.stringify(a.utilisation) === JSON.stringify(b.utilisation) && JSON.stringify(a.events) === JSON.stringify(b.events), "a run is what its seed says");
  check(JSON.stringify(a.prices) !== JSON.stringify(other.prices), "another seed is another path");
  check(a.prices.every((s) => s.length === c.hours + 1 && s.every((p) => p > 0)), "every price is positive, for every hour");
  check(a.utilisation.every((u) => u >= 0 && u <= 1), "utilisation stays in [0, 1]");
  check(a.borrowRate.every((r) => r >= c.base_rate), "the rate never goes below the base");
  check(a.totalDebt.every((d) => d >= 0), "debt is never negative");
  check(a.maxAccountingGap < 1e-6 * c.supply, `the pool's books balance every hour (largest gap ${a.maxAccountingGap.toExponential(2)})`);

  const fall = a.prices[0][c.shock_hour] / a.prices[0][c.shock_hour - 1];
  check(fall < 1 - c.shock_fraction + 0.05 && fall > 1 - c.shock_fraction - 0.05, `the stated shock happens at its hour (price ratio ${fall.toFixed(3)})`);
  check(a.events.some((e) => e.hour === c.shock_hour), `the shock liquidates positions (${a.events.filter((e) => e.hour === c.shock_hour).length} at its hour, ${a.events.length} in the run)`);
  check(a.events.every((e) => e.hfBefore < 1 && e.repaid >= 0 && e.seizedValue >= e.repaid - 1e-6), "every liquidation had a health factor below 1 and paid the bonus or all there was");
  check(a.bobHealth[0] > 1, `Bob starts healthy (${a.bobHealth[0].toFixed(3)})`);
  check(a.bobHealth.some((h) => h < 1), "Bob's position can be liquidated in the default run, so the private check has a positive case");

  for (const multiplier of c.sweep_multipliers) {
    for (let seed = 1; seed <= 60; seed++) {
      const r = m.simulate({ ...c, annual_vol: c.annual_vol.map((v) => v * multiplier), shock_fraction: 0 }, seed);
      if (r.utilisation.some((v) => !(v >= 0 && v <= 1)) || r.maxAccountingGap > 1e-3) { check(false, `a swept path broke the books at ${multiplier}x, seed ${seed}`); break; }
    }
  }
  check(a.checkpoints.length >= 2 && a.checkpoints.length <= c.check_points + 1 && a.checkpoints.every((k) => k.debtValue > 0), `${a.checkpoints.length} checkpoints, all on an open position`);
  const positive = a.checkpoints.filter((k) => k.margin < 0);
  check(positive.length >= 1, "a checkpoint sits on the first hour Bob can be liquidated");
  check(a.checkpoints.every((k) => near(k.margin, m.margin(k.amounts, k.prices, k.thresholds, k.debtValue)) && near(k.debtValue, k.scaledDebt * k.index, 1e-6)), "each checkpoint's margin is its own inputs'");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`${failed} check(s) failed`); process.exit(1); }
console.log("lending model: all checks passed");
