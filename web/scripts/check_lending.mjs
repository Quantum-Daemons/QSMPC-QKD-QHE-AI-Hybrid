#!/usr/bin/env node
/**
 * The lending simulator against closed forms and its own books (Round 86).
 *
 * The page draws what `web/src/lib/lending/model.ts` returns, so the model is checked here: the health factor on Aave's own worked figures, the two-slope rate at its
 * knots, the close factor at its boundary, a run's accounting, that a run is what its seed says, and (Round 86, batch 13) Aave's split of what a liquidation
 * takes between the liquidator and the protocol's fee, and the e-mode bonus rule.
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
    stdin: { contents: `export * from ${JSON.stringify(join(WEB, "src/lib/lending/model"))};
export { liquidationBonusFor, liquidationSplit } from ${JSON.stringify(join(WEB, "src/lib/lending/aaveRule"))};`, resolveDir: WEB, loader: "ts" },
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

  check(m.closeFactor(0.97, c) === c.close_factor && m.closeFactor(0.95, c) === 1 && m.closeFactor(0.94, c) === 1, "the close factor is 50% above 0.95 and 100% at or below it, as LiquidationLogic.sol sets it for positions of at least USD 2,000 on both sides (its rules for smaller positions are not modelled)");
  const params = m.withRunParams(c, { lending_borrowers: 12, lending_bob_ltv_pct: 50, lending_shock_hour: 100, lending_shock_pct: 20, lending_vol_mult: 2, lending_checks: 4 });
  check(params.borrowers === 12 && params.supply === c.supply / 2 && params.bob_initial_ltv === 0.5 && params.shock_hour === 100 && params.shock_fraction === 0.2
    && params.annual_vol.every((v, i) => v === c.annual_vol[i] * 2) && params.check_points === 4, "every control on the page reaches the simulation");
  check(JSON.stringify(m.withRunParams(c, {})) === JSON.stringify(c), "with no controls set the simulation runs on the configured values");
  check(c.custody === "keyholder" && m.withRunParams(c, { lending_custody: "borrower" }).custody === "borrower" && m.withRunParams(c, { lending_custody: "nonsense" }).custody === c.custody, "the key holder control is read, and an unknown value keeps the configured one");
  for (const seed of [1, 3, 5, 9]) {
    const r = m.simulate(c, seed), b0 = r.checkpoints[0];
    const off = r.checkpoints.flatMap((k) => [Math.abs(k.amounts[0] - (b0.amounts[0] - k.settled.seized[0])), Math.abs(k.amounts[1] - (b0.amounts[1] - k.settled.seized[1])), Math.abs(k.scaledDebt - (b0.scaledDebt - k.settled.scaledRepaid))]);
    if (Math.max(...off) > 1e-6) check(false, `seed ${seed}: Bob's position is not his opening one less what was settled (off by ${Math.max(...off)})`);
  }
  check(true, "Bob's position at every checkpoint is his opening one less what was settled in the open, so the pool can follow it");

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
  // Aave's split (LiquidationLogic.sol, v3.7.0): the borrower loses C, the treasury takes fee x (C - C / LB), the liquidator the rest.
  check(a.events.length > 0 && a.events.every((e) => near(e.liquidatorValue + e.protocolFee, e.seizedValue, 1e-9 * Math.max(1, e.seizedValue))
    && near(e.protocolFee, c.liquidation_protocol_fee * (e.seizedValue - e.seizedValue / (1 + c.liquidation_bonus)), 1e-9 * Math.max(1, e.seizedValue))),
        `every liquidation's two parts add up to what the borrower lost, the treasury's the fee's share of the bonus part (fee ${c.liquidation_protocol_fee})`);
  // A same-asset liquidation's ratio, liquidator's collateral over debt repaid, is LB - fee x (LB - 1): RED-LIQ-2026's USDC events sit at 1.036 (LB 1.045, fee 20%).
  const usdc = m.liquidationSplit(1.045, 0.045, 0.2);
  check(near(usdc.liquidator, 1.036, 1e-12) && near(usdc.protocolFee, 0.009, 1e-12), "a same-asset ratio is LB - fee x (LB - 1): 1.036 at LB 1.045 and a 20% fee");
  check(near(m.liquidationSplit(1.05, 0.05, 0.1).liquidator, 1.045, 1e-12) && near(m.liquidationSplit(1.01, 0.01, 0.1).liquidator, 1.009, 1e-12),
        "WETH's ratios before 2026-10-24, 1.045 at its reserve's bonus and 1.009 at e-mode's 1%, under the 10% fee of the time");
  check(m.liquidationSplit(100, 0.05, 0).protocolFee === 0, "no fee, no treasury part");
  const eth = { bonus: 0.01, collateral: ["WETH", "wstETH"] };
  check(m.liquidationBonusFor(0.05, "WETH", eth) === 0.01 && m.liquidationBonusFor(0.05, "WBTC", eth) === 0.05 && m.liquidationBonusFor(0.05, "WETH", null) === 0.05,
        "the e-mode bonus applies only when the borrower's category holds the collateral; otherwise the reserve's");
  let refused = false;
  try { m.liquidationSplit(1, 0.05, 1.5); } catch { refused = true; }
  check(refused, "a fee outside [0, 1] is refused");
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
  // The garbled circuit's ring (privateCheck.ts, signOfMargin): its sign is exact only while |M| / 1000 < 2^(31 - margin_scale_bits).
  const ringBound = 2 ** (31 - c.margin_scale_bits) * 1000;
  let widest = 0;
  for (const multiplier of c.sweep_multipliers) {
    for (let s = 1; s <= 24; s++) {
      const r = m.simulate({ ...c, annual_vol: c.annual_vol.map((v) => v * multiplier), shock_fraction: 0 }, s);
      for (const k of r.checkpoints) widest = Math.max(widest, Math.abs(k.margin));
    }
  }
  for (const k of a.checkpoints) widest = Math.max(widest, Math.abs(k.margin));
  check(widest < ringBound, `every checkpoint's margin stays inside the garbled circuit's ring: largest |M| ${Math.round(widest).toLocaleString("en-US")} against ${ringBound.toLocaleString("en-US")} stable units`);

  // A REAL MARKET'S PARAMETERS (batch 6): Aave V3 Ethereum Core, read on chain at one pinned block (data/defi/aave_v3_core_reads.json).
  const reads = JSON.parse(readFileSync(join(WEB, "..", "data", "defi", "aave_v3_core_reads.json"), "utf8"));
  const preset = c.market_presets.aave_v3_core;
  check(preset.block === reads.source.block, `the preset is the reads' block (${preset.block})`);
  // The page's two-slope rate, fed each reserve's own curve and its utilisation from the read totals, reproduces the rate Aave stored.
  for (const [name, r] of Object.entries(reads.reserves)) {
    const debt = r.reserve_data.total_variable_debt + r.reserve_data.total_stable_debt;
    const u = debt / (r.virtual_underlying_balance + debt);
    const q = { base_rate: r.rate_bps.base_variable_borrow_rate / 1e4, optimal_utilisation: r.rate_bps.optimal_usage_ratio / 1e4,
                slope1: r.rate_bps.variable_rate_slope1 / 1e4, slope2: r.rate_bps.variable_rate_slope2 / 1e4 };
    const stored = r.reserve_data.variable_borrow_rate_ray / 1e27, mine = m.twoSlopeRate(u, q);
    check(Math.abs(mine - stored) <= 1e-6 * stored, `${name}: the two-slope rate at utilisation ${u.toFixed(6)} is Aave's stored ${stored.toFixed(8)} (relative ${(Math.abs(mine - stored) / stored).toExponential(1)})`);
  }
  const aave = m.withRunParams(c, { lending_market: "aave_v3_core" });
  check(aave.liquidation_thresholds[0] === 0.84 && aave.liquidation_thresholds[1] === 0.85 && aave.liquidation_bonus === 0.05
    && aave.optimal_utilisation === 0.94 && aave.slope1 === 0.046 && aave.slope2 === 0.10 && aave.initial_prices[0] === preset.initial_prices[0],
    "the Aave preset reaches the simulation: thresholds, bonus, the USDC curve and the opening prices");
  check(JSON.stringify(m.withRunParams(c, { lending_market: "nonsense" })) === JSON.stringify(c), "an unknown market keeps the illustrative parameters");
  const ra = m.simulate(aave, 7);
  check(ra.maxAccountingGap < 1e-6 * aave.supply && ra.utilisation.every((v) => v >= 0 && v <= 1), "under Aave's parameters the pool's books still balance");
  check(ra.bobHealth[0] > 1 && ra.events.some((e) => e.hour === aave.shock_hour), `under Aave's parameters Bob starts healthy (${ra.bobHealth[0].toFixed(3)}) and the fall liquidates positions`);
  check(ra.checkpoints.every((k) => Math.abs(k.margin) < ringBound), "and every checkpoint's margin stays inside the ring");
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
if (failed) { console.error(`${failed} check(s) failed`); process.exit(1); }
console.log("lending model: all checks passed");
