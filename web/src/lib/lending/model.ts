/**
 * A dynamic lending market, simulated hour by hour (Round 86). No chain, no real prices: a pool, borrowers and a synthetic price path.
 *
 * The mechanics follow what Aave V3's public contracts implement, re-derived rather than copied (the current ones, aave-v3-origin, are under the Business Source License until 2027-03-06 at the latest):
 *   - a position's health factor is its collateral value, each asset weighted by its liquidation threshold, over its debt value;
 *   - below 1 it can be liquidated, up to 50% of the debt in one step, or 100% at or below 0.95;
 *   - the liquidator takes collateral worth the repaid debt plus a bonus;
 *   - the borrow rate is a two-slope function of utilisation, and debt grows through a public index, so a position is stored as a scaled balance.
 *
 * Every parameter here is this project's illustrative choice, read from config, and none is a claim about any live market.
 */

import type { LendingCfg } from "../config";

/** The borrow rate at utilisation u, per year. */
export function twoSlopeRate(u: number, r: Pick<LendingCfg, "base_rate" | "optimal_utilisation" | "slope1" | "slope2">): number {
  if (!(u >= 0 && u <= 1)) throw new Error(`utilisation ${u} is outside [0, 1]`);
  return u <= r.optimal_utilisation
    ? r.base_rate + (u / r.optimal_utilisation) * r.slope1
    : r.base_rate + r.slope1 + ((u - r.optimal_utilisation) / (1 - r.optimal_utilisation)) * r.slope2;
}

/** Collateral value weighted by liquidation threshold, in the same units as the debt value. */
export function weightedCollateral(amounts: readonly number[], prices: readonly number[], thresholds: readonly number[]): number {
  return amounts.reduce((sum, a, i) => sum + a * prices[i] * thresholds[i], 0);
}

/** Health factor. Infinite when there is no debt. */
export function healthFactor(amounts: readonly number[], prices: readonly number[], thresholds: readonly number[], debtValue: number): number {
  return debtValue <= 0 ? Infinity : weightedCollateral(amounts, prices, thresholds) / debtValue;
}

/** The margin M = weighted collateral - debt. HF < 1 exactly when M < 0 and debt > 0, and M is linear in the amounts, which a ratio is not. */
export function margin(amounts: readonly number[], prices: readonly number[], thresholds: readonly number[], debtValue: number): number {
  return weightedCollateral(amounts, prices, thresholds) - debtValue;
}

/** The share of the debt that may be repaid in one liquidation. */
export function closeFactor(hf: number, c: Pick<LendingCfg, "close_factor" | "full_close_at_or_below_hf">): number {
  return hf <= c.full_close_at_or_below_hf ? 1 : c.close_factor;
}

/** The page's Parameters panel over the market. Each key reaches `simulate`, and one that is absent keeps the configured value. */
export function withRunParams(c: LendingCfg, p: Record<string, unknown>): LendingCfg {
  const num = (v: unknown, fallback: number) => (v != null && Number.isFinite(Number(v)) ? Number(v) : fallback);
  const borrowers = num(p.lending_borrowers, c.borrowers);
  const volatility = num(p.lending_vol_mult, 1);
  return {
    ...c,
    borrowers,
    supply: c.supply * borrowers / c.borrowers,       // the pool grows with its borrowers, so the utilisation a run starts at does not depend on how many there are
    bob_initial_ltv: num(p.lending_bob_ltv_pct, c.bob_initial_ltv * 100) / 100,
    shock_hour: num(p.lending_shock_hour, c.shock_hour),
    shock_fraction: num(p.lending_shock_pct, c.shock_fraction * 100) / 100,
    annual_vol: c.annual_vol.map((v) => v * volatility),
    check_points: num(p.lending_checks, c.check_points),
    custody: p.lending_custody === "borrower" || p.lending_custody === "keyholder" ? p.lending_custody : c.custody,
  };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(u: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
}

export interface LiquidationEvent {
  hour: number;
  borrower: number;
  hfBefore: number;
  repaid: number;
  seizedValue: number;
  badDebt: number;
}

/** What the private check needs about one position at one hour, and the answer in the clear. */
export interface LendingCheckpoint {
  hour: number;
  amounts: number[];
  scaledDebt: number;
  /** What Bob's liquidations have settled so far, in the open: the collateral seized per asset and the scaled debt repaid. The pool applies it to his ciphertext. */
  settled: { seized: number[]; scaledRepaid: number };
  prices: number[];
  thresholds: number[];
  index: number;
  debtValue: number;
  margin: number;
  healthFactor: number;
}

export interface LendingRun {
  hours: number;
  prices: number[][];                 // [asset][hour]
  utilisation: number[];
  borrowRate: number[];
  supplyRate: number[];
  totalDebt: number[];
  liquidatedPerHour: number[];
  bobHealth: number[];
  bobDebt: number[];
  events: LiquidationEvent[];
  badDebt: number;
  checkpoints: LendingCheckpoint[];
  /** Largest absolute gap between the pool's books and the value they should hold, over the run. Accounting, so it should be rounding only. */
  maxAccountingGap: number;
  borrowers: number;
}

interface Position { amounts: number[]; scaledDebt: number; targetLtv: number }

export function simulate(c: LendingCfg, seed: number): LendingRun {
  const u = mulberry32(seed);
  const nAssets = c.initial_prices.length;
  const hours = c.hours, dt = 1 / (365 * 24);
  const prices: number[][] = c.initial_prices.map((p) => [p]);
  let index = 1;
  const bobSettled = { seized: c.initial_prices.map(() => 0), scaledRepaid: 0 };
  let accruedToSuppliers = 0, reserves = 0, badDebt = 0;

  const positions: Position[] = [];
  let openingDebt = 0;
  for (let b = 0; b < c.borrowers; b++) {
    const collateralUsd = Math.exp(c.collateral_log_mean + c.collateral_log_sd * normal(u));
    const split = b === 0 ? 0.5 : u();
    const amounts = [split * collateralUsd / c.initial_prices[0], (1 - split) * collateralUsd / c.initial_prices[1]];
    const ltv = b === 0 ? c.bob_initial_ltv : c.ltv_min + u() * (c.ltv_max - c.ltv_min);
    const debt = ltv * collateralUsd;
    positions.push({ amounts, scaledDebt: debt, targetLtv: ltv });
    openingDebt += debt;
  }
  if (openingDebt >= c.supply) throw new Error(`the opening debt ${openingDebt.toFixed(0)} exceeds the supply ${c.supply}`);
  let liquidity = c.supply - openingDebt;

  const utilisation: number[] = [], borrowRate: number[] = [], supplyRate: number[] = [], totalDebt: number[] = [], liquidatedPerHour: number[] = [];
  const bobHealth: number[] = [], bobDebt: number[] = [];
  const events: LiquidationEvent[] = [];
  let maxAccountingGap = 0;
  const debtOf = (p: Position) => p.scaledDebt * index;

  const checkpointHours = new Set<number>();
  for (let k = 0; k < c.check_points; k++) checkpointHours.add(Math.round((k * hours) / (c.check_points - 1)));
  const checkpoints: LendingCheckpoint[] = [];
  let bobLiquidatedFlagged = false;

  for (let hour = 0; hour <= hours; hour++) {
    const price = prices.map((series) => series[hour]);
    // Before any liquidation, so that the series shows the fall below 1 that triggers one.
    bobHealth.push(healthFactor(positions[0].amounts, price, c.liquidation_thresholds, debtOf(positions[0])));
    // Liquidations first, at this hour's prices.
    let liquidatedNow = 0;
    positions.forEach((p, b) => {
      const debt = debtOf(p);
      if (debt <= 0) return;
      const hf = healthFactor(p.amounts, price, c.liquidation_thresholds, debt);
      if (hf >= 1) return;
      if (b === 0 && !bobLiquidatedFlagged) { checkpointHours.add(hour); bobLiquidatedFlagged = true; }
      if (checkpointHours.has(hour) && b === 0) pushCheckpoint();
      const repay = closeFactor(hf, c) * debt;
      const want = repay * (1 + c.liquidation_bonus);
      const have = p.amounts.reduce((s, a, i) => s + a * price[i], 0);
      const seized = Math.min(want, have);
      const repaidCovered = seized / (1 + c.liquidation_bonus);
      const share = have > 0 ? seized / have : 0;
      if (b === 0) p.amounts.forEach((a, i) => { bobSettled.seized[i] += a * share; });
      p.amounts = p.amounts.map((a) => a * (1 - share));
      // Repaying all of it, `debt * (1 + bonus) / (1 + bonus)` can round above `debt`, and a debt of -1e-18 made the utilisation negative.
      let remaining = Math.max(0, debt - repaidCovered);
      let lost = 0;
      if (p.amounts.every((a) => a * price[0] < 1e-9) && remaining > 0) { lost = remaining; remaining = 0; }
      if (b === 0) bobSettled.scaledRepaid += p.scaledDebt - remaining / index;
      p.scaledDebt = remaining / index;
      liquidity += repaidCovered;
      badDebt += lost;
      liquidatedNow += repaidCovered;
      events.push({ hour, borrower: b, hfBefore: hf, repaid: repaidCovered, seizedValue: seized, badDebt: lost });
    });

    function pushCheckpoint() {
      const p = positions[0];
      if (checkpoints.some((x) => x.hour === hour)) return;
      const debtValue = debtOf(p);
      // A closed position has nothing to check. Its margin is exactly zero or rests on leftover collateral alone, and the closing liquidation was settled in the open.
      if (debtValue <= 0) return;
      checkpoints.push({
        hour, amounts: [...p.amounts], scaledDebt: p.scaledDebt, settled: { seized: [...bobSettled.seized], scaledRepaid: bobSettled.scaledRepaid },
        prices: price, thresholds: [...c.liquidation_thresholds], index, debtValue,
        margin: margin(p.amounts, price, c.liquidation_thresholds, debtValue),
        healthFactor: healthFactor(p.amounts, price, c.liquidation_thresholds, debtValue),
      });
    }
    if (checkpointHours.has(hour)) pushCheckpoint();

    const debtTotal = positions.reduce((s, p) => s + debtOf(p), 0);
    const util = debtTotal + liquidity > 0 ? Math.min(1, debtTotal / (debtTotal + liquidity)) : 0;
    const rate = twoSlopeRate(util, c);
    utilisation.push(util);
    borrowRate.push(rate);
    supplyRate.push(rate * util * (1 - c.reserve_factor));
    totalDebt.push(debtTotal);
    liquidatedPerHour.push(liquidatedNow);
    bobDebt.push(debtOf(positions[0]));

    // The books: what the pool is owed plus what it holds equals what it took in, plus interest, less bad debt.
    const accrued = c.supply + accruedToSuppliers + reserves - badDebt;
    maxAccountingGap = Math.max(maxAccountingGap, Math.abs(debtTotal + liquidity - accrued));
    if (hour === hours) break;

    // Borrowers react to the rate: above the optimum they repay, below it they draw, with some probability each hour.
    positions.forEach((p, b) => {
      if (b === 0 || u() >= c.adjust_probability) return;
      const debt = debtOf(p);
      const value = p.amounts.reduce((s, a, i) => s + a * price[i], 0);
      if (util > c.optimal_utilisation) {
        const pay = Math.min(debt, c.adjust_fraction * debt, liquidity + debt);
        p.scaledDebt -= pay / index;
        liquidity += pay;
      } else if (value > 0 && debt / value < p.targetLtv) {
        const draw = Math.min(c.adjust_fraction * debt, liquidity, p.targetLtv * value - debt);
        if (draw > 0) { p.scaledDebt += draw / index; liquidity -= draw; }
      }
    });

    // Interest accrues on the debt through the public index; the pool's depositors and the reserve receive it.
    const interest = positions.reduce((s, p) => s + p.scaledDebt, 0) * index * rate * dt;
    index *= 1 + rate * dt;
    reserves += interest * c.reserve_factor;
    accruedToSuppliers += interest * (1 - c.reserve_factor);

    // Prices: correlated diffusion, a common jump, and one stated shock.
    const common = normal(u);
    const jump = u() < c.jumps_per_year * dt ? c.jump_mean + c.jump_sd * normal(u) : 0;
    const shock = hour + 1 === c.shock_hour ? Math.log(1 - c.shock_fraction) : 0;
    for (let a = 0; a < nAssets; a++) {
      const z = Math.sqrt(c.price_correlation) * common + Math.sqrt(1 - c.price_correlation) * normal(u);
      const sigma = c.annual_vol[a];
      prices[a].push(prices[a][hour] * Math.exp(-0.5 * sigma * sigma * dt + sigma * Math.sqrt(dt) * z + jump + shock));
    }
  }
  return { hours, prices, utilisation, borrowRate, supplyRate, totalDebt, liquidatedPerHour, bobHealth, bobDebt, events, badDebt, checkpoints, maxAccountingGap, borrowers: c.borrowers };
}
