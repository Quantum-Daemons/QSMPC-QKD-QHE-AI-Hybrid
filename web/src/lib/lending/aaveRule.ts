/**
 * Aave V3's liquidation bonus and protocol fee, as `LiquidationLogic.sol` at aave-v3-origin v3.7.0 applies them (read
 * 2026-11-28; re-derived here, not copied: the contracts are under the Business Source License). Round 86, batch 13, the
 * owner's choice of 2026-11-28 to put the fee and e-mode in the lending page's rule.
 *
 *   - The bonus is the borrower's e-mode category's when the borrower is in a category that holds the collateral as
 *     collateral, and otherwise the collateral reserve's (LiquidationLogic.sol, lines 226-238).
 *   - The collateral taken from the borrower is the debt repaid in collateral times the bonus, C. Of it, the protocol's
 *     treasury takes the fee's share of the bonus part, fee x (C - C / LB), and the liquidator the rest (lines 590-624); the
 *     `LiquidationCall` event reports the liquidator's part. The fee is the collateral reserve's: e-mode has none.
 *
 * Aave V3 Ethereum Core raised the fee of WETH, WBTC and wstETH from 10% to 20% on 2026-10-24 (AIP 534); USDC's was 20%
 * already. Its only e-mode category holding WETH as collateral lends only WETH (bonus 1%), so no borrower of USDC there is
 * in an e-mode, and the page's simulated borrowers, who all borrow the stable asset, never are.
 */

/** An e-mode category as the rule needs it: its bonus (as a fraction above 1) and the assets it holds as collateral. */
export interface EModeCategory { bonus: number; collateral: string[] }

/** The bonus a liquidation of `collateral` gets: the borrower's category's when it holds the collateral, else the reserve's. */
export function liquidationBonusFor(reserveBonus: number, collateral: string, eMode?: EModeCategory | null): number {
  return eMode && eMode.collateral.includes(collateral) ? eMode.bonus : reserveBonus;
}

/** How collateral worth `taken`, seized at `bonus` under the reserve's protocol fee `fee`, splits between the liquidator
 *  and the protocol's treasury. The two parts add up to what the borrower loses. */
export function liquidationSplit(taken: number, bonus: number, fee: number): { liquidator: number; protocolFee: number } {
  if (!(fee >= 0 && fee <= 1)) throw new Error(`a protocol fee of ${fee} is outside [0, 1]`);
  if (!(bonus >= 0)) throw new Error(`a bonus of ${bonus} is negative`);
  const protocolFee = fee * (taken - taken / (1 + bonus));
  return { liquidator: taken - protocolFee, protocolFee };
}
