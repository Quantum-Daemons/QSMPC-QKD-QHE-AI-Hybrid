# Pre-registration: the lending page's liquidation arithmetic against Aave V3 liquidations that need no price

Written and committed on 2026-11-28, on the owner's choices of that day (download RED-LIQ-2026; pre-register its
check; freeze it with the protocol fee read on chain), and frozen at its SHA-256 in the script before any figure
below is computed. The EDA before it read the fields, the event counts by year and by asset pair, and nothing of the
amounts' ratios.

## What is known, said first

The DeFi lending page simulates Aave V3 liquidations with the bonus rule seized = repaid x (1 + bonus), under the
parameters of its `aave_v3_core` preset, read on chain at block 26,492,593 (`data/defi/aave_v3_core_reads.json`:
liquidation bonuses WETH 10,500, WBTC 10,500 and USDC 10,450 basis points; no protocol fee). Aave V3 gives part of the
bonus to its treasury (the liquidation protocol fee), so the collateral a `LiquidationCall` event reports is less than
repaid x bonus by that share: in `LiquidationLogic.sol` of aave-v3-origin v3.7.0 (read 2026-11-28), the collateral
taken is the base times the bonus LB, the fee is its share of the bonus part, C - C / LB, and the event reports the
collateral less the fee, so for one asset as debt and collateral r = LB - fee x (LB - 1) to integer rounding. The same
code takes LB from the borrower's e-mode category, not the collateral's reserve, when the borrower is in an e-mode
category in which the collateral is enabled; the deposit records no e-mode, so an event at an e-mode bonus has an r of
its own. RED-LIQ-2026 (Zenodo 23020998, CC BY 4.0; not peer-reviewed) holds 2,547 distinct 2026
`LiquidationCall` events of the Aave V3 Ethereum pool with raw debt and collateral amounts and no prices; 45 have the
same asset as debt and collateral, and 9 are between two dollar stablecoins.

## The data

The de-duplicated file, 2026 events only (the 1,000 of 2023 and 2024 are a collector error, as its README says), raw
integer amounts only (its scaled amounts are wrong for some tokens). The liquidation protocol fee of each preset asset
(WETH, WBTC, USDC) read on chain at the preset's block with one read-only call per asset to the AaveProtocolDataProvider
the reads name (`getLiquidationProtocolFee`), recorded with the endpoint's name and the block; no transaction is sent.

## What is measured

- For each same-asset event, r = collateral seized / debt repaid from the raw integers, and its asset.
- Against (a) the page's rule as the page computes it, 1 + its preset's one bonus (0.05, whatever the asset), and (b)
  Aave's rule with the fee, LB - fee x (LB - 1), at the collateral asset's own bonus LB as read and the fee read, for
  WETH, WBTC and USDC; per event the difference from each, and per asset their spread.
- The fee each event implies, (LB - r) / (LB - 1), beside the fee read; and the bonus each event implies under the
  fee read, 1 + (r - 1) / (1 - fee), so an event at another bonus (an e-mode category's) shows as its own value. The
  e-mode categories' parameters are not read: one fee read per asset is the only call made.
- An event is at (b) to integer rounding when |r - (b)| is at most 3 / D, D its raw debt repaid (the floors and
  ceilings of the code's percentage arithmetic move the collateral by at most a few base units); per asset, the
  events at (b) are counted.
- For the stablecoin pairs: r against (b) at par, with the difference reported, since the oracle prices of two
  stablecoins differ by their depegs.
- Assets the preset does not read are reported with their r only.

## Stated before any figure

- Expected: for same-asset WETH, WBTC and USDC events at the reserve's bonus, (b) matches r to integer rounding
  unless the parameters changed between the events (May to July 2026) and the read block (November 2026), which would
  show as a constant offset within an asset and is then reported as such; events of borrowers in an e-mode category
  show an r of their own and an implied bonus equal to that category's; (a) overstates r by the fee's share of the
  bonus. Stablecoin pairs within 0.5 percent of (b).
- Nothing on the page changes because of a result; whether the page adds the fee is the owner's choice after it.

## Published

Counts, ratios and differences only: no address, no transaction hash, no per-event row.

## Compute

CPU, seconds. Script `scripts/check_red_liq_arithmetic.py`; record `docs/measurements/defi-red-liq-arithmetic-<the
run's date>.json`, with the deposit's DOI and its files' SHA-256.
