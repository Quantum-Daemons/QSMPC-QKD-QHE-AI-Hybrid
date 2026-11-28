# Addendum to the RED-LIQ pre-registration: the fee at each event's time, after the result

Written and committed on 2026-11-28, after the pre-registered run, whose record is
`docs/measurements/defi-red-liq-arithmetic-2026-11-28.json`. The pre-registration
(`references/defi-red-liq-liquidation-arithmetic-preregistration.md`) is unchanged, and so is its record.

## What the run found, and what was read after it

The run read the liquidation protocol fee at the preset's block (2026-11-25): 20 percent for WETH, WBTC and USDC. USDC's
16 events with one asset as debt and collateral sat on Aave's rule with it; WETH's and WBTC's sat as a 10 percent fee
would put them, the constant offset the pre-registration named as a change of the parameters between the events (May to
June 2026) and the read. Read after that result, at the source (the owner's request of 2026-11-28 to fact-check the
checks against primary sources):

- LlamaRisk's ARFC of 2026-08-12 raised the fee of WETH, WBTC and wstETH on Aave V3 Ethereum Core from 10 to 20 percent;
  its payload (aave-dao/aave-proposals-v3, pull request 1222, merged 2026-10-19) changes those three from 1,000 to 2,000
  basis points, and AIP 534 executed it on 2026-10-24 at 14:16:59 UTC. The proposal states that USDC was at 20 percent
  already, as Aave's 2023 activation report lists it.
- In the storage the payload read before the change (a fork of 2026-10-16), WETH's and WBTC's bonuses were 10,500 basis
  points and wstETH's 10,600, with the fee at 1,000.
- Aave V3 Ethereum Core's e-mode category 1 holds WETH and wstETH (among others) as collateral, lends only WETH, and
  has a bonus of 10,100 basis points; WBTC and USDC are collateral in no e-mode category.

## The check, chosen after the result and labelled so

Every 2026 event of the de-duplicated file with one asset as debt and collateral, for the four assets whose fee and
bonus the sources above give (WETH, WBTC, wstETH, USDC). For each event, r is the collateral seized over the debt
repaid, from the raw integers, as the run computed it. The fee is the one in force at the event's block time: 10 percent
for WETH, WBTC and wstETH before 2026-10-24 14:16:59 UTC and 20 percent after, 20 percent for USDC throughout. The rule's
values are LB - fee x (LB - 1) at the reserve's bonus (WETH and WBTC 1.05, wstETH 1.06, USDC 1.045) and, for WETH and
wstETH, at e-mode category 1's bonus (1.01). An event is on the rule when |r - value| is at most 3 / D for one of its
asset's values, D its raw debt repaid, as the run's rounding test. Reported: per asset, the events on each value and the
events on none, the latter by their ratio only. No address and no transaction hash.

## Not done here

No call to any chain: the fee history and the bonuses are the cited sources'. A bonus change by Aave's risk stewards in
2026 was not searched for beyond them, and the record says so. Nothing on the page changes because of the result; the
page's rule has taken the fee since Round 86's batch 13.

Script `scripts/check_red_liq_fee_history.py`, frozen at this addendum's SHA-256; record
`docs/measurements/defi-red-liq-fee-history-2026-11-28.json`.
