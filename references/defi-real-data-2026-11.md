# DeFi lending and the FHEVM view: real Aave V3 parameters, real price data, measured FHEVM costs (read 2026-11-25)

On the owner's choice of 2026-11-25 (prefer practical large-scale REAL data to synthetic, include calibration and
preprocessing in the verification, and assess anything worth bringing to the encrypted paths), a survey read the
primary sources for the lending page and its modelled FHEVM view, and the agent then re-read the facts relied on
below at their sources on 2026-11-25: Aave's `aave-v3-origin` source at v3.7.0 and Zama's releases through `gh api`,
two read-only `eth_call`s per contract through a public Ethereum node at block 26,492,404, and the IACR ePrint pages.
Nothing was downloaded, no account was used and nothing was signed. Where a figure was read by the survey and not
re-read, it says so. None of this is investment advice; it is data provenance and mechanics.

## What the page uses today

- A simulated month with generated prices and **illustrative** parameters, which `LendingConfig` and
  `web/src/lib/lending/model.ts` say are "not any live market's settings": liquidation thresholds 0.83 and 0.78, a
  5 % bonus, a two-slope rate (80 % optimal, slopes 4 % and 60 %).
- The private check: the borrower's balances under CKKS, the margin
  $`M = \sum_i c_i p_i \mathrm{LT}_i / 1000 - (d / 1000) \, I`$ at public prices and thresholds, its sign in a garbled circuit
  (or a 2-of-3 threshold-BGV committee). No chain is involved.
- The FHEVM view: modelled from Zama's documentation (read 2026-11-21); nothing of Zama's runs.

## Real Aave V3 facts, re-read

| Fact | Where it was read | Value |
|---|---|---|
| Liquidation threshold, WETH, Ethereum Core | `getReserveConfigurationData` on the AaveProtocolDataProvider, block 26,492,404 | 8400 bps (84 %); LTV 8100 |
| Liquidation threshold, WBTC | same | 8500 bps (85 %); LTV 8100 |
| Liquidation bonus, both | same | 10500 (5 %) |
| Close factor | `LiquidationLogic.sol`, aave-v3-origin v3.7.0 | by default the whole reserve debt may be liquidated; the 50 % cap applies only when the reserve's collateral and debt are each at least USD 2,000 (`MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD`) and the health factor is above 0.95, and it is 50 % of the borrower's **total** debt; a liquidation must not leave less than USD 1,000 (`MIN_LEFTOVER_BASE`) |

The survey traced the change from the page's 0.83 and 0.78 to today's values to AIP payload 476 (2026-10-16), so the
page's illustrative thresholds are Aave's values from before that change (survey; the page claims no live market).
Aave V4 is live beside V3 (survey), so the page keeps saying V3. aave-v3-origin is under the Business Source License;
the page re-derives the mechanics and copies no code.

**Found while reading:** `model.ts` and `LendingConfig` say the mechanics *"follow what Aave V3's public contracts
implement"*, and state the close factor as 50 %, or 100 % at or below 0.95. That is V3's rule only for positions of at
least USD 2,000 on both sides; the USD 2,000 and USD 1,000 rules are not modelled. The simulated positions are large enough
for the difference not to arise, but the sentence overstates, and batch 6 corrects it.

**A bound the page did not state:** the garbled circuit compares additive shares in a 32-bit ring with 16 fractional
bits in thousands of stable units (`signOfMargin` in `privateCheck.ts`), so its sign is exact only while
$`|M| < 2^{15}`$ thousand units, about USD 32.8 million. The simulated pool stays far inside it; a check on real large
positions would not. Batch 6 states the bound and adds a check that every simulated margin stays inside it.

## Real price data a public repository could ship

| Source | Terms (survey) | A few thousand points in this repository? |
|---|---|---|
| Chainlink Data Feeds, on-chain rounds | readable without an account; Chainlink's terms grant no redistribution right | statistics only |
| *Stablecoin DeFi Hourly Dataset* (Zenodo, CC BY 4.0, hourly Chainlink ETH and BTC since 2022) | explicit CC BY 4.0 | yes, with credit; it is a third party's copy of Chainlink-sourced prices, which is the owner's call |
| Binance Vision | CC BY-NC-SA 4.0 | no |
| CoinGecko, CoinMarketCap, FRED (Coinbase series) | no redistribution | no |

The lowest-risk route is to ship no series, only statistics fitted to one (volatility, correlation, jumps), with the
window and source stated.

## The FHEVM view, re-read

| Fact | Where it was read | Value |
|---|---|---|
| HCU limit per transaction, Ethereum mainnet | `getMaxHCUPerTx()` on HCULimit, block 26,492,404 | 20,000,000 |
| HCU limit in sequence, Ethereum mainnet | `getMaxHCUDepthPerTx()` | 5,000,000 |
| Releases | `gh api repos/zama-ai/fhevm/releases` | v0.13.9 (2026-11-20) carries GitHub's "latest" marker; v0.14.7 (2026-11-16) is a stable release of a higher line |

The page takes these limits from Zama's documentation, which calls them the devnet's; the mainnet contract returns the
same values. The survey also found Zama's published per-operation timings, which, summed (not measured), put the
page's check near 270 ms on a 192-core CPU and 41 ms on eight H100s at 64 bits, and real mainnet transactions using the
same widen-multiply-divide pattern on 128-bit integers (survey). The docs pages end with a block of instructions
addressed to AI agents; the survey did not follow it.

## Research the page should know (ePrint pages re-read)

- *New IND-CPA-D Attacks on BFV and BGV* (ePrint 2026/2546, ACM CCS 2026): attacks when decryptions are revealed and
  parameters rest on heuristic noise bounds, which bears on the committee's threshold BGV.
- *Tulva: Practical Asynchronous Threshold FHE* (ePrint 2026/2485, IEEE S&P 2027): decryption shares from any $`t`$
  parties, who need not be online together.
- *Jambon* (ePrint 2026/2290) was withdrawn; it is not cited.
- No confidential liquidation engine was found running in production; Fhenix CoFHE went to an allowlisted mainnet on
  2026-10-21 with its key in trusted hardware (survey).

## Options for the owner

1. **A real-parameter preset pinned to a block** (about 2 KB of chain facts with their addresses and block), with the
   full close-factor and leftover rules and the protocol fee, and a test that the page's two-slope formula reproduces
   Aave's stored borrow and supply rates (the survey found agreement near $`10^{-8}`$).
2. **The price generator fitted to real hourly returns**, shipping only the fitted statistics and stress presets from
   primary sources. Needs either public-node reads or the CC BY snapshot (25.81 MB), each on the owner's consent.
3. **An offline check on 200 to 500 real positions**, with Aave's own health factor as the reference, publishing only
   counts; it exercises the ring bound above.
4. **The FHEVM view:** the mainnet limits and version pinned to a block; Zama's measured timings and real transaction
   mixes beside the modelled costs, each labelled; an overflow table at real magnitudes for 64 and 128 bits.

Not recommended: shipping CoinGecko, CoinMarketCap, FRED or Binance data; building on services that need an account
(Dune, The Graph, BigQuery); naming third parties' transaction hashes or addresses on the page without the owner's
decision; calling the page "calibrated" when only its prices would be real.
