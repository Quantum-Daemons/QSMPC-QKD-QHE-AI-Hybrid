"""The lending page's Aave V3 preset is derived from the raw chain reads it cites (Round 86, batch 6).

The owner chose on 2026-11-25 to add a real market's parameters beside the illustrative ones. They were read with
read-only eth_call at one pinned block (data/defi/aave_v3_core_reads.json, raw values only), and the preset in
`LendingConfig.market_presets` is derived from them by hand. This holds the two together, so a hand edit to either
fails here, and it pins the contracts to the addresses Aave's address book gives (bgd-labs/aave-address-book, MIT,
src/AaveV3Ethereum.sol at dae05ac). web/scripts/check_lending.mjs holds the page's rate formula to the stored rates.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
READS = json.loads((REPO / "data" / "defi" / "aave_v3_core_reads.json").read_text())

#: As the address book lists them (POOL, AAVE_PROTOCOL_DATA_PROVIDER, WETH_ and USDC_INTEREST_RATE_STRATEGY, ORACLE).
ADDRESS_BOOK = {
    "Pool": "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
    "AaveProtocolDataProvider": "0x0a16f2FCC0D44FaE41cc54e079281D84A363bECD",
    "DefaultReserveInterestRateStrategyV2": "0x9ec6F08190DeA04A54f8Afc53Db96134e5E3FdFB",
    "AaveOracle": "0x54586bE62E3c3580375aE3723C145253060Ca0C2",
}


def _preset() -> dict:
    from qsh.config.defaults import LendingConfig

    return LendingConfig().market_presets["aave_v3_core"]


def test_the_reads_name_their_block_and_the_address_books_contracts() -> None:
    src = READS["source"]
    assert src["chain"] == "Ethereum mainnet" and src["market"] == "Aave V3 Ethereum Core"
    assert src["contracts"] == ADDRESS_BOOK
    assert "dae05ac" in src["addresses_from"] and "MIT" in src["addresses_from"]
    assert set(READS["reserves"]) == {"WETH", "WBTC", "USDC"}


def test_the_preset_is_the_reads_derived() -> None:
    p, r = _preset(), READS["reserves"]
    assert p["block"] == READS["source"]["block"] and f"{p['block']:,}" in p["label"]
    assert tuple(p["liquidation_thresholds"]) == tuple(r[a]["liquidation_threshold_bps"] / 1e4 for a in ("WETH", "WBTC"))
    bonus = {(r[a]["liquidation_bonus_bps"] - 10_000) / 1e4 for a in ("WETH", "WBTC")}
    assert bonus == {p["liquidation_bonus"]}, bonus
    # The fee (Round 86, batch 13): read at the same block, the collateral reserves' own, one value for both.
    fee = {r[a]["liquidation_protocol_fee_bps"] / 1e4 for a in ("WETH", "WBTC")}
    assert fee == {p["liquidation_protocol_fee"]}, fee
    usdc = r["USDC"]["rate_bps"]
    assert (p["base_rate"], p["optimal_utilisation"], p["slope1"], p["slope2"]) == (
        usdc["base_variable_borrow_rate"] / 1e4, usdc["optimal_usage_ratio"] / 1e4,
        usdc["variable_rate_slope1"] / 1e4, usdc["variable_rate_slope2"] / 1e4)
    assert p["reserve_factor"] == r["USDC"]["reserve_factor_bps"] / 1e4
    assert tuple(p["initial_prices"]) == tuple(r[a]["oracle_price_usd_8dp"] / 1e8 for a in ("WETH", "WBTC"))


def test_the_page_offers_the_preset_and_says_the_market_is_still_simulated() -> None:
    panel = (REPO / "web" / "src" / "components" / "ParameterPanel.tsx").read_text()
    assert 'name="lending_market"' in panel and 'value="aave_v3_core"' in panel and 'value="illustrative"' in panel
    cc = (REPO / "web" / "src" / "lib" / "clientCompute.ts").read_text()
    assert "Aave's parameters on a simulated market, not Aave's market" in cc
    cfg = json.loads((REPO / "web" / "src" / "lib" / "config.defaults.json").read_text())
    assert cfg["lending"]["market_presets"]["aave_v3_core"] == json.loads(json.dumps(_preset()))
