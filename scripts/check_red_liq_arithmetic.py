#!/usr/bin/env python3
"""The lending page's liquidation arithmetic against Aave V3 liquidations that need no price, as pre-registered (Round 86,
batch 12): references/defi-red-liq-liquidation-arithmetic-preregistration.md.

Reads the deposit's de-duplicated liquidation file in place, the preset's on-chain reads, and, once, each preset
asset's liquidation protocol fee by one read-only `eth_call` at the preset's block (`--rpc URL`, the endpoint named in
the record; nothing is sent but the call). Writes docs/measurements/defi-red-liq-arithmetic-<date>.json: counts, ratios
and differences only, no address and no transaction hash.

    QSH_DATA_ROOT=... python scripts/check_red_liq_arithmetic.py --rpc URL
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
import urllib.request
from datetime import date
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets as D  # noqa: E402

PREREG = REPO / "references" / "defi-red-liq-liquidation-arithmetic-preregistration.md"
PREREG_SHA256 = "8fc961e67158e0a991dbd119e0d3607932d80d2c4d5ee334ed6a2ce41bf16b04"
READS = REPO / "data" / "defi" / "aave_v3_core_reads.json"
PAGE_BONUS = Fraction(5, 100)        # the preset's one bonus, as the page computes (src/qsh/config/defaults.py)
STABLES = {"USDC", "USDT", "DAI"}
DECIMALS = {"USDC": 6, "USDT": 6, "DAI": 18}      # the deposit's own table, correct for these three


def selector(signature: str) -> str:
    """The first four bytes of the Keccak-256 of a function signature (0x3cb8a622 for the fee getter)."""
    from Crypto.Hash import keccak

    return "0x" + keccak.new(digest_bits=256, data=signature.encode()).hexdigest()[:8]


SELECTOR = selector("getLiquidationProtocolFee(address)")
assert SELECTOR == "0x3cb8a622", SELECTOR


def fee_bps(rpc: str, provider: str, asset: str, block: int) -> int:
    data = SELECTOR + asset.lower().removeprefix("0x").rjust(64, "0")
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                       "params": [{"to": provider, "data": data}, hex(block)]}).encode()
    req = urllib.request.Request(rpc, data=body, headers={"Content-Type": "application/json",
                                                          "User-Agent": "qsh-check/1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
    if "error" in out:
        raise SystemExit(f"eth_call failed: {out['error']}")
    return int(out["result"], 16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rpc", required=True, help="an Ethereum JSON-RPC endpoint that serves state at the preset's block")
    args = ap.parse_args()
    got = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if got != PREREG_SHA256:
        raise SystemExit(f"{PREREG.name} hashes to {got[:12]}, not the committed {PREREG_SHA256[:12]}: run nothing")
    reads = json.loads(READS.read_text())
    block, provider = reads["source"]["block"], reads["source"]["contracts"]["AaveProtocolDataProvider"]
    preset = {sym: {"address": v["asset"].lower(), "lb": Fraction(v["liquidation_bonus_bps"], 10000)}
              for sym, v in reads["reserves"].items()}
    for sym, v in preset.items():
        v["fee"] = Fraction(fee_bps(args.rpc, provider, v["address"], block), 10000)
    path = Path(D.data_root()) / "raw" / "red-liq-2026" / "red_liq_2026_v1_1_liquidations_unique.jsonl.gz"
    rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    r26 = [r for r in rows if r["iso"].startswith("2026")]
    by_asset: dict[str, list[tuple[Fraction, int]]] = {}
    for r in r26:
        if r["debt_address"] == r["collateral_address"]:
            debt = int(r["debt_covered_raw"])
            by_asset.setdefault(r["debt_asset"], []).append((Fraction(int(r["collateral_seized_raw"]), debt), debt))
    same = {}
    for sym, events in sorted(by_asset.items()):
        ratios = [x for x, _ in events]
        entry = {"events": len(ratios), "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios)),
                 "ratio_median": float(statistics.median(ratios))}
        if sym in preset:
            lb, fee = preset[sym]["lb"], preset[sym]["fee"]
            aave = lb - fee * (lb - 1)
            at_rule = [abs(x - aave) <= Fraction(3, d) for x, d in events]
            implied_bonus = sorted(float(1 + (x - 1) / (1 - fee)) for x in ratios)
            entry.update({
                "preset_bonus": float(lb - 1), "fee_read": float(fee), "aave_rule": float(aave), "page_rule": float(1 + PAGE_BONUS),
                "events_at_aave_rule_to_rounding": sum(at_rule),
                "largest_abs_difference_from_aave_rule": float(max(abs(x - aave) for x in ratios)),
                "largest_abs_difference_from_aave_rule_among_events_at_it": (
                    float(max(abs(x - aave) for (x, _), ok in zip(events, at_rule, strict=True) if ok)) if any(at_rule) else None),
                "largest_abs_difference_from_page_rule": float(max(abs(x - (1 + PAGE_BONUS)) for x in ratios)),
                "implied_fee": {"min": float(min((lb - x) / (lb - 1) for x in ratios)),
                                "max": float(max((lb - x) / (lb - 1) for x in ratios))},
                "implied_bonus_under_fee_read": {"min": implied_bonus[0], "max": implied_bonus[-1],
                                                 "distinct_to_1e-6": sorted({round(v, 6) for v in implied_bonus})}})
        same[sym] = entry
    stable: dict[str, list[Fraction]] = {}
    for r in r26:
        d, c = r["debt_asset"], r["collateral_asset"]
        if d in STABLES and c in STABLES and r["debt_address"] != r["collateral_address"]:
            human = (Fraction(int(r["collateral_seized_raw"]), 10 ** DECIMALS[c])
                     / Fraction(int(r["debt_covered_raw"]), 10 ** DECIMALS[d]))
            stable.setdefault(f"{d} debt, {c} collateral", []).append(human)
    pairs = {}
    for name, ratios in sorted(stable.items()):
        coll = name.split(", ")[1].split()[0]
        entry = {"events": len(ratios), "ratio_min": float(min(ratios)), "ratio_max": float(max(ratios))}
        if coll in preset:
            lb, fee = preset[coll]["lb"], preset[coll]["fee"]
            entry["largest_abs_difference_from_aave_rule_at_par"] = float(max(abs(x - (lb - fee * (lb - 1))) for x in ratios))
        pairs[name] = entry
    record = {
        "preregistration": {"file": str(PREREG.relative_to(REPO)), "sha256": PREREG_SHA256},
        "deposit": {"doi": "10.5281/zenodo.23020998", "licence": "CC-BY-4.0",
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "fee_read": {"block": block, "contract": provider, "call": "getLiquidationProtocolFee(address)",
                     "endpoint": args.rpc.split("//")[-1].split("/")[0],
                     "fees": {s: float(v["fee"]) for s, v in preset.items()}},
        "events_2026": len(r26), "same_asset": same,
        "stable_pairs": pairs,
    }
    out = REPO / "docs" / "measurements" / f"defi-red-liq-arithmetic-{date.today().isoformat()}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("events_2026", "fee_read")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
