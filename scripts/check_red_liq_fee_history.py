#!/usr/bin/env python3
"""RED-LIQ-2026's same-asset liquidations against Aave's rule with the fee at each event's time, as the addendum written
after the pre-registered run states (`references/defi-red-liq-liquidation-arithmetic-preregistration-addendum-fee-history.md`,
Round 86, batch 13).

Reads the deposit's de-duplicated file in place, makes no chain call, and writes
`docs/measurements/defi-red-liq-fee-history-2026-11-28.json`: counts and ratios only, no address and no transaction hash.

    QSH_DATA_ROOT=... python scripts/check_red_liq_fee_history.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets as D  # noqa: E402

ADDENDUM = REPO / "references" / "defi-red-liq-liquidation-arithmetic-preregistration-addendum-fee-history.md"
ADDENDUM_SHA256 = "9cab3fa059b5f9aec0545bd013d5b8d8e2e487f1327a5162e55ae13e16f152f1"
RECORD = REPO / "docs" / "measurements" / "defi-red-liq-fee-history-2026-11-28.json"
#: AIP 534 executed 2026-10-24 14:16:59 UTC: WETH, WBTC and wstETH from 10 to 20 percent (the addendum's sources).
FEE_CHANGE = int(datetime(2026, 10, 24, 14, 16, 59, tzinfo=timezone.utc).timestamp())
RAISED = {"WETH", "WBTC", "wstETH"}
RESERVE_BONUS = {"WETH": Fraction(105, 100), "WBTC": Fraction(105, 100), "wstETH": Fraction(106, 100), "USDC": Fraction(1045, 1000)}
EMODE_1 = Fraction(101, 100)            # category 1: WETH and wstETH among its collateral, bonus 10,100 basis points
EMODE_1_COLLATERAL = {"WETH", "wstETH"}


def fee(asset: str, ts: int) -> Fraction:
    if asset in RAISED:
        return Fraction(1, 10) if ts < FEE_CHANGE else Fraction(1, 5)
    return Fraction(1, 5)               # USDC: 20 percent throughout


def main() -> int:
    got = hashlib.sha256(ADDENDUM.read_bytes()).hexdigest()
    if got != ADDENDUM_SHA256:
        raise SystemExit(f"{ADDENDUM.name} hashes to {got[:12]}, not the committed {ADDENDUM_SHA256[:12]}: run nothing")
    path = Path(D.data_root()) / "raw" / "red-liq-2026" / "red_liq_2026_v1_1_liquidations_unique.jsonl.gz"
    rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    events = [r for r in rows if r["iso"].startswith("2026") and r["debt_address"] == r["collateral_address"]
              and r["debt_asset"] in RESERVE_BONUS]
    per_asset: dict[str, dict] = {}
    for r in events:
        a, ts = r["debt_asset"], int(r["blockTs"])
        d = int(r["debt_covered_raw"])
        x = Fraction(int(r["collateral_seized_raw"]), d)
        f = fee(a, ts)
        values = {"reserve": RESERVE_BONUS[a] - f * (RESERVE_BONUS[a] - 1)}
        if a in EMODE_1_COLLATERAL:
            values["e-mode 1"] = EMODE_1 - f * (EMODE_1 - 1)
        hit = [k for k, v in values.items() if abs(x - v) <= Fraction(3, d)]
        e = per_asset.setdefault(a, {"events": 0, "on_the_rule": {}, "on_none": [],
                                     "rule_values": {k: float(v) for k, v in values.items()}, "fee_at_the_events": float(f)})
        e["events"] += 1
        if hit:
            e["on_the_rule"][hit[0]] = e["on_the_rule"].get(hit[0], 0) + 1
        else:
            e["on_none"].append(round(float(x), 6))
    record = {"addendum": {"file": str(ADDENDUM.relative_to(REPO)), "sha256": ADDENDUM_SHA256},
              "chosen_after_the_result": True,
              "deposit": {"doi": "10.5281/zenodo.23020998", "licence": "CC-BY-4.0",
                          "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
              "events_checked": len(events),
              "events_before_the_fee_change": sum(int(r["blockTs"]) < FEE_CHANGE for r in events),
              "per_asset": per_asset,
              "every_event_on_the_rule": all(not v["on_none"] for v in per_asset.values()),
              "not_covered": "bonus changes by Aave's risk stewards in 2026 beyond the cited payloads"}
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: (v["events"], v["on_the_rule"], v["on_none"]) for k, v in per_asset.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
