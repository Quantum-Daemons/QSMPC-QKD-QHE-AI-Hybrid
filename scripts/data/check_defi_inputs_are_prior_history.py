#!/usr/bin/env python3
"""Are the DeFi QCNN's eight inputs the account's history before the borrow? Checked on the table itself (Round 86, batch 8).

`scripts/data/defi_qcnn_inputs.py` chose eight inputs of the archive's Borrow-to-Liquidated table on the fit window
alone: counts and sums of the account's deposits and liquidations. A model fitted on them is only meaningful if they
describe the account before the borrow; if they counted the account's later liquidations, the borrow's own outcome
would be among its inputs. FinSurvival (Green et al., arXiv:2507.14160, appendix B.4) describes these columns as the
account's transactions in the past; the archive's paper (Spadea and Seneviratne, arXiv:2604.14583) does not define
them. This checks the table, after the pre-registration and not part of its criteria:

  1. along each account's borrows in time, no count of liquidations or deposits ever decreases;
  2. for every liquidated borrow (liquidated at its timestamp plus `timeDiff`), the same account's first borrow after
     that liquidation counts at least one more liquidation than the borrow did, so the borrow's count did not already
     include its own outcome;
  3. at each account's first borrow, the share with a liquidation already counted is about the same whether or not the
     account is liquidated later in the table (a whole-history count would be positive for every account liquidated
     later).

Writes counts only, no account or other identifier: `docs/measurements/defi-inputs-prior-history-2026-11-27.json`.

Usage: python -I scripts/data/check_defi_inputs_are_prior_history.py --zip <the archive's data.zip>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "measurements" / "defi-inputs-prior-history-2026-11-27.json"
TABLE = "data/Borrow/Liquidated/data.csv"
ZIP_BYTES = 13_863_498_879
COUNTS = ("userLiquidationCount", "userDepositCount")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", type=Path, required=True)
    args = ap.parse_args()
    if args.zip.stat().st_size != ZIP_BYTES:
        raise SystemExit(f"{args.zip} is not the recorded archive's size: run nothing")
    import pandas as pd

    started = time.time()
    inputs = json.loads((REPO / "docs" / "measurements" / "defi-qcnn-aave-v3-2026-11-27.json").read_text())["inputs"]
    features = inputs["features"]
    with zipfile.ZipFile(args.zip) as z, z.open(TABLE) as f:
        df = pd.read_csv(f, usecols=["user", "timestamp", "timeDiff", "status", *features], low_memory=False)
    df = df.sort_values(["user", "timestamp"], kind="stable").reset_index(drop=True)
    g = df.groupby("user", sort=False)
    out: dict = {"table": TABLE, "rows": int(len(df)), "accounts": int(df["user"].nunique()),
                 "accounts_with_two_or_more_borrows": int((g.size() > 1).sum()), "features": features}

    # 1. No count decreases along an account's borrows in time.
    out["1_counts_along_each_account"] = {c: {"decreases": int((g[c].diff() < 0).sum()), "increases": int((g[c].diff() > 0).sum())}
                                          for c in COUNTS}

    # 2. The first borrow after each liquidation counts at least one more.
    users, ts = df["user"].to_numpy(), df["timestamp"].to_numpy()
    cnt = df["userLiquidationCount"].to_numpy()
    outcome = ts + df["timeDiff"].to_numpy()
    liquidated = df["status"].to_numpy() == 1
    sizes = g.size().to_numpy()
    first_row = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    span = dict(zip(df["user"].drop_duplicates().to_numpy(), zip(first_row, first_row + sizes, strict=True), strict=True))
    grew = not_grew = no_later = 0
    for i in np.flatnonzero(liquidated):
        lo, hi = span[users[i]]
        j = lo + int(np.searchsorted(ts[lo:hi], outcome[i], side="right"))
        if j >= hi:
            no_later += 1
        elif cnt[j] >= cnt[i] + 1:
            grew += 1
        else:
            not_grew += 1
    out["2_after_each_liquidation"] = {"liquidated_borrows": int(liquidated.sum()), "with_a_later_borrow": grew + not_grew,
                                       "count_one_or_more_higher": grew, "count_not_higher": not_grew,
                                       "no_later_borrow": no_later}

    # 3. At each account's first borrow, a liquidation already counted, by whether the account is liquidated later.
    first = (g.cumcount() == 0).to_numpy()
    later = g["status"].transform("max").to_numpy() == 1
    out["3_first_borrow_with_a_liquidation_counted"] = {
        "accounts_liquidated_later": {"accounts": int((first & later).sum()), "share": round(float(np.mean(cnt[first & later] > 0)), 6)},
        "accounts_never_liquidated_in_the_table": {"accounts": int((first & ~later).sum()),
                                                   "share": round(float(np.mean(cnt[first & ~later] > 0)), 6)},
    }
    a = out["1_counts_along_each_account"]
    out["prior_history"] = bool(all(v["decreases"] == 0 for v in a.values()) and not_grew == 0
                                and out["3_first_borrow_with_a_liquidation_counted"]["accounts_liquidated_later"]["share"] < 0.01)
    out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out["wall_clock_s"] = round(time.time() - started, 1)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
