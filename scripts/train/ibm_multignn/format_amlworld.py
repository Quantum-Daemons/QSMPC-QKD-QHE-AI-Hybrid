"""IBM's `formatted_transactions.csv` for `IBM/Multi-GNN`, from this repository's AMLworld loader (Round 86).

The same encodings as the official `format_kaggle_files.py` (first-seen integer codes for accounts, currencies and payment formats,
seconds from the first day plus 10), with one change: the rows are in `load_sorted()` order, so a row's position is its position in
this project's temporal windows and `project_split.patch` can cut the fit, validation and test windows by position.

    .venv/bin/python scripts/train/ibm_multignn/format_amlworld.py <Multi-GNN data dir>/Small_HI
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from qsh.models import datasets  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    df = datasets.load_amlworld()                         # file order, as the official script reads it
    n = len(df)
    # First-seen integer codes, in the official script's order: per row, the receiving currency and then the paying one.
    cur = pd.factorize(np.ravel(np.column_stack([df.recv_currency.astype(str).to_numpy(), df.pay_currency.astype(str).to_numpy()])))[0]
    fmt = pd.factorize(df.payment_format.astype(str).to_numpy())[0]
    acc = pd.factorize(np.ravel(np.column_stack([(df.from_bank + df.from_account).to_numpy(), (df.to_bank + df.to_account).to_numpy()])))[0]
    start = df.timestamp.iloc[0].floor("D")
    ts = (df.timestamp - start).dt.total_seconds().to_numpy().astype(np.int64) + 10
    frame = pd.DataFrame({"EdgeID": np.arange(n), "from_id": acc[0::2], "to_id": acc[1::2], "Timestamp": ts,
                          "Amount Sent": df.amount_paid.to_numpy(), "Sent Currency": cur[1::2], "Amount Received": df.amount_received.to_numpy(),
                          "Received Currency": cur[0::2], "Payment Format": fmt, "Is Laundering": df.label.to_numpy()})
    # The order `load_sorted` produces (the same sort of the same frame), so positions are this project's temporal windows.
    order = df.sort_values("timestamp").index.to_numpy()
    frame = frame.iloc[order].copy()
    frame["EdgeID"] = np.arange(n)
    frame.to_csv(out / "formatted_transactions.csv", index=False, float_format="%.6f")
    print(f"rows {len(frame):,}, laundering {int(frame['Is Laundering'].sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
