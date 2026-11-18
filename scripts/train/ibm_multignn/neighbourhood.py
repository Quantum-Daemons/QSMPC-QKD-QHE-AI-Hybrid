"""What IBM's Multi-GNN read for each transfer it scored in a browser sample (Round 86).

The owner's choices of 2026-11-15: the finance page's IBM tab draws each scored transfer's neighbourhood from the
sample the page scores and counts the transfers in it that lie past the transfer's horizon, the horizon of the page's
own finance models: the end of its batch of 2,048 rows, and at most the reach after it (17 minutes on HI-Small, the
furthest any fit-window row sees to the end of its batch; `finance_aml_fan.batch_reach`). `export_for_browser.py`
writes these counts into the sample; the page re-derives them from the sample's own arrays
(`web/src/lib/graphNeighbourhood.ts`) and reports a mismatch. Both follow this definition:

- a transfer is a position in the time-ordered file; the sample can hold a position more than once, and it counts once;
- the payer's and the payee's transfers are the sample's edges at that account, out or in, other than the scored one;
- read: the transfers at the accounts within one hop of the payer or the payee, which is what two rounds of message
  passing in both directions take in;
- past its horizon: in a later batch of 2,048, or more than the reach after the scored transfer; the causal protocol
  reads none, IBM's loader samples them;
- later, within its horizon: after the scored transfer in time and not past its horizon;
- furthest later: how far after the scored transfer, in seconds, the latest transfer read is (0 when none is later).
"""

from __future__ import annotations

import numpy as np

BATCH_ROWS = 2048


def fit_reach(seconds, n_fit: int, rows: int = BATCH_ROWS) -> int:
    """The furthest any of the first `n_fit` rows sees to the end of its batch of `rows`, in whole minutes, as seconds:
    `finance_aml_fan.batch_reach` on the same rows, and the bound of the causal patch's batches."""
    ts = np.asarray(seconds, dtype=np.float64)
    n = len(ts)
    end = np.minimum(n - 1, (np.arange(n) // rows + 1) * rows - 1)
    return 60 * int((ts[end] - ts)[:n_fit].max() // 60)


def seed_fields(edge_index, position, seconds, seed_at, reach_s: float, batch_rows: int = BATCH_ROWS) -> list[dict]:
    """The counts above for each seed. `edge_index` is the sample's forward edges (2 x E, local account indices),
    `position` and `seconds` each edge's row in the time-ordered file and its time, `seed_at` the scored edges."""
    ei = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
    src, dst = ei[0], ei[1]
    pos = np.asarray(position, dtype=np.int64)
    sec = np.asarray(seconds, dtype=np.float64)
    if not (src.size == pos.size == sec.size):
        raise ValueError("every edge needs a position and a time")
    batch = pos // batch_rows
    out = []
    for s in seed_at:
        u, v, p, t = int(src[s]), int(dst[s]), int(pos[s]), float(sec[s])
        other = pos != p

        def count(mask: np.ndarray, other: np.ndarray = other) -> int:
            return int(np.unique(pos[mask & other]).size)

        near = np.isin(src, (u, v)) | np.isin(dst, (u, v))
        hop1 = np.union1d(src[near], dst[near])
        read = np.isin(src, hop1) | np.isin(dst, hop1)
        past = read & ((batch > p // batch_rows) | (sec > t + reach_s))
        ahead = sec[read & other & (sec > t)] - t
        out.append({
            "payer_out": count(src == u), "payer_in": count(dst == u),
            "payee_out": count(src == v), "payee_in": count(dst == v),
            "read": count(read),
            "later_within_horizon": count(read & ~past & (sec > t)),
            "past_horizon": count(past),
            "furthest_later_s": float(ahead.max()) if ahead.size else 0.0,
        })
    return out
