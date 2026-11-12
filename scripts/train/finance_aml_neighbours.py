#!/usr/bin/env python
"""The capped transaction neighbourhood the compact finance GNN reads (Round 86), computed by the holder of the transactions in the clear.

WHAT IT IS. For a transaction u -> v, the slots are the D most recent edges in each direction at u and at v, up to the end of the transaction's batch of 2,048 and at most a fixed number of
minutes past it (`finance_aml_fan.batch_reach`), the horizon of the published MLP's per-transaction features. The transaction itself is never one of its own slots. A slot describes its connecting edge: the amount, how long before
the end of the batch it happened, whether it crossed currencies or stayed in one bank, its payment format, whether its counterparty is the transaction's other endpoint, and
whether that counterparty also appears among the other endpoint's recent edges, the two graph signals that the GFP paper's scatter-gather and cycle patterns turn on. An empty
slot is all zeros, and because the per-neighbour map has no bias an empty slot contributes exactly zero downstream, so the tree can be padded to a fixed public shape.

WHY NOT NEIGHBOURS' OWN WINDOWED STATISTICS. They were tried, as sums the holder could compute in the clear, and validation preferred the model without them: mean validation
AUPRC 0.5262 against 0.5072 to 0.5211 with them, on three seeds each at the same learning rate and weight decay, and it was less seed-dependent. `references/REFERENCES.md` records the comparison.

Endpoint statistics, in the same windows as the MLP's fan features, are the first layer's input for the endpoints themselves.

No label is read here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from finance_aml_fan import BATCH_WINDOW, WINDOWS

#: Slots kept per direction at each endpoint, most recent first.
D = 4
#: Edges per direction searched for the "also at the other endpoint" flag: wider than D, so a counterparty a few edges back still counts.
SEARCH = 8
CHUNK = 200_000
N_STATS = 16


def statistic_names() -> list[str]:
    return [f"{side}_{kind}_{w}" for side in ("in", "out") for w in WINDOWS for kind in ("log_count", "log_sum_log_amount")]


def _table(acct: torch.Tensor, nbr: torch.Tensor, mins: torch.Tensor, amt: torch.Tensor) -> dict:
    """One direction's events sorted by account, then time, then position, with a key that a binary search can use."""
    n = len(acct)
    order = torch.tensor(np.lexsort((np.arange(n), mins.cpu().numpy(), acct.cpu().numpy())), device=acct.device)
    a, t = acct[order], mins[order]
    return {"key": (a << 21) + t, "acct": a, "nbr": nbr[order], "edge": order,
            "cum": torch.cat([torch.zeros(1, device=acct.device, dtype=torch.float64), amt[order].double().cumsum(0)])}


def _recent(r: torch.Tensor, at: torch.Tensor, me: torch.Tensor, tab: dict, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions of up to `depth` most recent events of account r with time at or before `at`, leaving out the transaction itself, and which of them exist."""
    end = torch.searchsorted(tab["key"], (r << 21) + at, right=True)
    pos = end.unsqueeze(1) - torch.arange(1, depth + 2, device=r.device)
    exists = pos >= 0
    p = pos.clamp(min=0)
    exists &= tab["acct"][p] == r.unsqueeze(1)
    exists &= tab["edge"][p] != me.unsqueeze(1)
    first = (~exists).long().argsort(dim=1, stable=True)
    return p.gather(1, first)[:, :depth], exists.gather(1, first)[:, :depth]


def _statistics(acct: torch.Tensor, at: torch.Tensor, tables: tuple[dict, dict], windows: torch.Tensor) -> torch.Tensor:
    """log1p count and log1p summed log amount, in and out, per window ending at `at`."""
    out: list[torch.Tensor] = []
    for tab in tables:
        end = torch.searchsorted(tab["key"], (acct << 21) + at, right=True)
        for w in windows:
            start = torch.searchsorted(tab["key"], (acct << 21) + (at - w).clamp(min=0), right=False)
            out += [torch.log1p((end - start).float()), torch.log1p((tab["cum"][end] - tab["cum"][start]).float())]
    return torch.stack(out, -1)


@torch.no_grad()
def neighbourhoods(df: pd.DataFrame, device: str, horizon_minutes: int | None = None) -> dict:
    """Every row's endpoints and slots, on the CPU: `endpoints` (n, 2, 16), `slots` (n, 4D, 13), `filled` (n, 4D) and `edges` (n, 4D), the slot's row in `df`, for the page's figure."""
    n = len(df)
    ids = pd.factorize(pd.concat([df.from_bank + df.from_account, df.to_bank + df.to_account], ignore_index=True))[0]
    src = torch.tensor(ids[:n].astype(np.int64), device=device)
    dst = torch.tensor(ids[n:].astype(np.int64), device=device)
    mins = torch.tensor(((df.timestamp - df.timestamp.iloc[0]).dt.total_seconds() / 60).to_numpy().astype(np.int64), device=device)
    amt = torch.tensor(np.log1p(np.clip(df.amount_paid.to_numpy(float), 0, None)), dtype=torch.float32, device=device)
    format_ids, formats = pd.factorize(df.payment_format.astype(str))
    fmt = torch.tensor(format_ids.astype(np.int64), device=device)
    cross = torch.tensor((df.pay_currency.astype(str).to_numpy() != df.recv_currency.astype(str).to_numpy()).astype(np.float32), device=device)
    same_bank = torch.tensor((df.from_bank.to_numpy() == df.to_bank.to_numpy()).astype(np.float32), device=device)
    idx = torch.arange(n, device=device)
    batch_time = mins[torch.minimum(torch.full((n,), n - 1, device=device), (idx // BATCH_WINDOW + 1) * BATCH_WINDOW - 1)]
    if horizon_minutes is not None:
        batch_time = torch.minimum(batch_time, mins + horizon_minutes)
    incoming, outgoing = _table(dst, src, mins, amt), _table(src, dst, mins, amt)
    windows = torch.tensor(WINDOWS, device=device)

    n_slot_features = 6 + len(formats)
    endpoints = torch.zeros((n, 2, N_STATS), dtype=torch.float32)
    slots = torch.zeros((n, 4 * D, n_slot_features), dtype=torch.float32)
    filled = torch.zeros((n, 4 * D), dtype=torch.bool)
    edges = torch.zeros((n, 4 * D), dtype=torch.int32)
    for a in range(0, n, CHUNK):
        i = idx[a:a + CHUNK]
        u, v, at = src[i], dst[i], batch_time[i]
        lists = [(_recent(u, at, i, incoming, SEARCH), incoming), (_recent(u, at, i, outgoing, SEARCH), outgoing),
                 (_recent(v, at, i, incoming, SEARCH), incoming), (_recent(v, at, i, outgoing, SEARCH), outgoing)]
        partners = [tab["nbr"][p] for (p, _), tab in lists]
        around_u, ok_u = torch.cat(partners[:2], 1), torch.cat([lists[0][0][1], lists[1][0][1]], 1)
        around_v, ok_v = torch.cat(partners[2:], 1), torch.cat([lists[2][0][1], lists[3][0][1]], 1)
        also_at_v = ((around_u.unsqueeze(2) == around_v.unsqueeze(1)) & ok_v.unsqueeze(1)).any(2)
        also_at_u = ((around_v.unsqueeze(2) == around_u.unsqueeze(1)) & ok_u.unsqueeze(1)).any(2)
        row_slots, row_filled, row_edges = [], [], []
        for k, ((p, ok), tab) in enumerate(lists):
            p, ok = p[:, :D], ok[:, :D]
            who = partners[k][:, :D]
            other = v if k < 2 else u
            e = tab["edge"][p]
            flag = (also_at_v if k < 2 else also_at_u)[:, (k % 2) * SEARCH:(k % 2) * SEARCH + D]
            connecting = torch.stack([amt[e], torch.log1p((at.unsqueeze(1) - mins[e]).clamp(min=0).float()), cross[e], same_bank[e],
                                      (who == other.unsqueeze(1)).float(), flag.float()], -1)
            row_slots.append(torch.cat([connecting, torch.nn.functional.one_hot(fmt[e], len(formats)).float()], -1) * ok.unsqueeze(-1).float())
            row_filled.append(ok)
            row_edges.append(e)
        slots[a:a + CHUNK] = torch.cat(row_slots, 1).cpu()
        filled[a:a + CHUNK] = torch.cat(row_filled, 1).cpu()
        edges[a:a + CHUNK] = torch.cat(row_edges, 1).int().cpu()
        endpoints[a:a + CHUNK, 0] = _statistics(u, at, (incoming, outgoing), windows).cpu()
        endpoints[a:a + CHUNK, 1] = _statistics(v, at, (incoming, outgoing), windows).cpu()
    return {"endpoints": endpoints, "slots": slots, "filled": filled, "edges": edges, "formats": [str(f) for f in formats]}
