"""The four splits of `references/telecom-nidd-clean-protocols-preregistration.md` (Round 86, batch 9).

Each function returns index arrays (fit, validation, test) into the loader's rows, in the file's order. Cuts in every
ordered group of n rows fall at floor(0.70 n) and floor(0.85 n); P1's validation is the rows from floor(0.85 n) on, per
class of the training station.
"""
from __future__ import annotations

import numpy as np


def cuts(n: int) -> tuple[int, int]:
    return int(np.floor(0.70 * n)), int(np.floor(0.85 * n))


def file_ids(file_label: np.ndarray) -> np.ndarray:
    """One id per contiguous run of the file-naming column (one value per capture file), in the file's order."""
    change = np.r_[True, file_label[1:] != file_label[:-1]]
    return np.cumsum(change) - 1


def p0(y: np.ndarray):
    """The shipped split: 75/25 stratified at seed 42, then a stratified 10 percent of the training part at seed 0."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, random_state=42, stratify=y)
    fi, vi = train_test_split(np.arange(len(tr)), test_size=0.10, random_state=0, stratify=y[tr])
    return tr[fi], tr[vi], te


def p1(y: np.ndarray, station: np.ndarray, train_station: int):
    """Cross-station: fit and validate on one station (per class, the last 15 percent in file order validates), test on
    every row of the other."""
    fit, val = [], []
    for c in np.unique(y[station == train_station]):
        rows = np.flatnonzero((station == train_station) & (y == c))   # already in file order
        _, b = cuts(len(rows))
        fit.append(rows[:b])
        val.append(rows[b:])
    test = np.flatnonzero(station != train_station)
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(val)), test


def p2(y: np.ndarray, fid: np.ndarray):
    """A time split in each capture file: per (file, class), in file order, 70 / 15 / 15."""
    parts = ([], [], [])
    for f in np.unique(fid):
        for c in np.unique(y[fid == f]):
            rows = np.flatnonzero((fid == f) & (y == c))
            a, b = cuts(len(rows))
            parts[0].append(rows[:a])
            parts[1].append(rows[a:b])
            parts[2].append(rows[b:])
    return tuple(np.sort(np.concatenate(p)) for p in parts)


def p3(y: np.ndarray, seq: np.ndarray):
    """LEAF's rule on this export: per class over the whole file, a stable sort by Seq (ties keep the file's order)."""
    parts = ([], [], [])
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        rows = rows[np.argsort(seq[rows], kind="stable")]
        a, b = cuts(len(rows))
        parts[0].append(rows[:a])
        parts[1].append(rows[a:b])
        parts[2].append(rows[b:])
    return tuple(np.sort(np.concatenate(p)) for p in parts)


def _self_test() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    fid = np.repeat(np.arange(10), n // 10)
    station = (fid >= 5).astype(int)
    y = np.where(rng.random(n) < 0.6, 0, 1 + fid % 5)          # Benign everywhere, one attack per file
    seq = np.concatenate([rng.permutation(n // 10) for _ in range(10)])
    for name, (f, v, t) in {"p0": p0(y), "p1a": p1(y, station, 0), "p1b": p1(y, station, 1),
                            "p2": p2(y, fid), "p3": p3(y, seq)}.items():
        allr = np.concatenate([f, v, t])
        assert len(np.intersect1d(f, v)) == len(np.intersect1d(f, t)) == len(np.intersect1d(v, t)) == 0, name
        if name.startswith("p1"):
            assert set(station[f]) == set(station[v]) == {int(name == "p1b")} and set(station[t]) == {int(name == "p1a")}
        else:
            assert len(allr) == n and len(np.unique(allr)) == n, name
    # P2 is time-ordered within each (file, class): every fit row precedes every test row of its group.
    f, v, t = p2(y, fid)
    for g in range(10):
        for c in np.unique(y[fid == g]):
            fr, tr = f[(fid[f] == g) & (y[f] == c)], t[(fid[t] == g) & (y[t] == c)]
            assert len(fr) == 0 or len(tr) == 0 or fr.max() < tr.min()
    print("self-test ok")


if __name__ == "__main__":
    _self_test()
