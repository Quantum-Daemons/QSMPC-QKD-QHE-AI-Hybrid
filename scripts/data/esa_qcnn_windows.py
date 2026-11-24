#!/usr/bin/env python3
"""ESA-ADB Mission1's lightweight channels as 64-step windows with eight features each, for the satellite QCNN.

references/satellite-qcnn-tn-preregistration.md, section B. The grid is the shipped Telemanom student's
(`prepare` in scripts/train/satellite_telemanom.py: channels 41 to 46, a zero-order hold at 30 s, ESA-ADB's split at
2007-01-01 with validation from 2006-10-01, and a z-score per channel from the unlabelled training rows before the
validation months). The grid is cut into non-overlapping windows of 64 steps (32 minutes) from its first row; a window
is positive when any of its rows overlaps an Anomaly interval (`Prepared.mask`: ESA-ADB's lightweight positives, not
rare nominal events or communication gaps). A window that crosses the validation or the test boundary is dropped, so
no window mixes two parts.

The eight features of a window, on the z-scored grid: the standard deviation of each of the six channels; the largest
one-step change of any channel; the largest absolute value of any channel.

    python scripts/data/esa_qcnn_windows.py      # writes $QSH_DATA_ROOT/artifacts/esa_qcnn_windows_w64.npz and .json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "train"))

WINDOW = 64
FEATURES = tuple([f"std_channel_{c}" for c in range(41, 47)] + ["max_abs_step", "max_abs_value"])


def window_features(z: np.ndarray) -> np.ndarray:
    """(windows, 64, 6) z-scored blocks -> (windows, 8)."""
    std = z.std(axis=1)
    step = np.abs(np.diff(z, axis=1)).max(axis=(1, 2))
    peak = np.abs(z).max(axis=(1, 2))
    return np.column_stack([std, step, peak]).astype(np.float64)


def main() -> int:
    from qsh.models import datasets
    from satellite_telemanom import ESA_ADB_SPLIT, ESA_ADB_VALIDATION, prepare

    t0 = time.time()
    prep = prepare("30s", ESA_ADB_SPLIT, ESA_ADB_VALIDATION, hold="last")
    z = prep.zscore(prep.raw_plain).astype(np.float32)
    rows = (len(z) // WINDOW) * WINDOW
    starts = np.arange(0, rows, WINDOW)
    ends = starts + WINDOW                                   # exclusive
    part = np.full(len(starts), -1, dtype=np.int8)           # 0 fit, 1 validation, 2 test, -1 crosses a boundary
    part[ends <= prep.vstart] = 0
    part[(starts >= prep.vstart) & (ends <= prep.cut)] = 1
    part[starts >= prep.cut] = 2
    keep = part >= 0
    blocks = z[:rows].reshape(-1, WINDOW, z.shape[1])[keep]
    label = prep.mask[:rows].reshape(-1, WINDOW).any(axis=1)[keep].astype(np.int8)
    X = window_features(blocks)
    part = part[keep]
    start_time = np.asarray(prep.df.index[starts[keep]].astype("int64"))
    out = datasets.artifacts("esa_qcnn_windows_w64.npz")
    np.savez_compressed(out, X=X, y=label, part=part, start_ns=start_time)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    counts = {name: {"windows": int((part == k).sum()), "positive": int(label[part == k].sum())}
              for k, name in enumerate(("fit", "validation", "test"))}
    meta = {"about": "ESA-ADB Mission1 lightweight channels 41-46 (CC BY 3.0 IGO), 64-step windows of the shipped "
                     "Telemanom grid, eight features each; references/satellite-qcnn-tn-preregistration.md section B",
            "features": list(FEATURES), "window_steps": WINDOW, "grid": "30s, zero-order hold",
            "split": {"validation_from": ESA_ADB_VALIDATION, "test_from": ESA_ADB_SPLIT},
            "dropped_crossing_a_boundary": int((~keep).sum()), "parts": counts, "sha256": digest,
            "seconds": round(time.time() - t0, 1)}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
