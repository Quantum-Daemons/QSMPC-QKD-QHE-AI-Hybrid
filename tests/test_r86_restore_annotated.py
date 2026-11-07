"""ESA-ADB's restoration of annotated samples, as `restore_annotated` in the satellite trainer applies it (Round 86).

ESA-ADB's Mission1 preparation (notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py in kplabs-pl/ESA-ADB, MIT,
read at commit 67194d3 on 2026-11-07) writes an Anomaly or Rare Event sample that the zero-order hold dropped back onto
the grid. The first tests pin each condition of its rule; the last runs a transcription of its pandas loop on random
telemetry and labels and requires the same grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train.satellite_telemanom import restore_annotated  # noqa: E402

RULE = "30s"
T0 = pd.Timestamp("2010-01-01")


def _sec(s: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(seconds=s)


def _labels(*rows):
    return pd.DataFrame([{"ID": f"id_{i}", "Channel": ch, "StartTime": _sec(a), "EndTime": _sec(b), "Category": c}
                         for i, (ch, a, b, c) in enumerate(rows)])


def _grid_and_hold(series: pd.Series, n: int = 4):
    grid = pd.date_range(T0, periods=n, freq=RULE)
    held = series.reindex(series.index.union(grid)).ffill().reindex(grid).to_numpy(dtype=np.float64)
    return grid, held[:, None]


def _run(times, values, labels):
    s = pd.Series(np.asarray(values, dtype=np.float32), index=[_sec(t) for t in times])
    grid, vals = _grid_and_hold(s)
    changed = restore_annotated(vals, grid, ["c"], {"c": s}, labels, RULE)
    return vals[:, 0], changed


def test_an_annotated_sample_the_hold_dropped_is_written_onto_the_next_grid_point() -> None:
    vals, changed = _run([0, 10, 20], [1, 9, 2], _labels(("c", 10, 10, "Anomaly")))
    assert vals[1] == 9 and changed.tolist() == [False, True, False, False]


def test_a_rare_event_is_restored_too() -> None:
    vals, _ = _run([0, 10, 20], [1, 9, 2], _labels(("c", 10, 10, "Rare Event")))
    assert vals[1] == 9


def test_nothing_is_restored_when_the_bins_last_sample_is_itself_annotated() -> None:
    # The hold already carries it: the last sample of [0, 30) is the anomaly.
    vals, changed = _run([0, 10, 20], [1, 2, 9], _labels(("c", 20, 20, "Anomaly")))
    assert vals[1] == 9 and not changed.any()


def test_a_communication_gap_is_not_restored() -> None:
    vals, changed = _run([0, 10, 20], [1, 9, 2], _labels(("c", 10, 10, "Communication Gap")))
    assert vals[1] == 2 and not changed.any()


def test_a_bin_with_one_sample_is_left_alone() -> None:
    vals, changed = _run([10, 40], [9, 2], _labels(("c", 10, 10, "Anomaly")))
    assert not changed.any() and vals[1] == 9      # the hold itself carries the one sample


def test_a_label_on_another_channel_restores_nothing_here() -> None:
    vals, changed = _run([0, 10, 20], [1, 9, 2], _labels(("other", 10, 10, "Anomaly")))
    assert vals[1] == 2 and not changed.any()


def _esa_reference(times: pd.DatetimeIndex, values: np.ndarray, labels: pd.DataFrame, rule: str) -> pd.Series:
    """ESA-ADB's own steps for one channel, transcribed: label each raw sample, hold, restore."""
    param_df = pd.DataFrame({"value": values, "label": np.uint8(0)}, index=times)
    for _, row in labels.iterrows():
        if row["Channel"] != "c":
            continue
        code = 1 if row["Category"] == "Anomaly" else 2 if row["Category"] == "Rare Event" else 3
        param_df.loc[row["StartTime"]:row["EndTime"], "label"] = code
    first = pd.Timestamp(param_df.index[0]).floor(freq=rule)
    last = pd.Timestamp(param_df.index[-1]).ceil(freq=rule)
    out = param_df.reindex(pd.date_range(first, last, freq=rule), method="ffill")
    out.iloc[0] = param_df.iloc[0]
    for timestamp, group in param_df.groupby(pd.Grouper(freq=rule)).indices.items():
        if len(group) <= 1:
            continue
        org = param_df.iloc[group]
        if org.label.values[-1] != 0:
            continue
        is_annotated = (org.label == 1) | (org.label == 2)
        if is_annotated.any():
            out.loc[timestamp + pd.Timedelta(rule)] = org[is_annotated].iloc[-1]
    return out["value"]


def test_it_matches_esa_adbs_loop_on_random_telemetry() -> None:
    rng = np.random.default_rng(7)
    # Irregular sampling, sometimes several samples per bin and sometimes none, over two hours.
    t = np.sort(rng.choice(np.arange(0, 7200, 0.5), size=900, replace=False))
    times = pd.DatetimeIndex([_sec(x) for x in t])
    values = rng.normal(size=len(t)).astype(np.float32)
    cats = ["Anomaly", "Rare Event", "Communication Gap"]
    rows = []
    for _ in range(40):
        a = float(rng.uniform(0, 7100))
        rows.append(("c" if rng.random() < 0.85 else "other", a, a + float(rng.choice([0.0, 3.0, 12.0, 45.0])),
                     cats[int(rng.integers(0, 3))]))
    labels = _labels(*rows)

    ref = _esa_reference(times, values, labels, RULE)
    s = pd.Series(values, index=times)
    grid = pd.date_range(times[0].ceil(RULE), times[-1].floor(RULE), freq=RULE)
    held = s.reindex(s.index.union(grid)).ffill().reindex(grid).to_numpy(dtype=np.float64)[:, None]
    changed = restore_annotated(held, grid, ["c"], {"c": s}, labels, RULE)
    assert changed.sum() >= 5, "the random case should exercise the restoration"
    np.testing.assert_array_equal(held[:, 0], ref.reindex(grid).to_numpy(dtype=np.float64))
