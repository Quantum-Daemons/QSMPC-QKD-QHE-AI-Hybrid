"""`qsh.models.esa_scores` scores the way ESA-ADB's `ESAScores` does (Round 86), on cases whose answers follow from its rules.

ESA-ADB's own test file (tests/metrics/test_metrics.py in kplabs-pl/ESA-ADB, MIT) checks the same rules; these restate
the ones the satellite score depends on: per-ID detection, no false positive for a rare nominal event, the closed-open
event intervals, point anomalies, and the time-based true-negative-rate correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsh.models.esa_scores import esa_event_wise, events_from_series

T0 = pd.Timestamp("2010-01-01")
STEP = pd.Timedelta("30s")
TIMES = pd.date_range(T0, periods=100, freq="30s")


def _labels(*rows):
    return pd.DataFrame([{"ID": i, "StartTime": T0 + a * STEP, "EndTime": T0 + b * STEP, "Category": c}
                         for i, a, b, c in rows])


def _flags(*runs):
    f = np.zeros(len(TIMES), dtype=int)
    for a, b in runs:
        f[a:b] = 1
    return f


SELECT = {"Category": ["Anomaly"]}


def test_events_are_closed_open_runs_and_closed_at_the_end() -> None:
    ev = events_from_series(TIMES, _flags((3, 5), (98, 100)))
    ns = TIMES.as_unit("ns").asi8
    assert ev[0] == (ns[3], ns[5], False)          # [t3, t5): the run's end is the next timestamp, excluded
    assert ev[1] == (ns[98], ns[99], True)         # reaching the end: closed [t98, t99]


def test_a_detected_anomaly_is_one_true_positive_however_many_events_hit_it() -> None:
    s = esa_event_wise(TIMES, _flags((10, 12), (14, 16)), _labels(("a", 9, 20, "Anomaly")), select=SELECT)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0) and s["redundant_detections"] == 1


def test_a_rare_nominal_event_is_neither_credited_nor_a_false_alarm() -> None:
    lab = _labels(("a", 40, 45, "Anomaly"), ("r", 60, 62, "Rare Event"))
    s = esa_event_wise(TIMES, _flags((60, 61)), lab, select=SELECT)
    assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 1)


def test_an_event_over_nothing_is_a_false_positive() -> None:
    s = esa_event_wise(TIMES, _flags((70, 72)), _labels(("a", 40, 45, "Anomaly")), select=SELECT)
    assert (s["tp"], s["fp"], s["fn"]) == (0, 1, 1)


def test_a_point_anomaly_on_a_grid_timestamp_is_detected_and_one_just_before_the_run_is_not() -> None:
    s = esa_event_wise(TIMES, _flags((5, 6)), _labels(("p", 5, 5, "Anomaly")), select=SELECT)
    assert (s["tp"], s["fn"]) == (1, 0)
    # The run [t6, t7) does not reach back to t5.
    s2 = esa_event_wise(TIMES, _flags((6, 7)), _labels(("p", 5, 5, "Anomaly")), select=SELECT)
    assert (s2["tp"], s2["fn"]) == (0, 1)


def test_precision_is_corrected_by_the_time_based_true_negative_rate() -> None:
    # Full range t0..t99 = 99 steps; the anomaly [t20, t30] is 10 steps, so nominal time is 89 steps. The prediction
    # [t25, t35) spends t30..t35 = 5 steps in nominal time: TNR = 1 - 5/89, and with TP 1, FP 0 that is the precision.
    s = esa_event_wise(TIMES, _flags((25, 35)), _labels(("a", 20, 30, "Anomaly")), select=SELECT)
    assert s["tp"] == 1 and s["fp"] == 0
    assert s["tnr"] == pytest.approx(1 - 5 / 89)
    assert s["precision"] == pytest.approx(1 - 5 / 89) and s["recall"] == 1.0
    p, r = s["precision"], s["recall"]
    assert s["f_0.50"] == pytest.approx(1.25 * p * r / (0.25 * p + r))


# ---- ESA-ADB's own cases ---------------------------------------------------------------------------------------------
# From tests/metrics/test_metrics.py::TestAffiliationMetrics::test_time_aware in kplabs-pl/ESA-ADB (MIT, Copyright (c)
# the ESA-ADB authors; read at commit 67194d3 on 2026-11-07): the label table, the predictions and their EW_* results.
# Only the event-wise keys are compared; the affiliation-based ones are not implemented here.

_D = pd.to_datetime
_FULL = (_D("2015-01-01"), _D("2015-01-15"))
_Y_TRUE = pd.DataFrame(
    [["id_0", _D("2015-01-01"), _D("2015-01-02"), "Rare Event"],
     ["id_1", _D("2015-01-04"), _D("2015-01-05"), "Anomaly"],
     ["id_1", _D("2015-01-06"), _D("2015-01-07"), "Anomaly"],
     ["id_2", _D("2015-01-09"), _D("2015-01-10"), "Communication Gap"],
     ["id_3", _D("2015-01-12"), _D("2015-01-14"), "Rare Event"]],
    columns=["ID", "StartTime", "EndTime", "Category"])
_RARE_AND_ANOMALIES = {"Category": ["Rare Event", "Anomaly"]}
_ANOMALIES = {"Category": ["Anomaly"]}


def _series(*points):
    return [_D(t) for t, _ in points], [v for _, v in points]


_CASES = [
    (_series(("2015-01-01", 1), ("2015-01-02", 0), ("2015-01-04", 1), ("2015-01-05", 0), ("2015-01-09", 1),
             ("2015-01-10", 0)),
     (0.9090909090909091, 1.0, 0.6666666666666666, 1.0), (1.0, 1.0, 1.0, 1.0)),
    (_series(("2015-01-01", 1), ("2015-01-02", 0), ("2015-01-04", 1), ("2015-01-05", 0), ("2015-01-06", 1),
             ("2015-01-07", 0), ("2015-01-09", 1), ("2015-01-10", 0)),
     (0.9090909090909091, 1.0, 0.6666666666666666, 1.0), (1.0, 1.0, 1.0, 1.0)),
    (_series(("2015-01-01", 1), ("2015-01-02", 0), ("2015-01-04", 1), ("2015-01-05", 0), ("2015-01-06", 1),
             ("2015-01-07", 0), ("2015-01-09", 1), ("2015-01-10", 0), ("2015-01-11", 1), ("2015-01-12", 0)),
     (0.5982905982905982, 0.5833333333333333, 0.6666666666666666, 1.0),
     (0.49295774647887325, 0.4375, 1.0, 1.0)),
    (_series(("2015-01-01", 0), ("2015-01-04", 1), ("2015-01-06 12:00", 0), ("2015-01-06 14:00", 1),
             ("2015-01-08", 0)),
     (0.6000000000000001, 0.75, 0.3333333333333333, 0.5), (0.7894736842105263, 0.75, 1.0, 0.5)),
]


@pytest.mark.parametrize(("series", "rare_and_anomalies", "anomalies"), _CASES)
def test_esa_adbs_own_cases(series, rare_and_anomalies, anomalies) -> None:
    times, flags = series
    for select, (f, p, r, alarming) in ((_RARE_AND_ANOMALIES, rare_and_anomalies), (_ANOMALIES, anomalies)):
        s = esa_event_wise(times, flags, _Y_TRUE, select=select, full_range=_FULL)
        assert s["f_0.50"] == pytest.approx(f, abs=1e-12)
        assert s["precision"] == pytest.approx(p, abs=1e-12)
        assert s["recall"] == pytest.approx(r, abs=1e-12)
        assert s["alarming_precision"] == pytest.approx(alarming, abs=1e-12)


def test_timestamps_of_different_resolutions_are_compared_as_instants() -> None:
    # pandas 3 parses the label file at microsecond resolution while a grid can be at nanoseconds; integer timestamps
    # taken at each one's own resolution are a thousandfold apart (Round 86).
    times_ns = TIMES.as_unit("ns")
    lab = _labels(("a", 40, 45, "Anomaly"))
    lab = lab.assign(StartTime=lab["StartTime"].astype("datetime64[us]"), EndTime=lab["EndTime"].astype("datetime64[us]"))
    s = esa_event_wise(times_ns, _flags((41, 43)), lab, select=SELECT, full_range=(times_ns[0], times_ns[-1]))
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0)
