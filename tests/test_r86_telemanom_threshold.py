"""Telemanom's thresholding, as `qsh.models.telemanom_threshold` implements it (Round 86).

The satellite forecaster's earlier rule set one threshold over the whole test window and "pruned" individual sorted
error values, which left it near the single largest error. These pin the properties of Telemanom's procedure that the
earlier rule lacked, on errors whose answer is known.
"""

from __future__ import annotations

import numpy as np

from qsh.models import telemanom_threshold as tt


def _series(n: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    values = np.sin(np.arange(n) / 50.0)[:, None] + 0.05 * rng.normal(size=(n, 1))
    errors = 0.02 * np.abs(rng.normal(size=(n, 1)))
    return values, errors


def test_a_planted_anomaly_is_flagged_and_nothing_else() -> None:
    values, errors = _series()
    errors[12_000:12_300] += 1.0
    _, flagged = tt.flag(errors, values)
    idx = np.flatnonzero(flagged)
    assert idx.size and idx.min() >= 12_000 - tt.DEFAULT.error_buffer and idx.max() <= 12_300 + 2 * tt.DEFAULT.error_buffer
    assert flagged[12_000:12_300].all()


def test_noise_alone_flags_nothing() -> None:
    values, errors = _series(seed=1)
    _, flagged = tt.flag(errors, values)
    assert not flagged.any()


def test_two_anomalies_of_different_size_are_both_kept() -> None:
    # The old rule pruned individual values from the top and stopped at the first small drop, so only the largest
    # survived. Telemanom prunes SEQUENCE maxima, and a clear gap between them keeps both.
    values, errors = _series(seed=2)
    errors[6_000:6_200] += 1.0
    errors[15_000:15_200] += 0.6
    _, flagged = tt.flag(errors, values)
    assert flagged[6_000:6_200].all() and flagged[15_000:15_200].all()


def test_any_channel_flags_the_timestep() -> None:
    values, errors = _series(seed=3)
    values = np.repeat(values, 2, axis=1)
    errors = np.repeat(errors, 2, axis=1)
    errors[9_000:9_250, 1] += 1.0
    per, flagged = tt.flag(errors, values)
    assert per[9_000:9_250, 1].all() and not per[:, 0].any()
    assert np.array_equal(flagged, per.any(axis=1))


def test_pruning_drops_a_near_tie_with_the_noise_and_keeps_a_clear_peak() -> None:
    e = np.zeros(1_000)
    e[100:110], e[500:510] = 1.0, 0.95
    # Two sequences whose maxima sit within p of the largest non-anomalous error are both pruned...
    assert tt._prune([(100, 109), (500, 509)], e, non_anom_max=0.9, p=0.13) == []
    # ...and a clear drop to the noise keeps both, which the reset after a large drop is for.
    assert tt._prune([(100, 109), (500, 509)], e, non_anom_max=0.1, p=0.13) == [(100, 109), (500, 509)]


def test_the_trainer_uses_it() -> None:
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "train" / "satellite_telemanom.py").read_text()
    assert "telemanom_threshold.flag(errors, vals, telemanom_threshold.TelemanomConfig(" in src
    assert "def dynamic_threshold" not in src and "esa_event_wise(t_times, flagged" in src
    assert 'ESA_ADB_SPLIT = "2007-01-01"' in src and "prepare(args.resample, args.split_at, args.validation_from" in src
    assert "preprocess(raw, resample, split_at=split_at, hold=hold)" in src
    # Both of ESA-ADB's readings for every run: anomalies alone (Supp. Table 9) and all events (Table 2).
    assert "for sel in (ANOMALIES, ALL_EVENTS)" in src and '"all_events": _all_events_brief(' in src


def test_the_strided_view_is_the_copied_windows() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train.satellite_telemanom import window_view, windows

    x = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    a, b = windows(x, 20, 1)
    c, d = window_view(x, 20)
    assert np.array_equal(a, c) and np.array_equal(b, d)


def test_the_forecaster_starts_from_kerass_initialisers() -> None:
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train.satellite_telemanom import HORIZON, TelemanomLSTM

    torch.manual_seed(0)
    net = TelemanomLSTM.build(6, horizon=HORIZON)
    TelemanomLSTM.keras_init(net)
    h = net.lstm.hidden_size
    w_hh = net.lstm.weight_hh_l0.detach()
    # Orthonormal columns, as Keras's orthogonal recurrent kernel is once transposed into PyTorch's layout.
    assert torch.allclose(w_hh.T @ w_hh, torch.eye(h), atol=1e-5)
    b = net.lstm.bias_ih_l1.detach()
    assert torch.all(b[h:2 * h] == 1) and torch.all(b[:h] == 0) and torch.all(b[2 * h:] == 0)
    assert torch.all(net.lstm.bias_hh_l0 == 0) and torch.all(net.head.bias == 0)
    limit = (6 / (6 + 4 * h)) ** 0.5           # glorot-uniform on the first layer's input kernel
    assert float(net.lstm.weight_ih_l0.detach().abs().max()) <= limit


# ---- ESA-ADB's own procedure ---------------------------------------------------------------------------------------
# The expectations below are what ESA-ADB's TelemanomThresholding (timeeval/metrics/thresholding.py in kplabs-pl/ESA-ADB,
# MIT, commit 67194d3) returned for these inputs when run on 2026-11-07; its code is not copied here. In 480 comparisons
# over 210 random series with planted anomalies, at floors 0, 0.007 and 0.05, this module's flags equalled that class's
# row for row.


def test_a_sequence_is_flagged_first_to_last_exclusive() -> None:
    n = 3000
    y = np.sin(np.arange(n) / 50.0)
    e_s = np.full(n, 0.01)
    e_s[2000:2011] = 1.0
    f = tt.flag_channel(e_s, y, tt.TelemanomConfig())
    # The sequence is 1900..2099; ESA-ADB writes it as [1900, 2099), so its last index is not flagged.
    assert f[1900:2099].all() and not f[2099] and not f[1899]


def test_every_batch_is_covered_up_to_the_end() -> None:
    # 2480 rows: the original Telemanom's last full window ends at 2380 and its final one starts its newest batch at 2410,
    # so a spike at 2390 was in no window's newest batch. ESA-ADB walks one window per batch to the end.
    n = 2480
    y = np.sin(np.arange(n) / 50.0)
    e_s = np.full(n, 0.01)
    e_s[2390:2396] = 1.0
    assert tt.flag_channel(e_s, y, tt.TelemanomConfig())[2390:2396].all()


def _random_series(seed: int, n: int):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(size=n)) * 0.01 + np.sin(np.arange(n) / rng.uniform(30, 400))
    yhat = y + rng.normal(scale=rng.uniform(0.005, 0.05), size=n)
    for _ in range(int(rng.integers(0, 6))):
        a = int(rng.integers(0, n - 50))
        b = a + int(rng.integers(1, 40))
        yhat[a:b] += rng.choice([-1, 1]) * rng.uniform(0.05, 2.0)
    return y, yhat


def test_it_reproduces_esa_adbs_flags_on_random_series() -> None:
    """Each digest is of ESA-ADB's own output, packed bits; these three cases separate the earlier inclusive end, the
    earlier window count, and the single points ESA-ADB's pruning leaves in place when a window holds no sequence."""
    import hashlib

    for seed, n, flagged, digest in (
            (10, 8769, 432, "7a32e0b5497504efbe2c378e0c56b04607c0c2a2359d6d091a1d47123c61ff31"),
            (32, 5983, 210, "f799f40549cf4145a2b4f2c9af23ee965d80e5f3ea4e7e6d4b1f1ad67d5e1556"),
            (57, 10353, 571, "83b2ff34aedb4fe2e9ba7821d6f782e21662c3b6eb75c2f7cfcd83bb4b6bbd52")):
        y, yhat = _random_series(seed, n)
        _, f = tt.flag(np.abs(yhat - y)[:, None], y[:, None], tt.TelemanomConfig())
        assert int(f.sum()) == flagged and hashlib.sha256(np.packbits(f).tobytes()).hexdigest() == digest, seed


def test_the_agreement_reading_chooses_k_by_the_floors_rule() -> None:
    """The third reading (Round 86): k channels must agree, k chosen as the middle of the best plateau."""
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train.satellite_telemanom import _choose_agreement

    times = pd.date_range("2010-01-01", periods=200, freq="30s")
    per = np.zeros((200, 4), dtype=bool)
    per[40:46, :3] = True               # an anomaly three channels flag
    per[120:123, 0] = True              # a false alarm one channel raises
    labels = pd.DataFrame([{"ID": "a", "StartTime": times[40], "EndTime": times[45], "Category": "Anomaly"}])
    k, table = _choose_agreement(per, times, labels, (times[0], times[-1]))
    by_k = {r["k"]: (r["tp"], r["fp"], r["fn"]) for r in table}
    assert by_k[1] == (1, 1, 0) and by_k[2] == by_k[3] == (1, 0, 0) and by_k[4] == (0, 0, 1)
    assert k == 3, "two and three tie at the best score; the middle of [2, 3] by the floor's rule is 3"
