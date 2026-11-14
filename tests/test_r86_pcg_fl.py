"""The heart-sound federation (Round 86): the preparation and the analysis's arithmetic, without the data or a GPU.

- `scripts/data/prepare_physionet2016.py` on a synthetic zip in the Challenge's layout: every recording named by a
  source's REFERENCE-SQI.csv lands at its offset with its label and quality, band-passed and standardised; a member the
  reference does not name is never read, and a missing one or a recording at another rate is refused;
- the mel filterbank's triangles, the window starts (the last window ends at the end), the per-recording mean of the
  windows' logits, the AUROC against scikit-learn and the Challenge's balanced accuracy;
- the card shows the record and credits PhysioNet under ODC-By 1.0.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
PREP = REPO / "scripts" / "data" / "prepare_physionet2016.py"
TRAIN = REPO / "scripts" / "train" / "medical_pcg_fl.py"


@pytest.fixture(scope="module")
def pcg():
    spec = importlib.util.spec_from_file_location("medical_pcg_fl_under_test", TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wav(seconds: float, rate: int = 2000, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    x = (3000 * np.sin(2 * np.pi * 60 * t) + 500 * rng.normal(size=len(t))).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return buf.getvalue()


def _zip(tmp: Path, *, missing: bool = False, bad_rate: bool = False) -> Path:
    path = tmp / "challenge.zip"
    root = "classification-of-heart-sound-recordings-1.0.0"
    with zipfile.ZipFile(path, "w") as z:
        for k, s in enumerate("abcdef"):
            rows = [(f"{s}0001", "1", "1"), (f"{s}0002", "-1", "0")]
            z.writestr(f"{root}/training-{s}/REFERENCE-SQI.csv", "".join(",".join(r) + "\n" for r in rows))
            for j, (rec, *_rest) in enumerate(rows):
                if missing and s == "c" and j == 1:
                    continue
                rate = 4000 if bad_rate and s == "d" else 2000
                z.writestr(f"{root}/training-{s}/{rec}.wav", _wav(3 + j + k / 10, rate=rate, seed=k * 2 + j))
            z.writestr(f"{root}/training-{s}/unnamed.wav", _wav(1.0))
    return path


def _run(zip_path: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(PREP), "--zip", str(zip_path), "--out", str(out)], capture_output=True, text=True)


def test_every_named_recording_lands_with_its_label_quality_and_offset(tmp_path: Path) -> None:
    out = tmp_path / "out"
    r = _run(_zip(tmp_path), out)
    assert r.returncode == 0, r.stderr
    with open(out / "index.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [x["record"] for x in rows] == [f"{s}000{j}" for s in "abcdef" for j in (1, 2)], "only the named recordings, in order"
    assert [x["abnormal"] for x in rows[:2]] == ["1", "0"] and [x["good_quality"] for x in rows[:2]] == ["1", "0"]
    sig = np.load(out / "signals.npy")
    off = [int(x["offset"]) for x in rows]
    length = [int(x["length"]) for x in rows]
    assert off == list(np.cumsum([0] + length[:-1])) and sum(length) == len(sig)
    seg = sig[off[0]:off[0] + length[0]].astype(np.float64)
    assert abs(seg.mean()) < 0.01 and abs(seg.std() - 1) < 0.01, "each recording is standardised"


def test_a_missing_recording_or_another_rate_is_refused(tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    (tmp_path / "r").mkdir()
    assert _run(_zip(tmp_path / "m", missing=True), tmp_path / "m" / "out").returncode != 0
    r = _run(_zip(tmp_path / "r", bad_rate=True), tmp_path / "r" / "out")
    assert r.returncode != 0 and "expected 2000 Hz" in r.stderr


def test_the_mel_filters_are_triangles_inside_the_band(pcg) -> None:
    fb = pcg.mel_filterbank()
    assert fb.shape == (pcg.N_MELS, pcg.N_FFT // 2 + 1)
    assert (fb >= 0).all() and (fb <= 1).all() and (fb.sum(axis=1) > 0).all()
    freqs = np.linspace(0, pcg.RATE / 2, pcg.N_FFT // 2 + 1)
    support = freqs[fb.sum(axis=0) > 1e-9]
    assert support.min() >= pcg.F_MIN and support.max() < pcg.F_MAX, "the filters stay inside the band"
    peaks = fb.argmax(axis=1)
    assert (np.diff(peaks) >= 0).all(), "the filters rise in frequency"


def test_band_mean_removal_leaves_each_band_centred(pcg) -> None:
    import torch
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, pcg.WINDOW, generator=g) * torch.linspace(0.5, 2.0, pcg.WINDOW)
    plain = pcg.Features().to("cpu")
    plain.fb, plain.win = plain.fb.cpu(), plain.win.cpu()
    centred = pcg.Features(band_mean=True)
    m0, m1 = plain(x), centred(x)
    assert m1.shape == m0.shape
    assert m1.mean(dim=3).abs().max() < 1e-4, "each band's mean over the window is removed"
    assert m0.mean(dim=3).abs().max() > 0.05, "without the option the bands keep their levels"
    assert abs(float(m1.std(dim=(1, 2, 3)).mean()) - 1) < 1e-3, "the window is still standardised"


def test_the_windows_cover_the_recording(pcg) -> None:
    assert pcg.window_starts(100, 400, 50).tolist() == [0]
    assert pcg.window_starts(1000, 400, 200).tolist() == [0, 200, 400, 600]
    s = pcg.window_starts(1050, 400, 200)
    assert s[-1] == 650 and s[-1] + 400 == 1050


def test_the_per_recording_score_and_the_metrics(pcg) -> None:
    logits = np.array([1.0, 3.0, -2.0, 0.0, 2.0])
    recs = np.array([7, 7, 9, 9, 9])
    assert pcg.per_record_mean(logits, recs, np.array([9, 7])).tolist() == [0.0, 2.0]
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.3).astype(int)
    p = np.round(rng.normal(y, 1.0), 1)
    r = pcg.auc_boot(y, p, n_boot=200)
    assert r["auroc"] == pytest.approx(roc_auc_score(y, p)) and r["lo"] <= r["auroc"] <= r["hi"]
    assert pcg.macc(np.array([1, 0]), np.array([1.0, -1.0]))["macc"] == 1.0
    assert pcg.macc(np.array([1, 0]), np.array([-1.0, 1.0]))["macc"] == 0.0


def test_the_card_shows_the_record_and_credits_physionet() -> None:
    record = json.loads((REPO / "docs" / "measurements" / "pcg-fl-2026-11-14.json").read_text())
    text = (REPO / "web" / "public" / "data" / "medical" / "pcg_fl.json").read_text()
    card = json.loads(text)
    for key in ("summary", "sources", "attribution", "recipe", "training_recording_counts"):
        assert card[key] == record[key], f"the card's {key} is not the record's"
    att = card["attribution"]
    assert att["licence"] == "Open Data Commons Attribution License v1.0" and att["licence_url"].startswith("https://opendatacommons.org/")
    assert "2181-2213" in att["citation"] and att["download"] == "https://physionet.org/content/challenge-2016/1.0.0/"
    assert "/home/" not in text and "/home/" not in json.dumps(record), "a local path reached a shipped file"
    assert set(card["summary"]) == set("abcdef") | {"random 20% of recordings"}
    assert card["training_recording_counts"]["released_zip"] == 3240
    band = record["band_mean"]
    assert card["band_mean"] == {"recipe": band["recipe"], "summary": band["summary"]}, "the card's band-mean figures are the record's"
    assert band["recipe"]["features"]["band_mean_removed"] is True and not record["recipe"]["features"].get("band_mean_removed")
    assert {r["seed"] for r in band["runs"]} == {1, 2, 3} and set(band["summary"]) == set(card["summary"])
