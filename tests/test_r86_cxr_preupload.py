"""The chest X-ray model's pre-upload checks (Round 86, `scripts/data/check_cxr_preupload.py`), without the data or a GPU.

What `docs/measurements/cxr-preupload-2026-11-14.json` rests on, held here on small synthetic inputs:

- an archive is hashed in the same pass that decodes it: the digest equals the file's, however tarfile reads;
- one archive's pass: every PNG counted by colour mode and size, each test image resized by all seven filters, and the
  reference filter compared with the prepared set (a planted difference is counted);
- the manifest comparison names a file whose bytes or digest differ, and one that is missing;
- bilinear without a low-pass equals PyTorch's interpolate without antialias, pixel for pixel after rounding;
- ECE over equal-count bins is zero for scores equal to the outcome rates and equals a constant offset; the temperature
  fit recovers a known temperature; the recalibration slope is about 1 for scores drawn from their own model;
- the group statistics, age bands, co-occurring findings and pixel flags count what they say.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "data" / "check_cxr_preupload.py"


@pytest.fixture(scope="module")
def ck():
    spec = importlib.util.spec_from_file_location("check_cxr_preupload_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _png(arr: np.ndarray, mode: str = "L") -> bytes:
    from PIL import Image

    im = Image.fromarray(arr)
    if mode != "L":
        im = im.convert(mode)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def _archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(f"images/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_hashing_reader_digests_the_whole_file(ck, tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "images_001.tar.gz"
    _archive(p, {f"{i:08d}_000.png": _png(rng.integers(0, 256, (40, 40), dtype=np.uint8)) for i in range(5)})
    with open(p, "rb") as raw:
        reader = ck.HashingReader(raw)
        buf = io.BufferedReader(reader, 1 << 10)
        with tarfile.open(fileobj=buf, mode="r|gz") as tf:
            names = [m.name for m in tf]
        while buf.read(1 << 10):
            pass
    assert len(names) == 5
    assert reader.n == p.stat().st_size
    assert reader.sha.hexdigest() == hashlib.sha256(p.read_bytes()).hexdigest()
    assert ck.sha256_of(p) == (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())


def test_one_archive_counts_resizes_and_compares(ck, tmp_path):
    rng = np.random.default_rng(1)
    imgs = {f"0000000{i}_000.png": rng.integers(0, 256, (64, 64), dtype=np.uint8) for i in range(4)}
    data = {n: _png(a, "RGBA" if i == 0 else "L") for i, (n, a) in enumerate(imgs.items())}
    tgz = tmp_path / "images_001.tar.gz"
    _archive(tgz, data)
    from PIL import Image

    names = sorted(imgs)
    prep = np.zeros((len(names), ck.SIZE, ck.SIZE), np.uint8)
    for i, n in enumerate(names):
        prep[i] = np.asarray(Image.fromarray(imgs[n]).resize((ck.SIZE, ck.SIZE), Image.Resampling.BILINEAR))
    prep[3] ^= 1                                   # the prepared copy of one test image differs
    np.save(tmp_path / "prep.npy", prep)
    test_row = {names[1]: 0, names[3]: 1}
    res = ck.one_archive((str(tgz), str(tmp_path / "prep.npy"), {n: i for i, n in enumerate(names)}, test_row))
    assert res["members"] == 4
    assert res["modes"] == {"RGBA": 1, "L": 3}
    assert res["sizes"] == {"64x64": 4}
    assert res["test_written"] == 2 and sorted(res["rows"]) == [0, 1]
    assert res["arrays"].shape == (2, len(ck.FILTERS), ck.SIZE, ck.SIZE)
    assert res["reference_mismatched"] == 1
    assert (res["bytes"], res["sha256"]) == ck.sha256_of(tgz)
    k = ck.FILTERS.index(ck.REFERENCE)
    got = dict(zip(res["rows"], res["arrays"][:, k], strict=True))
    assert np.array_equal(got[0], prep[1])


def test_manifest_comparison_names_what_differs(ck):
    manifest = {"a": {"bytes": 3, "sha256": "x"}, "b": {"bytes": 4, "sha256": "y"}, "c": {"bytes": 5, "sha256": "z"}}
    found = {"a": (3, "x"), "b": (4, "not y")}
    r = ck.compare_manifest(manifest, found)
    assert r == {"files": 3, "bytes": 12, "matching": 1, "differing_or_missing": ["b", "c"]}


def test_bilinear_without_lowpass_is_torch_interpolate_without_antialias(ck):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (97, 131), dtype=np.uint8)
    for size in (23, 40, 224):
        ours = ck.bilinear_no_lowpass(a, size)
        ref = torch.nn.functional.interpolate(torch.from_numpy(a).double()[None, None], size=(size, size), mode="bilinear",
                                              align_corners=False, antialias=False)[0, 0].numpy()
        assert np.abs(ours.astype(int) - np.clip(np.rint(ref), 0, 255).astype(int)).max() <= 1


def test_ece_is_zero_when_scores_are_the_rates_and_equals_an_offset(ck):
    ks = np.arange(10, 190, 12)                    # 15 rates, each a whole number of 200 outcomes
    p = np.repeat(ks / 200, 200)
    y = np.concatenate([np.r_[np.ones(k), np.zeros(200 - k)] for k in ks])
    assert ck.ece_equal_count(y, p) == pytest.approx(0.0, abs=1e-3)
    assert ck.ece_equal_count(y, np.clip(p + 0.03, 0, 1)) == pytest.approx(0.03, abs=1e-3)
    rel = ck.reliability(y, p)
    assert sum(r["n"] for r in rel) == len(p) and len(rel) == ck.ECE_BINS


def test_temperature_and_recalibration_line(ck):
    rng = np.random.default_rng(3)
    z = rng.normal(0, 2.5, 60_000)
    y = (rng.random(z.size) < ck.expit(z)).astype(int)
    assert ck.fit_temperature(y, 2.0 * z) == pytest.approx(2.0, rel=0.05)
    line = ck.recalibration_line(y, z)
    assert line["slope"] == pytest.approx(1.0, abs=0.05) and line["intercept"] == pytest.approx(0.0, abs=0.05)
    assert ck.expit(np.array([-800.0, 0.0, 800.0])).tolist() == [0.0, 0.5, 1.0]


def test_calibration_of_returns_both_maps_scored_on_test(ck):
    rng = np.random.default_rng(4)
    zv, zt = rng.normal(0, 3, 4000), rng.normal(0, 3, 4000)
    yv = (rng.random(zv.size) < ck.expit(zv / 2)).astype(int)
    yt = (rng.random(zt.size) < ck.expit(zt / 2)).astype(int)
    r = ck.calibration_of(yv, zv, yt, zt, rng.random(zt.size) < 0.5)
    assert r["temperature"] == pytest.approx(2.0, rel=0.15)
    assert r["test_after_temperature"]["ece"] < r["test"]["ece"]
    assert r["test_after_platt"]["log_loss"] < r["test"]["log_loss"]
    assert r["test_by_view"]["AP"]["n"] + r["test_by_view"]["PA"]["n"] == r["test"]["n"]
    corp = r["test"]["corp"]
    assert corp["mcb"] - corp["dsc"] + corp["unc"] == pytest.approx(r["test"]["brier"], abs=1e-12)


def test_group_statistics_age_bands_and_cooccurrence(ck):
    rows = [{"view": "AP", "sex": "F", "age": 17, "patient": 1, "follow_up": 0},
            {"view": "AP", "sex": "F", "age": 45, "patient": 1, "follow_up": 1},
            {"view": "PA", "sex": "M", "age": 85, "patient": 2, "follow_up": 0},
            {"view": "PA", "sex": "M", "age": 60, "patient": 3, "follow_up": 2}]
    y = np.array([1.0, 0.0, 1.0, 0.0])
    s = ck.group_stats(rows, y)
    assert s["images"] == 4 and s["patients"] == 3 and s["ap_share"] == 0.5 and s["female_share"] == 0.5
    assert s["by_view"]["AP"] == {"images": 2, "effusion": 0.5}
    assert s["by_age_band"]["80+"] == {"images": 1, "effusion": 1.0}
    assert s["images_per_patient"] == {"median": 1.0, "max": 2}
    assert [ck.age_band(a) for a in (0, 17, 18, 79, 80, 120)] == ["0-17", "0-17", "18-39", "60-79", "80+", "80+"]
    entries = [{"labels": ["Effusion"]}, {"labels": ["Effusion", "Infiltration"]}, {"labels": ["Infiltration", "Effusion", "Mass"]},
               {"labels": ["No Finding"]}]
    c = ck.cooccurring(entries)
    assert c["images"] == 3 and c["alone"] == 1
    assert c["top"][0] == {"finding": "Infiltration", "share": 2 / 3}


def test_pixel_flags_and_separation(ck):
    mean = np.array([5.0, 100.0, 250.0, 120.0])
    sd = np.array([3.0, 50.0, 40.0, 60.0])
    corr = np.array([0.9, 0.1, 0.8, -0.3])
    f = ck.pixel_flags(mean, sd, corr)
    assert (f["near_uniform"], f["very_dark"], f["very_bright"], f["correlation_below_0.2"], f["correlation_negative"]) == (1, 1, 1, 2, 1)
    assert ck.pixel_flags(mean, sd, np.array([0.9, 0.1, -0.6, -0.3]))["correlation_below_minus_0.5"] == 1
    y = np.array([0, 0, 1, 1])
    assert ck.separation(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert ck.separation(np.array([0.9, 0.8, 0.2, 0.1]), y) == 1.0
    assert Counter(ck.FILTERS)[ck.REFERENCE] == 1 and len(ck.FILTERS) == 7
